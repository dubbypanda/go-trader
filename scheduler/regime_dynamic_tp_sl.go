package main

import (
	"fmt"
	"math"
	"strings"
)

const (
	dynamicCloseStrategyName       = "tiered_tp_atr_live_regime_dynamic"
	defaultRegimeConfirmCycles     = 2
	dynamicCloseParamConfirmCycles = "regime_confirm_cycles"
)

// strategyUsesDynamicRegimeClose reports whether the strategy's close ref is
// tiered_tp_atr_live_regime_dynamic with the unified per-regime block (#843).
func strategyUsesDynamicRegimeClose(sc StrategyConfig) bool {
	for _, ref := range sc.closeRefs() {
		if strings.ToLower(strings.TrimSpace(ref.Name)) == dynamicCloseStrategyName {
			return closeParamsAreUnifiedRegime(ref.Params)
		}
	}
	return false
}

// dynamicCloseConfirmCycles returns regime_confirm_cycles from the close ref
// (default 2). Values < 1 are treated as 1.
func dynamicCloseConfirmCycles(sc StrategyConfig) int {
	for _, ref := range sc.closeRefs() {
		if strings.ToLower(strings.TrimSpace(ref.Name)) != dynamicCloseStrategyName {
			continue
		}
		if ref.Params == nil {
			return defaultRegimeConfirmCycles
		}
		v, ok := ref.Params[dynamicCloseParamConfirmCycles]
		if !ok {
			return defaultRegimeConfirmCycles
		}
		n, err := floatFromAnyChecked(v)
		if err != nil || n < 1 {
			return 1
		}
		return int(n)
	}
	return defaultRegimeConfirmCycles
}

// strategyCurrentATRRegime returns the live ATR-window regime label from the
// strategy's most recent check output, mirroring strategyCurrentDirectionalRegime.
func strategyCurrentATRRegime(stratState *StrategyState, sc StrategyConfig) string {
	if stratState == nil {
		return ""
	}
	key := normalizeRegimeWindowKey(sc.RegimeATRWindow)
	if key != "" && key != regimeWindowDefaultKey && len(stratState.RegimeWindows) > 0 {
		if label, ok := stratState.RegimeWindows[key]; ok && strings.TrimSpace(label) != "" {
			return strings.TrimSpace(label)
		}
	}
	return strings.TrimSpace(stratState.Regime)
}

// dynamicCloseATRRegimeLabel returns the ATR regime label used to resolve SL,
// TP tiers, and sl_after for dynamic close strategies. While a pending label
// is counting toward confirm cycles, the last applied label stays in effect.
func dynamicCloseATRRegimeLabel(pos *Position, sc StrategyConfig) string {
	if pos == nil {
		return ""
	}
	if label := strings.TrimSpace(pos.RegimeAppliedLabel); label != "" {
		return label
	}
	return positionATRRegimeLabel(pos, sc)
}

// advanceDynamicCloseRegime updates confirm-cycle state on pos and reports
// whether the applied ATR regime label changed this cycle. Call under the
// state lock while a position is open.
func advanceDynamicCloseRegime(pos *Position, stratState *StrategyState, sc StrategyConfig) (regimeChanged bool) {
	if pos == nil || stratState == nil || !strategyUsesDynamicRegimeClose(sc) {
		return false
	}
	current := strategyCurrentATRRegime(stratState, sc)
	if current == "" {
		return false
	}
	confirm := dynamicCloseConfirmCycles(sc)
	if pos.RegimeAppliedLabel == "" {
		pos.RegimeAppliedLabel = current
		pos.RegimePendingLabel = ""
		pos.RegimePendingCount = 0
		return false
	}
	if current == pos.RegimeAppliedLabel {
		pos.RegimePendingLabel = ""
		pos.RegimePendingCount = 0
		return false
	}
	if current == pos.RegimePendingLabel {
		pos.RegimePendingCount++
	} else {
		pos.RegimePendingLabel = current
		pos.RegimePendingCount = 1
	}
	if pos.RegimePendingCount < confirm {
		return false
	}
	old := pos.RegimeAppliedLabel
	pos.RegimeAppliedLabel = current
	pos.RegimePendingLabel = ""
	pos.RegimePendingCount = 0
	return old != current
}

// protectionATRRegimeLabel selects the ATR regime for on-chain protection and
// post-TP SL resolution: dynamic close uses the applied label; everything else
// uses the stamped position label.
func protectionATRRegimeLabel(pos *Position, sc StrategyConfig) string {
	if strategyUsesDynamicRegimeClose(sc) {
		return dynamicCloseATRRegimeLabel(pos, sc)
	}
	return positionATRRegimeLabel(pos, sc)
}

// atrTierTriggerPx computes the reduce-only TP trigger for one tier.
func atrTierTriggerPx(side string, avgCost, entryATR, atrMult float64) float64 {
	if avgCost <= 0 || entryATR <= 0 || atrMult <= 0 {
		return 0
	}
	offset := atrMult * entryATR
	switch strings.ToLower(strings.TrimSpace(side)) {
	case "long":
		return avgCost + offset
	case "short":
		return avgCost - offset
	default:
		return 0
	}
}

// atrStopLossTriggerPx computes the fixed ATR stop trigger for one side.
func atrStopLossTriggerPx(side string, avgCost, entryATR, slMult float64) float64 {
	if avgCost <= 0 || entryATR <= 0 || slMult <= 0 {
		return 0
	}
	switch strings.ToLower(strings.TrimSpace(side)) {
	case "long":
		return avgCost - slMult*entryATR
	case "short":
		return avgCost + slMult*entryATR
	default:
		return 0
	}
}

// triggerPxMoveExceedsMinPct reports whether replacing oldPx with newPx clears
// the trailing-stop min-move debounce (same 0.5% default as HL trailing SL).
func triggerPxMoveExceedsMinPct(oldPx, newPx, minMovePct float64) bool {
	if newPx <= 0 {
		return false
	}
	if oldPx <= 0 {
		return true
	}
	if minMovePct <= 0 {
		return true
	}
	movePct := math.Abs(newPx-oldPx) / oldPx * 100.0
	return movePct >= minMovePct
}

// dynamicProtectionForceReplace computes per-tier TP and SL force-replace flags
// after the applied regime changes. Filled tiers (armed with OID 0) are skipped.
func dynamicProtectionForceReplace(
	sc StrategyConfig,
	pos *Position,
	plan hlProtectionPlan,
	oldRegime string,
	regimeChanged bool,
) (forceSL bool, forceTP []bool) {
	if !regimeChanged || pos == nil {
		return false, nil
	}
	minMove := effectiveTrailingStopMinMovePct(sc)

	if plan.StopLossATRMult > 0 {
		oldSL := 0.0
		if v, ok := unifiedCloseStopLossATR(sc, oldRegime); ok {
			oldSL = atrStopLossTriggerPx(plan.Side, plan.AvgCost, plan.EntryATR, v)
		}
		newSL := atrStopLossTriggerPx(plan.Side, plan.AvgCost, plan.EntryATR, plan.StopLossATRMult)
		cur := pos.StopLossTriggerPx
		if cur > 0 {
			oldSL = cur
		}
		forceSL = triggerPxMoveExceedsMinPct(oldSL, newSL, minMove)
	}

	oldTiers := strategyTPTiersForRegime(sc, oldRegime)
	if len(plan.Tiers) == 0 {
		return forceSL, nil
	}
	forceTP = make([]bool, len(plan.Tiers))
	for i := range plan.Tiers {
		if i < len(pos.TPArmedTiers) && pos.TPArmedTiers[i] && (i >= len(pos.TPOIDs) || pos.TPOIDs[i] == 0) {
			continue
		}
		if i < len(pos.TPOIDs) && pos.TPOIDs[i] == 0 && (i >= len(pos.TPArmedTiers) || !pos.TPArmedTiers[i]) {
			// Never armed — sync will place fresh.
			continue
		}
		oldMult := 0.0
		if i < len(oldTiers) {
			oldMult = oldTiers[i].Multiple
		}
		newPx := atrTierTriggerPx(plan.Side, plan.AvgCost, plan.EntryATR, plan.Tiers[i].Multiple)
		oldPx := atrTierTriggerPx(plan.Side, plan.AvgCost, plan.EntryATR, oldMult)
		if i < len(pos.TPOIDs) && pos.TPOIDs[i] > 0 {
			forceTP[i] = triggerPxMoveExceedsMinPct(oldPx, newPx, minMove)
		}
	}
	return forceSL, forceTP
}

// validateDynamicRegimeClose validates tiered_tp_atr_live_regime_dynamic params.
func validateDynamicRegimeClose(params map[string]interface{}, labels []string, ctxLabel string) []string {
	var errs []string
	if params == nil {
		errs = append(errs, fmt.Sprintf("%s: params required", ctxLabel))
		return errs
	}
	for k := range params {
		if k != regimeClassifierKey && k != "atr_source" && k != dynamicCloseParamConfirmCycles {
			errs = append(errs, fmt.Sprintf("%s: unknown param %q (allowed: trend_regime, atr_source, regime_confirm_cycles)", ctxLabel, k))
		}
	}
	if v, ok := params[dynamicCloseParamConfirmCycles]; ok {
		if f, err := floatFromAnyChecked(v); err != nil || f < 1 {
			errs = append(errs, fmt.Sprintf("%s.%s: must be >= 1", ctxLabel, dynamicCloseParamConfirmCycles))
		}
	}
	errs = append(errs, validateUnifiedRegimeClose(params, labels, ctxLabel)...)
	return errs
}
