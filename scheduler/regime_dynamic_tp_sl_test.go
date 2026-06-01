package main

import (
	"testing"
)

func TestStrategyCurrentATRRegime(t *testing.T) {
	sc := StrategyConfig{RegimeATRWindow: "medium"}
	st := &StrategyState{
		RegimeWindows: map[string]string{"medium": "ranging_quiet"},
	}
	if got := strategyCurrentATRRegime(st, sc); got != "ranging_quiet" {
		t.Fatalf("strategyCurrentATRRegime = %q, want ranging_quiet", got)
	}
}

func TestAdvanceDynamicCloseRegime_ConfirmCycles(t *testing.T) {
	sc := StrategyConfig{
		CloseStrategy: &StrategyRef{
			Name: dynamicCloseStrategyName,
			Params: map[string]interface{}{
				"regime_confirm_cycles": 2,
				"trend_regime": map[string]interface{}{
					"trending_up": map[string]interface{}{
						"stop_loss_atr": 1.5,
						"tp_tiers": []interface{}{
							map[string]interface{}{"atr_multiple": 2.0, "close_fraction": 0.5},
							map[string]interface{}{"atr_multiple": 4.0, "close_fraction": 1.0},
						},
					},
					"ranging": map[string]interface{}{
						"stop_loss_atr": 0.8,
						"tp_tiers": []interface{}{
							map[string]interface{}{"atr_multiple": 1.0, "close_fraction": 0.5},
							map[string]interface{}{"atr_multiple": 2.0, "close_fraction": 1.0},
						},
					},
				},
			},
		},
	}
	pos := &Position{RegimeAppliedLabel: "trending_up"}
	st := &StrategyState{
		Regime:        "ranging",
		RegimeWindows: map[string]string{"medium": "ranging"},
	}
	sc.RegimeATRWindow = "medium"

	if changed := advanceDynamicCloseRegime(pos, st, sc); changed {
		t.Fatal("first pending cycle should not apply")
	}
	if pos.RegimeAppliedLabel != "trending_up" {
		t.Fatalf("applied=%q want trending_up", pos.RegimeAppliedLabel)
	}
	if changed := advanceDynamicCloseRegime(pos, st, sc); !changed {
		t.Fatal("second pending cycle should apply ranging")
	}
	if pos.RegimeAppliedLabel != "ranging" {
		t.Fatalf("applied=%q want ranging", pos.RegimeAppliedLabel)
	}
}

func TestTriggerPxMoveExceedsMinPct(t *testing.T) {
	if !triggerPxMoveExceedsMinPct(100, 100.6, 0.5) {
		t.Fatal("0.6% move should exceed 0.5% threshold")
	}
	if triggerPxMoveExceedsMinPct(100, 100.4, 0.5) {
		t.Fatal("0.4% move should not exceed 0.5% threshold")
	}
}

func TestDynamicCloseInSuppressionSet(t *testing.T) {
	sc := StrategyConfig{
		Type:          "perps",
		Platform:      "hyperliquid",
		Args:          []string{"hold", "ETH", "1h", "--mode=live"},
		CloseStrategy: &StrategyRef{Name: dynamicCloseStrategyName},
	}
	if !strategyUsesTieredTPATRClose(sc) {
		t.Fatal("dynamic close must be recognized by strategyUsesTieredTPATRClose")
	}
	if !closeStrategySuppressedByOnChainProtection(sc) {
		t.Fatal("dynamic close must be suppressed on HL live when on-chain TPs active")
	}
}
