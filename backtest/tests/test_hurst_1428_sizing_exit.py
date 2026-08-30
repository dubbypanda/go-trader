import json
import math
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "research")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import hurst_1428_sizing_exit as study
import hurst_1427_change_sort as study1427
import hurst_1426_two_sided_sort as study1426
import hurst_1424_gate_resolution as study1424
import hurst_1410_gate_calibration as study1410
import hurst_gate as parity

_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_GO_GATE = os.path.join(_REPO, "scheduler", "hurst_gate.go")
CONTRACT = os.path.join(os.path.dirname(study._DEFAULT_REPORT_OUT),
                        "hurst_gate_calibration.md")


# --- the shipped sizing form -------------------------------------------------

def test_the_study_calls_the_parity_module_rather_than_a_copy():
    gate = parity.HurstGate({"enabled": True,
                             "mode": parity.HURST_GATE_MODE_SIZE,
                             "size_floor": study.SHIPPED_SIZE_FLOOR})
    for h in (0.0, 0.2, 0.33, 0.42, 0.5, 0.58, 0.65, 0.8, 1.0):
        assert study.shipped_size_multiplier(h) == gate.size_multiplier(h)


def test_the_shipped_form_never_exceeds_one():
    for i in range(0, 201):
        h = i / 200.0
        assert study.shipped_size_multiplier(h) <= study.SHIPPED_SIZE_CEILING


def test_the_shipped_form_floors_at_size_floor():
    for i in range(0, 201):
        h = i / 200.0
        assert study.shipped_size_multiplier(h) >= study.SHIPPED_SIZE_FLOOR


def test_the_shipped_form_reaches_the_ceiling_exactly_at_the_span():
    span = study.SHIPPED_SIZE_SPAN
    assert study.shipped_size_multiplier(0.5 + span) == 1.0
    assert study.shipped_size_multiplier(0.5 - span) == 1.0
    assert study.shipped_size_multiplier(0.5 + span / 2) == pytest.approx(0.5)


def test_undefined_h_sizes_at_exactly_one():
    assert study.shipped_size_multiplier(None) == 1.0
    assert study.shipped_size_multiplier(float("nan")) == 1.0
    assert study.shipped_size_multiplier(float("inf")) == 1.0
    assert study.SHIPPED_NAN_MULTIPLIER == study1410.SIZING_NAN_MULTIPLIER


def test_the_go_source_still_implements_the_form_this_study_scores():
    src = open(_GO_GATE).read()
    assert f"hurstSizeSpan = {study.SHIPPED_SIZE_SPAN}" in src
    assert f"hurstDefaultSizeFloor = {study.SHIPPED_SIZE_FLOOR}" in src
    body = src.split("func hurstSizeMultiplier(", 1)[1].split("\n}", 1)[0]
    assert "math.Abs(h-0.5) / hurstSizeSpan" in body.replace(" ", " ")
    assert "m > 1.0" in body and "m = 1.0" in body
    assert "m < floor" in body and "m = floor" in body
    assert "return 1.0" in body


def test_the_study_does_not_score_the_1410_curve():
    assert study1410.SIZING_CLAMP_HI > study.SHIPPED_SIZE_CEILING
    h = 0.9
    assert (study1410.size_multiplier(h, study1410.SENSE_HIGH,
                                      study1410.SIZING_GAINS[0])
            != study.shipped_size_multiplier(h))


# --- the sizing validity contrast --------------------------------------------

def _sizing_trade(h, eff=0.1, pnl=1.0, day=0, window="2021",
                  symbol="BTC/USDT", timeframe="1h"):
    entry = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
    return {
        "strategy": "momentum", "exchange": "binanceus", "symbol": symbol,
        "base_symbol": symbol, "timeframe": timeframe, "window": window,
        "cohort": study.COHORT_PRIMARY,
        "entry_date": str(entry), "entry_ns": int(entry.value),
        "pnl_pct_net": float(pnl),
        "efficiency": None if eff is None else float(eff),
        "adx": None, "h": {512: h},
        "size_mult": {512: study.shipped_size_multiplier(h)},
    }


def test_the_sizing_kept_side_is_the_tail_not_an_interval():
    rows = [_sizing_trade(h, day=i) for i, h in enumerate(
        [0.50, 0.52, 0.58, 0.66, 0.34, 0.44])]
    _keep, _v, _r, mults, suppressed = study.sizing_rows(rows, 512)
    kept = [h for h, s in zip([0.50, 0.52, 0.58, 0.66, 0.34, 0.44], suppressed)
            if not s]
    assert kept == [0.66, 0.34]
    for h in kept:
        assert abs(h - 0.5) >= study.SIZE_CEILING_BAND
    assert all(m == 1.0 for m, s in zip(mults, suppressed) if not s)
    assert all(m < 1.0 for m, s in zip(mults, suppressed) if s)


def test_sizing_rows_drop_a_row_with_no_primary_target():
    rows = [_sizing_trade(0.7, eff=None, day=0), _sizing_trade(0.7, day=1)]
    keep, _v, _r, _m, _s = study.sizing_rows(rows, 512)
    assert len(keep) == 1


def test_a_nan_h_row_sizes_at_one_and_lands_on_the_kept_side():
    keep, _v, _r, mults, suppressed = study.sizing_rows(
        [_sizing_trade(None)], 512)
    assert len(keep) == 1
    assert mults == [1.0]
    assert suppressed == [False]


# --- the exit form -----------------------------------------------------------

def test_exit_buckets_read_the_family_sense():
    assert study.exit_bucket(0.7, study1410.SENSE_HIGH) == "persistent"
    assert study.exit_bucket(0.7, study1410.SENSE_LOW) == "anti_persistent"
    assert study.exit_bucket(0.3, study1410.SENSE_HIGH) == "anti_persistent"
    assert study.exit_bucket(0.3, study1410.SENSE_LOW) == "persistent"
    assert study.exit_bucket(0.5, study1410.SENSE_HIGH) == "neutral"
    assert study.exit_bucket(0.5, study1410.SENSE_LOW) == "neutral"


def test_an_undefined_h_is_its_own_exit_bucket_and_never_neutral():
    for sense in (study1410.SENSE_HIGH, study1410.SENSE_LOW):
        assert study.exit_bucket(None, sense) == study.BUCKET_NAN
        assert study.exit_bucket(float("nan"), sense) == study.BUCKET_NAN
    assert study.BUCKET_NAN in study.EXIT_BUCKETS
    assert study.BUCKET_NAN not in study.EXIT_SCALED_BUCKETS


def test_the_exit_edges_are_1410s_committed_landmarks():
    committed = "".join(study1410.BUCKETS)
    assert f"{study.EXIT_ANTIPERSISTENT_EDGE:.2f}" in committed
    assert f"{study.EXIT_PERSISTENT_EDGE:.2f}" in committed


def test_the_exit_ladder_moves_only_the_two_scaled_buckets():
    base = study.EXIT_BASE_TRAIL_ATR_MULT
    assert study.exit_trail_mult("neutral") == base
    assert study.exit_trail_mult(study.BUCKET_NAN) == base
    assert study.exit_trail_mult("persistent") > base
    assert study.exit_trail_mult("anti_persistent") < base
    assert study.exit_trail_mult("anti_persistent") > 0


def test_the_exit_arm_scores_exactly_one_hurst_window():
    assert f"W{study.EXIT_HURST_WINDOW}" in study1424.PRIMARY_CONFIG_ID
    grid = study._sweep_grid(study1410.HURST_WINDOWS)
    exit_hw = sorted({hw for _f, arm, hw in grid if arm == study.ARM_EXIT})
    assert exit_hw == [study.EXIT_HURST_WINDOW]


def test_the_sizing_arm_scores_every_hurst_window():
    grid = study._sweep_grid(study1410.HURST_WINDOWS)
    size_hw = sorted({hw for _f, arm, hw in grid if arm == study.ARM_SIZING})
    assert size_hw == sorted(study1410.HURST_WINDOWS)


def test_the_hypothesis_family_sizes_match_the_grid():
    grid = study._sweep_grid(study1410.HURST_WINDOWS)
    for arm in study.ARMS:
        assert (study.FAMILY_SIZE_BY_ARM[arm]
                == sum(1 for _f, a, _hw in grid if a == arm))


# --- the paired exit contrast ------------------------------------------------

def _leg(samples):
    return {"trade_samples": samples}


def _sample(day, pnl, side="long", hold_days=1):
    entry = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
    exit_ = entry + pd.Timedelta(days=hold_days)
    return {"entry_date": str(entry), "exit_date": str(exit_), "side": side,
            "pnl_pct_net": pnl}


def test_a_paired_row_carries_a_holding_interval_the_cluster_model_can_use():
    closes = [100.0] * 40
    key_pos = {str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)): i
               for i in range(40)}
    base = _leg([_sample(0, 1.0, hold_days=1)])
    scaled = _leg([_sample(0, 3.0, hold_days=5)])
    rows, _ = study.pair_exit_rows(base, scaled, closes, key_pos, 4)
    row = rows[0]
    assert row["exit_ns"] is not None
    assert row["exit_ns"] == row["scaled_exit_ns"]
    assert row["exit_ns"] > row["base_exit_ns"]


def test_the_paired_row_takes_the_later_exit_so_overlap_is_never_understated():
    closes = [100.0] * 40
    key_pos = {str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)): i
               for i in range(40)}
    base = _leg([_sample(0, 1.0, hold_days=9)])
    scaled = _leg([_sample(0, 3.0, hold_days=2)])
    rows, _ = study.pair_exit_rows(base, scaled, closes, key_pos, 4)
    row = rows[0]
    assert row["exit_ns"] == row["base_exit_ns"]
    assert row["exit_ns"] > row["scaled_exit_ns"]


def test_an_unreadable_exit_leaves_the_row_droppable_rather_than_imputed():
    closes = [100.0] * 40
    key_pos = {str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)): i
               for i in range(40)}
    bad = _sample(0, 1.0)
    bad["exit_date"] = "not-a-timestamp"
    rows, _ = study.pair_exit_rows(_leg([bad]), _leg([bad]), closes,
                                   key_pos, 4)
    assert rows[0]["exit_ns"] is None
    assert study.effective_n(
        [dict(rows[0], symbol="BTC/USDT", timeframe="1h")], {}) == 0.0


def test_exit_rows_survive_the_effective_n_correction():
    closes = [100.0] * 40
    key_pos = {str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)): i
               for i in range(40)}
    base = _leg([_sample(i, 1.0) for i in range(0, 12, 2)])
    scaled = _leg([_sample(i, 2.0) for i in range(0, 12, 2)])
    rows, _ = study.pair_exit_rows(base, scaled, closes, key_pos, 4)
    enriched = [dict(r, symbol="BTC/USDT", timeframe="1h") for r in rows]
    assert study.effective_n(enriched, {}) > 0.0


def test_pairing_matches_on_the_entry_bar_and_signs_the_difference():
    closes = [100.0] * 40
    key_pos = {str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)): i
               for i in range(40)}
    base = _leg([_sample(0, 1.0), _sample(1, -2.0)])
    scaled = _leg([_sample(0, 3.0), _sample(1, -1.0)])
    rows, unpaired = study.pair_exit_rows(base, scaled, closes, key_pos, 4)
    assert unpaired == 0
    assert [r["delta_primary"] for r in rows] == [2.0, 1.0]


def test_an_entry_on_only_one_side_is_dropped_and_counted():
    closes = [100.0] * 40
    key_pos = {str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)): i
               for i in range(40)}
    base = _leg([_sample(0, 1.0), _sample(1, -2.0)])
    scaled = _leg([_sample(0, 3.0)])
    rows, unpaired = study.pair_exit_rows(base, scaled, closes, key_pos, 4)
    assert unpaired == 1
    assert len(rows) == 1
    assert rows[0]["delta_primary"] == 2.0


def test_a_missing_leg_pairs_nothing():
    assert study.pair_exit_rows(None, _leg([]), [], {}, 4) == ([], 0)
    assert study.pair_exit_rows(_leg([]), None, [], {}, 4) == ([], 0)


def _exit_row(bucket, delta, day=0, window="2021"):
    entry = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
    return {
        "strategy": "momentum", "exchange": "binanceus",
        "symbol": "BTC/USDT", "base_symbol": "BTC/USDT", "timeframe": "1h",
        "window": window, "cohort": study.COHORT_PRIMARY,
        "entry_date": str(entry), "entry_ns": int(entry.value),
        "bucket": bucket, "scaled": bucket in study.EXIT_SCALED_BUCKETS,
        "trail_atr_mult": study.exit_trail_mult(bucket),
        "h": {512: 0.7}, "efficiency": 0.1,
        "base_pnl_pct_net": 0.0, "scaled_pnl_pct_net": delta,
        "pnl_pct_net": delta, "delta_primary": delta,
    }


def test_the_control_side_is_the_unscaled_buckets():
    rows = [_exit_row("persistent", 2.0, day=0),
            _exit_row("neutral", 0.0, day=1),
            _exit_row(study.BUCKET_NAN, 0.0, day=2),
            _exit_row("anti_persistent", -1.0, day=3)]
    keep, values, _r, suppressed = study.exit_contrast_rows(rows)
    assert len(keep) == 4
    assert suppressed == [False, True, True, False]
    assert values == [2.0, 0.0, 0.0, -1.0]


def test_a_control_row_difference_is_zero_by_construction():
    for bucket in study.EXIT_BUCKETS:
        if bucket in study.EXIT_SCALED_BUCKETS:
            continue
        assert study.EXIT_BUCKET_SCALES[bucket] == study.EXIT_NEUTRAL_SCALE


def test_a_row_with_no_paired_difference_is_dropped():
    row = _exit_row("persistent", 1.0)
    row["delta_primary"] = None
    keep, _v, _r, _s = study.exit_contrast_rows([row])
    assert keep == []


# --- the per-arm validity gate ----------------------------------------------

def _mde(arm, limit, sep, n=100):
    return {"by_arm": {arm: {"cluster": {study.PRIMARY_FAMILY: limit},
                             "separation": {study.PRIMARY_FAMILY: sep},
                             "n": {study.PRIMARY_FAMILY: n},
                             "units": "units"}}}


def _cfg(arm, kept_effective):
    return {"arm": arm, "family": study.PRIMARY_FAMILY,
            "cohort": study.COHORT_PRIMARY,
            "hurst_window": study.EXIT_HURST_WINDOW,
            "n_kept_effective": kept_effective}


def test_the_gate_reads_the_magnitude_because_the_null_is_two_sided():
    gate = study.validity_gate(
        study.ARM_SIZING, _mde(study.ARM_SIZING, 0.01, -0.05),
        [_cfg(study.ARM_SIZING, 99.0)])
    assert gate["passed"] is True
    assert gate["mode"] == study.MODE_OK
    assert gate["largest_separation"] == -0.05


def test_a_separation_below_the_limit_fails_as_a_power_statement():
    gate = study.validity_gate(
        study.ARM_EXIT, _mde(study.ARM_EXIT, 0.05, 0.01),
        [_cfg(study.ARM_EXIT, 99.0)])
    assert gate["passed"] is False
    assert gate["mode"] == study.MODE_BELOW_LIMIT


def test_a_kept_floor_breach_beats_every_other_gate_outcome():
    gate = study.validity_gate(
        study.ARM_SIZING, _mde(study.ARM_SIZING, 0.001, 0.9),
        [_cfg(study.ARM_SIZING, study.MIN_KEPT_EFFECTIVE - 0.1)])
    assert gate["passed"] is False
    assert gate["mode"] == study.MODE_FLOOR_BREACH
    assert "kept" in gate["reason"].lower()


def test_the_sizing_floor_breach_names_the_ceiling_cap_as_its_cause():
    gate = study.validity_gate(
        study.ARM_SIZING, _mde(study.ARM_SIZING, 0.001, 0.9),
        [_cfg(study.ARM_SIZING, 1.0)])
    assert "cap" in gate["reason"]
    assert str(study.SIZE_CEILING_BAND) in gate["reason"]


def test_an_unresolvable_limit_fails_rather_than_passes():
    gate = study.validity_gate(
        study.ARM_EXIT, _mde(study.ARM_EXIT, None, 0.5),
        [_cfg(study.ARM_EXIT, 99.0)])
    assert gate["passed"] is False
    assert gate["mode"] == study.MODE_UNRESOLVABLE


def test_a_degenerate_limit_is_labelled_even_when_it_passes():
    gate = study.validity_gate(
        study.ARM_EXIT, _mde(study.ARM_EXIT, 0.0, 0.2),
        [_cfg(study.ARM_EXIT, 99.0)])
    assert gate["passed"] is True
    assert gate["degenerate"] is True


def test_every_gate_carries_its_arms_own_contrast_statement():
    for arm in study.ARMS:
        gate = study.validity_gate(arm, _mde(arm, 0.01, 0.02),
                                   [_cfg(arm, 99.0)])
        assert gate["arm"] == arm
        assert gate["contrast"]
    sizing = study.validity_gate(
        study.ARM_SIZING, _mde(study.ARM_SIZING, 0.01, 0.02),
        [_cfg(study.ARM_SIZING, 99.0)])["contrast"]
    exit_ = study.validity_gate(
        study.ARM_EXIT, _mde(study.ARM_EXIT, 0.01, 0.02),
        [_cfg(study.ARM_EXIT, 99.0)])["contrast"]
    assert sizing != exit_


# --- the decision and the contract-path claim --------------------------------

def _both_arm_mde(limit, sep):
    return {"by_arm": {arm: {"cluster": {study.PRIMARY_FAMILY: limit},
                             "separation": {study.PRIMARY_FAMILY: sep},
                             "n": {study.PRIMARY_FAMILY: 100},
                             "units": "units"} for arm in study.ARMS}}


def _decision_cfg(arm, bh_reject, kept_effective=99.0, p=0.001):
    return {"arm": arm, "family": study.PRIMARY_FAMILY,
            "cohort": study.COHORT_PRIMARY,
            "hurst_window": study.EXIT_HURST_WINDOW,
            "config_id": f"{study.PRIMARY_FAMILY}/{arm}/W512/x",
            "n_kept_effective": kept_effective,
            "p_cluster": p, "bh_reject": bh_reject, "separation": 0.5}


def test_an_inconclusive_run_never_claims_the_contract_path():
    cfgs = [_decision_cfg(a, False) for a in study.ARMS]
    decision = study.decide_recommendation(cfgs, _both_arm_mde(0.9, 0.01))
    assert decision["verdict"] == study.VERDICT_INCONCLUSIVE
    assert study.claims_contract_path(decision) is False


def test_a_resolved_null_never_claims_the_contract_path():
    cfgs = [_decision_cfg(a, False) for a in study.ARMS]
    decision = study.decide_recommendation(cfgs, _both_arm_mde(0.01, 0.5))
    assert decision["verdict"] == study.VERDICT_RESOLVED_NULL
    assert study.claims_contract_path(decision) is False


def test_a_confirmatory_arm_claims_the_contract_path():
    cfgs = [_decision_cfg(study.ARM_SIZING, True),
            _decision_cfg(study.ARM_EXIT, False)]
    decision = study.decide_recommendation(cfgs, _both_arm_mde(0.01, 0.5))
    assert decision["verdict"] == study.VERDICT_SORTS
    assert study.claims_contract_path(decision) is True
    assert decision["confirmatory_arms"] == [study.ARM_SIZING]


def test_a_significant_result_on_a_degenerate_limit_does_not_claim_the_path():
    cfgs = [_decision_cfg(a, True) for a in study.ARMS]
    decision = study.decide_recommendation(cfgs, _both_arm_mde(0.0, 0.5))
    assert decision["verdict"] == study.VERDICT_SORTS
    assert study.claims_contract_path(decision) is False
    assert decision["confirmatory_arms"] == []


def test_a_significant_result_behind_a_failed_gate_does_not_claim_the_path():
    cfgs = [_decision_cfg(a, True) for a in study.ARMS]
    decision = study.decide_recommendation(cfgs, _both_arm_mde(0.9, 0.01))
    assert study.claims_contract_path(decision) is False


def test_a_floor_breached_arm_cannot_carry_the_claim():
    cfgs = [_decision_cfg(study.ARM_SIZING, True, kept_effective=1.0),
            _decision_cfg(study.ARM_EXIT, False)]
    decision = study.decide_recommendation(cfgs, _both_arm_mde(0.01, 0.5))
    assert study.claims_contract_path(decision) is False


def test_the_decision_has_no_promotion_branch():
    src = open(study.__file__).read()
    body = src.split("def decide_recommendation(", 1)[1].split("\ndef ", 1)[0]
    assert "winner" not in body
    assert "VERDICT_CONFIG" not in src
    assert study.NO_PROMOTION_SENTENCE


# --- pinned payload and rendering -------------------------------------------

def test_the_module_refuses_the_contract_path_unconditionally(tmp_path):
    with pytest.raises(SystemExit) as exc:
        study.main(["--report-out", CONTRACT])
    assert study.CONTRACT_REPORT_BASENAME in str(exc.value)


def test_the_contract_refusal_survives_render_only(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        study.main(["--render-only", "--json-out", str(path),
                    "--report-out", CONTRACT])
    assert study.CONTRACT_REPORT_BASENAME in str(exc.value)


def test_a_scoped_run_may_not_overwrite_the_committed_artefacts():
    with pytest.raises(SystemExit) as exc:
        study.main(["--windows", "2021"])
    assert "refusing to overwrite" in str(exc.value)


def test_a_deviating_run_may_not_overwrite_the_committed_artefacts():
    with pytest.raises(SystemExit) as exc:
        study.main(["--seed", "7"])
    assert "refusing to overwrite" in str(exc.value)


def test_the_exit_arm_cannot_be_dropped_silently():
    with pytest.raises(SystemExit) as exc:
        study.main(["--no-exit-arm"])
    assert "no-exit-arm" in str(exc.value)


@pytest.mark.skipif(not os.path.exists(study._DEFAULT_JSON_OUT),
                    reason="committed run not present")
def test_the_committed_decision_is_a_mechanical_render_of_its_configs():
    payload = json.load(open(study._DEFAULT_JSON_OUT))
    rebuilt = study.decide_recommendation(payload["configs"],
                                          payload["detection_limits"])
    assert study.decision_payload(rebuilt) == payload["decision_payload"]


@pytest.mark.skipif(not os.path.exists(study._DEFAULT_JSON_OUT),
                    reason="committed run not present")
def test_render_only_reproduces_the_committed_report_byte_for_byte():
    payload = json.load(open(study._DEFAULT_JSON_OUT))
    assert (study.report_from_payload(payload)
            == open(study._DEFAULT_REPORT_OUT).read())


@pytest.mark.skipif(not os.path.exists(study._DEFAULT_JSON_OUT),
                    reason="committed run not present")
def test_the_committed_run_is_stamped_complete_and_pre_registered():
    payload = json.load(open(study._DEFAULT_JSON_OUT))
    scope = payload["run_summary"]["scope"]
    assert scope["complete"] is True
    assert scope["pre_registered_inference"] is True
    assert payload["run_summary"]["exit_arm_ran"] is True


@pytest.mark.skipif(not os.path.exists(study._DEFAULT_JSON_OUT),
                    reason="committed run not present")
def test_the_committed_run_scores_the_shipped_form():
    payload = json.load(open(study._DEFAULT_JSON_OUT))
    pre = payload["pre_registered"]
    assert pre["shipped_size_span"] == parity.HURST_SIZE_SPAN
    assert pre["shipped_size_floor"] == parity.HURST_DEFAULT_SIZE_FLOOR
    assert pre["shipped_size_ceiling"] == 1.0


@pytest.mark.skipif(not os.path.exists(study._DEFAULT_JSON_OUT),
                    reason="committed run not present")
def test_the_committed_claim_agrees_with_the_report_and_the_siblings():
    payload = json.load(open(study._DEFAULT_JSON_OUT))
    claimed = study.claims_contract_path(payload["decision"])
    report = open(study._DEFAULT_REPORT_OUT).read()
    if claimed:
        assert f"CLAIMS `{study.CONTRACT_REPORT_BASENAME}`" in report
    else:
        assert f"DEFERS `{study.CONTRACT_REPORT_BASENAME}`" in report
        assert study1424._DEFAULT_REPORT_OUT.endswith(
            study.CONTRACT_REPORT_BASENAME)


# --- the siblings and the registry -------------------------------------------

def test_1427_hands_the_supersede_clause_to_this_study():
    assert study1427.CONTRACT_PATH_CLAIMED is False
    assert study.ISSUE in study1427.SIBLING_DEFERRAL
    assert 1426 in study1427.DEFERRING_SIBLINGS


def test_1426_still_refuses_the_contract_path():
    assert study1426.CONTRACT_REPORT_BASENAME == study.CONTRACT_REPORT_BASENAME


def test_the_deleted_architecture_doc_is_not_re_introduced():
    for path in (study.__file__, study._DEFAULT_REPORT_OUT):
        if not os.path.exists(path):
            continue
        assert "ARCHITECTURE.md" not in open(path).read()


def test_the_contract_path_rule_cites_only_the_surviving_readers():
    rule = study.CONTRACT_PATH_CLAIM_RULE
    assert "scheduler/hurst_gate.go" in rule
    assert "#1412" in rule
    assert "ARCHITECTURE.md" not in rule


def test_the_registry_carries_a_row_for_this_harness():
    registry = os.path.join(_REPO, "docs", "backtesting-registry.md")
    text = open(registry).read()
    assert "research/hurst_1428_sizing_exit.py" in text
    assert f"#{study.ISSUE}" in text


def test_the_estimator_stays_the_1409_single_source_of_truth():
    src = open(study.__file__).read()
    assert "rolling_hurst = study1410.rolling_hurst" in src
    assert "def hurst_exponent" not in src
    assert "def rolling_hurst" not in src


def test_the_inference_is_two_sided_on_both_arms():
    assert study.TWO_SIDED is True
    assert study.INFERENCE_DIRECTION == "two_sided"
    src = open(study.__file__).read()
    body = src.split("def build_configs(", 1)[1].split("\ndef ", 1)[0]
    assert "two_sided_permutation_pvalue_weighted" in body
    assert "two_sided_cluster_permutation_pvalue_group_diff" in body
    assert re.search(r"[^_]permutation_pvalue_group_diff\(", body) is None


def test_the_arms_carry_the_pre_registered_target_swap():
    assert study.ARM_TARGETS[study.ARM_SIZING]["primary"] == study1424.PRIMARY_TARGET
    assert study.ARM_TARGETS[study.ARM_EXIT]["primary"] == "pnl_pct_net"
    assert (study.ARM_TARGETS[study.ARM_EXIT]["continuity"]
            == study1424.PRIMARY_TARGET)
