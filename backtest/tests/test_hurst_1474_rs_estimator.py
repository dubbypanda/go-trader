import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "research")))

import hurst_1474_rs_estimator as study
import hurst_1426_two_sided_sort as study1426
import hurst_1424_gate_resolution as study1424
import hurst_1422_gate_power as study1422
import hurst_1410_gate_calibration as study1410

from indicators_core import hurst_exponent, hurst_rescaled_range

CONTRACT = os.path.join(os.path.dirname(study._DEFAULT_REPORT_OUT),
                        "hurst_gate_calibration.md")

_DFA = study.ESTIMATOR_DFA
_RS = study.ESTIMATOR_RS
_RS_RAW = study.ESTIMATOR_RS_RAW


def _prices(n=1400, seed=11, sigma=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n))),
                     index=idx)


def _trade(symbol="BTC/USDT", timeframe="1h", window="2021", day=0, pnl=1.0,
           eff=0.1, hold_days=1, h=None, cohort=None, exchange="binanceus",
           estimator_h=None):
    entry = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
    stamps = estimator_h
    if stamps is None:
        stamps = {est: {str(w): h for w in study.AGREEMENT_WINDOWS}
                  for est in study.ESTIMATORS}
    return {
        "strategy": "momentum",
        "exchange": exchange,
        "symbol": symbol,
        "base_symbol": symbol.split("@", 1)[0],
        "timeframe": timeframe,
        "window": window,
        "cohort": cohort or study.COHORT_PRIMARY,
        "entry_date": str(entry),
        "entry_ns": int(entry.value),
        "exit_ns": int((entry + pd.Timedelta(days=hold_days)).value),
        "pnl_pct_net": float(pnl),
        "efficiency": None if eff is None else float(eff),
        "adx": None,
        "h_by_estimator": stamps,
    }


# --- the estimators stay the shared single sources of truth -----------------

def test_neither_estimator_is_reimplemented_here():
    source = open(study.__file__).read()
    assert "def hurst_exponent" not in source
    assert "def hurst_rescaled_range" not in source
    assert study.hurst_exponent is hurst_exponent
    assert study.hurst_rescaled_range is hurst_rescaled_range


def test_the_dfa_estimator_is_still_1409s_and_1410s_rolling_wrapper():
    assert study.rolling_hurst is study1410.rolling_hurst
    assert study.estimator_fn(_DFA) is hurst_exponent


def test_the_rolling_wrapper_delegates_the_dfa_arm_verbatim():
    close = _prices(400, seed=3)
    mine = study.rolling_estimator(close, 128, _DFA)
    theirs = study1410.rolling_hurst(close, 128)
    pd.testing.assert_series_equal(mine, theirs)


def test_the_rolling_wrapper_mirrors_1410s_window_and_first_needed_semantics():
    close = _prices(400, seed=4)
    first_needed = close.index[300]
    reference = study1410.rolling_hurst(close, 128, first_needed=first_needed)
    mine = study.rolling_estimator(close, 128, _DFA, first_needed=first_needed)
    pd.testing.assert_series_equal(mine, reference)
    rs = study.rolling_estimator(close, 128, _RS, first_needed=first_needed)
    assert list(np.isfinite(rs.to_numpy(dtype=float))) == \
        list(np.isfinite(reference.to_numpy(dtype=float)))


def test_the_rolling_wrapper_computes_the_named_estimator_bar_for_bar():
    close = _prices(200, seed=5)
    rolled = study.rolling_estimator(close, 150, _RS)
    for pos in (149, 175, 199):
        expected = hurst_rescaled_range(close.iloc[pos - 149: pos + 1])
        assert rolled.iloc[pos] == pytest.approx(expected, abs=1e-12)


def test_the_rolling_wrapper_refuses_a_degenerate_window():
    with pytest.raises(ValueError):
        study.rolling_estimator(_prices(200), 1, _RS)


def test_an_unknown_estimator_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        study.estimator_fn("hurst")


def test_the_look_ahead_shifts_are_the_inherited_ones():
    series = pd.Series(np.arange(10.0))
    assert study.decision_series(series).equals(series.shift(1))
    assert study.entry_stamp_series(series).equals(series.shift(2))


# --- the design is inherited, not restated ---------------------------------

def test_the_design_is_inherited_from_1424_rather_than_restated():
    assert study.WINDOWS == study1424.WINDOWS
    assert study.WINDOW_ORDER == study1424.WINDOW_ORDER
    assert study.DATASETS == study1424.DATASETS
    assert study.WINDOW_OWNER == study1424.WINDOW_OWNER
    assert study.PRIMARY_CONFIG_ID == study1424.PRIMARY_CONFIG_ID
    assert study.PRIMARY_FAMILY == study1424.PRIMARY_FAMILY
    assert study.PRIMARY_TARGET == study1424.PRIMARY_TARGET
    assert study.CONTINUITY_TARGET == study1424.CONTINUITY_TARGET
    assert study.HORIZON_HOURS == study1424.HORIZON_HOURS
    assert study.build_leg is study1424.build_leg
    assert study.signed_efficiency is study1424.signed_efficiency
    assert study.cell_cohort is study1424.cell_cohort


def test_the_inference_is_1426s_two_sided_machinery():
    assert study.TWO_SIDED is True
    assert study.two_sided_min_detectable_effect_eff is \
        study1426.two_sided_min_detectable_effect_eff
    assert study.two_sided_min_detectable_effect_pp is \
        study1426.two_sided_min_detectable_effect_pp
    assert study.two_sided_cluster_permutation_pvalue_group_diff is \
        study1426.two_sided_cluster_permutation_pvalue_group_diff
    assert study.validity_gate is study1426.validity_gate
    assert study.TWO_SIDED_P_DEFINITION == study1426.TWO_SIDED_P_DEFINITION


_ONE_SIDED = (
    (study1410, "permutation_pvalue_group_diff"),
    (study1410, "permutation_pvalue_weighted"),
    (study1422, "cluster_permutation_pvalue_group_diff"),
    (study1422, "cluster_permutation_pvalue_weighted"),
    (study1424, "permutation_pvalue_group_diff"),
    (study1424, "permutation_pvalue_weighted"),
    (study1424, "cluster_permutation_pvalue_group_diff"),
    (study1424, "cluster_permutation_pvalue_weighted"),
    (study1424, "min_detectable_effect_on_grid"),
    (study1424, "min_detectable_effect_eff"),
    (study1424, "min_detectable_effect_pp"),
)


@pytest.fixture()
def one_sided_is_a_landmine(monkeypatch):
    def _boom(*_a, **_kw):
        raise AssertionError("a one-sided p-value function was reached from "
                             "the two-sided estimator comparison")

    for module, name in _ONE_SIDED:
        monkeypatch.setattr(module, name, _boom, raising=False)
    return _boom


def _contrasting_rows(n=40):
    rows = [_trade(day=i * 5, h=0.4 + 0.01 * (i % 5)) for i in range(n)]
    rows += [_trade(symbol="ETH/USDT", day=i * 5, h=0.55 + 0.01 * (i % 5))
             for i in range(n)]
    _, matched = study.matched_rows(rows, 512, study.ESTIMATORS)
    return matched


def test_no_one_sided_null_is_reachable_from_a_measurement(
        one_sided_is_a_landmine):
    out = study.estimator_measurement(_contrasting_rows(), 512, _DFA,
                                      study.PRIMARY_FAMILY, {}, n_perm=200,
                                      n_perm_mde=200, seed=study.SEED)
    assert out["separation"] is not None
    assert out["p"] is not None


def test_no_one_sided_null_is_reachable_from_the_separation_section(
        one_sided_is_a_landmine):
    pooled = {study.PRIMARY_FAMILY: _contrasting_rows(),
              "mean_reversion": []}
    section = study.separation_section(pooled, (512,), {}, n_perm=200,
                                       n_perm_mde=200, seed=study.SEED)
    block = section["by_window"]["512"]
    assert block["measurements"][_DFA]["p"] is not None


def test_the_seed_is_the_issue_number():
    assert study.SEED == study.ISSUE == 1474


def test_the_effective_n_floors_are_1424s():
    assert study.MIN_KEPT_EFFECTIVE == study1424.MIN_KEPT_EFFECTIVE
    assert study.MIN_SUPPRESSED_EFFECTIVE == study1424.MIN_SUPPRESSED_EFFECTIVE


def test_the_pinned_hypothesis_is_still_the_committed_1410_argmin():
    assert study.resolve_primary_config_id(study._JSON_1410) == \
        study.PRIMARY_CONFIG_ID


# --- section 1: agreement ---------------------------------------------------

def test_agreement_of_a_series_with_itself_is_perfect():
    values = pd.Series([0.4, 0.45, 0.55, 0.6, 0.52, 0.48])
    stats = study.agreement_stats(values, values)
    assert stats["pearson"] == pytest.approx(1.0)
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["mean_signed_difference"] == 0.0
    assert stats["side_disagreement_share"] == 0.0


def test_agreement_scores_only_the_bars_where_both_are_defined():
    ref = pd.Series([0.4, np.nan, 0.6, 0.5, np.nan])
    cand = pd.Series([0.45, 0.5, np.nan, 0.55, 0.6])
    stats = study.agreement_stats(ref, cand)
    assert stats["n_rows"] == 2
    assert stats["n_reference_only"] == 1
    assert stats["n_candidate_only"] == 2


def test_the_signed_difference_is_candidate_minus_reference():
    ref = pd.Series([0.5, 0.5, 0.5])
    cand = pd.Series([0.4, 0.4, 0.4])
    assert study.agreement_stats(ref, cand)["mean_signed_difference"] == \
        pytest.approx(-0.1)
    assert study.agreement_stats(cand, ref)["mean_signed_difference"] == \
        pytest.approx(0.1)


def test_the_side_share_counts_the_gates_own_edge():
    ref = pd.Series([0.51, 0.49, 0.50, 0.60])
    cand = pd.Series([0.49, 0.51, 0.50, 0.61])
    stats = study.agreement_stats(ref, cand)
    assert stats["side_disagreement_share"] == pytest.approx(0.5)
    assert stats["reference_persistent_share"] == pytest.approx(0.75)
    assert stats["candidate_persistent_share"] == pytest.approx(0.75)
    assert study.PERSISTENT_SIDE_EDGE == 0.5


def test_spearman_matches_a_naive_rank_pearson_with_average_ties():
    rng = np.random.default_rng(1474)
    for _ in range(20):
        a = rng.normal(size=50).round(2)
        b = (a * 0.5 + rng.normal(size=50)).round(2)
        expected = np.corrcoef(pd.Series(a).rank().to_numpy(),
                               pd.Series(b).rank().to_numpy())[0, 1]
        assert study._spearman(a, b) == pytest.approx(expected, abs=1e-6)


def test_a_monotone_transform_keeps_spearman_and_moves_pearson():
    a = np.linspace(0.3, 0.7, 40)
    b = a ** 3
    assert study._spearman(a, b) == pytest.approx(1.0)
    assert study._pearson(a, b) < 1.0


def test_agreement_is_untestable_rather_than_zero_without_rows():
    empty = pd.Series([], dtype=float)
    stats = study.agreement_stats(empty, empty)
    assert stats["n_rows"] == 0
    assert stats["pearson"] is None
    assert stats["spearman"] is None
    assert stats["side_disagreement_share"] is None


def test_the_reference_window_is_named_as_unreachable_live():
    assert study.REFERENCE_WINDOW == 1000
    assert study.REFERENCE_WINDOW not in study.HURST_WINDOWS
    assert study.AGREEMENT_WINDOWS == tuple(study.HURST_WINDOWS) + (1000,)
    assert "cannot run live" in study.REFERENCE_WINDOW_STATEMENT
    assert "hurst_live_frame_bars" in study.REFERENCE_WINDOW_STATEMENT


def test_the_reference_window_exceeds_the_live_fetch_depth():
    from hurst_gate import hurst_live_frame_bars

    assert hurst_live_frame_bars(None) < study.REFERENCE_WINDOW
    assert hurst_live_frame_bars({"trend": {"period": 200}}) < study.REFERENCE_WINDOW


def test_the_agreement_section_pools_the_same_bars_on_both_sides():
    close = _prices(500, seed=9)
    ds = ("binanceus", "BTC/USDT", "1h")
    rolling = {}
    for hw in (128,):
        for est in study.ESTIMATORS:
            rolling[(ds, hw, est)] = study.rolling_estimator(close, hw, est)
    section = study.agreement_section(rolling, [ds], (128,))
    block = section["by_window"]["128"]
    pooled = block["pooled"][_RS]
    per_dataset = block["by_dataset"][study.dataset_key("BTC/USDT", "1h")][_RS]
    assert pooled == per_dataset
    assert pooled["n_rows"] > 0


# --- the Anis-Lloyd correction is a constant shift at a fixed window --------

def test_the_correction_is_a_constant_shift_at_a_fixed_window():
    close = _prices(1400, seed=13)
    for window in study.HURST_WINDOWS:
        corrected = study.rolling_estimator(close, window, _RS)
        raw = study.rolling_estimator(close, window, _RS_RAW)
        offsets = (corrected - raw).dropna().to_numpy(dtype=float)
        assert offsets.size > 10
        assert float(np.ptp(offsets)) < 1e-9, (window, float(np.ptp(offsets)))


def test_the_constant_shift_leaves_the_ranking_alone_and_moves_the_edge():
    close = _prices(1400, seed=14)
    corrected = study.rolling_estimator(close, 256, _RS)
    raw = study.rolling_estimator(close, 256, _RS_RAW)
    dfa = study.rolling_estimator(close, 256, _DFA)
    corr_stats = study.agreement_stats(dfa, corrected)
    raw_stats = study.agreement_stats(dfa, raw)
    assert corr_stats["spearman"] == pytest.approx(raw_stats["spearman"],
                                                   abs=1e-9)
    assert corr_stats["side_disagreement_share"] != \
        raw_stats["side_disagreement_share"]


def test_the_report_states_the_constant_shift_rather_than_folding_the_two():
    assert "CONSTANT SHIFT" in study.CONSTANT_OFFSET_STATEMENT
    assert _RS_RAW in study.ESTIMATORS


# --- section 2: bias --------------------------------------------------------

def test_the_bias_draws_are_reproducible_from_the_issue_seed():
    assert study.bias_draw_seed(101, 0) == study.SEED * 1_000_000 + 101_000
    assert study.bias_draw_seed(2000, 7) == study.SEED * 1_000_000 + 2_000_000 + 7
    a = study._random_walk_prices(200, study.bias_draw_seed(200, 3))
    b = study._random_walk_prices(200, study.bias_draw_seed(200, 3))
    pd.testing.assert_series_equal(a, b)


def test_every_bias_draw_is_its_own_seed():
    first = study._random_walk_prices(200, study.bias_draw_seed(200, 0))
    second = study._random_walk_prices(200, study.bias_draw_seed(200, 1))
    assert not first.equals(second)


def test_the_bias_table_covers_every_pre_registered_sample_size():
    assert study.BIAS_SAMPLE_SIZES == (101, 128, 256, 512, 1000, 2000)
    section = study.bias_section((101, 128), draws=12, jobs=2)
    assert sorted(section["by_n"]) == ["101", "128"]
    for row in section["by_n"].values():
        assert sorted(row) == sorted(study.ESTIMATORS)
        for stats in row.values():
            assert stats["n_draws"] == 12
            assert stats["mean"] is not None
            assert stats["iqr"] is not None


def test_the_bias_column_is_measured_against_one_half():
    summary = study._summarize([0.4, 0.6, 0.5])
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["bias"] == pytest.approx(0.0)
    assert study._summarize([0.6, 0.6, 0.6])["bias"] == pytest.approx(0.1)


def test_an_all_undefined_column_is_reported_as_undefined_not_as_a_half():
    summary = study._summarize([float("nan"), float("nan")])
    assert summary["n_defined"] == 0
    assert summary["mean"] is None
    assert summary["bias"] is None


def test_the_iqr_is_the_quartile_spread():
    summary = study._summarize(list(np.linspace(0.0, 1.0, 101)))
    assert summary["q25"] == pytest.approx(0.25, abs=1e-6)
    assert summary["q75"] == pytest.approx(0.75, abs=1e-6)
    assert summary["iqr"] == pytest.approx(0.5, abs=1e-6)


def test_the_bias_section_measures_the_1409_noise_claim():
    section = study.bias_section((128,), draws=60, jobs=2)
    row = section["by_n"]["128"]
    assert row[_RS]["sd"] > row[_DFA]["sd"]
    assert "too noisy" in section["claim_under_test"]


def test_the_raw_slope_is_biased_up_and_the_correction_recentres_it():
    section = study.bias_section((512,), draws=60, jobs=2)
    row = section["by_n"]["512"]
    assert row[_RS_RAW]["bias"] > 0.0
    assert abs(row[_RS]["bias"]) < abs(row[_RS_RAW]["bias"])


# --- stamping and row matching ---------------------------------------------

def test_the_stamp_is_the_same_two_bar_shift_build_leg_applies():
    close = _prices(400, seed=17)
    ds = ("binanceus", "BTC/USDT", "1h")
    rolling = study.rolling_estimator(close, 128, _RS)
    stamps = {(ds, 128, est): study.entry_stamp_series(
        study.rolling_estimator(close, 128, est)) for est in study.ESTIMATORS}
    entry = close.index[300]
    row = {"exchange": "binanceus", "base_symbol": "BTC/USDT",
           "timeframe": "1h", "entry_date": str(entry)}
    study.stamp_rows([row], stamps, (128,))
    assert row["h_by_estimator"][_RS]["128"] == \
        pytest.approx(round(float(rolling.iloc[298]), 6), abs=1e-12)


def test_the_stamp_lookup_agrees_with_build_legs_reindex_of_the_slice():
    close = _prices(400, seed=18)
    rolling = study.rolling_estimator(close, 128, _RS)
    stamp = study.entry_stamp_series(rolling)
    window = close.iloc[200:]
    reindexed = stamp.reindex(window.index).to_numpy(dtype=float)
    for pos, ts in enumerate(window.index):
        direct = float(stamp.loc[ts])
        if math.isnan(direct):
            assert math.isnan(reindexed[pos])
        else:
            assert reindexed[pos] == pytest.approx(direct, abs=1e-12)


def test_an_undefined_stamp_is_none_rather_than_a_half():
    close = _prices(300, seed=19)
    ds = ("binanceus", "BTC/USDT", "1h")
    stamps = {(ds, 256, est): study.entry_stamp_series(
        study.rolling_estimator(close, 256, est)) for est in study.ESTIMATORS}
    row = {"exchange": "binanceus", "base_symbol": "BTC/USDT",
           "timeframe": "1h", "entry_date": str(close.index[10])}
    study.stamp_rows([row], stamps, (256,))
    for est in study.ESTIMATORS:
        assert row["h_by_estimator"][est]["256"] is None


def test_a_bar_outside_the_stamped_frame_is_none_rather_than_a_crash():
    close = _prices(300, seed=20)
    ds = ("binanceus", "BTC/USDT", "1h")
    stamps = {(ds, 128, est): study.entry_stamp_series(
        study.rolling_estimator(close, 128, est)) for est in study.ESTIMATORS}
    row = {"exchange": "binanceus", "base_symbol": "BTC/USDT",
           "timeframe": "1h", "entry_date": "1999-01-01 00:00:00"}
    study.stamp_rows([row], stamps, (128,))
    assert row["h_by_estimator"][_DFA]["128"] is None


def test_row_matching_drops_a_row_undefined_for_any_single_estimator():
    good = _trade(day=0, h=0.6)
    partial = _trade(day=5, h=0.6)
    partial["h_by_estimator"] = dict(partial["h_by_estimator"])
    partial["h_by_estimator"][_RS] = {str(w): None
                                      for w in study.AGREEMENT_WINDOWS}
    audit, rows = study.matched_rows([good, partial], 512, study.ESTIMATORS)
    assert audit["n_matched"] == 1
    assert audit["n_dropped_undefined_h"] == 1
    dfa_audit, _ = study.matched_rows([good, partial], 512, (_DFA,))
    assert dfa_audit["n_matched"] == 2


def test_row_matching_drops_a_row_with_no_target():
    rows = [_trade(day=0, h=0.6), _trade(day=5, h=0.6, eff=None)]
    audit, kept = study.matched_rows(rows, 512, study.ESTIMATORS)
    assert audit["n_dropped_no_target"] == 1
    assert audit["n_matched"] == 1


def test_row_matching_scores_the_primary_cohort_only():
    rows = [_trade(day=0, h=0.6),
            _trade(day=5, h=0.6, cohort=study.COHORT_EXPLORATORY)]
    audit, _ = study.matched_rows(rows, 512, study.ESTIMATORS)
    assert audit["n_matched"] == 1


def test_every_estimator_reads_the_identical_matched_row_set():
    rows = [_trade(day=i * 5, h=0.45 + 0.01 * (i % 7)) for i in range(40)]
    rows += [_trade(symbol="ETH/USDT", day=i * 5, h=0.52 + 0.01 * (i % 7))
             for i in range(40)]
    _, matched = study.matched_rows(rows, 512, study.ESTIMATORS)
    measurements = {
        est: study.estimator_measurement(matched, 512, est,
                                         study.PRIMARY_FAMILY, {}, 50, 200,
                                         study.SEED)
        for est in study.ESTIMATORS
    }
    counts = {m["n_rows"] for m in measurements.values()}
    assert len(counts) == 1


# --- section 3: separation --------------------------------------------------

def _split_pool(h_by_estimator_for):
    rows = []
    for i in range(60):
        rows.append(_trade(day=i * 5, pnl=1.0, eff=0.2,
                           estimator_h=h_by_estimator_for(i, "BTC/USDT")))
        rows.append(_trade(symbol="ETH/USDT", day=i * 5, pnl=-1.0, eff=-0.2,
                           estimator_h=h_by_estimator_for(i, "ETH/USDT")))
    return rows


def test_the_separation_is_signed_and_never_an_absolute_value():
    def stamps(i, symbol):
        h = 0.6 if symbol == "BTC/USDT" else 0.4
        return {est: {str(w): h for w in study.AGREEMENT_WINDOWS}
                for est in study.ESTIMATORS}

    rows = _split_pool(stamps)
    _, matched = study.matched_rows(rows, 512, study.ESTIMATORS)
    out = study.estimator_measurement(matched, 512, _DFA, study.PRIMARY_FAMILY,
                                      {}, 50, 200, study.SEED)
    assert out["separation"] == pytest.approx(0.4, abs=1e-9)

    def flipped(i, symbol):
        h = 0.4 if symbol == "BTC/USDT" else 0.6
        return {est: {str(w): h for w in study.AGREEMENT_WINDOWS}
                for est in study.ESTIMATORS}

    _, matched = study.matched_rows(_split_pool(flipped), 512,
                                    study.ESTIMATORS)
    out = study.estimator_measurement(matched, 512, _DFA, study.PRIMARY_FAMILY,
                                      {}, 50, 200, study.SEED)
    assert out["separation"] == pytest.approx(-0.4, abs=1e-9)


def test_a_measurement_with_no_contrast_is_untestable_not_zero():
    rows = [_trade(day=i * 5, h=0.6) for i in range(40)]
    _, matched = study.matched_rows(rows, 512, study.ESTIMATORS)
    out = study.estimator_measurement(matched, 512, _DFA,
                                      study.PRIMARY_FAMILY, {}, 50, 200,
                                      study.SEED)
    assert out["separation"] is None
    assert out["limit"] is None
    assert out["p"] is None
    assert out["reason"] == "no testable contrast"


def test_the_gate_reads_each_estimators_own_row_matched_limit():
    def stamps(i, symbol):
        h = 0.6 if symbol == "BTC/USDT" else 0.4
        return {est: {str(w): h for w in study.AGREEMENT_WINDOWS}
                for est in study.ESTIMATORS}

    _, matched = study.matched_rows(_split_pool(stamps), 512,
                                    study.ESTIMATORS)
    out = study.estimator_measurement(matched, 512, _DFA,
                                      study.PRIMARY_FAMILY, {}, 200, 400,
                                      study.SEED)
    gate = out["gate"]
    assert gate["family"] == study.PRIMARY_FAMILY
    assert gate["limit"] == out["limit"]
    assert gate["largest_separation"] == out["separation"]
    assert gate["two_sided"] is True


def test_the_suppressed_side_follows_the_family_sense():
    assert study.FAMILY_SENSE[study.PRIMARY_FAMILY] == study.SENSE_HIGH
    assert study.anti_signal_side(0.4, study.SENSE_HIGH) is True
    assert study.anti_signal_side(0.6, study.SENSE_HIGH) is False


def test_the_confirmatory_bar_is_alpha_for_a_family_of_one():
    assert study.PRIMARY_FAMILY_SIZE == 1
    assert study._rank1_threshold(study.PRIMARY_FAMILY_SIZE,
                                  study.ALPHA) == study.ALPHA


# --- the estimator-risk verdict --------------------------------------------

def _separation_payload(delta, limit, dfa_sep=-0.005, rs_limit=0.012):
    pinned = str(int(max(study.HURST_WINDOWS)))
    rs_sep = None if (delta is None or dfa_sep is None) else dfa_sep + delta
    return {
        "by_window": {
            pinned: {
                "measurements": {
                    _DFA: {"separation": dfa_sep, "limit": limit},
                    _RS: {"separation": rs_sep, "limit": rs_limit},
                },
                "separation_delta_vs_dfa": {_RS: delta},
            }
        }
    }


def test_a_move_below_the_limit_bounds_the_estimator_risk():
    verdict = study.estimator_risk_verdict(_separation_payload(0.002, 0.013))
    assert verdict["verdict"] == study.VERDICT_BOUNDED
    assert verdict["bounded"] is True
    assert "BOUNDED" in verdict["statement"]
    assert "no threshold ships" in verdict["statement"]


def test_a_move_at_or_above_the_limit_is_the_opposite_finding():
    verdict = study.estimator_risk_verdict(_separation_payload(0.013, 0.013))
    assert verdict["verdict"] == study.VERDICT_MOVES
    assert verdict["bounded"] is False
    assert "licenses NO threshold" in verdict["statement"]


def test_the_verdict_reads_the_magnitude_of_the_move_in_either_direction():
    up = study.estimator_risk_verdict(_separation_payload(0.02, 0.013))
    down = study.estimator_risk_verdict(_separation_payload(-0.02, 0.013))
    assert up["verdict"] == down["verdict"] == study.VERDICT_MOVES


def test_a_missing_limit_is_unresolved_rather_than_bounded():
    verdict = study.estimator_risk_verdict(_separation_payload(0.002, None))
    assert verdict["verdict"] == study.VERDICT_UNRESOLVED
    assert verdict["bounded"] is False
    assert "bounds nothing" in verdict["statement"]


def test_a_missing_separation_is_unresolved_rather_than_bounded():
    verdict = study.estimator_risk_verdict(_separation_payload(None, 0.013))
    assert verdict["verdict"] == study.VERDICT_UNRESOLVED
    assert verdict["bounded"] is False


def test_the_verdict_reads_the_pinned_window_not_the_smallest():
    payload = _separation_payload(0.002, 0.013)
    payload["by_window"]["128"] = {
        "measurements": {_DFA: {"separation": -0.9, "limit": 0.001},
                         _RS: {"separation": 0.9, "limit": 0.001}},
        "separation_delta_vs_dfa": {_RS: 1.8},
    }
    verdict = study.estimator_risk_verdict(payload)
    assert verdict["hurst_window"] == max(study.HURST_WINDOWS)
    assert verdict["verdict"] == study.VERDICT_BOUNDED


def test_no_verdict_ever_recommends_an_estimator_swap():
    for delta, limit in ((0.002, 0.013), (0.02, 0.013), (None, 0.013)):
        statement = study.estimator_risk_verdict(
            _separation_payload(delta, limit))["statement"]
        assert "recommend" not in statement.lower()
        assert "threshold" not in statement.lower() or "NO threshold" in statement \
            or "no threshold ships" in statement


# --- non-goals --------------------------------------------------------------

def test_the_study_declares_its_non_goals_as_data():
    for needle in ("hurst_rs", "shared_tools/regime.py",
                   "scheduler/hurst_gate.go", "config.example.json"):
        assert needle in study.NON_GOALS
    assert study.CONTRACT_PATH_CLAIMED is False


def _code_lines():
    lines = []
    for line in open(study.__file__).read().split("\n"):
        stripped = line.strip()
        if stripped.startswith(('"', "'", "f\"", "f'", "#")):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_the_study_imports_no_live_regime_or_scheduler_module():
    code = _code_lines()
    for banned in ("import regime", "from regime", "regime_unified",
                   "hurst_gate_state", "config.example.json"):
        assert banned not in code, banned


def test_the_study_writes_only_its_own_two_artifacts():
    code = _code_lines()
    writes = [ln for ln in code.split("\n") if 'open(' in ln and '"w"' in ln]
    assert writes, "the study must write its own report and JSON"
    for line in writes:
        assert "args.json_out" in line or "args.report_out" in line, line


def test_the_live_payload_key_is_never_written_by_this_study():
    assert 'metrics["hurst_rs"]' not in _code_lines()
    assert "hurst_rs" in study.NON_GOALS


# --- the contract path ------------------------------------------------------

def test_1474_does_not_default_to_the_contract_path():
    assert os.path.abspath(study._DEFAULT_REPORT_OUT) != os.path.abspath(CONTRACT)
    assert study._DEFAULT_REPORT_OUT.endswith("hurst_1474_rs_estimator.md")
    assert study._DEFAULT_JSON_OUT.endswith("hurst_1474_rs_estimator.json")


def test_1424_still_owns_the_contract_path():
    assert os.path.abspath(study1424._DEFAULT_REPORT_OUT) == \
        os.path.abspath(CONTRACT)


def test_1474_may_not_write_the_contract_path_even_when_asked(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--report-out", CONTRACT, "--write-report",
                    "--json-out", str(tmp_path / "x.json")])
    assert "DEFERS" in str(excinfo.value)
    assert study.CONTRACT_REPORT_BASENAME in str(excinfo.value)


def test_the_contract_refusal_survives_render_only(tmp_path):
    payload = tmp_path / "p.json"
    payload.write_text("{}")
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--render-only", "--json-out", str(payload),
                    "--report-out", CONTRACT, "--write-report"])
    assert "DEFERS" in str(excinfo.value)


def test_the_contract_refusal_is_checked_before_every_other_refusal(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--report-out", CONTRACT, "--only", "momentum",
                    "--n-perm", "3", "--json-out", str(tmp_path / "x.json")])
    assert "DEFERS" in str(excinfo.value)


def test_the_contract_path_statement_names_who_keeps_it():
    assert "hurst_1424_gate_resolution.py" in study.CONTRACT_PATH_STATEMENT
    assert "DEFERS" in study.CONTRACT_PATH_STATEMENT


# --- committed-artifact protection -----------------------------------------

def test_a_scoped_run_may_not_overwrite_the_committed_json():
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--only", "momentum"])
    assert "refusing to overwrite the committed aggregate" in str(excinfo.value)


@pytest.mark.parametrize("flag,value", [
    ("--only", "momentum"),
    ("--datasets", "BTC/USDT:1h"),
    ("--windows", "2021"),
    ("--hurst-windows", "128"),
])
def test_every_scoping_flag_protects_the_committed_report(tmp_path, flag, value):
    with pytest.raises(SystemExit) as excinfo:
        study.main([flag, value, "--json-out", str(tmp_path / "x.json")])
    assert "refusing to target the committed report" in str(excinfo.value)


@pytest.mark.parametrize("argv,needle", [
    (["--n-perm", "5"], "--n-perm 5"),
    (["--n-perm-mde", "5"], "--n-perm-mde 5"),
    (["--seed", "1"], "--seed 1"),
    (["--bias-draws", "5"], "--bias-draws 5"),
    (["--no-mirror-check"], "--no-mirror-check"),
])
def test_a_deviating_run_may_not_write_the_committed_artifacts(tmp_path, argv,
                                                               needle):
    with pytest.raises(SystemExit) as excinfo:
        study.main(argv)
    message = str(excinfo.value)
    assert "refusing to overwrite the committed aggregate" in message
    assert needle in message


def test_stating_the_pre_registered_settings_explicitly_is_not_a_deviation():
    class _Args:
        n_perm = study.N_PERM
        n_perm_mde = study.N_PERM_MDE
        seed = study.SEED
        bias_draws = study.BIAS_DRAWS
        no_mirror_check = False

    assert study.inference_deviations(_Args()) == []


def test_render_only_refuses_an_unstamped_payload_on_the_committed_report(tmp_path):
    payload = tmp_path / "p.json"
    payload.write_text(json.dumps({"run_summary": {"scope": {"complete": False}}}))
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--render-only", "--json-out", str(payload),
                    "--write-report"])
    assert "not stamped as a complete run" in str(excinfo.value)


def test_render_only_refuses_a_payload_not_stamped_pre_registered(tmp_path):
    payload = tmp_path / "p.json"
    payload.write_text(json.dumps({"run_summary": {
        "scope": {"complete": True, "pre_registered_inference": False}}}))
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--render-only", "--json-out", str(payload),
                    "--write-report"])
    assert "pre-registered inference" in str(excinfo.value)


def test_render_only_to_the_committed_report_needs_write_report(tmp_path):
    payload = tmp_path / "p.json"
    payload.write_text(json.dumps({"run_summary": {
        "scope": {"complete": True, "pre_registered_inference": True}}}))
    with pytest.raises(SystemExit) as excinfo:
        study.main(["--render-only", "--json-out", str(payload)])
    assert "--write-report" in str(excinfo.value)


# --- the report -------------------------------------------------------------

def _render_payload():
    with open(study._DEFAULT_JSON_OUT) as fh:
        return json.load(fh)


def test_the_registry_row_records_the_second_estimator_and_the_deferral():
    path = os.path.join(os.path.dirname(study.__file__), "..", "..", "docs",
                        "backtesting-registry.md")
    with open(os.path.abspath(path)) as fh:
        rows = [ln for ln in fh if "hurst_1474_rs_estimator.py" in ln]
    assert len(rows) == 1
    row = rows[0]
    assert "hurst_rescaled_range" in row
    assert "Anis-Lloyd" in row
    assert "DEFERS" in row and "hurst_gate_calibration.md" in row
    assert "#1474" in row
    assert "one-shot" in row


def test_the_committed_run_exists_and_is_complete_and_pre_registered():
    payload = _render_payload()
    scope = payload["run_summary"]["scope"]
    assert scope["complete"] is True
    assert scope["pre_registered_inference"] is True
    assert payload["issue"] == study.ISSUE
    assert payload["schema_version"] == study.SCHEMA_VERSION


def test_the_committed_run_scores_the_1424_pool():
    payload = _render_payload()
    pre = payload["pre_registered"]
    assert pre["primary_config_id"] == study.PRIMARY_CONFIG_ID
    assert pre["hurst_windows"] == list(study.HURST_WINDOWS)
    assert pre["agreement_windows"] == list(study.AGREEMENT_WINDOWS)
    assert pre["estimators"] == list(study.ESTIMATORS)
    assert payload["run_summary"]["legs"] > 0
    assert payload["run_summary"]["pooled_primary"][study.PRIMARY_FAMILY] > 0


def test_the_committed_bias_table_reports_every_estimator_at_every_n():
    bias = _render_payload()["bias"]
    assert bias["sample_sizes"] == list(study.BIAS_SAMPLE_SIZES)
    for n in study.BIAS_SAMPLE_SIZES:
        row = bias["by_n"][str(n)]
        assert sorted(row) == sorted(study.ESTIMATORS)
        for stats in row.values():
            assert stats["mean"] is not None
            assert stats["iqr"] is not None


def test_the_committed_separation_prints_both_estimators_row_matched():
    separation = _render_payload()["separation"]
    for hw in study.HURST_WINDOWS:
        block = separation["by_window"][str(hw)]
        counts = {m["n_rows"] for m in block["measurements"].values()}
        assert len(counts) == 1
        for est in study.ESTIMATORS:
            measurement = block["measurements"][est]
            assert measurement["separation"] is not None
            assert measurement["p"] is not None


def test_the_committed_verdict_is_what_the_current_rule_produces():
    payload = _render_payload()
    assert study.estimator_risk_verdict(payload["separation"]) == \
        payload["verdict"]


def test_the_committed_report_is_what_the_committed_json_renders():
    payload = _render_payload()
    with open(study._DEFAULT_REPORT_OUT) as fh:
        assert fh.read() == study.report_from_payload(payload)


def test_the_committed_report_never_licenses_a_threshold():
    with open(study._DEFAULT_REPORT_OUT) as fh:
        report = fh.read()
    assert "NON-GOALS" in report
    assert "hurst_gate_calibration.md" in report
    assert "cannot run live" in report
    assert "1. Agreement between the two estimators" in report
    assert "2. Bias and spread on a memoryless series" in report
    assert "3. Gate separation under each estimator" in report
