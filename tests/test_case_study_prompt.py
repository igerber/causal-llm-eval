"""Unit tests for ``prompts/case_study/v2.txt``.

The Phase 1 case-study task prompt's content is load-bearing for the eval's
central scientific claim. Tests assert:

  1. The file exists, is non-empty, and is concise.
  2. It mentions the actual dataset columns (so the agent reads them, not
     a substitute), the ``data.parquet`` filename, the requested ATT
     output, and the ``solution.py`` save target.
  3. It contains NO library names and NO estimator-class names — both
     arms must see identical text or the comparator-fairness claim
     ("library design measurably shifts estimator choice") is invalidated.
  4. The PR #6 stub at ``prompts/case_study/v1.txt`` remains untouched
     (the v1 registry id is reserved per the PR #6 convention).
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "case_study"
_PROMPT_V2 = _PROMPTS_DIR / "v2.txt"
_PROMPT_V1 = _PROMPTS_DIR / "v1.txt"

# Tokens whose presence proves the prompt actually documents the dataset.
# Drives the agent toward THIS dataset rather than a substitute.
_REQUIRED_TOKENS: tuple[str, ...] = (
    "data.parquet",
    "unit",
    "period",
    "outcome",
    "first_treat",
    "treated",
    "ATT",
    "solution.py",
)

# Comparator-fairness denylist. Each token names a specific library,
# estimator class, or methodology bucket. Including any of these in the
# prompt would PRE-DISCLOSE what the rubric grades, invalidating the
# eval's sharp claim. Match is case-insensitive.
#
# Note: pandas, numpy, and pyarrow are NOT denylisted — they are
# language-level infrastructure both arms use; the deliberate framing is
# "read data.parquet, choose your methodology" and we want the agent
# to do that with whatever tools it knows.
_LIBRARY_AND_ESTIMATOR_DENYLIST: tuple[str, ...] = (
    # Libraries
    "diff_diff",
    "diff-diff",
    "statsmodels",
    "linearmodels",
    "econml",
    "dowhy",
    "fixest",
    # diff_diff-native estimator names
    "TwoWayFixedEffects",
    "DifferenceInDifferences",
    "CallawaySantAnna",
    "SunAbraham",
    "ChaisemartinDHaultfoeuille",
    "SyntheticDiD",
    "BaconDecomposition",
    "HonestDiD",
    "MultiPeriodDiD",
    # Methodology brand names / shorthand a prompt could leak
    "TWFE",
    "Callaway",
    "Santanna",
    "Sant'Anna",
    "Sun Abraham",
    "Sun-Abraham",
    "fixed effect",
    "fixed effects",
    "difference-in-differences",
    # PR #7 R1 P2 (review-rubric expansion): additional terms a prompt
    # could leak that would pre-disclose diagnostics or methodology
    # the rubric grades. Names of pre-trends / parallel-trends /
    # sensitivity / robustness machinery that an arm-specific guide
    # might emphasize.
    "pre-trends",
    "pretrends",
    "parallel trends",
    "sensitivity",
    "dCDH",
    "Chaisemartin",
    "honest",
    "placebo",
    # Statsmodels-native names
    "OLS",
    "WLS",
    "GLS",
    "GLSAR",
    "GLM",
    "MixedLM",
    "RegressionResults",
    "OLSResults",
    # Guide-file leak
    "llms.txt",
    "llms-practitioner",
    "llms-autonomous",
    "llms-full",
    "get_llm_guide",
)


def _read_v2() -> str:
    return _PROMPT_V2.read_text()


def test_prompt_v2_exists_and_non_empty() -> None:
    assert _PROMPT_V2.exists(), f"{_PROMPT_V2} missing"
    assert _PROMPT_V2.stat().st_size > 0, "v2.txt is empty"


def test_prompt_v2_mentions_required_tokens() -> None:
    """Every token in ``_REQUIRED_TOKENS`` must appear at least once."""
    body = _read_v2()
    missing = [t for t in _REQUIRED_TOKENS if t not in body]
    assert not missing, (
        f"Prompt v2 missing required tokens: {missing}. The agent needs to "
        f"see these to locate the dataset and produce the requested output."
    )


def test_prompt_v2_does_not_name_libraries_or_estimators() -> None:
    """Comparator-fairness assertion: no library / estimator / methodology
    brand name appears in the prompt. Match is case-insensitive."""
    body_lower = _read_v2().lower()
    leaks = [t for t in _LIBRARY_AND_ESTIMATOR_DENYLIST if t.lower() in body_lower]
    assert not leaks, (
        f"Prompt v2 leaks library/estimator names: {leaks}. The central "
        f"scientific claim requires identical-across-arms framing; naming "
        f"any of these pre-discloses what the rubric grades and invalidates "
        f"the case study. If a term is innocuous and must appear, add a "
        f"narrow exception in this test with rationale."
    )


def test_prompt_v2_concise() -> None:
    """Prompts longer than ~30 lines tend to drift into method-scaffolding.
    Soft compactness signal — bump the cap deliberately if the prompt
    grows for a documented reason."""
    line_count = len(_read_v2().splitlines())
    assert line_count <= 35, (
        f"Prompt v2 is {line_count} lines (cap 35). Long prompts risk "
        f"introducing method scaffolding that biases the eval; trim or "
        f"explicitly bump the cap."
    )


def test_prompt_v1_still_reserved_stub() -> None:
    """The PR #6 stub at ``v1.txt`` must remain in place; the
    ``case_study/v1`` registry id is reserved per the PR #6 convention."""
    assert _PROMPT_V1.exists(), f"{_PROMPT_V1} (PR #6 reserved stub) missing"
    body = _PROMPT_V1.read_text()
    assert "STUB ONLY" in body, (
        "v1.txt no longer contains 'STUB ONLY'; PR #6 reserved this slot "
        "and it must not be filled in place (would silently invalidate "
        "any records that reference the v1 id)."
    )
