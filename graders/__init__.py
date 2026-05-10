"""Graders apply pre-defined rubrics to per-run records.

Phase 1 uses one grader: an AI judge (Claude API direct call) that produces
structured JSON matching the rubric schema. Phase 2 adds human-rater interfaces
and computes inter-rater reliability (Cohen's kappa) across raters.
"""
