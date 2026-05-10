.PHONY: help smoke test lint format type case-study-v1 calibration preflight clean

help:
	@echo "Available targets:"
	@echo "  smoke           - smoke test: verify cold-start invocation + inheritance probe"
	@echo "  test            - run pytest suite (excludes slow + live tests)"
	@echo "  lint            - run ruff check"
	@echo "  format          - run black + ruff format"
	@echo "  type            - run mypy"
	@echo "  preflight       - run comparator-asymmetry pre-flight (3-5 sample agents per arm)"
	@echo "  calibration     - run DGP calibration loop (statsmodels arm vs candidate DGP)"
	@echo "  case-study-v1   - run the full Phase 1 case study (N=15 per arm = 30 runs)"
	@echo "  clean           - remove venv pool, run outputs, caches"

smoke:
	@echo "Smoke test not yet implemented (Phase 0 skeleton)."
	@echo "Will assert: cold-start invocation includes --bare; inheritance probe returns no leakage."
	@false

test:
	pytest -v --tb=short -m "not slow and not live"

lint:
	ruff check harness graders analysis tests

format:
	black harness graders analysis tests
	ruff check --fix harness graders analysis tests

type:
	mypy harness graders analysis

preflight:
	@echo "Comparator-asymmetry pre-flight not yet implemented (Phase 0 skeleton)."
	@false

calibration:
	@echo "DGP calibration loop not yet implemented (Phase 0 skeleton)."
	@false

case-study-v1:
	@echo "Case study runner not yet implemented (Phase 0 skeleton)."
	@false

clean:
	rm -rf .venv-pool/
	rm -rf runs/case_study_v*/
	rm -rf runs/preflight/
	rm -rf runs/scratch/
	rm -rf .pytest_cache/
	find . -name "__pycache__" -type d -exec rm -rf {} +
