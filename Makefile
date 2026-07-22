PY ?= python
TRACE ?= data/synthetic_trace.jsonl
BASE_URL ?= http://localhost:8000

.PHONY: help gen analyze mock replay demo clean

help:
	@echo "make gen      - generate synthetic 3-regime trace -> $(TRACE)"
	@echo "make analyze  - analyze TRACE (stats + plots in results/)"
	@echo "make mock     - run the mock streaming server on :8000 (foreground)"
	@echo "make replay   - open-loop replay TRACE against BASE_URL and score"
	@echo "make demo     - full local validation (mock server + replay + score)"
	@echo "make clean    - remove generated data/results"

gen:
	$(PY) -m harness.gen_trace --n 2000 --rps 12 --out $(TRACE)

analyze:
	$(PY) analysis/analyze_trace.py --trace $(TRACE)

mock:
	$(PY) -m harness.mock_server --port 8000

replay:
	$(PY) -m harness.replay --trace $(TRACE) --base-url $(BASE_URL) --out results/run.json

demo:
	bash scripts/demo.sh

clean:
	rm -rf data/*.jsonl results/*.json results/*.png
