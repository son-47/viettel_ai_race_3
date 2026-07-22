# ai-race-2026 — Serving optimization cho Qwen3.5 / Qwen3-Next MoE trên trace Kimi

Bộ công cụ đo lường + tối ưu serving stack cho bài toán: phục vụ LLM MoE + hybrid-attention
sao cho tối đa **goodput-under-SLO** (kèm throughput & chất lượng) trên GPU hữu hạn, với workload
là production trace kiểu Kimi (hội thoại ngắn, agent system-prompt dài, doc-QA 100K token).

- **Chiến lược**: [PLAN.md](PLAN.md)
- **Kịch bản tác động từng đòn bẩy** (định lượng + điều kiện + ngưỡng): [IMPACT_SCENARIOS.md](IMPACT_SCENARIOS.md)
- **Kịch bản chạy turnkey trên H200** (ablation ladder): [RUN_PLAN.md](RUN_PLAN.md)
- **Nền tảng kỹ thuật** (landscape): [research.md](research.md) · ghi chú research có nguồn: [research/](research/)
- **Trạng thái Phase 0**: [PHASE0_REPORT.md](PHASE0_REPORT.md)

## Cài đặt
```bash
pip install -r requirements.txt   # aiohttp, numpy, pandas, matplotlib
```
Harness không cần GPU/model để chạy thử (dùng mock server). Serving thật cần H200 + vLLM/SGLang.

## Cấu trúc
```
config/slo.json          SLO per-regime + trọng số scoring (PLACEHOLDER — thay bằng spec thật)
harness/
  trace.py               schema + loaders (JSONL + Mooncake) + prompt synthesis
  regime.py              phân loại 3 regime corpus-level (source-aware + hot-prefix)
  gen_trace.py           sinh trace tổng hợp 3 regime (python -m harness.gen_trace)
  merge_traces.py        hợp nhất 3 trace Kimi thật -> workload thống nhất (python -m harness.merge_traces)
  metrics.py             TTFT / TPOT / E2E + percentiles
  scoring.py             composite score (goodput + throughput + quality)
  replay.py              open-loop replay tới endpoint OpenAI-compatible (python -m harness.replay)
  mock_server.py         LLM giả lập tải-phụ-thuộc để validate pipeline (python -m harness.mock_server)
analysis/analyze_trace.py  thống kê phân phối/burstiness/prefix-sharing + plots
eval/
  tasks.py               bộ eval guardrail: GSM8K + RULER NIAH + chat probes + scorers
  run_eval.py            chạy guardrail 1 endpoint, hoặc baseline-vs-candidate gate theo budget
  quality.py             A/B agreement đơn giản (phụ)
research/                ghi chú research có nguồn cho từng cluster đòn bẩy
serve/serve_{vllm,sglang}.sh  template khởi động baseline trên H200
scripts/demo.sh          validate end-to-end local
data/mooncake/           3 trace Kimi/Mooncake thật (tải về)
```

## Quy trình
```bash
# 1) Validate pipeline ngay trên máy thường (không cần GPU)
make demo

# 2) Phân tích workload
make gen && make analyze          # -> results/trace_summary.json, results/trace_analysis.png

# 3) Đo một serving config thật (trên H200)
bash serve/serve_sglang.sh        # hoặc serve_vllm.sh
python -m harness.replay --trace data/kimi.jsonl --format mooncake \
       --base-url http://localhost:8000 --out results/baseline.json

# 4) Quét tải để tìm "goodput knee"
for rs in 1 2 4 8; do
  python -m harness.replay --trace data/kimi.jsonl --rate-scale $rs --out results/rs$rs.json
done
```

## Khái niệm chính
- **Open-loop replay**: request bắn theo đúng thời điểm arrival, không chờ request trước xong →
  đo đúng hành vi server dưới tải (closed-loop sẽ che giấu queueing). `--rate-scale > 1` nén thời gian = tăng tải.
- **Composite score**: `w_goodput·goodput + w_throughput·thr_norm + w_quality·quality` (xem `config/slo.json`).
  Goodput = tỉ lệ request đạt SLO (TTFT/TPOT/E2E) theo regime.
- **Regime**: phân loại theo `input_tokens` (chat ≤4K, agent ≤32K, docqa còn lại) trừ khi trace đã có field `regime`.

## ⚠️ Placeholder cần thay khi có spec cuộc thi
- `config/slo.json` — ngưỡng SLO & trọng số chấm điểm.
- `harness/scoring.py::score()` — nếu công thức chính thức khác.
- `eval/quality.py` — thay PROBES + metric bằng eval có nhãn thật (LongBench/RULER/GSM8K).
- `serve/*.sh` — model id, số GPU, và kiểm tra tên flag theo phiên bản framework.
