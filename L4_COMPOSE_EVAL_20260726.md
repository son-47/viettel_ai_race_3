# L4 compose evaluation — 2026-07-26

## Thiết lập

- GPU: NVIDIA L4, 23,034 MiB, driver 580.173.02.
- Host: Lightning Studio qua SSH.
- Workload: replay đúng 70 conversations × 6 turns = 420 requests, seed 42,
  shared system 1,000 tokens, conversation prefix 1,000 tokens, 150 user
  tokens/turn, 300 output tokens pinned, `RATE_SCALE=1`.
- Mỗi case chạy trong một Compose project riêng, mount cùng checkpoint
  `LiquidAI/LFM2.5-1.2B-Instruct`, teardown sau khi hoàn thành.
- `ERS` bên dưới là điểm của synthetic replay trong harness, không phải điểm
  private portal. Tất cả case đều hoàn tất 420/420 và không có error.

## Kết quả

| Compose | ERS | Δ ERS vs 63.94 | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
|---|---:|---:|---:|---:|---:|---:|
| `docker-compose_rmsnorm_fp8_candidate.yml` | 0.425014 | −1.856% | 50.061 ms | 71.304 ms | 7.800 ms | 9.000 ms |
| `docker-compose_silu_fp8_candidate.yml` | **0.436072** | **+0.697%** | 43.718 ms | 71.817 ms | **7.822 ms** | 8.909 ms |
| `docker-compose_shortconv_fused_64.35.yml` | 0.434950 | +0.438% | 43.733 ms | 72.499 ms | 7.842 ms | 8.911 ms |
| `docker-compose_63.94.yml` | 0.433053 | baseline | 43.816 ms | 71.559 ms | 7.899 ms | 8.944 ms |
| `docker-compose_processing.yml` | 0.422525 | −2.431% | 49.916 ms | 71.647 ms | 7.942 ms | 9.131 ms |

## Nhận xét

1. Trên L4, candidate SiLU-FP8 đứng đầu trong bốn compose, cải thiện ERS
   khoảng 0.70% so với baseline và khoảng 0.26% so với image ShortConv/QK.
2. ShortConv/QK fusion vẫn có lợi ích nhỏ, nhưng thấp hơn candidate mới.
3. `docker-compose_processing.yml` làm TTFT p50 tăng khoảng 6.1 ms và ERS
   giảm 2.43%; không nên chọn cấu hình này cho portal.
4. Kết quả này xác nhận candidate chạy được trên SM89/L4; vẫn cần xem đây là
   xếp hạng tham khảo cho H200 MIG, không suy diễn trực tiếp thành điểm portal.
5. Candidate RMSNorm→FP8 mới hoàn tất 420/420 nhưng ERS 0.425014, thấp hơn
   SiLU-FP8 khoảng 2.54% và thấp hơn baseline 63.94 khoảng 1.86%; chưa nên
   promote candidate này lên portal. TTFT p50 của nó cũng cao hơn 6.34 ms so
   với SiLU-FP8. Kết quả dùng scheduler của compose 65.71 (max-model-len 8192,
   max-num-batched-tokens 4096, max-num-seqs 32), nên đây là A/B tham khảo chứ
   không phải cô lập riêng chi phí RMSNorm.

## Artifact

JSON và server log của các case trước đã kéo về `eval/l4_compose_20260726/`; JSON
candidate RMSNorm mới là `eval/l4_compose_20260726/rmsnorm_fp8.json`. Runner provisioning
nằm ở `scripts/run_l4_compose_benchmark.sh`; override chỉ mount model/cache/
benchmark và không sửa bốn compose gốc.
