# LFM2.5 / Viettel AI Race 2026 — báo cáo hợp nhất tối ưu suy luận

Ngày hợp nhất: 2026-07-29  
Phạm vi nguồn:

- `L4_COMPOSE_EVAL_20260726.md`
- `L4_LFM25_CONFIG_REPORT.md`
- `LFM25_KERNEL_FUSIONS_20260726.md`
- `LFM25_MASTER_OPTIMIZATION_SUMMARY_20260726.md`
- `LFM25_SPECULATIVE_AND_NEXT_OPTIMIZATIONS_20260728.md`

> Các kết quả trong tài liệu được chia thành **điểm portal H200** và **ERS
> synthetic trên L4**. ERS L4 chỉ dùng để xếp hạng tương đối, phát hiện crash
> và đo xu hướng; không được coi là điểm portal.

## 1. Kết luận điều hành

### Cấu hình nên giữ để nộp

Mốc tốt nhất đã được portal xác nhận là:

```text
final_score: 65.71
TTFT p50/p95: 32/47 ms
TBT median: 4 ms
failed: 4/420
accuracy_drop: 0
penalty: 1
f_delta: 1
image: misokaio/ghfjdk
digest: sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4
compose: submission/docker-compose_silu_fp8_65.71.yml
```

Image này gộp ShortConv fusion, Q/K RMSNorm + NeoX RoPE fusion, bỏ singleton
`torch.vstack` và SiLU×Mul + dynamic per-token FP8. Tuy nhiên compose 65.71
cũng đổi scheduler/batching sang `8192/4096/32`, nên số điểm 65.71 chưa phải
ablation cô lập riêng cho SiLU-FP8.

Mốc kernel fusion trước đó đạt 64.35, cao hơn control 63.94 nhưng có 6 failed
request. Candidate ShortConv decode→FP8 out-projection đã được build/public và
portal chấm **59.95** (53/71 ms, TBT 4 ms, 5 failed/420, accuracy_drop 0), thấp
hơn control nên đã loại. External draft `LFM2.5-350M` cũng bị loại sau lượt
portal 34.9. Vì vậy 65.71 vẫn là fallback/production choice; mọi kernel mới
phải A/B xen kẽ với image này.

### Quyết định ngắn gọn

- Giữ đúng runtime flags của compose 65.71 làm control H200:
  `max-model-len=8192`, `max-num-batched-tokens=4096`, `max-num-seqs=32`,
  FCFS, chunked prefill và prefix caching.
- Giữ image/compose 65.71 làm phương án nộp đã có bằng chứng portal.
- Không dùng `processing`, interactivity, custom scheduler, `4096/32`,
  `8192/24`, Medusa hoặc external draft.
- Không dùng GPTQ/AWQ offline nếu BTC chưa xác nhận bằng văn bản rằng checkpoint
  biến đổi ngoài online quantization là hợp lệ.
- Mọi candidate mới phải dùng image pin theo digest, không dùng mutable tag, và
  phải qua smoke, paired A/B, failed-count gate và accuracy/output parity.

## 2. Bài toán, workload và hàm điểm

Model bắt buộc là `LiquidAI/LFM2.5-1.2B-Instruct`, chạy bằng vLLM. Kiến trúc
hybrid có 16 decoder layer: 10 block ShortConv/recurrent và 6 block GQA/
attention. Mỗi block ShortConv có các projection `in_proj`/`out_proj`; tổng cộng
20 ShortConv projection linear cần được xem xét khi quantize.

Workload portal:

- 70 conversation × 6 turn = **420 request**.
- Shared system prefix: 1.000 token.
- Prefix riêng mỗi conversation: 1.000 token.
- Mỗi turn thêm 150 user token.
- Mỗi request bị pin đúng 300 output token; tổng decode tối đa 126.000 token.
- Poisson arrival, seed 42; context cuối turn khoảng 4.400 input token trước
  chat-template/output overhead.
- Request lỗi, timeout, thiếu turn hoặc trả 0 token nhận 0 điểm.
- Ngưỡng hiện tại: TTFT 10–400 ms, TPOT/TBT 1–10 ms, gamma 2, trọng số
  TTFT/TPOT = 0,5/0,5.

Prefix caching và chunked prefill giúp phần prefill, nhưng không làm 300 token
decode nhanh hơn. Vì vậy sau khi flags và online quantization gần bão hòa, decode
TPOT/launch overhead là đòn bẩy chính.

Ràng buộc grader H200 được ghi nhận: 1 MiG H200, 18 GB VRAM, 3 CPU, 8 GB RAM,
Ubuntu 24.04, driver 590.x, tensor parallel 1. Harness cũ
`harness/scoring.py` dùng ngưỡng TTFT/TPOT cũ và không được dùng để quyết định
candidate hiện tại.

## 3. Kết quả portal H200 và cấu hình runtime

### Các mốc chính

| Mốc | Kết quả | TTFT p50/p95 | TBT median | Failed | Quyết định |
|---|---:|---:|---:|---:|---|
| Stock FP8 exact | 63.82 | — | — | — | Mốc sớm |
| Stock/control history | 63.94 | 39/52 ms | 4 ms | 5 | Control cho fusion |
| Warmup nội bộ | 62.97 | xấu hơn | — | — | Loại |
| HTTP keep-alive 600 | 62.16 | — | — | — | Không tái hiện trên H200, loại |
| ShortConv + QK + no-vstack | **64.35** | **38/51 ms** | 4 ms | 6 | Gain có thật nhưng cần lặp |
| Combined SiLU-FP8 | **65.71** | **32/47 ms** | 4 ms | 4 | Phương án tốt nhất hiện tại |

### H200 A/B với cùng image `misokaio/ghfjdk:v0.25.2`

| Compose | Scheduler | `max-model-len` | Batch tokens | ERS | TTFT p50 | TTFT p95 | TBT median | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `docker-compose copy.yml` | stock async FCFS | 32768 | 8192 | **63.76** | **37 ms** | **58 ms** | 4 ms | 5/420 |
| custom priority v5 | score-aware | 8192 | 4096 | 61.78 | 46 ms | 60 ms | 4 ms | 5/420 |
| `v0252-interactivity-only.yml` | stock + interactivity | 32768 | 8192 | 59.87 | 50 ms | 70 ms | 4 ms | 6/420 |
| A/B batch tokens | stock async FCFS | 32768 | 4096 | 60.74 | 48 ms | 64 ms | 4 ms | 5/420 |
| `v0252-seqs24.yml` | stock async FCFS | 32768 | 8192 | 60.34 | 47 ms | 68 ms | 4 ms | 6/420 |

Kết luận chắc chắn từ A/B H200: stock FCFS + `8192/32` thắng custom priority +
`4096/32`; giữ riêng `8192` tốt hơn `4096` **3.02 điểm**, TTFT p50 tốt hơn
11 ms và p95 tốt hơn 6 ms. Giảm `max-num-seqs` từ 32 xuống 24 cũng làm điểm
giảm 3.42 điểm. Không thử `4096/24` vì cả hai thành phần đều đã cho tín hiệu
âm. Năm lỗi tương đương trần mất điểm `5/420 × 100 = 1.19` điểm; cần log
per-request để phân loại timeout, HTTP error, 0-token hay lỗi model trước khi
tối ưu failure count.

`config_hash` giống nhau dù command khác nhau là giá trị opaque của grader,
không dùng để suy luận runtime config giống nhau.

### Baseline H200 được chọn

```text
--quantization=fp8
--max-model-len=32768
--max-num-batched-tokens=8192
--max-num-seqs=32
--optimization-level=3
--gpu-memory-utilization=0.85
--enable-prefix-caching
--enable-chunked-prefill
tensor parallel = 1
```

Không ép FlashInfer, không bật KV quantization, CPU/NVMe offload, custom
scheduler, internal warmup hoặc `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600`. Interactivity
và `VLLM_USE_SPINLOOP_EXT=1` bị loại: source audit không cho thấy spinloop nằm
trên critical path frontend–EngineCore ZMQ với TP=1.

## 4. Benchmark L4: lịch sử và compose evaluation

Máy L4: NVIDIA L4 23.034 MiB, driver 580.x/CUDA 13.0, Docker 28.0.1,
container giới hạn 3 CPU/8 GB RAM. L4 không phải H200 MIG; mọi ERS dưới đây là
synthetic replay, không bao gồm accuracy penalty và không phải private portal.

### Sweep lịch sử

Mọi case đạt 330/330 request và prompt trung bình 4001,6 token; tải chính 2×,
trừ dòng cuối 1×.

| Case | Rate | ERS | TTFT mean/p95 | TPOT mean/p95 |
|---|---:|---:|---:|---:|
| BF16, PC off, 8192/16 | 2× | 0.01633 | 884.6/2007.7 ms | 23.12/27.51 ms |
| BF16, PC on, 4096/32 | 2× | 0.04377 | 572.8/1356.8 ms | 15.83/22.23 ms |
| BF16, PC on, 8192/8 | 2× | 0.00141 | 7420.7/10422.8 ms | 17.24/18.01 ms |
| BF16, PC on, 8192/16 | 2× | 0.03649 | 687.3/1475.1 ms | 17.30/23.10 ms |
| BF16, PC on, 8192/32 | 2× | 0.03179 | 805.7/1777.3 ms | 16.40/24.25 ms |
| FP8, PC on, 8192/16 | 2× | 0.08490 | 395.5/877.5 ms | 11.15/14.81 ms |
| FP8, PC on, 4096/16 | 2× | 0.08989 | 332.2/659.3 ms | 11.08/15.19 ms |
| **FP8, PC on, 4096/32** | **2×** | **0.09476** | **335.6/789.4 ms** | **10.95/14.85 ms** |
| FP8, PC on, 8192/32 | 2× | 0.06738 | 476.8/1132.3 ms | 11.24/16.04 ms |
| FP8, PC on, 8192/60 | 2× | 0.08763 | 440.8/1132.2 ms | 11.05/16.09 ms |
| **FP8, PC on, 4096/32** | **1×** | **0.13928** | **254.4/625.9 ms** | **9.22/12.83 ms** |

Kết quả cũ của L4 cho thấy FP8 + PC + `4096/32` thắng ở tải đó: ERS 0.09476
so với 0.06738 ở `8192/32`, TTFT mean giảm 29% và TPOT giảm nhẹ. Nhưng H200
hiện tại đảo thứ hạng; không chuyển nguyên kết luận L4 sang portal.

### Compose evaluation L4 ngày 2026-07-26

Replay đúng 70 conversation × 6 turn = 420 request, `RATE_SCALE=1`, output
300 token, tất cả case hoàn tất 420/420.

| Compose | ERS proxy | Δ vs 63.94 | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
|---|---:|---:|---:|---:|---:|---:|
| `docker-compose_rmsnorm_fp8_candidate.yml` | 0.425014 | −1.856% | 50.061 ms | 71.304 ms | 7.800 ms | 9.000 ms |
| `docker-compose_silu_fp8_candidate.yml` | **0.436072** | **+0.697%** | 43.718 ms | 71.817 ms | **7.822 ms** | 8.909 ms |
| `docker-compose_shortconv_fused_64.35.yml` | 0.434950 | +0.438% | 43.733 ms | 72.499 ms | 7.842 ms | 8.911 ms |
| `docker-compose_63.94.yml` | 0.433053 | baseline | 43.816 ms | 71.559 ms | 7.899 ms | 8.944 ms |
| `docker-compose_processing.yml` | 0.422525 | −2.431% | 49.916 ms | 71.647 ms | 7.942 ms | 9.131 ms |

SiLU-FP8 đứng đầu L4, hơn baseline khoảng 0,70%; ShortConv/QK fusion chỉ hơn
0,44%. RMSNorm thấp hơn SiLU khoảng 2,54% và TTFT p50 chậm hơn 6,34 ms. Kết quả
RMSNorm dùng scheduler `8192/4096/32`, nên không cô lập hoàn toàn chi phí kernel.
`processing` làm TTFT p50 tăng khoảng 6,1 ms và ERS giảm 2,43%, không chọn.

Artifact JSON/log: `eval/l4_compose_20260726/`; runner:
`scripts/run_l4_compose_benchmark.sh`.

## 5. Những gì đã sweep và đã loại

### Flags, scheduler và runtime

Đã thử batch token 4096/6144/7168/8192/9216/12288, sequence 24/40,
`max-model-len=8192`, block size 8/32, Mamba block 8/32/64,
`gpu-memory-utilization=0.92`, optimization level 2, CUDA graph capture 1–32,
output processing chunk 256, renderer/frontend variants, Rust frontend, MRV2,
cascade attention, DBO, reserve-full-ISL, hybrid KV manager, stream interval,
throughput/interactivity và FlashInfer ép buộc. Phần lớn là noise hoặc làm ERS
giảm; một số fail startup do backend/cache không đồng nhất.

Đã loại: warmup, keep-alive 600, custom score-aware scheduler v5,
interactivity, `4096/32`, `8192/24`, `4096/24`, `processing` và mọi flag chưa có
giả thuyết từ server/per-request log.

### Online quantization và KV cache

Đã thử FP8 per-tensor/per-block/per-channel, INT8 weight-only, MXFP8, FP8 KV và
INT8 per-token-head. `fp8_per_tensor` chỉ hơn stock `+0.000385 ERS` ở r1,
không giảm TPOT, coi là noise. FP8 per-block, INT8 weight-only, MXFP8 và FP8 KV
làm ERS giảm; FP8 KV còn thiếu calibration scale trong checkpoint và làm output
thay đổi. Không bật `--kv-cache-dtype=fp8`.

### Compliance và dữ liệu

- `data/model-real/config.json` là Qwen3.5, không phải LFM2.5; không dùng để
  benchmark bài này.
- Compose v2–v29 và nhiều research comment thuộc workload Qwen cũ, prompt tới
  27k token/SLO cũ; không chuyển kết luận sang LFM2.5.
- Không dùng `served-model-name=Qwen3.5-2B`, `LAMBDA_GATE`, cắt prompt,
  head+tail truncation, cap output, trace-only gate hoặc dual-path gaming.
- FP8 là latency winner nhưng phải chạy GPQA full cùng seed/template với BF16;
  chưa có accuracy validation đầy đủ trong các kết quả L4.

## 6. Kernel fusion 64.35

### ShortConv decode fusion

Đường gốc tạo các tensor tạm:

```text
B*x → causal_conv1d_update → recurrent state → C*output → vstack
```

Triton kernel gộp `B*x`, causal convolution width 3, state update và `C*output`
trong một launch. Guard: CUDA, width 3, không bias, decode thường, không
speculative accepted-token state, số decode token bằng số decode request và shape
đúng LFM2.5. Kernel giữ nguyên dtype/state rounding và không approximation.

### Q/K RMSNorm + NeoX RoPE

Kernel đọc trực tiếp hai view Q/K có stride từ packed QKV, tính RMSNorm FP32 và
sinh Q/K contiguous sau RoPE trong một launch. Guard giới hạn BF16/FP16, NeoX
RoPE và `rotary_dim == head_dim`; đường khác fallback vLLM.

### Bỏ singleton `vstack`

Ở iteration chỉ có một tensor decode hoặc prefill, bỏ `torch.vstack([tensor])`
bằng flag độc lập. Tất cả nhánh đều opt-in và có fallback nguyên bản.

### Bằng chứng portal và artifact

| Chỉ số | Control 63.94 | Fusion |
|---|---:|---:|
| Final score | 63.94 | **64.35** |
| TTFT p50/p95 | 39/52 ms | **38/51 ms** |
| TBT median | 4 ms | 4 ms |
| Failed | 5 | 6 |
| Accuracy drop | 0 | 0 |

Digest image fused theo báo cáo:

```text
misokaio/ghfjdk:v0.25.1-lfm25-fused
sha256:53d1892ca842ffa2f5e3113f0f775450701bacb7c014d8f497bd63e6ad61d401
```

Artifact: `submission/lfm25_fused_short_conv.py`,
`submission/lfm25_fused_qk_norm_rope.py`,
`submission/apply_lfm25_shortconv_fusion.py`,
`submission/Dockerfile.shortconv-fused`, compose, smoke tests và paired matrix.
Các biến rollback:

```text
VLLM_LFM25_BYPASS_SINGLE_VSTACK
VLLM_LFM25_FUSED_SHORTCONV
VLLM_LFM25_FUSED_QK_NORM_ROPE
```

Smoke GTX 1650 pass; portal 64.35 mới là bằng chứng end-to-end. Cần chấm lặp
hoặc paired control vì failed tăng một request. Nếu kernel fusion không thắng
trên H200, dùng Nsight Systems/CUDA-event để xác định launch còn chiếm tỷ trọng
đáng kể trước khi viết fusion thứ ba.

## 7. SiLU×Mul + dynamic per-token FP8 fusion

### Khoảng trống trong vLLM 0.25.1

Đường online per-token FP8 cũ:

```text
w13 FP8 GEMM → SiLU×gate ghi BF16 → đọc BF16 để quant E4M3 → w2 CUTLASS
```

Candidate gộp:

```text
w13 FP8 GEMM → fused SiLU×gate + row-absmax + E4M3 quant → w2 CUTLASS
```

Có thể bỏ tối đa 16 launch và lượt ghi/đọc tensor BF16 mỗi model step, khoảng
`16 × batch × 8192` phần tử (xấp xỉ 512 KiB × batch traffic trung gian).

Fast path chỉ bật khi `VLLM_LFM25_FUSED_SILU_FP8=1`,
`Fp8PerTensorOnlineLinearMethod`, `CutlassFP8ScaledMMLinearKernel`, TP=1 và
linear không bias; backend khác fallback nguyên bản.

Giữ nguyên hai lần rounding của `silu_and_mul`: SiLU về BF16/FP16, sau đó tích
với `up` làm tròn lần hai trước row maximum. Scale E4M3:

```text
scale = max(absmax / 448, 1 / (448 × 512))
q = clamp(rounded_activation / scale, -448, 448)
```

### Kiểm chứng

- SM90 compile pass với Triton 3.6.0; 48 registers, 32 bytes shared memory,
  không spill.
- GTX 1650: BF16 activation khớp tuyệt đối; scale error tối đa
  `1.862645149230957e-09`; FP16 activation error tối đa `2.44140625e-04`.
- E4M3 runtime trên SM75 bị skip vì Triton không hỗ trợ store E4M3 ở GPU đó.
- L4 SM89 chạy đủ 420 request; H200 output-FP8 correctness vẫn phải kiểm tra.

Artifact: `submission/lfm25_fused_silu_fp8.py`, patcher, test,
`Dockerfile.silu-fp8-fused`, compose và paired H200 matrix.

### Bằng chứng L4 và portal 65.71

L4 candidate SiLU đạt ERS 0.436072, TTFT p50 43.718 ms, TPOT p50 7.822 ms,
420/420. Portal 65.71 gộp cả kernel fusion và thay đổi scheduler `8192/4096/32`,
do đó cần paired same-image ablation trước khi quy toàn bộ +1.36 điểm so với
64.35 cho riêng SiLU.

## 8. Candidate RMSNorm→dynamic-FP8

Hướng này nối fused CUDA op add-RMSNorm-dynamic-FP8 có sẵn trong vLLM vào 16 W13
và 6 QKV projection của LFM2.5. L4 compose evaluation:

```text
ERS: 0.425014
TTFT p50/p95: 50.061/71.304 ms
TPOT p50/p95: 7.800/9.000 ms
420/420 request, không error
```

Kết quả thấp hơn SiLU-FP8 2,54% và thấp hơn baseline 63.94 khoảng 1,86%; TTFT
p50 chậm hơn SiLU 6,34 ms. Đây là A/B tham khảo vì dùng scheduler của compose
65.71 (`max-model-len=8192`, `max-num-batched-tokens=4096`, `max-num-seqs=32`),
không phải ablation hoàn toàn cô lập. Không promote chỉ dựa trên build/smoke
local. Artifact: `submission/lfm25_fused_rmsnorm_fp8.py`,
`submission/docker-compose_rmsnorm_fp8_candidate.yml` và JSON/log trong
`eval/l4_compose_20260726/rmsnorm_fp8.json`.

## 9. Speculative decoding, PLD và rollback

### Vì sao external draft bị loại

External `LFM2.5-350M` từng làm điểm portal rơi xuống **34.9**:

```text
TTFT p50: 32 → 67 ms
TBT median: 4 → 14 ms
Failed: 4 → 6
```

TPOT bị chặn về 0 khi từ 10 ms trở lên, nên TBT 14 ms đã mất gần nửa cơ hội
ERS trước cả failed penalty. Với target 1–2B, verify rẻ khiến chi phí drafter
dễ lớn hơn phần target compute tiết kiệm. Bài EACL 2026 báo cáo draft độc lập
Qwen 1.5B chỉ 0,83× và SmolLM 1.7B 0,67×; EAGLE-2 đạt khoảng 1,44–1,81× nhưng
cần train drafter riêng. PLD training-free đạt khoảng 1,25–1,52× trong bài,
nhưng không thể chuyển thẳng sang vLLM/LFM2.

### Lỗi correctness trong vLLM 0.25.1

LFM2 ShortConv có recurrent state. Bản vLLM hiện tại:

1. State chỉ rộng `conv_kernel - 1`, không giữ speculative tail để rollback.
2. `num_accepted_tokens` không tới `ShortConvAttentionMetadataBuilder`.
3. Khi draft bị reject, state vẫn chứa token chưa accept và greedy output có thể
   lệch từ lần reject đầu.

PR upstream #44296 sửa cả ba điểm. `accuracy_drop=0` trong record portal 34.9
không chứng minh candidate lossless vì lượt đó chưa phải GPQA gate và prompt lặp
có thể chưa kích hoạt rejection đa dạng. Issue #49112 còn chỉ ra hybrid target/
draft cần route metadata/block-table theo từng KV group và quản lý draft state
đầy đủ; patch hybrid hiện tại chưa đủ bằng chứng để promote.

### Ứng viên A: ShortConv online-FP8 đầy đủ

PR #48917 truyền `quant_config` vào `ShortConv.in_proj`/`out_proj` và
`Lfm2ShortConvDecoderLayer`; ảnh FP8 cũ bỏ sót 40 projection linear BF16.

Artifact:

- `submission/apply_lfm25_shortconv_fp8.py`
- `submission/Dockerfile.shortconv-fp8-online`
- `submission/docker-compose_shortconv_fp8_candidate.yml`
- `scripts/lfm25_matrix_shortconv_fp8_fused_20260728.sh`

Digest public theo báo cáo:

```text
misokaio/ghfjdk@sha256:fdc694b7282a591428debbcbb9ae2424bfb5c2905d7950f536c13495a04ac829
```

Candidate dùng base digest 65.71 và scheduler `8192/4096/32`, nên chỉ thay đổi
phạm vi FP8. Không mặc định promote vì GEMM nhỏ có thể không bù được overhead.

### Ứng viên B: PLD/ngram có rollback ShortConv

Đã backport PR #44296:

- State thêm `num_spec`.
- `num_accepted_tokens` và decode query offsets được route vào metadata.
- Spec verify gọi `causal_conv1d_update` rollback-aware.
- Decode thường vẫn dùng fused ShortConv; spec verify không rơi vào kernel
  non-spec.

Artifact:

- `submission/apply_lfm25_shortconv_spec_fix.py`
- `submission/Dockerfile.pld-safe`
- `submission/docker-compose_pld_safe_candidate.yml`
- `scripts/lfm25_matrix_pld_safe_20260728.sh`
- `harness/check_speculative_parity.py`
- `scripts/lfm25_spec_parity_gate.sh`

Digest public theo báo cáo:

```text
misokaio/ghfjdk@sha256:d1a4d9bab96cfcaaffbfb531bf7935abcd97ab70787fc4a08eafdc593494eff1
```

Compose thử CPU ngram, matching window 3, `k=2`; matrix control/k1/k2/k3/control.
`k=15` chỉ thử khi `INCLUDE_PAPER_K15=1`, vì ngram có thể chậm hơn target nhỏ
ngay cả acceptance khoảng 70% do proposal/verify/scoring overhead.

## 10. Medusa và GPTQ/AWQ

### Medusa — loại trên L4

Checkpoint có 3 residual head, hidden 2048, vocab 65.536,
`original_lm_head=true`. Matrix L4:

| Case | Thành công | ERS | TPOT p50 |
|---|---:|---:|---:|
| Medusa control | 420/420 | ~0.404 | 7.905 ms |
| K1 | 420/420 | 0.3664 | 12.125 ms |
| K2 | 418/420 | 0.3629 | 13.276 ms |
| K3 | 414/420 | 0.3783 | 11.774 ms |

K2/K3 disconnect, không OOM/traceback; K1–K3 thua control 49–68% TPOT. Không
build/push Medusa fused; H200 chưa bị phủ định tuyệt đối nhưng không ưu tiên.

### GPTQ/AWQ — nghiên cứu, chưa hợp lệ

| Weight | ERS r1 | TPOT mean | Thành công | GPQA mirror |
|---|---:|---:|---:|---:|
| Stock online FP8 | 0.424552 | 8.013 ms | 420/420 | 47/198 |
| GPTQ W4A16 | **0.517258** | **5.620 ms** | 420/420 | 44/198 |
| AWQ W4A16 | 0.382721 ở r4 | — | — | 45/198 |

GPTQ tăng ERS proxy khoảng 21,84%, TPOT giảm 29,87%, nhưng GPQA mirror giảm
3/198 = 1,52 điểm phần trăm so với stock. Đây chỉ là mirror; thể lệ mô tả
online quantization, nên cần BTC xác nhận trước khi đưa checkpoint offline vào
submission.

## 11. Quy trình build, benchmark và promote

### Build và pin image

```bash
bash scripts/build_lfm25_research_candidates.sh
```

Chỉ đặt `PUSH=1` khi đã chọn đúng registry/repository. Sau push phải thay tag
mutable bằng digest trong compose.

Với kernel fusion cũ:

```bash
bash scripts/build_lfm25_shortconv_images.sh
bash scripts/test_lfm25_kernel_image.sh
PUSH=1 bash scripts/build_lfm25_shortconv_images.sh
```

Script tạo control, nostack, shortconv, qk và fused; ba bit ENV theo thứ tự
ShortConv/no-vstack/QK fusion. Matrix phải kẹp candidate giữa control cùng image
ở đầu/cuối để phân biệt gain kernel với clock, layer image hoặc noise.

### Gate bắt buộc

1. Smoke regression ShortConv, QK, SiLU-FP8, RMSNorm-FP8.
2. Với PLD: chạy `lfm25_spec_parity_gate.sh`, 16 deterministic multi-turn,
   yêu cầu `PARITY_FAILURES=0`, đồng thời xác nhận Prometheus có ít nhất một
   draft round; sau đó chạy GPQA mirror.
3. Chạy paired control/candidate/control/candidate/control ở `RATE_SCALE=1`,
   `WARMUP=0`.
4. Promote chỉ khi cả hai candidate run thắng control lân cận ở ERS/TPOT,
   420/420 request thành công, failed không tăng và output-FP8 correctness/
   GPQA nằm trong guardrail.
5. Không dùng profile có logging stats để so latency cuối; profile speculative
   chỉ dùng để đo acceptance/accepted-length thực tế.

Lệnh L4 tổng quát:

```bash
RATE_SCALE=1 WARMUP=0 bash scripts/run_l4_compose_benchmark.sh
```

L4 historical orchestration: `scripts/l4_config_matrix.sh`; kết quả:
`eval/l4_compose_20260726/`. Local GTX 1650 chỉ đủ smoke/build; workspace không
có checkpoint `/model`, nên không có ERS local mới cho PLD/ShortConv-FP8.

## 12. Ma trận quyết định hiện tại

| Hướng | Bằng chứng | Trạng thái |
|---|---|---|
| Stock online FP8 `8192/32` | H200 63.76/63.82/63.94 | Baseline/control |
| Combined kernel + SiLU-FP8 | H200 **65.71**, 4 failed | **Giữ để nộp** |
| ShortConv + QK + no-vstack | H200 64.35, 6 failed | Fallback/ứng viên cần lặp |
| ShortConv online-FP8 đầy đủ | H200 60.85 | Loại |
| ShortConv decode→FP8 out_proj fusion | H200 **59.95**, 5 failed | **Loại, không tune tiếp** |
| PLD/ngram rollback-safe | Parity harness 16/16, chưa E2E | Thí nghiệm |
| RMSNorm dynamic-FP8 | L4 0.425014 | Không promote |
| External 350M draft | Portal 34.9 | Loại |
| Medusa K1–K3 | L4 thua mạnh | Loại |
| GPTQ/AWQ offline | L4 mạnh, luật chưa rõ | Chỉ thử khi BTC cho phép |
| Processing/interactivity/custom scheduler | H200/L4 giảm điểm | Loại |

## 13. Artifact, log và nguồn kỹ thuật

Artifact/compose chính:

- `submission/docker-compose_silu_fp8_65.71.yml`
- `submission/docker-compose_shortconv_fused_64.35.yml`
- `submission/docker-compose_shortconv_fp8_out_candidate_59.95.yml` (đã pin digest public, portal 59.95)
- `submission/docker-compose_rmsnorm_fp8_candidate.yml`
- `submission/docker-compose_shortconv_fp8_candidate.yml`
- `submission/docker-compose_pld_safe_candidate.yml`
- `submission/lfm25_fused_short_conv.py`
- `submission/lfm25_fused_qk_norm_rope.py`
- `submission/lfm25_fused_silu_fp8.py`
- `submission/lfm25_fused_rmsnorm_fp8.py`
- `scripts/lfm25_matrix_shortconv_fused_20260726.sh`
- `scripts/lfm25_matrix_shortconv_fp8_fused_20260728.sh`
- `scripts/lfm25_matrix_pld_safe_20260728.sh`
- `scripts/lfm25_matrix_fusion_ablation_20260726.sh`
- `scripts/lfm25_matrix_rmsnorm_fp8_20260726.sh`
- `harness/check_speculative_parity.py`
- `eval/l4_compose_20260726/`
- `submission/benchmark_lfm25_w2_norm_quant_pair.py`

Nguồn kỹ thuật:

- [LFM2 architecture — Liquid AI](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- [vLLM LFM2 source](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/lfm2.py)
- [vLLM ShortConv source](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/mamba/short_conv.py)
- [vLLM PR #48917 — ShortConv FP8](https://github.com/vllm-project/vllm/pull/48917)
- [vLLM PR #44296 — ShortConv speculative rollback](https://github.com/vllm-project/vllm/pull/44296)
- [vLLM issue #49112 — hybrid draft-model gaps](https://github.com/vllm-project/vllm/issues/49112)
- [vLLM issue #16258 — ngram overhead](https://github.com/vllm-project/vllm/issues/16258)
- [vLLM PR #29184 — GPU ngram proposer](https://github.com/vllm-project/vllm/pull/29184)
- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [Medusa paper](https://arxiv.org/abs/2401.10774)
- EACL 2026 paper: `2026.eacl-long.255.pdf`

## 14. Cập nhật mới nhất: portal candidate và CUTLASS SM90 EVT

### Candidate đã push public nhưng không được promote

Image `misokaio/ghfjdk@sha256:c6a8e1f733e7af11c27a90d3f35800d9c78315af76c5f6ef6aaa06e4770e4c10`
đã được registry xác nhận manifest v2 và portal đã xử lý 420 request (5 failed).
Compose
tương ứng là `submission/docker-compose_shortconv_fp8_out_candidate_59.95.yml`;
file đã được workspace đổi tên theo kết quả portal. Kết quả 59.95 xác nhận rằng
giảm một quantizer bằng cách ghi FP8 sau ShortConv không bù được parallelism/
traffic cost của đường decode. Giữ image này để tái lập, nhưng không thay control
65.71 và không tiếp tục sweep block/warp hoặc tăng batch-token.

### Audit CUTLASS 4.2.1 trên SM90/H200

Môi trường image: vLLM 0.25.1, PyTorch 2.11.0+cu130, CUDA 13.0.88, CUTLASS
headers 4.2.1; LFM2.5 có hidden 2048, intermediate 8192, 16 layer (10
ShortConv, 6 attention). `w2` là `A[M,8192] × B[8192,2048]`, decode `M=1..32`.
SM90 FP8 dispatch của vLLM dùng `swap_ab=true`: tile `64×16×256` cho `M≤16`,
`64×64×256` cho `16<M≤64`; 2048 channel bị chia thành 32 CTA. Vì
`Sm90RowReduction` chỉ trả partial/final statistic ở `end()`, EVT thuần không
thể lấy inverse-RMS rồi phát ngược kết quả cho các fragment đã ghi. Do đó mục
tiêu “w2 → residual add → RMSNorm → FP8 trong đúng một EVT launch” không khả thi
với tile hiện tại nếu không đổi sang cooperative/cluster epilogue.

### Thiết kế được khuyến nghị

Giữ đúng hai BF16 rounding boundary của baseline:

```text
w2_bf16       = bf16(scale_a * scale_b * accumulator_fp32)
residual_out  = bf16(residual_bf16 + w2_bf16)
sumsq         = sum(float(residual_out)^2)
weighted_amax = max(abs(float(residual_out) * float(next_norm_weight)))
inv_rms       = rsqrt(sumsq / 2048 + 1e-5)
fp8_scale     = weighted_amax * inv_rms / 448
```

**M0 (prototype):** CUTLASS EVT gộp dequant, residual add, BF16 store và hai
partial reductions; kernel phụ đọc residual/stats rồi RMSNorm/FP8. Dùng PDL để
giảm gap/preamble, nhưng không giả định overlap là bắt buộc. Workspace cố định
cho `M≤32` khoảng 9 KiB và phải capture-safe.

**M1 (hướng có gain lớn nhất):** không materialize tensor normalized; projection
kế tiếp đọc `residual_out`, stats và `next_norm_weight` trong mainloop. Sáu
boundary vào attention có thể convert FP8 ngay trước QKV MMA; chín boundary vào
ShortConv giữ BF16 `in_proj` vì full ShortConv online-FP8 đã chỉ đạt 60.85.
Đây mới là cách có thể bỏ một launch trên mỗi layer mà không cần global barrier.

**M2 (rủi ro cao):** cooperative/cluster epilogue tự giữ tile, reduce rồi ghi
normalized FP8. Hopper portable cluster tối đa 8 CTA (H100/H200 có thể opt-in
16), trong khi dispatch hiện tại dùng 32 CTA theo chiều output; chỉ thử nếu M0
đo được gain ≥8% ở M=4..32 và M1 không tích hợp được.

### Gate triển khai tiếp theo

Benchmark trên H200 với CUDA Graph on/off, 2.000 replay cho `M=1,2,4,8,16,32`:
`stock w2 + fused RMSNorm/quant` so với M0, M1-attention và M1-ShortConv. Thu
Nsight về launch gap, DRAM bytes, registers, spills, occupancy và CTA count.
Chỉ promote khi residual bit-exact, FP8 scale/bytes giữ parity, không spill/
runtime allocation và paired portal score/TPOT tốt hơn control với failed không
tăng. Chi tiết source audit và API tham khảo CUTLASS Hopper GEMM, visitor store,
PDL/CUDA Graph ở các link chính thức dưới đây.

- [vLLM SM90 FP8 dispatch](https://github.com/vllm-project/vllm/blob/v0.25.1/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh)
- [vLLM fused RMSNorm + dynamic FP8](https://github.com/vllm-project/vllm/blob/v0.25.1/csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu)
- [CUTLASS SM90 visitor reductions](https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp)
- [CUTLASS Hopper GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md)
- [CUTLASS dependent kernel launch](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/dependent_kernel_launch.md)
- [CUDA Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [Hopper thread-block clusters](https://docs.nvidia.com/cuda/archive/11.8.0/hopper-tuning-guide/index.html#thread-block-clusters)

## Tóm tắt cuối

Trên H200, stock FCFS với `max-model-len=8192`, `max-num-batched-tokens=4096`,
`max-num-seqs=32` là baseline tốt hơn các biến thể scheduler/
batch đã đo. Combined ShortConv/QK/no-vstack + SiLU-FP8 đạt 65.71, là mốc portal
tốt nhất và nên giữ. Candidate ShortConv→FP8 out đạt 59.95 nên đã dừng. L4
xác nhận xu hướng SiLU-FP8 nhưng không thay thế H200. Flags, online
quantization và external speculative draft đã gần/chạm trần; không tiếp tục
sweep mù. Hướng phát triển còn lại có giá trị nhất là CUTLASS SM90 EVT M0 rồi
M1 normalized-load, với attention QKV là nhánh FP8 và ShortConv in-proj giữ BF16.
