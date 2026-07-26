# LFM2.5 / AI Race 2026 — bản tổng hợp tối ưu suy luận

Ngày tổng hợp: 2026-07-26

## Cập nhật buổi chiều: portal 65.71 và fusion kế tiếp

`submission/docker-compose_silu_fp8_65.71.yml` đã đạt **65.71**, với TTFT
p50/p95 là **32/47 ms**, TBT median 4 ms và 4 request lỗi. Đây là mốc portal cao
nhất mới, thay cho 64.35.

Image 65.71 đã gộp hai nhóm fusion: Dockerfile SiLU-FP8 kế thừa trực tiếp image
ShortConv/QK/no-vstack, và compose bật tất cả các flag. Tuy nhiên compose cũng
đổi scheduler sang `8192/4096/32`; vì vậy cần paired same-image ablation trước
khi kết luận riêng SiLU-FP8 đóng góp bao nhiêu.

Hướng kế tiếp đã được cài là nối fused CUDA op add-RMSNorm-dynamic-FP8 có sẵn
trong vLLM vào 16 W13 và 6 QKV projection của LFM2.5. Chi tiết và quy trình A/B
nằm trong `LFM25_RMSNORM_FP8_FUSION_20260726.md`.

Tài liệu này hợp nhất bảy báo cáo nghiên cứu/benchmark:

- `L4_COMPOSE_EVAL_20260726.md`
- `LFM25_DEEP_OPTIMIZATION_20260726.md`
- `LFM25_KERNEL_FUSIONS_20260726.md`
- `LFM25_OPTIMIZATION_20260722.md`
- `LFM25_RESEARCH_OPTIMIZATION_20260723.md`
- `LFM25_SILU_FP8_FUSION_20260726.md`
- `LFM25_RMSNORM_FP8_FUSION_20260726.md`

## 1. Kết luận ngắn gọn

### Kết quả đã được portal xác nhận

Image combined fusion là điểm cao nhất đã quan sát:

```text
misokaio/ghfjdk
digest: sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4
portal final_score: 65.71
```

So với mốc fusion buổi sáng 64.35, điểm tăng 1.36; TTFT p50/p95 giảm từ 38/51
ms xuống 32/47 ms và failed count giảm từ 6 xuống 4. Image này đã chứa cả
ShortConv/QK/no-vstack và SiLU-FP8. Scheduler cũng thay đổi, nên số portal chưa
phải là ablation riêng cho SiLU-FP8.

### Candidate mới có tín hiệu tốt nhất trên L4

Candidate fuse `SiLU×Mul + dynamic per-token FP8 quantization` đã được push
public:

```text
misokaio/ghfjdk@sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4
```

Replay đủ 420 request trên Lightning L4 cho kết quả tốt nhất trong bốn compose
đã kiểm tra:

| Compose | ERS proxy L4 | Δ so với control 63.94 | TTFT p50 | TPOT p50 | Thành công |
|---|---:|---:|---:|---:|---:|
| SiLU-FP8 candidate | **0.436072** | **+0.697%** | 43.718 ms | **7.822 ms** | 420/420 |
| ShortConv/QK fusion | 0.434950 | +0.438% | 43.733 ms | 7.842 ms | 420/420 |
| Stock control | 0.433053 | — | 43.816 ms | 7.899 ms | 420/420 |
| Processing flags | 0.422525 | −2.431% | 49.916 ms | 7.942 ms | 420/420 |

ERS trên bảng là điểm synthetic replay của harness, không phải điểm private
portal. Sau replay này, combined SiLU-FP8 đã được portal xác nhận ở 65.71;
paired same-image ablation vẫn cần thiết để tách gain của kernel khỏi scheduler.

### Quyết định thực tế

- Nếu cần dùng image đã có điểm portal tốt nhất: dùng compose 65.71 đã pin digest.
- Nếu có slot thử nghiệm: build candidate RMSNorm-FP8 mới, chạy đủ smoke H200
  và paired same-image A/B trước khi nộp portal.
- Không dùng `docker-compose_processing.yml`.
- Không dùng Medusa trên L4.
- Không dùng GPTQ/AWQ trong submission nếu BTC chưa xác nhận checkpoint offline
  được phép.

## 2. Bài toán, workload và hàm điểm

Model là `LiquidAI/LFM2.5-1.2B-Instruct`, kiến trúc hybrid gồm 16 decoder layer:
10 ShortConv/recurrent block và 6 GQA/attention block.

Workload công bố:

- 70 conversations × 6 turns = 420 request.
- Shared system prefix: 1.000 token.
- Prefix riêng mỗi conversation: 1.000 token.
- Mỗi turn thêm 150 user token.
- Mỗi request phải sinh chính xác 300 output token.
- Poisson arrival, seed 42.
- Context cuối turn xấp xỉ 4.400 input token, chưa tính chat-template overhead.

ERS cân bằng TTFT và TPOT/TBT, với các ngưỡng:

- TTFT: sàn 10 ms, trần 400 ms.
- TPOT: sàn 1 ms, trần 10 ms.
- Gamma: 2.
- Trọng số TTFT/TPOT: 0,5/0,5.
- Request lỗi hoặc turn bị thiếu được tính 0 điểm.

Vì mỗi request sinh 300 token, decode chi phối điểm nhiều hơn sau khi prefix
caching và chunked prefill đã xử lý prefill. APC giúp reuse shared/multi-turn
prefix nhưng không làm 300 decode token nhanh hơn.

## 3. Mốc kết quả và cách đọc đúng

Các báo cáo dùng nhiều mốc theo thời điểm khác nhau; không nên trộn proxy L4
với điểm portal H200.

| Mốc | Nền tảng | Kết quả | Ý nghĩa |
|---|---|---:|---|
| Stock FP8 exact | H200 portal | **63.82** | Mốc submission hợp lệ stock được xác nhận sớm |
| Warmup1 | H200 portal | 62.97 | Warmup giảm failure nhưng làm TTFT p50 xấu; loại |
| Stock + HTTP keep-alive 600 | H200 portal | 62.16 | L4 từng có lợi nhưng không tái hiện trên H200; loại |
| Stock/control artifact sau đó | Portal/history | 63.94 | Mốc control được dùng trong các compose mới |
| ShortConv + QK + no-vstack kernel fusion | H200 portal | **64.35** | Điểm cao nhất đã quan sát; failed tăng 5 → 6 |
| SiLU-FP8 candidate | L4 synthetic | 0.436072 ERS | Tín hiệu tốt nhất mới; chưa có H200 portal |

Điểm portal 64.35 đáng tin cậy hơn mọi ERS L4. L4 chỉ dùng để xếp hạng,
phát hiện crash và đo xu hướng; L4 không đại diện hoàn toàn cho H200 MIG 18 GB.

## 4. Baseline được chọn

Cấu hình stock thắng đã được giữ làm control:

- Image vLLM `0.25.1`, pin digest.
- `--quantization=fp8` (online weight quantization).
- `--max-model-len=32768`.
- `--max-num-batched-tokens=8192`.
- `--max-num-seqs=32`.
- `--optimization-level=3`.
- `--gpu-memory-utilization=0.85`.
- Tensor parallel = 1.
- Automatic prefix caching.
- Chunked prefill.
- Không ép FlashInfer, không KV quantization, không custom scheduler.
- Không dùng internal warmup trong compose chốt.
- Không dùng `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600`: proxy L4 có vẻ giảm cold
  failure nhưng H200 chính thức giảm từ 63.82 xuống 62.16.

Cấu hình `processing` thử `max-num-batched-tokens=9216`, `max-num-seqs=32`,
`max-model-len=32768` và bỏ performance mode interactivity; trên L4 nó làm
TTFT p50 tăng khoảng 6.1 ms và ERS giảm 2.43%.

## 5. Những nhóm kỹ thuật đã sweep và quyết định

### Flags/scheduler/runtime — không thắng ổn định

Đã thử batch token 4.096/6.144/7.168/8.192/9.216/12.288, sequence 24/40,
max-model-len 8.192, block size 8/32, Mamba block 8/32/64, GPU memory
utilization 0.92, optimization level 2, CUDA graph capture 1–32, output
processing chunk 256, renderer/frontend variants, Rust frontend, MRV2,
cascade attention, DBO, reserve-full-ISL, hybrid KV manager, stream interval,
performance mode throughput/interactivity và FlashInfer ép buộc.

Kết luận: đa số nằm trong noise hoặc làm ERS giảm; một số fail ngay khi
startup do backend/heterogeneous cache. Không tiếp tục sweep flag nhỏ vì đã
không còn tỷ lệ lợi ích/rủi ro tốt.

### Online quantization/KV cache

Đã thử online FP8 per-tensor/per-block/per-channel, INT8 weight-only, MXFP8,
FP8 KV và INT8 per-token-head.

- `fp8_per_tensor` chỉ hơn stock `+0.000385 ERS` ở r1, không giảm TPOT; coi là
  noise.
- FP8 per-block, INT8 weight-only, MXFP8 và FP8 KV làm ERS giảm ở tải chuẩn.
- FP8 KV không có calibration scale trong checkpoint; dù một slice GPQA 100 câu
  có kết quả tốt hơn, latency chuẩn giảm và output thay đổi nhiều, nên loại.

### Speculative decoding / n-gram / draft model

N-gram CPU/GPU, suffix và draft `LFM2.5-350M` đã được thử hoặc fail. Model nhỏ
1.2B và hybrid recurrent cache làm chi phí verify/rollback lớn; các hướng này
không tạo gain bền vững.

## 6. Kernel fusion 64.35 — implementation và bằng chứng

Image fused kế thừa stock vLLM nhưng thêm ba nhánh opt-in, fallback về đường
gốc nếu guard không khớp.

### ShortConv decode fusion

Trong mỗi ShortConv decode, đường cũ materialize nhiều tensor tạm:

```text
B*x → causal_conv1d_update → recurrent state → C*output → vstack
```

Triton fusion gộp `B*x`, causal convolution width 3, state update và `C*output`
vào một launch. Kernel giữ các ranh giới dtype/state của CUDA gốc, không dùng
approximation.

Guard chính: CUDA, width 3, không bias, decode thường, không speculative
accepted-token state, số decode token = số decode request và shape đúng LFM2.

### Q/K RMSNorm + NeoX RoPE fusion

Đọc trực tiếp hai view Q/K có stride từ packed QKV, tính RMSNorm FP32 rồi sinh
Q/K contiguous sau RoPE trong một Triton launch. Guard giới hạn BF16/FP16,
NeoX RoPE và `rotary_dim == head_dim`.

### Bỏ singleton vstack

Ở iteration chỉ có một tensor decode/prefill, bỏ `torch.vstack([tensor])` với
flag riêng. Đây là tối ưu nhỏ, độc lập và rollback được.

### Kết quả portal

| Chỉ số | Stock/mốc 63.94 | Kernel fusion |
|---|---:|---:|
| Final score | 63.94 | **64.35** |
| TTFT p50 | 39 ms | **38 ms** |
| TTFT p95 | 52 ms | **51 ms** |
| TBT median | 4 ms | 4 ms |
| Failed | 5 | 6 |
| Accuracy drop | 0 | 0 |

CUDA smoke test trên GTX 1650 pass; image public đã pin digest. Cần chấm lại
nếu muốn kết luận statistical stability vì failed count xấu hơn một request.

Artifact chính: `submission/lfm25_fused_short_conv.py`,
`submission/lfm25_fused_qk_norm_rope.py`, exact-source patcher,
`submission/Dockerfile.shortconv-fused`, smoke tests và paired matrix script.

## 7. SiLU-FP8 fusion candidate — hướng mới

### Khoảng trống trong vLLM 0.25.1

Với online per-token FP8, mỗi MLP hiện chạy:

```text
w13 FP8 GEMM → SiLU×gate ghi BF16 → đọc BF16 để quant E4M3 → w2 CUTLASS
```

vLLM 0.25.1 có activation fusion cho static FP8/NVFP4/block FP8, nhưng chưa có
fusion cho `kFp8DynamicTokenSym` theo token. Candidate gộp thành:

```text
w13 FP8 GEMM → fused SiLU×gate + row-absmax + E4M3 quant → w2 CUTLASS
```

Mỗi model step có thể loại tối đa 16 launch và bỏ lượt ghi/đọc tensor BF16
`16 × batch × 8192`, tương đương khoảng `512 KiB × batch` traffic trung gian.

### Correctness/guard

Kernel giữ hai ranh giới rounding của CUDA `silu_and_mul`:

1. SiLU được làm tròn về BF16/FP16.
2. Kết quả nhân với `up` được làm tròn lần hai trước row maximum.

Scale E4M3:

```text
scale = max(absmax / 448, 1 / (448 × 512))
q = clamp(rounded_activation / scale, -448, 448)
```

Fast path chỉ bật khi `VLLM_LFM25_FUSED_SILU_FP8=1`, online
`Fp8PerTensorOnlineLinearMethod`, backend `CutlassFP8ScaledMMLinearKernel`,
TP=1 và linear không bias. Các backend khác fallback nguyên bản.

### Kiểm chứng

- SM90 production kernel compile pass bằng Triton 3.6.0.
- 48 registers, 32 bytes shared memory, không spill.
- GTX 1650: BF16 activation khớp tuyệt đối; scale error tối đa
  `1.862645149230957e-09`; FP16 activation error tối đa `2.44140625e-04`.
- E4M3 runtime trên SM75 bị skip vì Triton không hỗ trợ store E4M3 ở GPU đó;
  L4 SM89 đã chạy full 420 request, nhưng H200 output-FP8 correctness vẫn cần
  kiểm chứng riêng.

Artifact: `submission/lfm25_fused_silu_fp8.py`, patcher, test,
`Dockerfile.silu-fp8-fused`, compose digest và paired H200 matrix.

## 8. Medusa — đã loại trên L4

Medusa checkpoint có 3 head residual, hidden size 2048, vocab 65.536,
`original_lm_head=true`. Về lý thuyết có thể ghép ba residual GEMM và ba vocab
GEMM, nhưng LFM2 cần rollback hybrid ShortConv state khi verify draft.

Matrix L4 với image control đã vá cache và K1/K2/K3:

| Case | Thành công | ERS L4 | TPOT p50 | So với Medusa control |
|---|---:|---:|---:|---:|
| Medusa control | 420/420 | ~0.404 | 7.905 ms | — |
| K1 | 420/420 | 0.3664 | 12.125 ms | +53.4% |
| K2 | 418/420 | 0.3629 | 13.276 ms | +68.0% |
| K3 | 414/420 | 0.3783 | 11.774 ms | +48.9% |

K2/K3 mất request do disconnect; không có OOM/traceback. K1–K3 đều thua rõ
ràng, nên chưa build/push fused Medusa. Kết quả L4 chưa phủ định hoàn toàn H200,
nhưng Medusa hiện không đáng ưu tiên so với kernel fusion.

## 9. GPTQ/AWQ — latency winner nghiên cứu, chưa hợp lệ để nộp

Offline W4A16 có kết quả L4 rất mạnh:

| Weight | ERS r1 | TPOT mean | Thành công | GPQA mirror |
|---|---:|---:|---:|---:|
| Stock online FP8 | 0.424552 | 8.013 ms | 420/420 | 47/198 |
| GPTQ W4A16 | **0.517258** | **5.620 ms** | 420/420 | 44/198 |
| AWQ W4A16 | 0.382721 ở r4 | — | — | 45/198 |

GPTQ tăng ERS proxy khoảng 21.84%, TPOT giảm 29.87%, nhưng TTFT tăng nhẹ.
GPQA mirror GPTQ giảm 3/198 = 1.52 điểm phần trăm so với stock, thấp hơn
guardrail 10 điểm phần trăm; đây vẫn chỉ là mirror, không phải accuracy test
chính thức.

Thể lệ mô tả phạm vi là online quantization. GPTQ/AWQ đã biến đổi checkpoint
offline, nên chỉ được thử trong submission nếu BTC xác nhận bằng văn bản.

## 10. Bảng quyết định cuối

| Hướng | Đã code | Đã chạy L4 | Đã có điểm portal | Quyết định |
|---|---|---|---|---|
| Stock online FP8 8192/32 | Có | Có | 63.82/63.94 history | Baseline/control |
| Warmup nội bộ | Có | Có | 62.97 | Loại |
| Keep-alive 600 | Có | Có | 62.16 | Loại |
| ShortConv + QK + no-vstack | Có | Smoke + L4 | **64.35** | Bản điểm cao nhất hiện tại |
| SiLU + dynamic FP8 | Có | **420/420 L4** | Chưa | Candidate cần H200 A/B |
| Medusa K1/K2/K3 | Có/public | Có, thua mạnh | Chưa | Loại trên L4 |
| GPTQ/AWQ offline | Research | Có | Chưa | Chỉ dùng nếu luật cho phép |
| Processing flags | Có | 420/420, ERS giảm | Chưa | Không dùng |

## 11. Quy trình kiểm chứng và promote

### Nếu chỉ có L4

```bash
RATE_SCALE=1 WARMUP=0 \
  bash scripts/run_l4_compose_benchmark.sh
```

Runner hiện tại đã dùng để tạo `eval/l4_compose_20260726/` và báo cáo
`L4_COMPOSE_EVAL_20260726.md`.

### Trên H200/H100 trước khi promote RMSNorm-FP8

1. Chạy smoke regression ShortConv, QK, SiLU-FP8 và RMSNorm-FP8.
2. Chạy paired matrix xen kẽ control/candidate/control/candidate/control với
   `RATE_SCALE=1`, `WARMUP=0`.
3. Chỉ promote nếu cả hai candidate run thắng control lân cận ở ERS/TPOT,
   không tăng failed count và output-FP8 correctness pass.
4. Pin registry digest trong compose; không dùng tag mutable.

Combined fusion 65.71 là fallback tốt nhất vì đã có portal evidence. Candidate
RMSNorm-FP8 không được coi là thắng chỉ dựa trên build/smoke local.

## 12. Artifact và tài liệu tham khảo

Các file quan trọng:

- `submission/docker-compose_shortconv_fused_64.35.yml`
- `submission/docker-compose_silu_fp8_65.71.yml`
- `submission/docker-compose_rmsnorm_fp8_candidate.yml`
- `submission/lfm25_fused_short_conv.py`
- `submission/lfm25_fused_qk_norm_rope.py`
- `submission/lfm25_fused_silu_fp8.py`
- `submission/lfm25_fused_rmsnorm_fp8.py`
- `scripts/lfm25_remote_ab.sh`
- `scripts/lfm25_matrix_shortconv_fused_20260726.sh`
- `scripts/lfm25_matrix_silu_fp8_20260726.sh`
- `scripts/lfm25_matrix_fusion_ablation_20260726.sh`
- `scripts/lfm25_matrix_rmsnorm_fp8_20260726.sh`
- `eval/l4_compose_20260726/`

Nguồn kỹ thuật chính:

- [LFM2 architecture — Liquid AI](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- [vLLM ShortConv](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/mamba/short_conv.py)
- [vLLM LFM2](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/lfm2.py)
- [vLLM PR #48917 — FP8 coverage for ShortConv](https://github.com/vllm-project/vllm/pull/48917)
- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [Medusa paper](https://arxiv.org/abs/2401.10774)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)

## Tóm tắt một câu

Sau khi stock flags, online quantization và speculative decoding chạm trần,
combined kernel fusion đã đạt 65.71 trên portal; bước kế tiếp là fused
add-RMSNorm-dynamic-FP8, nhưng chỉ promote sau paired H200 A/B và correctness.
