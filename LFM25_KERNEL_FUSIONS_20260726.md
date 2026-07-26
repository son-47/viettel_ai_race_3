# LFM2.5 kernel fusions cho workload AI Race 2026

Ngày: 2026-07-26

## Kết luận

Candidate kernel fusion đã được chấm **64.3500**, cao hơn mốc 63.94 trước đó
**0.41 điểm** (khoảng **0.64%**). Đây là điểm cao nhất đã quan sát trong báo cáo
này. TTFT p50/p95 cùng giảm 1 ms, TBT median giữ nguyên 4 ms, nhưng failed count
tăng từ 5 lên 6; vì vậy vẫn cần thêm lần chấm để xác nhận độ ổn định.

Đã cài hai tối ưu opt-in mới vào một image recipe độc lập:

1. Fused LFM2.5 ShortConv decode: gộp `B*x`, causal convolution width 3,
   cập nhật recurrent state và `C*output` vào một Triton launch.
2. Fused LFM2.5 Q/K RMSNorm + NeoX RoPE: đọc trực tiếp hai view Q/K có stride từ
   packed QKV và sinh Q/K contiguous sau RoPE bằng một Triton launch.

Ngoài ra có một công tắc độc lập bỏ `torch.vstack([tensor])` ở iteration chỉ có
decode hoặc chỉ có prefill. Tất cả nhánh đều có fallback nguyên bản và mặc định
tắt nếu không dùng image/biến môi trường mới.

Image public đã được build và push tại
`misokaio/ghfjdk:v0.25.1-lfm25-fused`, pin digest
`sha256:53d1892ca842ffa2f5e3113f0f775450701bacb7c014d8f497bd63e6ad61d401`.
CUDA smoke test trên GTX 1650 đã pass trước khi push; số portal dưới đây mới là
kết quả end-to-end quyết định trên grader.

## Kết quả portal: 64.3500

Portal hiển thị **đã chấm 1 giờ trước** tại thời điểm cập nhật báo cáo.

| Chỉ số | Giá trị |
|---|---:|
| `ers` | **64.35** |
| `f_delta` | **1** |
| `penalty` | **1** |
| `final_score` | **64.35** |
| `total_count` | **420** |
| `ttft_p50_ms` | **38** |
| `ttft_p95_ms` | **51** |
| `failed_count` | **6** |
| `warmup_count` | **0** |
| `accuracy_drop` | **0** |
| `tbt_median_ms` | **4** |
| `tokens_per_sec` | **0.0572** |

So với mốc 63.94:

| Chỉ số | Mốc 63.94 | Kernel fusion 64.35 | Delta |
|---|---:|---:|---:|
| ERS/final score | 63.94 | 64.35 | **+0.41** |
| TTFT p50 (ms) | 39 | 38 | **-1** |
| TTFT p95 (ms) | 52 | 51 | **-1** |
| TBT median (ms) | 4 | 4 | 0 |
| Failed count | 5 | 6 | **+1** |

`accuracy_drop=0`, `penalty=1` và `f_delta=1` xác nhận kết quả không bị trừ điểm
do accuracy hay hệ số phạt. Gain hiện đến từ phần TTFT và tổng phân phối latency;
TBT median làm tròn vẫn là 4 ms. Không suy diễn `tokens_per_sec=0.0572` sang một
đơn vị khác vì portal không ghi đơn vị chi tiết ngoài tên trường.

## Vì sao chọn kernel fusion

Workload có 70 conversation × 6 turn = 420 request, prompt dùng shared prefix,
context tăng dần và mỗi request bắt buộc sinh đúng 300 token. Prefix caching và
chunked prefill đã xử lý phần prefill thuận lợi; tài liệu vLLM cũng nêu rõ APC
không làm decode của một request nhanh hơn. Vì vậy, sau khi flag, online FP8,
speculative decoding và MTP đã bão hòa, TPOT của decode path là đòn bẩy còn lại.

LFM2.5-1.2B có 16 layer, gồm 10 ShortConv và 6 attention:

- Mỗi ShortConv decode trong vLLM v0.25.1 materialize `B*x`, chạy
  `causal_conv1d_update`, materialize `C*Bx`, rồi luôn gọi `torch.vstack`.
- Mỗi attention layer materialize hai bản Q/K contiguous, chạy hai RMSNorm rồi
  mới chạy RoPE.
- Hai fusion mới nhắm đúng các tensor tạm và kernel-launch overhead này; không
  thay weight, không đổi thuật toán sampling, không dùng approximation.

Đếm từ source cho thấy candidate có thể loại tối đa 20 elementwise launch và 10
single-tensor stack/copy qua 10 ShortConv layer, cùng tối đa 24 launch/copy qua 6
attention layer trong mỗi model step. Đây là số operation ở source, không phải
số kernel đã đo; CUDA graph/TorchInductor có thể làm số thực tế khác. Nsight hoặc
CUDA-event microbenchmark trong image mới là bằng chứng quyết định.

Hướng này phù hợp với roofline của model nhỏ: decode batch nhỏ thường nhạy với
launch latency và traffic tensor tạm hơn là chỉ giảm số FLOP. Nó cũng bám theo
chính upstream vLLM: v0.25.1 đã có một Triton kernel gộp QK RMSNorm + RoPE cho
Qwen3.5; phần cài đặt mới chuyên biệt hóa cùng ranh giới làm tròn cho RMSNorm
chuẩn của LFM2.

Nguồn chính:

- [LFM2 architecture, Liquid AI](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- [vLLM v0.25.1 ShortConv source](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/mamba/short_conv.py)
- [vLLM v0.25.1 LFM2 source](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/lfm2.py)
- [Upstream fused QK norm/RoPE source](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/fused_qk_norm_rope.py)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)

## Phạm vi correctness

ShortConv fusion chỉ chạy khi đồng thời thỏa:

- CUDA;
- convolution width đúng 3 và không bias;
- không có speculative accepted-token state;
- iteration decode thường, số decode token bằng số decode request;
- tensor/state có đúng shape LFM2.5.

Kernel giữ các ranh giới làm tròn của đường gốc:

- `B*x` làm tròn về dtype projection;
- giá trị mới làm tròn tiếp về dtype recurrent state;
- convolution làm tròn qua dtype state rồi quay về dtype projection trước gate C;
- `NULL_BLOCK_ID` không đọc hoặc ghi state.

QK fusion chỉ chạy khi đồng thời thỏa CUDA, BF16/FP16, NeoX RoPE và
`rotary_dim == head_dim`. RMSNorm được tính FP32, làm tròn về input dtype trước
RoPE giống fusion đã có trong upstream vLLM. Các đường CPU, dtype/shape khác và
kiểu RoPE khác dùng nguyên code vLLM.

## Artifact đã cài

- `submission/lfm25_fused_short_conv.py`: Triton ShortConv kernel.
- `submission/lfm25_fused_qk_norm_rope.py`: Triton QK norm/RoPE kernel.
- `submission/apply_lfm25_shortconv_fusion.py`: exact-source patcher; Docker build
  fail ngay nếu source base image drift.
- `submission/Dockerfile.shortconv-fused`: image recipe pin đúng digest của image
  v0.25.1 đang dùng ở mốc tốt nhất.
- `submission/docker-compose_shortconv_fused.yml`: combined candidate, giữ nguyên
  flags của mốc 63.94.
- `submission/test_lfm25_fused_short_conv.py`: correctness state/output và
  CUDA-event microbenchmark theo batch 1, 2, 4, 8, 16, 32.
- `submission/test_lfm25_fused_qk_norm_rope.py`: correctness và microbenchmark
  BF16/FP16 theo cùng các token count.
- `scripts/build_lfm25_shortconv_images.sh`: build năm image chỉ khác ENV.
- `scripts/test_lfm25_kernel_image.sh`: chạy hai GPU smoke test.
- `scripts/lfm25_matrix_shortconv_fused_20260726.sh`: full workload paired A/B.

Ba biến rollback độc lập:

```text
VLLM_LFM25_BYPASS_SINGLE_VSTACK
VLLM_LFM25_FUSED_SHORTCONV
VLLM_LFM25_FUSED_QK_NORM_ROPE
```

## Build và kiểm thử

Từ root repository trên máy Linux có Docker/NVIDIA runtime:

```bash
bash scripts/build_lfm25_shortconv_images.sh
bash scripts/test_lfm25_kernel_image.sh
```

Script build tạo:

```text
v0.25.1-lfm25-control    0/0/0
v0.25.1-lfm25-nostack    0/1/0
v0.25.1-lfm25-shortconv  1/1/0
v0.25.1-lfm25-qk         0/0/1
v0.25.1-lfm25-fused      1/1/1
```

Thứ tự ba bit là ShortConv / no-vstack / QK fusion. Chỉ push khi smoke test pass:

```bash
PUSH=1 bash scripts/build_lfm25_shortconv_images.sh
```

Trên máy remote có harness:

```bash
RATE_SCALE=1 WARMUP=0 \
  bash /home/zeus/content/lfm25_matrix_shortconv_fused_20260726.sh
```

Matrix chạy stock control và patched-image control ở cả đầu/cuối, kẹp bốn
candidate ở giữa. `lfm25-control` rất quan trọng: nó phân biệt gain của kernel
với chênh lệch do layer image, clock hoặc phiên benchmark.

## Trạng thái promote

Candidate hiện có thể dùng làm submission điểm cao nhất đã đo vì:

- hai GPU smoke test pass; ShortConv output và state exact với đường gốc;
- portal chấm 64.35, cao hơn mốc 63.94;
- `accuracy_drop=0`, `penalty=1`, `f_delta=1`;
- TTFT p50/p95 đều tốt hơn 1 ms và TBT median không giảm chất lượng.

Tuy nhiên đây mới là một lần chấm và có 6 failed request, nhiều hơn mốc trước
một request. Nên chấm lặp lại hoặc chạy paired control trước khi coi +0.41 là
gain ổn định. FP8 + concurrency cũng không bảo đảm output hash tuyệt đối ổn định
giữa hai control; quality mirror riêng vẫn hữu ích.

Nếu combined không thắng nhưng một nhánh riêng thắng, dùng đúng image nhánh đó;
không giữ fusion thua chỉ vì nó hợp lý trên lý thuyết.

## Những hướng chưa ưu tiên

- Backport hybrid prefix-cache tracking mới hơn: liên quan nhiều module cache và
  rủi ro correctness cao; APC hiện đã tốt và không giải quyết 300 decode token.
- Sparse/approximate attention: sáu attention layer và context khoảng vài nghìn
  token chưa đủ để biện minh rủi ro chất lượng.
- Offline W4A8/GPTQ/AWQ: có thể đáng thử nếu luật cho phép checkpoint biến đổi,
  nhưng là trục quantization khác và cần quality validation; không trộn vào A/B
  kernel hiện tại.
- Thêm scheduler/engine flags: sweep cũ đã rộng và dễ chỉ dịch chuyển TTFT/TPOT
  thay vì hạ tổng GPU work.

Nếu kernel fusion vẫn không thắng trên H200 MIG, bước tiếp theo nên là profile
Nsight Systems một decode step của control và fused image. Chỉ viết fusion thứ ba
khi trace cho thấy một chuỗi launch cụ thể còn chiếm tỷ trọng đáng kể.
