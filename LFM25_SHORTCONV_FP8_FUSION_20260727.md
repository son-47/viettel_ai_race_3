# LFM2.5 ShortConv output-FP8 fusion — 27/07/2026

## Kết luận ngắn

Đã cài một candidate mới trên đúng image portal 65.71, không thay thế submission
tốt nhất hiện tại. Candidate chỉ lượng tử online FP8 cho `ShortConv.out_proj`,
giữ `ShortConv.in_proj` ở BF16, đồng thời gộp:

```text
B*x -> recurrent width-3 convolution -> C gate -> BF16 rounding
    -> dynamic per-token FP8 quantization
```

thành một Triton kernel trước khi gọi CUTLASS FP8 scaled-MM. Hướng này **không**
fuse Add/RMSNorm và không bật lại candidate RMSNorm-FP8 đã có kết quả thấp.

Image local đã build thành công:

```text
misokaio/ghfjdk:v0.25.1-lfm25-shortconv-fp8
local image ID: sha256:a1314eec9fbd338e0bd04329663dca07406e127f52405e922efdd7df54134dbe
```

Image này chưa được push và chưa có điểm H200/portal. Không dùng tag local trong
portal trước khi push public, pin digest và hoàn tất A/B.

## Vì sao chọn đúng điểm nối này

Workload có 420 request, mỗi request sinh cố định 300 token, tức 126.000 output
token. Model có 16 layer, gồm 10 block LIV convolution và 6 block GQA. Vì vậy
đường ShortConv được lặp lại rất nhiều trong decode. Model card chính thức xác
nhận cấu trúc 10/16 này:

- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

vLLM 0.25.1 không truyền `quant_config` vào hai projection của ShortConv; MLP và
attention đã FP8 nhưng ShortConv vẫn BF16. Khoảng trống này được upstream xác
nhận và sửa trong PR #48917 ngày 21/07/2026:

- https://github.com/vllm-project/vllm/pull/48917

PR upstream truyền quantization vào cả `in_proj` và `out_proj`, nhưng không có
benchmark latency và được viết chủ yếu để load checkpoint quantized đúng. Bản
candidate này cố ý hẹp hơn:

- chỉ `out_proj` FP8: giảm khoảng 40 MiB trọng số phải đọc cho mười ma trận
  2048×2048 trong mỗi model step so với BF16;
- `in_proj` lớn hơn vẫn BF16 để giảm phạm vi sai số;
- chi phí quant activation được hấp thụ vào chính ShortConv kernel thay vì tạo
  thêm một kernel và tensor BF16 2.048 chiều;
- nhánh prefill/mixed vẫn đi qua đường vLLM tổng quát;
- bất kỳ backend/shape/TP không đúng contract đều không vào direct fused path.

Việc ưu tiên kernel fusion cho decode batch nhỏ cũng phù hợp với phân tích chính
thức của vLLM: ở batch thấp, launch overhead của các phép nhỏ như normalization,
RoPE và quantization có thể chi phối; giảm số lần launch và round-trip bộ nhớ đã
cho cải thiện đáng kể trên các model khác:

- https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-05-11-vllm-tops-artificial-analysis.md

## Thay đổi đã cài

- `submission/lfm25_fused_shortconv_fp8.py`: Triton kernel, guard CUTLASS và
  direct scaled-MM helper.
- `submission/apply_lfm25_shortconv_fp8_fusion.py`: exact-source patcher. Docker
  build dừng nếu source image 65.71 lệch dù chỉ một fragment.
- `submission/test_lfm25_fused_shortconv_fp8.py`: SM90 compile test, kiểm tra
  state/rounding/scale/FP8 và microbenchmark cho GPU.
- `submission/Dockerfile.shortconv-fp8-fused`: pin digest image 65.71.
- `submission/docker-compose_shortconv_fp8_candidate.yml`: giữ nguyên scheduler
  8192/4096/32 đã đạt 65.71, chỉ thêm opt-in mới.
- `scripts/build_lfm25_shortconv_fp8_image.sh`: build; chỉ push khi `PUSH=1`.
- `scripts/lfm25_matrix_shortconv_fp8_20260727.sh`: A/B cùng image, đổi thứ tự
  OFF/ON qua từng repeat.
- `scripts/lfm25_remote_ab.sh`: thêm hai case
  `shortconv_fp8_fused_off/on`.

Công tắc `VLLM_LFM25_FUSED_SHORTCONV_FP8` được đọc trước khi dựng model:

- `0`: `out_proj` không nhận quant config, quay lại BF16 và đường 65.71;
- `1`: `out_proj` nhận FP8 config và decode-only dùng kernel mới.

Điều này quan trọng vì A/B dùng cùng image, tránh nhiễu do layer hoặc dependency
khác nhau giữa hai Docker build.

## Tính đúng được bảo toàn như thế nào

Kernel giữ nguyên ba ranh giới làm tròn của đường cũ:

1. `B*x` được làm tròn về BF16/FP16;
2. recurrent convolution được làm tròn qua dtype của state rồi về dtype
   projection;
3. `C*conv` được làm tròn BF16/FP16 trước khi tính absmax và chuyển E4M3.

`NULL_BLOCK_ID` của CUDA graph được mask hoàn toàn nên không đọc/ghi `state[-1]`.
State width-3 vẫn cập nhật đúng `[old_1, current_Bx]`.

Các kiểm tra đã qua trên máy hiện tại:

| Kiểm tra | Kết quả |
|---|---:|
| Exact-source patch trên image digest 65.71 | PASS |
| CPU recurrence/rounding/scale self-test trong Docker build | PASS |
| `py_compile` host và trong image | PASS |
| `docker compose config --quiet` | PASS |
| `bash -n` cho build/matrix/remote A/B scripts | PASS |
| Triton compile + `ptxas` cho SM90 | PASS |
| Registers/thread | 64 |
| Shared memory | 32 B |
| Spill stores / loads | 0 / 0 B |

Máy Windows hiện tại không có NVIDIA runtime (`nvidia-smi` không tồn tại), nên
chưa chạy được numerical GPU test hay latency microbenchmark. SM90 compile chỉ
chứng minh kernel biên dịch/assemble sạch; không được diễn giải thành speedup.

## Cách chạy A/B trước khi dùng portal

Trên máy Linux có NVIDIA GPU và model/workload hiện tại:

```bash
PUSH=0 scripts/build_lfm25_shortconv_fp8_image.sh

IMAGE=misokaio/ghfjdk:v0.25.1-lfm25-shortconv-fp8 \
REPEATS=3 \
scripts/lfm25_matrix_shortconv_fp8_20260727.sh
```

Trước A/B end-to-end, chạy test kernel trên H100/H200:

```bash
docker run --rm --gpus all \
  --entrypoint python3 \
  misokaio/ghfjdk:v0.25.1-lfm25-shortconv-fp8 \
  /opt/lfm25/test_lfm25_fused_shortconv_fp8.py \
  --iterations 500
```

Chỉ promote khi đồng thời đạt:

1. state khớp tuyệt đối; scale/dequant FP8 qua tolerance của test;
2. microbenchmark median batch 8/16/32 nhanh hơn đường
   `fused_shortconv + standalone quant`;
3. median ERS của ba cặp ON > OFF; TTFT/TPOT không đổi lấy tăng failure;
4. GPQA full cùng seed/template không tạo accuracy drop có ý nghĩa.

Nếu qua các gate trên, push public và pin digest thay vì nộp mutable tag:

```bash
PUSH=1 scripts/build_lfm25_shortconv_fp8_image.sh
docker buildx imagetools inspect \
  misokaio/ghfjdk:v0.25.1-lfm25-shortconv-fp8
```

Sau đó thay `image:` trong compose candidate bằng đúng `@sha256:...`.

## Các hướng đã khảo sát nhưng không chọn cho vòng này

| Hướng | Quyết định | Lý do |
|---|---|---|
| Add + RMSNorm + dynamic FP8 | Loại | Đã đo thấp; yêu cầu không tiếp tục hướng này |
| Thêm tuning flag/scheduler | Không ưu tiên | Ma trận cũ đã phủ rộng và hầu hết neutral/xấu; scheduler 8192/4096/32 đã có portal evidence |
| Medusa/MTP/speculative | Không làm lại | Đã chạm trần theo thử nghiệm hiện có; workload batch động làm chi phí verify nhạy |
| Quant toàn bộ ShortConv `in_proj+out_proj` | Chưa dùng | Tiềm năng bandwidth cao hơn nhưng phạm vi sai số lớn hơn và quay lại hướng online quant đã bão hòa |
| Online INT4/Machete cho projection/LM head | Chưa dùng | Rủi ro GPQA lớn, không có portal evidence tốt hơn 65.71 trong repo |
| Cascade attention | Loại | Replay cũ cho kết quả xấu rõ rệt |
| Fused greedy LM-head/argmax | Nghiên cứu tiếp | Tiềm năng cao vì vocab 65.536 và 300 output token/request, nhưng cần custom GEMM/argmax chính xác; phạm vi triển khai lớn hơn candidate này |

Danh sách paper trong `LLM-inference-optimization-paper/README.md` hữu ích để lập
bản đồ giải pháp, nhưng nhiều hướng trong đó nhắm multi-GPU, disaggregation,
long-context hoặc model retraining. Chúng không khớp bài toán một MIG H200, model
1.17B, output cố định và accuracy gate. Với trace này, tối ưu đường decode
model-specific và giảm launch/memory round-trip vẫn là hướng có tỷ lệ rủi ro/lợi
ích tốt nhất sau khi các flag/speculation/quantization chung đã bão hòa.

## Trạng thái an toàn

Submission portal 65.71 ở
`submission/docker-compose_silu_fp8_65.71.yml` không bị sửa. File bị xóa sẵn
`submission/docker-compose_silu_fp8_processing.yml` cũng không được khôi phục.
Candidate mới chỉ là nhánh thử nghiệm cho đến khi có số H200 và GPQA.
