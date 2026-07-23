# LFM2.5 — nghiên cứu phương pháp mới và kế hoạch thực nghiệm (2026-07-23)

## Phạm vi và nguyên tắc

Mục tiêu của tài liệu này là bổ sung các phương pháp chưa xuất hiện trong:

- `LFM25_RESEARCH_OPTIMIZATION_20260723.md`
- `LFM25_OPTIMIZATION_20260722.md`

Baseline hợp lệ vẫn là compose FP8 chính thức. Không thay đổi baseline, không lặp lại các sweep đã có, và không coi một phương pháp là thắng nếu chỉ tăng tốc nhưng làm giảm chất lượng GPQA mirror hoặc vi phạm giới hạn cuộc thi.

Thiết lập A/B giữ nguyên workload đã được dùng trong report trước: 70 hội thoại × 6 lượt, 420 request, shared prefix 1000 token, private prefix 1000 token, 150 token/lượt, output tối đa 300, Poisson seed 42, cùng image vLLM pinned và cấu hình prefix caching/chunked prefill/max-num-batched-tokens 8192/max-num-seqs 32. Screening dùng `RATE_SCALE=4`, sau đó xác nhận paired run ở scale 1 nếu có tín hiệu.

## Những gì đã loại khỏi vòng lặp

Đã đọc lại report cũ và loại khỏi matrix mới: stock FP8, các biến thể FP8 online đã đo, INT8 weight-only, MXFP8, FP8 KV/INT8 KV/TurboQuant KV, prefix caching on/off, chunked prefill, mọi batch-token/sequence sweep, block size, Mamba cache/backend/layout, CUDA graph, fused/FlashInfer/POD attention, scheduler/reserve/DBO/frontend/renderer/stream/output tuning, n-gram speculative decoding, warmup/keepalive và GPTQ/AWQ W4A16 research-only.

Các kết quả cũ vẫn được dùng làm guardrail: stock FP8 là đường submit chính; GPTQ/AWQ chỉ là nghiên cứu vì checkpoint offline không thuộc đường online-only hợp lệ.

## Sàng lọc từ README và paper

| Phương pháp | Nguồn | Quyết định |
|---|---|---|
| Draft-model speculative decoding | [vLLM draft-model docs](https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/draft_model/), [LFM2.5-350M](https://docs.liquid.ai/lfm/models/lfm25-350m) | Chọn. Dùng draft cùng họ LFM2.5, các mức 2/4/6/8 token để đo acceptance/TPOT/ERS và kiểm tra chất lượng. |
| SuffixDecoding | [paper](https://arxiv.org/abs/2411.04975), [vLLM suffix docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/suffix/) | Chọn. Đây là model-free speculative decoding, tận dụng suffix của prompt và generation trước; thử depth 8/16/32 và cấu hình loose. Cần `arctic-inference`. |
| Online BitsAndBytes 4-bit | [vLLM BitsAndBytes docs](https://docs.vllm.ai/en/v0.25.1/features/quantization/bnb/), [LLM.int8 paper](https://arxiv.org/abs/2208.07339) | Chọn để probe. Đây là inflight 4-bit từ model chưa quantize offline; cần đo thật vì bandwidth giảm nhưng dequant/kernel overhead có thể làm chậm trên L4/H200. |
| Hydragen | [paper](https://arxiv.org/abs/2402.05099), [implementation](https://github.com/ScalingIntelligence/hydragen) | Không đưa vào runner. Đây là kernel shared-prefix attention cấp engine, chủ yếu tích hợp Llama/gpt-fast; chưa có đường cắm an toàn cho LFM2.5 hybrid trong vLLM 0.25.1. |
| RelayAttention | [paper](https://arxiv.org/abs/2402.14808) | Không đưa vào runner. Cần reformulation/kernel attention theo kiến trúc Transformer và sửa engine; không phải flag độc lập của vLLM. |
| ChunkAttention | [paper](https://arxiv.org/abs/2402.15220) | Không đưa vào runner. Cùng nhóm prefix-aware attention kernel cấp engine; LFM2.5 có nhánh SSM nên việc áp dụng một kernel Transformer-only không còn là A/B hợp lệ. |
| TorchAO | [paper](https://arxiv.org/abs/2507.16099), [repo](https://github.com/pytorch/ao) | Không đưa vào matrix online hiện tại. vLLM cần torchao config/checkpoint serialization; tự ép BF16→TorchAO tại load sẽ không còn là phép thử cấu hình đơn giản và có rủi ro thay đổi semantics/legality. |
| Component-aware self-speculative decoding | [paper](https://arxiv.org/abs/2605.01106) | Ghi nhận cho hướng tương lai. Chưa có proposer/engine support tương ứng trong image vLLM pinned; paper cũng cho thấy hybrid model cần chọn component cẩn thận vì acceptance có thể thấp. |

Hydragen/RelayAttention/ChunkAttention là các ý tưởng đáng nghiên cứu nếu được phép sửa sâu vLLM hoặc viết kernel riêng. Chúng không bị “quên”; chúng bị loại khỏi thử nghiệm trực tiếp vì không thể cô lập thành một flag mới mà vẫn giữ đúng LFM2.5 hybrid và serving framework được phép.

## Các case mới đã tích hợp

`ai_race_2026/scripts/lfm25_remote_ab.sh` có các case mới:

- `suffix8`, `suffix16`, `suffix32`, `suffix16_loose`
- `bnb4`
- `draft350m2`, `draft350m4`, `draft350m6`, `draft350m8`
- `bnb4_suffix16`

Runner cài dependency nghiên cứu vào thư mục mount riêng (`researchdeps`), không sửa image vLLM pinned. Với draft model, đặt `DRAFT_MODEL_DIR` trỏ đến checkpoint local hoặc dùng `DOWNLOAD_DRAFT=1` để tải `LiquidAI/LFM2.5-350M` một lần vào workspace Lightning.

Matrix mới chỉ gọi các case trên trong `ai_race_2026/scripts/lfm25_matrix_new_methods_20260723.sh`. Guardrail GPQA mirror tương ứng nằm trong `ai_race_2026/scripts/lfm25_remote_gpqa.sh`.

## Cách đánh giá

1. Screening tất cả case mới ở `RATE_SCALE=4`, `WARMUP=0`.
2. Giữ lại cấu hình có ERS tăng so với paired `cold_control`, đồng thời kiểm tra 420/420 request thành công, không timeout và không lỗi server.
3. Chạy lại winner ở scale 1 với seed/workload cũ để giảm nhiễu proxy L4.
4. Chạy GPQA mirror trên cùng model/serving config. Loại cấu hình có tụt chất lượng rõ ràng; chỉ xem GPTQ/AWQ cũ là tham chiếu research-only.
5. Chỉ khi có winner ổn định và đạt guardrail mới cân nhắc tạo compose thử nghiệm riêng; compose submit chính thức không bị thay đổi trong nghiên cứu này.

Lưu ý: L4 là proxy để xếp hạng tương đối; không suy diễn trực tiếp latency tuyệt đối sang H200. Đặc biệt BitsAndBytes có thể cho VRAM thấp hơn nhưng chưa chắc có throughput tốt hơn do dequantization overhead.

## Kết quả Lightning L4

SSH đã kết nối thành công bằng identity tường minh:

```text
ssh -i C:\Users\admin\.ssh\id_ed25519_lightning \
  -o IdentitiesOnly=yes \
  s_01ky4bakadv2q9t17ay429chdv@ssh.lightning.ai
```

Máy đánh giá là NVIDIA L4 23,034 MiB. Image được tải đúng digest đã dùng trong report trước:

```text
misokaio/ghfjdk:v0.25.1
sha256:1d03088f685d6c8ddab5078d3b3374ba3e85ed557e5baa50aca9770b6cabdf18
```

Kết quả screening `RATE_SCALE=4`, `WARMUP=0`:

| Case | ERS | Δ so với control | Thành công | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| `cold_control` | 0.343036804 | 0 | 420/420 | 92.530 / 111.653 | 11.521 / 12.748 | Baseline |
| `suffix8` | 0.138589324 | -0.204447480 | 420/420 | 730.669 / 1920.013 | 20.473 / 24.625 | Loại |
| `suffix16` | 0.180821312 | -0.162215492 | 419/420 | 557.565 / 1792.590 | 19.835 / 25.427 | Loại |
| `suffix32` | 0.123690265 | -0.219346538 | 419/420 | 1030.418 / 2691.121 | 21.791 / 26.782 | Loại |
| `suffix16_loose` | 0.121201959 | -0.221834845 | 420/420 | 1174.007 / 3235.858 | 21.252 / 26.768 | Loại |
| `bnb4` | 0.105095560 | -0.237941244 | 417/420 | 763.088 / 2085.582 | 20.146 / 21.936 | Loại |
| `bnb4_suffix16` | 0.157868833 | -0.185167971 | 418/420 | 1378.138 / 3888.511 | 21.816 / 31.730 | Loại |

Cluster bootstrap 95% cho mọi delta đều âm:

```text
suffix16          [-0.190414660, -0.132074831]
suffix32          [-0.244941477, -0.191843305]
suffix16_loose    [-0.246534932, -0.195558048]
suffix8           [-0.228156556, -0.179566538]
bnb4              [-0.258219927, -0.216238827]
bnb4_suffix16     [-0.210544850, -0.157532032]
```

BitsAndBytes được cài `--no-deps` để dùng đúng Torch 2.11/CUDA 13 của image; không cho pip kéo Torch/CUDA khác vào `PYTHONPATH`. Kết quả xấu vì dequant/kernel overhead làm cả TTFT và TPOT tăng mạnh, không phải do checkpoint offline.

`draft350m2`, `draft350m4`, `draft350m6`, `draft350m8` đều được thử với checkpoint `LiquidAI/LFM2.5-350M` tải trực tiếp từ Hugging Face. Cả bốn dừng trước benchmark tại cùng assertion của vLLM:

```text
AssertionError: All drafting layers should belong to the same kv cache group
```

Lỗi xảy ra trong `llm_base_proposer.py` khi khởi tạo attention backend cho draft model hybrid. Vì lỗi không phụ thuộc số speculative token, đây là giới hạn hỗ trợ hybrid KV groups của vLLM 0.25.1, không phải một kết quả latency.

Rà lại CLI/source đúng image còn cho thấy:

- `async_scheduling` đã được vLLM tự bật mặc định khi executor tương thích, nên thêm `--async-scheduling` không tạo case mới.
- LFM2 không khai báo `SupportsMambaPrefixCaching`; khi prefix caching bật, `mamba_cache_mode=all` tự fallback về `align`. Explicit `align` vì vậy trùng baseline.
- `watermark` chỉ hữu ích khi có KV preemption/thrashing; workload hiện không cung cấp bằng chứng cần thay đổi admission policy.

Không có case mới nào đủ điều kiện chạy scale 1 hoặc GPQA mirror. Chạy thêm guardrail chất lượng cho các cấu hình chậm hơn rõ rệt và có request lỗi sẽ không thay đổi quyết định.

## Kết luận

Trong phạm vi vLLM 0.25.1, LFM2.5 hybrid, online-only và single L4/H200, baseline stock FP8 + prefix caching + chunked prefill vẫn là cấu hình submit tốt nhất. Không thay đổi `submission/docker-compose.yml`.

Toàn bộ JSON, metrics và server logs đã được tải về `ai_race_2026/results/lfm25_new_methods_20260723/`.
