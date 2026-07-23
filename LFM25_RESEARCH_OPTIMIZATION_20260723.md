# LFM2.5 — nghiên cứu tối ưu mới và thực nghiệm Lightning ngày 23/07/2026

## 1. Kết luận

- Cấu hình nộp hợp lệ tốt nhất vẫn là **exact stock FP8 online** đã đạt `63,82`
  trên H200: `max-num-batched-tokens=8192`, `max-num-seqs=32`,
  `max-model-len=32768`, prefix caching và chunked prefill.
- Đã bỏ `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600` khỏi
  `submission/docker-compose.yml`. Biến này không cải thiện phép đo chính thức:
  stock đạt `63,82`, còn stock + keep-alive 600 chỉ đạt `62,16`.
- Trong các kỹ thuật mới hợp lệ, không có phương án nào thắng stock đủ lớn và
  tái hiện được ở tải `r1`. `fp8_per_tensor` chỉ hơn stock `+0,000385 ERS`
  (`+0,09%`), còn Mamba FlashInfer chỉ hơn `+0,000144 ERS`; cả hai nằm trong
  nhiễu ở `r1`.
- **GPTQ W4A16** là latency winner kỹ thuật trên Lightning L4:
  `0,517258 ERS` so với stock `0,424552` ở `r1`, tăng `21,84%`; TPOT giảm
  `8,013 → 5,620 ms`. GPQA-Diamond mirror giảm `47/198 → 44/198`, tức
  `1,52` điểm phần trăm, nhỏ hơn guardrail 10 điểm.
- Không đưa GPTQ/AWQ vào compose nộp. Checkpoint của chúng đã được lượng tử
  offline, trong khi mục 3 của thể lệ ghi rõ phạm vi quantization là
  **“Các kỹ thuật Online Quantization”**. Chỉ nên thử nộp W4 nếu BTC xác nhận
  bằng văn bản rằng checkpoint offline được phép.

Các con số Lightning là A/B trên **NVIDIA L4**, không phải điểm private grader
MiG H200. Kết quả H200 `63,82` ở trên là kết quả chính thức đã có từ vòng trước,
không phải số suy diễn từ L4.

## 2. Môi trường thực nghiệm

- Studio: Lightning AI qua SSH.
- GPU proxy: NVIDIA L4, 23.034 MiB VRAM.
- Image: `misokaio/ghfjdk:v0.25.1`.
- Digest bất biến:
  `sha256:1d03088f685d6c8ddab5078d3b3374ba3e85ed557e5baa50aca9770b6cabdf18`.
- vLLM `0.25.1`, PyTorch `2.11.0+cu130`, Transformers `5.13.1`.
- Mỗi server bị giới hạn `3 CPU`, `8 GiB RAM`, `2 GiB shm`, một GPU.
- Workload latency: 70 hội thoại × 6 turn = 420 request, shared prefix 1.000
  token, private prefix 1.000 token, thêm 150 token/turn, output 300 token,
  Poisson seed 42.
- `r4` dùng để sàng lọc nhanh; `r1` dùng để xác nhận gần nhịp grader hơn.
- ERS proxy dùng đúng các ngưỡng TTFT `10–400 ms`, TPOT `1–10 ms`, gamma `2`,
  trọng số `0,5/0,5`.

## 3. Audit kỹ thuật cũ — không chạy lặp

Đã đọc toàn bộ `LFM25_OPTIMIZATION_20260722.md` trước khi thiết kế matrix mới.
Các họ sau đã có dữ liệu hoặc đã fail ở vòng trước nên không bị chạy lại:

- Stock FP8 online; prefix caching; chunked prefill; batch `8192/32`.
- Batch token `4096`, `6144`, `7168`, `9216`, `12288`; sequence `24`, `40`;
  `max-model-len=8192`.
- Block size `8/32`; Mamba block `8`; output processing chunk `256`;
  GPU memory utilization `0,92`; optimization level `2`.
- CUDA graph capture chính xác `1..32`; fused prefill/decode attention.
- Custom/Nexus/deadline scheduler; DBO; hai API frontend; frontend in-process.
- Speculative n-gram CPU/GPU; stream interval `2`; performance mode
  interactivity/throughput; ép FlashInfer.
- FP8 KV cache ở `r4` và `r1`; internal warm-up; HTTP keep-alive 600.

Việc không chạy lại các case trên giúp dành ngân sách GPU cho các nhánh thật sự
mới từ kho `LLM-inference-optimization-paper`.

## 4. Chọn kỹ thuật mới từ kho nghiên cứu

### Được triển khai và đo

- Cascade attention cho shared prefix.
- Model Runner V2 (MRV2).
- Rust frontend và nhiều renderer worker.
- SSM convolution layout `DS`.
- Mamba block `32/64`, Mamba cache FP16.
- Mamba backend FlashInfer, gồm cả FP16 + stochastic rounding.
- Tắt scheduler reserve-full-ISL và thử tắt hybrid KV manager.
- KV cache INT8 per-token-head và TurboQuant 4-bit.
- Online weight quantization:
  `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`,
  `int8_per_channel_weight_only`, `mxfp8`.
- Offline W4A16 GPTQ và AWQ, chỉ để xác định trần latency/quality.

### Không phù hợp kiến trúc hoặc tài nguyên

- TP/PP/PD, disaggregated prefill/decode và network collectives không phù hợp
  một GPU MiG.
- MoE routing, expert parallelism, EPLB không áp dụng cho LFM2.5-1.2B dense.
- LoRA serving không có adapter trong bài.
- CPU/NVMe offload không cần thiết khi model nhỏ hơn VRAM rất nhiều và sẽ thêm
  PCIe latency.
- Multimodal/video/diffusion, training-time pruning/NAS và serving nhiều model
  không thuộc workload.
- Spinloop extension chỉ nằm trên đường multiprocessing/tensor-parallel; TP=1
  không đi qua đường code này.
- Stochastic rounding với Mamba backend Triton yêu cầu compute capability 10.0
  (Blackwell). Source vLLM 0.25.1 cho phép backend FlashInfer trên GPU cũ hơn,
  nên đường này đã được triển khai và đo riêng thay vì loại theo suy đoán:
  [vLLM Mamba config](https://docs.vllm.ai/en/stable/api/vllm/config/mamba/).

MRV2 được vLLM mô tả là GPU-native/async-first nhưng vẫn experimental; chính
tài liệu cũng nêu hạn chế với linear-attention models. Kết quả thực tế dưới đây
xác nhận không nên bật cho LFM2.5 ở image hiện tại:
[vLLM MRV2](https://vllm.ai/blog/2026-03-24-mrv2).

Online quantization của vLLM lượng tử Linear/MoE weights lúc load từ checkpoint
BF16/FP16, không cần checkpoint đã lượng tử hay calibration. Đây là nhánh bám
sát thể lệ:
[vLLM Online Quantization](https://docs.vllm.ai/en/v0.21.0/features/quantization/online/).

## 5. Kết quả sàng lọc `r4`

Hai control lạnh kẹp đầu/cuối đạt `0,360640` và `0,364595`; trung bình
`0,362618`. CI dưới đây bootstrap theo **70 cụm hội thoại**, không coi 420
request là IID. Request/turn bị thiếu sau lỗi được tính `0`, đúng logic grader.

| Case mới | ERS | Δ so với control mean | CI95% của Δ | Quyết định |
|---|---:|---:|---:|---|
| Cascade attention | 0,336442 | -0,026176 | [-0,030681; -0,021593] | Loại |
| MRV2 | 0,340171 | -0,022447 | [-0,027550; -0,017537] | Loại |
| Rust frontend | 0,362792 | +0,000175 | [-0,002896; +0,003259] | Nhiễu |
| Renderer workers = 2 | 0,364333 | +0,001716 | [-0,001327; +0,004740] | Nhiễu |
| SSM layout DS | 0,363205 | +0,000587 | [-0,002407; +0,003650] | Nhiễu |
| Mamba block 32 | 0,361996 | -0,000622 | [-0,003550; +0,002344] | Không thắng |
| Mamba block 64 | 0,361749 | -0,000868 | [-0,005881; +0,003296] | Không thắng |
| Scheduler reserve off | 0,361915 | -0,000703 | [-0,003721; +0,002402] | Không thắng |
| KV INT8/token/head | 0,347582 | -0,015036 | [-0,020939; -0,009187] | Loại |
| FP8 per-block online | 0,348843 | -0,013774 | [-0,019392; -0,008215] | Loại |
| FP8 per-tensor online | 0,365487 | +0,002869 | [+0,000163; +0,005528] | Xác nhận `r1` |
| FP8 per-channel online | 0,364549 | +0,001931 | [-0,001009; +0,004909] | Nhiễu |
| INT8 weight-only online | 0,331100 | -0,031518 | [-0,037258; -0,025734] | Loại |
| MXFP8 online | 0,353403 | -0,009215 | [-0,014168; -0,004205] | Loại |
| Mamba cache FP16 | 0,364188 | +0,001570 | [-0,002543; +0,005675] | Nhiễu |
| Mamba FlashInfer | 0,366706 | +0,004089 | [+0,001368; +0,006966] | Xác nhận `r1` |
| Mamba FlashInfer + FP16/SR | 0,365669 | +0,003051 | [-0,000830; +0,006946] | Không hơn backend thường |
| GPTQ W4A16 offline | **0,391919** | **+0,029301** | **[+0,022944; +0,035794]** | Research-only winner |
| AWQ W4A16 offline | 0,382721 | +0,020103 | [+0,012515; +0,027578] | Research-only |

Các case fail khi startup:

- `hybrid_manager_off`: LFM2.5 có heterogeneous cache specs, không thể ép về
  một unified type.
- `kvturbo4`: recurrent cache group truyền dtype `auto` vào TurboQuant và vLLM
  báo `Unknown TurboQuant cache dtype: 'auto'`.
- `mamba_fp16_sr`: lần chạy Triton đầu thiếu explicit SSM dtype. Sau khi đọc
  source đúng phiên bản, đã chuyển sang FlashInfer để tránh giới hạn Blackwell;
  case sửa đúng chạy thành công nhưng không thắng.

TurboQuant vẫn có cơ sở nghiên cứu tốt (Hadamard rotation và scalar
quantization), nhưng implementation hiện tại không tương thích hybrid cache
group của LFM2.5 trong image này:
[vLLM TurboQuant](https://docs.vllm.ai/en/v0.22.1/api/vllm/model_executor/layers/quantization/turboquant/).

## 6. Xác nhận `r1`

| Case | ERS | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | Thành công |
|---|---:|---:|---:|---:|---:|---:|
| Stock `--quantization=fp8` | 0,424552 | 56,657 ms | 70,798 ms | 8,013 ms | 9,070 ms | 420/420 |
| `fp8_per_tensor` | 0,424937 | 56,467 ms | 70,010 ms | 8,015 ms | 9,065 ms | 420/420 |
| Mamba FlashInfer | 0,424696 | 56,520 ms | 71,972 ms | 8,022 ms | 9,088 ms | 420/420 |
| GPTQ W4A16 | **0,517258** | 57,610 ms | 73,912 ms | **5,620 ms** | **6,400 ms** | 420/420 |

`fp8_per_tensor` chỉ tăng `0,000385 ERS` (`0,09%`) so với stock và không giảm
TPOT. Tín hiệu dương nhỏ ở `r4` không tái hiện ở `r1`, nên không thay main
compose.

Mamba FlashInfer cũng không tái hiện lợi thế `r4`: ở `r1` chỉ tăng
`0,000144 ERS`, CI95% `[-0,001443; +0,001730]`, trong khi TPOT hơi xấu hơn
stock. Stochastic rounding còn chậm hơn backend FlashInfer thường ở `r4`.

GPTQ tăng `0,092706 ERS`, CI95% theo cụm
`[+0,088221; +0,097040]`, tương đương `+21,84%` tương đối. Lợi ích đến từ TPOT
giảm khoảng `29,87%`; TTFT trung bình tăng nhẹ khoảng `1,68%`.

## 7. GPQA-Diamond mirror đủ 198 câu

Client dùng cùng prompt, greedy decode, concurrency `8`, `max_tokens=1536`.
Package `datasets==4.4.1` được cài riêng vào
`/home/zeus/content/evaldeps`; nó không đi vào image inference/nộp bài.

| Weight | Đúng | Accuracy | Unparsed | Request error |
|---|---:|---:|---:|---:|
| Stock FP8 online | 47/198 | 23,74% | 39 | 0 |
| GPTQ W4A16 | 44/198 | 22,22% | 44 | 0 |
| AWQ W4A16 | 45/198 | 22,73% | 40 | 0 |

So với stock, GPTQ giảm `3/198 = 1,52` điểm phần trăm. Đối chiếu từng câu:

- Prediction giống nhau: `101/198`.
- Cả hai đúng: `23`.
- Chỉ stock đúng: `24`.
- Chỉ GPTQ đúng: `21`.
- Stock parsed nhưng GPTQ unparsed: `15`; chiều ngược lại: `10`.

Đây chỉ là mirror guardrail, không phải GPQA full chính thức của BTC. Tuy nhiên,
mức giảm quan sát nhỏ hơn rõ rệt ngưỡng không phạt 10 điểm phần trăm. Báo cáo cũ
đã xác nhận BF16 và FP8 online cùng đạt `46/198`, nên không lặp lại cặp đó.

## 8. Vì sao giữ prefix cache stock

LFM2.5 là hybrid attention/recurrent nên prefix cache phức tạp hơn Transformer
thuần. Marconi chỉ ra recurrent in-place state khiến partial overlaps cần
exact-state handling và policy admission/eviction riêng:
[Marconi, MLSys 2025](https://arxiv.org/abs/2411.19379).

Trong image hiện tại, vLLM đã dùng hybrid-aware Mamba cache mode; các thử nghiệm
tắt hybrid manager, đổi block `8/32/64`, cascade attention, đổi SSM layout và
đổi Mamba backend sang FlashInfer đều không thắng ở `r1`. Vì vậy giữ prefix
cache stock là lựa chọn có dữ liệu, không phải bỏ qua kỹ thuật hybrid caching.

## 9. Cấu hình chốt

`submission/docker-compose.yml`:

- Image được pin theo digest.
- Online `--quantization=fp8`.
- `--max-model-len=32768`.
- `--max-num-batched-tokens=8192`.
- `--max-num-seqs=32`.
- Prefix caching + chunked prefill.
- Không warm-up nội bộ.
- Không override HTTP keep-alive.
- Không KV quantization, speculative decoding, custom scheduler hay frontend
  experimental.

GPTQ/AWQ chỉ tồn tại trong runner nghiên cứu, không nằm trong compose nộp.

## 10. Artifacts và tái lập

Artifacts local:

- `results/lightning_20260723/lfm25_research_20260723/`: JSON từng request,
  metrics và server logs.
- `results/lightning_20260723/gpqa/`: JSON từng câu của stock/GPTQ/AWQ.

Scripts:

- `scripts/lfm25_remote_ab.sh`: runner tất cả case mới.
- `scripts/lfm25_matrix_research_20260723.sh`: matrix chỉ gồm các hướng chưa có
  trong báo cáo 22/07.
- `scripts/lfm25_remote_gpqa.sh`: GPQA paired runner cho online FP8 và W4.
- `scripts/analyze_lfm25_ab.py`: bootstrap theo cụm; đã sửa để turn bị thiếu
  sau request fail được tính điểm 0.

Lệnh sàng lọc:

```bash
RESULT_DIR=/home/zeus/content/results/lfm25_research_20260723 \
RATE_SCALE=4 WARMUP=0 \
bash /home/zeus/content/lfm25_matrix_research_20260723.sh
```

Lệnh xác nhận:

```bash
RESULT_DIR=/home/zeus/content/results/lfm25_research_20260723 \
RATE_SCALE=1 WARMUP=0 \
bash /home/zeus/content/lfm25_remote_ab.sh cold_control

RESULT_DIR=/home/zeus/content/results/lfm25_research_20260723 \
RATE_SCALE=1 WARMUP=0 \
bash /home/zeus/content/lfm25_remote_ab.sh fp8_per_tensor

RESULT_DIR=/home/zeus/content/results/lfm25_research_20260723 \
RATE_SCALE=1 WARMUP=0 \
bash /home/zeus/content/lfm25_remote_ab.sh mamba_flashinfer

RESULT_DIR=/home/zeus/content/results/lfm25_research_20260723 \
RATE_SCALE=1 WARMUP=0 \
bash /home/zeus/content/lfm25_remote_ab.sh gptq_w4
```

Lệnh GPQA:

```bash
N=198 CONC=8 MAX_TOKENS=1536 \
bash /home/zeus/content/lfm25_remote_gpqa.sh control

N=198 CONC=8 MAX_TOKENS=1536 \
bash /home/zeus/content/lfm25_remote_gpqa.sh gptq_w4

N=198 CONC=8 MAX_TOKENS=1536 \
bash /home/zeus/content/lfm25_remote_gpqa.sh awq_w4
```

## 11. Hành động tiếp theo có giá trị

Không nên tiếp tục sweep nhỏ trên L4: Rust frontend, renderer2, SSM layout,
Mamba FP16/FlashInfer và FP8 per-tensor đều nằm trong noise hoặc không tái hiện
ở `r1`.

Hai hành động duy nhất có kỳ vọng dương:

1. Nộp lại **exact stock compose không keep-alive 600** nếu submission tương ứng
   với `63,82` chưa được giữ trong top 5.
2. Hỏi BTC bằng văn bản liệu offline GPTQ/AWQ checkpoint có hợp lệ không. Nếu
   được phép, GPTQ W4A16 là ứng viên rõ ràng để dùng một slot submission riêng;
   nếu không, tuyệt đối không đưa nó vào bài.
