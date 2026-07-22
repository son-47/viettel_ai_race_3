# Báo cáo phân tích cấu hình LFM2.5 trên NVIDIA L4 và MiG H200

Ngày chạy L4: 2026-07-16; cập nhật leaderboard H200: 2026-07-22 (Asia/Bangkok)  
Phạm vi: vòng 1 Viettel AI Race 2026, `LiquidAI/LFM2.5-1.2B-Instruct`, vLLM 0.22.1 và image vLLM 0.25.1-derived `misokaio/ghfjdk:v0.25.2`.

## 0. Cập nhật từ grader MiG H200 ngày 21–22/07/2026

Năm lượt chấm chính thức đều dùng cùng image `misokaio/ghfjdk:v0.25.2`, FP8, prefix caching và chunked prefill:

| Compose | Scheduler | `max-model-len` | Batch tokens | ERS | TTFT p50 | TTFT p95 | TBT median | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `docker-compose copy.yml` | stock async FCFS | 32768 | 8192 | **63.76** | **37 ms** | **58 ms** | 4 ms | 5/420 |
| `docker-compose.yml` (custom, lượt cũ) | score-aware priority v5 | 8192 | 4096 | 61.78 | 46 ms | 60 ms | 4 ms | 5/420 |
| `docker-compose.v0252-interactivity-only.yml` | stock FCFS + interactivity | 32768 | 8192 | 59.87 | 50 ms | 70 ms | 4 ms | 6/420 |
| `docker-compose.yml` (A/B batch tokens) | stock async FCFS | 32768 | 4096 | 60.74 | 48 ms | 64 ms | 4 ms | 5/420 |
| `docker-compose.v0252-seqs24.yml` | stock async FCFS, seqs 24 | 32768 | 8192 | 60.34 | 47 ms | 68 ms | 4 ms | 6/420 |

Chênh lệch thực đo là **-1.98 điểm**, TTFT p50 xấu hơn 9 ms và p95 xấu hơn 2 ms. TBT median và số request lỗi không đổi. Vì lượt 61.78 thay đồng thời ba biến (scheduler, batch tokens và context cap), không thể quy toàn bộ chênh lệch cho riêng scheduler; kết luận chắc chắn là **tổ hợp stock FCFS + 8192/32 thắng tổ hợp custom + 4096/32 trên H200**.

`config_hash=sha256:603c84f...` giống nhau ở cả năm lượt dù command khác nhau, nên coi đây là giá trị opaque của grader, không dùng nó để suy luận các server có cùng runtime config.

Ước lượng từ median chỉ để giải thích độ lớn, không thay thế ERS trung bình per-request:

- tại TTFT 37 ms và TBT 4 ms, proxy score là khoảng 65.54;
- tại TTFT 46 ms và TBT 4 ms, proxy score là khoảng 63.42;
- chênh proxy 2.12 điểm gần với chênh ERS thực 1.98 điểm;
- quanh TBT 4 ms, tăng 1 ms có thể đắt hơn nhiều so với tăng 1 ms TTFT. Vì vậy bước tiếp theo ưu tiên giảm decode/IPC latency, không tăng prefill batch quá 8192.

Năm request lỗi của hai lượt đầu đặt trần mất điểm tuyệt đối là `5/420 × 100 = 1.19` điểm; nếu chúng có chất lượng tương đương các request thành công hiện tại, phần có thể lấy lại gần 0.77 điểm. Lượt interactivity tăng lên 6 lỗi, đồng thời TTFT xấu đi, nên không có lý do giữ mode này. Cần per-request error/server log để xác định timeout, HTTP error, 0-token hay lỗi model trước khi tối ưu trực tiếp failed count.

### Kết quả interactivity ngày 22/07/2026: loại hướng này

`--performance-mode=interactivity` đạt **59.87**, thấp hơn control 63.76 tới **3.89 điểm**. TTFT p50 tăng 37 → 50 ms, p95 tăng 58 → 70 ms, TBT median vẫn 4 ms và failed tăng 5 → 6. Đây là thoái lui đồng thời về ERS, TTFT và độ tin cậy; không ghép thêm spinloop để cố cứu nhánh này. File interactivity-only chỉ được giữ làm bằng chứng/audit.

Đề xuất `VLLM_USE_SPINLOOP_EXT=1` cũng được rút lại. Audit source vLLM 0.25.1 cho thấy extension này được gọi trong `shm_broadcast`, không phải đường ZMQ giữa frontend và EngineCore; với tensor parallel 1 chưa có bằng chứng nó nằm trên critical path của workload này.

### Kết quả hai A/B sạch: giữ `8192/32`

- A/B batch tokens `4096/32` đạt **60.74**, thấp hơn control **3.02 điểm**. TTFT p50 tăng 37 → 48 ms, p95 tăng 58 → 64 ms; failed giữ 5/420 và TBT median giữ 4 ms.
- A/B sequence `8192/24` đạt **60.34**, thấp hơn control **3.42 điểm**. TTFT p50 tăng 37 → 47 ms, p95 tăng 58 → 68 ms; failed tăng 5 → 6 và TBT median giữ 4 ms.

Hai thử nghiệm đã tách được nguyên nhân: trên grader H200 này, giảm riêng batch tokens xuống 4096 hoặc giảm riêng `max-num-seqs` xuống 24 đều làm TTFT và ERS xấu hơn. Không thử `4096/24`, vì cả hai thành phần đều đã cho tín hiệu âm. `docker-compose copy.yml` với stock FCFS + `8192/32` tiếp tục là cấu hình tốt nhất đã đo; không cần build/push lại image và không ghi đè tag `v0.25.2` đã nộp.

## 1. Kết luận sweep L4 lịch sử

Cấu hình latency tốt nhất trong sweep L4 là:

```yaml
--quantization=fp8
--max-model-len=8192
--gpu-memory-utilization=0.85
--enable-prefix-caching
--enable-chunked-prefill
--max-num-batched-tokens=4096
--max-num-seqs=32
```

So với cấu hình gần với file ban đầu (`FP8`, `8192`, `max-num-seqs=60`) ở cùng tải 2×, cấu hình `4096/32`:

- tăng ERS giả định từ 0.08763 lên 0.09476, tương đương khoảng **+8.1%**;
- giảm TTFT trung bình từ 440.8 xuống 335.6 ms (**-23.9%**);
- giảm p95 TTFT từ 1132.2 xuống 789.4 ms (**-30.3%**);
- giảm TPOT trung bình từ 11.05 xuống 10.95 ms và p95 TPOT từ 16.09 xuống 14.85 ms;
- giữ 330/330 request chấm điểm thành công.

Kết quả xác nhận ở đúng nhịp trace 1×, prompt trung bình 4001.6 token: 330/330 thành công, TTFT trung bình 254.4 ms, p95 TTFT 625.9 ms, TPOT trung bình 9.22 ms, p95 TPOT 12.83 ms. ERS giả định `w=0.5` là 0.13928.

Đây là **latency winner trên L4 ngày 16/07**, không còn là cấu hình nộp được ưu tiên sau khi có dữ liệu H200. FP8 vẫn phải được đối chiếu GPQA full với BF16; `submission/docker-compose.lfm25-bf16-safe.yml` là phương án accuracy-safe lịch sử.

## 2. Ràng buộc lấy từ đề hiện tại

Nguồn cục bộ: [`problem.md`](../problem.md).

- Máy chấm: 1 MiG H200, 18GB VRAM, 3 CPU, 8GB RAM; Ubuntu 24.04, driver 590.x.
- Model bắt buộc: `LiquidAI/LFM2.5-1.2B-Instruct`; chỉ được dùng vLLM.
- 70 hội thoại Poisson, mỗi hội thoại 6 turn, tổng cộng **420 request được chấm**; leaderboard xác nhận `total_count=420`, `warmup_count=0`.
- Shared system prefix 1000 token, prefix riêng mỗi hội thoại 1000 token, user thêm 150 token mỗi turn và output bị pin 300 token.
- Theo đặc tả rút gọn, input tăng từ khoảng 2150 token ở turn 1 lên 4400 token ở turn 6; input + output tối đa khoảng 4700 token.
- Request lỗi, timeout hoặc trả 0 token nhận điểm 0.
- Ngưỡng mới trong đề: TTFT 10–400 ms, TPOT 1–10 ms, số mũ 2.

Tài liệu portal mới xác nhận trọng số TTFT `w=0.5`. Hai lượt H200 hiện có `f_delta=1`, `penalty=1`, `accuracy_drop=0`.

Harness cũ `harness/scoring.py` **không còn đúng với đề này**: nó đang dùng TTFT 100–1500 ms và TPOT 20–45 ms. Không dùng con số Score từ harness cũ để chọn config LFM2.5.

## 3. Phân tích `trace_grading_public.jsonl` lịch sử

Phần này mô tả trace công khai cũ dùng cho sweep L4 ngày 16/07. File `grading-workload-spec.json` mới và grader 420 request đã thay thế các giả định primer/output 200 token bên dưới; không dùng phần lịch sử này để diễn giải trực tiếp leaderboard hiện tại.

Nguồn: [`trace_grading_public.jsonl`](../input_part2/trace_grading_public.jsonl).

### 3.1. Cấu trúc và khối lượng

| Thuộc tính | Giá trị |
|---|---:|
| Tổng dòng/request | 420 |
| Tổng hội thoại | 70 |
| Primer | 15 hội thoại, 90 request |
| Được chấm | 55 hội thoại, 330 request |
| Turn mỗi hội thoại | 6 |
| `think_ms` | luôn 3000 ms |
| Input scored | 3997–4000 token, trung bình 3998.87 |
| Input scored | 11992–11999 ký tự, trung bình 11996.52 |
| Tổng input scored | 1,319,628 token ước lượng |
| `out_tokens_max` | luôn 200; tổng tối đa 66,000 token scored |

Turn 0 luôn 4000 token. Kích thước giảm rất nhẹ qua các turn và turn 5 còn 3997–3999 token. Workload vì thế gần như đơn cỡ; không cần tối ưu cho hỗn hợp short/long prompt.

### 3.2. Arrival process

Chỉ turn đầu có `timestamp_ms` arrival thực; turn sau có timestamp 0 và phải chạy tuần tự sau khi response trước kết thúc cộng 3 giây think time. Replay 420 dòng như 420 request độc lập là sai semantics.

- Arrival đầu: 0 ms; cuối: 302,866 ms.
- 69 inter-arrival có mean 4389.4 ms, standard deviation 4404.0 ms, CV = 1.003; khớp đặc trưng Poisson/exponential.
- Tốc độ arrival hội thoại trung bình khoảng 0.228 hội thoại/giây.
- Burst lớn nhất: 3 hội thoại/1 giây, 6/5 giây, 8/10 giây.

Vì mỗi hội thoại còn sống qua 6 lượt và có think time, số sequence đồng thời lớn hơn arrival rate tức thời. Đây là lý do `max-num-seqs` vừa phải có ích, nhưng cap 8 gây queueing rất nặng.

### 3.3. Những gì trace công khai không cho biết

Trace không có prompt thật, output token thực tế hay tỷ lệ prefix giống nhau. Do đó không thể suy ra cache hit-rate từ file này. Bất kỳ khẳng định kiểu “prefix reuse X%” từ trace công khai đều không có cơ sở.

## 4. Phương pháp benchmark

Harness mới: [`harness/benchmark_public_round1.py`](harness/benchmark_public_round1.py).

- Mỗi hội thoại là một coroutine: chờ arrival của turn 0, gửi 6 turn tuần tự, chờ `think_ms` sau mỗi response.
- Gửi đủ 15 primer nhưng loại chúng khỏi ERS; chấm đúng 330 request còn lại.
- Prompt tổng hợp được hiệu chuẩn thành trung bình 4001.6 token theo tokenizer thật của LFM2.5.
- Prompt có 3072 ký tự prefix chung, marker riêng ngay sau prefix và phần còn lại ổn định giữa các turn. Điều này tạo workload cache-friendly hợp lý cho multi-turn nhưng không khẳng định giống private prompt.
- Ép 200 output token bằng `min_tokens=max_tokens=200` để mọi config chịu cùng worst-case decode load. Private grader có thể sinh ít hơn.
- Sweep chính dùng rate 2× để làm lộ queueing knee; cấu hình thắng được xác nhận lại ở rate 1×.
- Mỗi server chạy trong Docker với `--cpus=3`, `--memory=8g`, `--shm-size=2g`, đúng giới hạn CPU/RAM của đề.
- Server dùng `vllm/vllm-openai:v0.22.1`; cờ CLI được kiểm tra trực tiếp từ image.

Máy thử thực tế:

- Ubuntu Linux, NVIDIA L4 23,034 MiB, driver 580.159.03/CUDA 13.0;
- host có 31GB RAM nhưng container server bị giới hạn 8GB;
- Docker 28.0.1, Compose 2.27.0.

Máy này **không phải** MiG H200 18GB/driver 590.x. Số tuyệt đối không dự đoán leaderboard; kết quả chủ yếu dùng để xếp hạng tương đối và loại config xấu. H200 có kiến trúc/băng thông khác nên cần dùng tối đa 5 lượt submission như một final H200 sweep.

## 5. Kết quả L4 lịch sử

Mọi case hiệu chuẩn dưới đây đều đạt 330/330 request thành công và prompt trung bình 4001.6 token. ERS dùng ngưỡng mới và giả định `w=0.5`; không bao gồm accuracy penalty.

| Case | Rate | ERS giả định | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 |
|---|---:|---:|---:|---:|---:|---:|
| BF16, PC off, 8192/16 | 2× | 0.01633 | 884.6 | 2007.7 | 23.12 | 27.51 |
| BF16, PC on, 4096/32 | 2× | 0.04377 | 572.8 | 1356.8 | 15.83 | 22.23 |
| BF16, PC on, 8192/8 | 2× | 0.00141 | 7420.7 | 10422.8 | 17.24 | 18.01 |
| BF16, PC on, 8192/16 | 2× | 0.03649 | 687.3 | 1475.1 | 17.30 | 23.10 |
| BF16, PC on, 8192/32 | 2× | 0.03179 | 805.7 | 1777.3 | 16.40 | 24.25 |
| FP8, PC on, 8192/16 | 2× | 0.08490 | 395.5 | 877.5 | 11.15 | 14.81 |
| FP8, PC on, 4096/16 | 2× | 0.08989 | 332.2 | 659.3 | 11.08 | 15.19 |
| **FP8, PC on, 4096/32** | **2×** | **0.09476** | **335.6** | **789.4** | **10.95** | **14.85** |
| FP8, PC on, 8192/32 | 2× | 0.06738 | 476.8 | 1132.3 | 11.24 | 16.04 |
| FP8, PC on, 8192/60 | 2× | 0.08763 | 440.8 | 1132.2 | 11.05 | 16.09 |
| **FP8, PC on, 4096/32** | **1×** | **0.13928** | **254.4** | **625.9** | **9.22** | **12.83** |

Một run BF16 1× ban đầu đạt TTFT 43.6/TPOT 10.39 ms nhưng prompt chỉ 2819 token do bản harness mới chưa copy thành công. File vẫn được lưu để audit nhưng bị loại khỏi quyết định.

Toàn bộ per-request JSON và log nằm ở [`results/l4_20260716`](results/l4_20260716).

## 6. Diễn giải từng đòn bẩy

### `max-num-batched-tokens`: 4096 thắng trên L4 cũ, 8192 thắng trên H200 hiện tại

Với prompt khoảng 4k, chunk 8192 cho phép prefill lớn chiếm GPU lâu hơn và làm decode bị gián đoạn. Cả BF16 lẫn FP8 đều cho kết quả 4096 tốt hơn trong các cặp có thể so sánh. FP8 4096/32 thắng 8192/32 ở cả TTFT lẫn TPOT, nên lựa chọn không phụ thuộc chữ số `w` bị thiếu.

Kết quả H200 ban đầu đảo thứ hạng tổ hợp: stock FCFS + 8192/32 đạt 63.76, còn custom priority + 4096/32 đạt 61.78. A/B sạch sau đó đã xác nhận trực tiếp: cùng stock FCFS, context 32768 và seqs 32, giảm riêng batch tokens 8192 → 4096 chỉ đạt 60.74. Trên grader hiện tại, **8192 thắng 4096** với chênh lệch 3.02 điểm, TTFT p50 tốt hơn 11 ms và p95 tốt hơn 6 ms.

Kết quả này cũng cho thấy thứ hạng từ sweep L4 cũ không chuyển nguyên sang MiG H200/grader. Giữ 8192 cho cấu hình tốt nhất hiện tại.

Khuyến nghị chung của vLLM rằng model nhỏ thường hưởng lợi throughput với batch-token lớn hơn 8192 là điểm xuất phát, không thay thế benchmark latency có SLO rất chặt; xem [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).

### `max-num-seqs`: 32 thắng A/B H200; 24 và 8 bị loại

- Cap 8 làm TTFT trung bình tăng lên 7.4 giây ở tải 2×.
- Cap 16 ổn nhưng có tail lớn hơn.
- Cap 32 có ERS FP8 tốt nhất và tail ổn định.
- Cap 24 trong A/B H200 chỉ đạt 60.34 so với 63.76 của cap 32; TTFT p50/p95 tăng 10/10 ms và failed tăng từ 5 lên 6.
- Cap 60 giảm một ít TPOT mean so với 8192/32 nhưng TTFT vẫn xấu hơn nhiều; tổng thể thua 4096/32.

Giá trị 60/69 trong compose cũ không được xác nhận cho workload LFM2.5; nó có nguồn gốc từ sweep Qwen cũ trong repo.

### Prefix caching: giữ lại nhưng coi là experimental

So sánh trực tiếp BF16 8192/16 ở tải 2×:

- PC on: TTFT 687.3 ms, TPOT 17.30 ms;
- PC off: TTFT 884.6 ms, TPOT 23.12 ms.

vLLM 0.22.1 tự chuyển LFM2 sang Mamba cache mode `align` và log cảnh báo experimental. LFM2 không hỗ trợ mode `all`; xem [implementation LFM2 của vLLM 0.22](https://docs.vllm.ai/en/v0.22.0/api/vllm/model_executor/models/lfm2/) và [Mamba cache config](https://docs.vllm.ai/en/v0.21.0/api/vllm/model_executor/models/config/). Vì benchmark vẫn 330/330 thành công và latency tốt hơn, giữ `--enable-prefix-caching` cùng `--enable-chunked-prefill`.

### FP8: latency thắng, accuracy còn phải gate

So sánh trực tiếp 8192/16 cho thấy FP8 giảm TTFT 687.3 → 395.5 ms và TPOT 17.30 → 11.15 ms. vLLM mô tả online FP8 là quantize Linear/MoE weights lúc load và scale activation động; xem [Online Quantization](https://docs.vllm.ai/en/latest/features/quantization/online/). Model card xác nhận checkpoint gốc là BF16, 1.17B tham số và context 32768; xem [LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).

Không dùng kết quả latency để suy diễn GPQA. Cần chạy đúng script/dataset BTC cho cả BF16 và FP8 cùng seed/template. Nên nộp cả hai biến thể trong quota tối đa 5 submission, rồi chỉ chọn FP8 hậu kiểm nếu accuracy drop nằm trong vùng an toàn.

### Context và memory

`max-model-len=8192` đủ cho workload mới khoảng 4400 input + 300 output và overhead chat template. Tuy nhiên cấu hình H200 tốt nhất giữ 32768 để bám đúng control 63.76 và tránh thêm một biến liên quan GPQA. Model chỉ chiếm khoảng 2.2 GiB GPU khi nạp BF16 trên L4; với `gpu-memory-utilization=0.85`, KV capacity vẫn rộng. Vì thế chưa có lý do bật FP8 KV cache, CPU offload hay NVMe offload.

## 7. Audit repo và compliance

- `data/model-real/config.json` là `qwen3_5`, không phải LFM2.5. Không dùng model cục bộ đó để benchmark bài mới.
- Nhiều compose v2–v29 và research comment trong repo thuộc workload Qwen cũ (prompt đến 27k token, SLO cũ). Không chuyển nguyên các kết luận đó sang LFM2.5.
- `submission/docker-compose.v29-fp8-window32.yml` còn `served-model-name=Qwen3.5-2B`, `LAMBDA_GATE`, `LAMBDA_CHARCUT`, `LAMBDA_WINDOW` và scheduler tùy chỉnh. Không dùng file này cho đề hiện tại.
- `submission/lambda/chat_serving_lambda_baked.py` sửa prompt token IDs bằng head+tail truncation. Các biến thể có trace-only gate, cắt prompt hoặc cap output có rủi ro cao vi phạm điều khoản cấm dual-path, gaming đo lường và can thiệp tokenizer. Compose cuối dùng image vLLM chính thức và không dùng các cơ chế đó.
- Không bật `--kv-cache-dtype=fp8`: cache memory không thiếu và đây là một trục accuracy khác chưa được gate.
- Không bật speculative decoding: LFM2 prefix-cache align mode có giới hạn tương thích; workload mới pin 300 output token nhưng chưa có draft model/acceptance benchmark chứng minh lợi ích.
- Custom score-aware scheduler v5 đã được port đúng API vLLM 0.25.1 và SLO 400 ms, nhưng tổ hợp có scheduler đạt 61.78, thấp hơn control stock 63.76. Main compose hiện tắt custom scheduler; module vẫn nằm trong image nhưng không được import nếu không truyền `--scheduler-cls`.
- `--performance-mode=interactivity` đạt 59.87, TTFT p50/p95 xấu hơn và failed tăng lên 6; loại khỏi candidate tiếp theo. Không bật `VLLM_USE_SPINLOOP_EXT` vì source audit không đặt nó trên đường frontend–EngineCore ZMQ của cấu hình tensor parallel 1.
- Hai A/B stock mới cũng đã bị loại: `4096/32` đạt 60.74 và `8192/24` đạt 60.34, đều thua stock `8192/32` đạt 63.76. Không kết hợp thành `4096/24`.

## 8. Cấu hình tốt nhất và kế hoạch tiếp theo

File tốt nhất đã đo: [`submission/docker-compose copy.yml`](submission/docker-compose%20copy.yml), ERS 63.76.  
File A/B batch tokens đã đo: [`submission/docker-compose.yml`](submission/docker-compose.yml), stock FCFS + `4096/32`, ERS 60.74 — loại.  
File A/B sequence đã đo: [`submission/docker-compose.v0252-seqs24.yml`](submission/docker-compose.v0252-seqs24.yml), stock FCFS + `8192/24`, ERS 60.34 — loại.  
File BF16 fallback: [`submission/docker-compose.lfm25-bf16-safe.yml`](submission/docker-compose.lfm25-bf16-safe.yml).

Ưu tiên tiếp theo sau năm kết quả:

1. Giữ stock FCFS + `8192/32` + context 32768 làm baseline và cấu hình tốt nhất hiện tại.
2. Nếu quota cho phép, rerun nguyên trạng `docker-compose copy.yml` để định lượng noise quanh 63.76.
3. Thu thập per-request error/server log của 5 request lỗi; đây là hướng còn dư địa rõ hơn việc giảm batch hoặc sequence cap.
4. Không nộp lại interactivity, `4096/32`, `8192/24`, custom scheduler hoặc tổ hợp `4096/24`.
5. Chỉ mở A/B một biến mới khi có giả thuyết từ log; không thay đổi đồng thời scheduler, context và batching.

`docker-compose.v0252-interactivity-only.yml` đã bị loại với ERS 59.87 và chỉ được giữ để audit, không nộp lại và không dùng làm nền cho candidate mới.

Sau vòng online, chạy GPQA full trên image bất biến của các candidate. Nếu FP8 có accuracy drop không an toàn, chọn BF16 dù ERS thấp hơn.

## 9. Tái lập

Script orchestration: [`scripts/l4_config_matrix.sh`](scripts/l4_config_matrix.sh).

```bash
# trên máy L4 đã có image/model cache
bash scripts/l4_config_matrix.sh main
bash scripts/l4_config_matrix.sh fp8-followup
bash scripts/l4_config_matrix.sh final
```

Các giới hạn quan trọng của kết luận:

1. Prompt thật đã bị BTC lược; synthetic prompt chỉ giữ đúng size và một giả định prefix-sharing.
2. Sweep L4 lịch sử ép 200 output token; workload grader mới pin 300 token.
3. L4 không tương đương MiG H200; chỉ dùng thứ hạng tương đối.
4. Portal mới xác nhận `w=0.5`; các kết quả H200 được ưu tiên hơn giả định từ tài liệu cũ.
5. Chưa chạy GPQA full trong phiên này, nên FP8 được gọi là latency winner, không phải accuracy-validated winner.
