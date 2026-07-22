# LFM2.5-1.2B optimization report — 2026-07-22

## Kết luận

Phiên bản warmup lịch sử được ghi nhận ban đầu trong báo cáo dùng image công
khai được pin bằng digest:

```text
misokaio/ghfjdk:v0.25.1-warmup1@sha256:a17c8a90ebbdf45956611f7a88ba4eb156f70c5f4e4c411fc15f5a9d6fa53230
```

Thay đổi duy nhất so với engine stock thắng A/B trước đây là warmup nội bộ trước
khi HTTP endpoint báo sẵn sàng. Không thêm custom scheduler, speculative decode,
FP8 KV cache hoặc backend attention cưỡng bức vì các phương án đó không thắng ở
phép đo sát workload.

**Kết quả H200 chính thức của warmup1: 62,97 điểm, thấp hơn baseline 63,82 đúng 0,85 điểm.**
Vì vậy phiên bản warmup1 này được giữ để audit nhưng không còn được khuyến nghị
làm submission tốt nhất nếu mục tiêu là vượt mốc 63,82.

**Cập nhật sau A/B trực tiếp trên Lightning ngày 2026-07-22:** file
`submission/docker-compose.yml` hiện đã được đưa về engine stock
`misokaio/ghfjdk:v0.25.1@sha256:1d03088f...`, bỏ warmup nội bộ và thêm duy nhất
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600`. Biến này loại ổn định lỗi TCP làm mất trọn
một hội thoại 6 turn trong hai lần lặp ở tải chuẩn, không đổi đường tính model.
Trước khi có kết quả H200 bên dưới, đây là submission ERS được chọn để kiểm
chứng; BF16 fallback dùng cùng keep-alive fix.

**Kết quả H200 chính thức mới của compose stock + keep-alive: 62,16 điểm, thấp
hơn baseline 63,82 đúng 1,66 điểm và thấp hơn warmup1 0,81 điểm.** Grader vẫn
ghi nhận 5 failed request, do đó kết quả proxy L4 420/420 không tái hiện trên
MiG H200. Cấu hình này được giữ làm artifact A/B nhưng không còn được khuyến
nghị là phương án tốt nhất nếu mục tiêu là điểm leaderboard.

## Cập nhật A/B trực tiếp trên Lightning — keep-alive fix

### Phương pháp

- Studio: NVIDIA L4 23.034 MiB, driver 580.159.03, Docker 28.0.1.
- Container giữ đúng giới hạn chấm: 3 CPU, 8GB RAM, 2GB shared memory.
- Image stock: `misokaio/ghfjdk:v0.25.1` tại digest
  `sha256:1d03088f685d6c8ddab5078d3b3374ba3e85ed557e5baa50aca9770b6cabdf18`.
- Workload token-exact: 70 conversation × 6 turn, 420 request; 1.000 token
  system chung, 1.000 token riêng, 150 token user/turn, 300 output token.
- Screening dùng rate 4×; final validation dùng đúng rate 1×. Mọi case bắt đầu
  bằng engine và prefix cache rỗng, không warmup từ client.
- So sánh thống kê gom theo 70 conversation vì sáu turn trong một conversation
  không độc lập. JSON/log/metrics gốc nằm ở
  `results/lightning_20260722/lfm25_ab/`.

### Screening một biến ở tải 4×

Hai control stock cách nhau `0,003117` ERS, lớn hơn mọi mức “cải thiện” một-run.
Khoảng bootstrap theo conversation của tất cả candidate dưới đây đều cắt qua 0
khi so với trung bình control. Vì vậy không ghép các candidate có vẻ thắng.

| Case | ERS proxy | TTFT mean (ms) | TPOT mean (ms) | Kết luận |
|---|---:|---:|---:|---|
| Stock cold control #1 | 0,362389 | 80,408 | 11,321 | Biên noise thấp |
| Stock cold control #2 | **0,365506** | **78,785** | 11,422 | Biên noise cao |
| xxHash prefix | 0,363481 | 79,473 | 11,425 | Trong noise; cần image mới, loại |
| `max-model-len=8192` | 0,362826 | 80,323 | 11,399 | Trong noise, giảm safety margin |
| batch tokens 7.168 | 0,361383 | 80,451 | 11,470 | Thua |
| batch tokens 9.216 | 0,362234 | 80,007 | 11,443 | Thua control tốt |
| max sequences 40 | 0,364459 | 79,614 | 11,462 | Trong noise |
| Mamba block 8 | 0,363893 | 79,001 | 11,422 | Trong noise |
| output processing chunk 256 | 0,363148 | 79,740 | 11,470 | Trong noise |
| GPU memory utilization 0,92 | 0,365045 | 80,433 | 11,467 | Trong noise |
| optimization level 2 | 0,364102 | 79,193 | 11,464 | Trong noise |
| exact CUDA graphs 1–32 | 0,362443 | 80,526 | 11,444 | Không hơn control |
| fused prefill/decode attention | 0,364580 | 79,071 | 11,432 | Trong noise |

Ba nhánh bị loại ở startup/validation:

- attention `block-size=8`: không backend CUDA nào trong image hỗ trợ;
- `--disable-frontend-multiprocessing`: flag không còn trong vLLM 0.25.1;
- dual-batch overlap: validation yêu cầu backend all-to-all chuyên dụng, không
  phù hợp LFM2 dense trên một GPU.

Kết luận screening: giữ O3, FP8 weight, prefix caching, chunked prefill,
8.192 batch token, 32 sequence, memory utilization 0,85, backend/layout tự động.

### Phát hiện failure theo conversation ở tải 1×

Control stock trả lỗi `ServerDisconnectedError` tại conversation 25, turn 0,
arrival offset 85,049 giây. Harness không thể xây ngữ cảnh cho năm turn sau nên
chỉ hoàn thành 415 request, 414 thành công và ghi đúng 6 zero trên 420 request.
Server log không có OOM, traceback hay engine error.

vLLM 0.25.1 mặc định `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=5`. Với Poisson arrival và
HTTP connection pool, client có thể tái sử dụng socket đúng lúc server vừa đóng
sau 5 giây idle. A/B tăng server keep-alive lên 600 giây:

| Chỉ số | Stock 5s | Keep-alive 600s #1 | Keep-alive 600s #2 |
|---|---:|---:|---:|
| ERS proxy | 0,418626 | **0,423716** | **0,423679** |
| Completed / successful | 415 / 414 | **420 / 420** | **420 / 420** |
| Request error | 1 | **0** | **0** |
| Zero do error/missing | 6 | **0** | **0** |
| TTFT mean / p50 / p95 (ms) | 56,049 / 48,962 / 71,612 | 56,480 / 49,168 / 71,199 | 56,429 / 48,760 / 71,827 |
| TPOT mean / p50 / p95 (ms) | 8,074 / 7,932 / 9,125 | 8,073 / 7,942 / 9,101 | 8,076 / 7,948 / 9,096 |

Hai lần keep-alive lặp chỉ lệch `0,0000365` ERS và đều hoàn thành 420/420. ERS
trung bình tăng khoảng `0,005071`, tương đương **+0,507 điểm proxy trên thang
100**, trong khi TTFT/TPOT thực chất không đổi. Đây là thay đổi lossless: chỉ
đổi vòng đời TCP, không đổi weight, tokenizer, sampling, KV cache, scheduler hay
output. Dữ liệu L4 không thay thế lượt H200 chính thức. Tín hiệu failure
elimination đủ ổn định để đưa đi kiểm chứng và có rủi ro accuracy bằng 0, nhưng
kết quả H200 bên dưới cho thấy lợi ích reliability này không tái hiện trên
grader chính thức.

## Workload và hàm điểm đã dùng

- 70 hội thoại, 6 lượt/hội thoại, tổng 420 request.
- Shared system prefix 1.000 token; prefix riêng mỗi hội thoại 1.000 token.
- Mỗi lượt thêm 150 user token và bắt buộc sinh 300 output token.
- Lịch đến Poisson, seed 42.
- Điểm từng request cân bằng TTFT/TPOT; ngưỡng mới là TTFT 10–400 ms và TPOT
  1–10 ms, gamma 2.
- Accuracy gate chỉ chạy sau vòng online. Không bị phạt khi độ giảm GPQA so với
  BF16 không quá 10 điểm phần trăm.

Baseline leaderboard do người dùng cung cấp: 63,82 điểm; TTFT p50/p95 34/61 ms;
TBT median 4 ms; 6 failed; `accuracy_drop=0`, `f_delta=1`.

## Kết quả H200 chính thức của compose stock + keep-alive

Kết quả được chấm ngày 2026-07-22 cho `submission/docker-compose.yml` hiện tại:

| Chỉ số | Baseline 63,82 | Stock + keep-alive | Chênh lệch |
|---|---:|---:|---:|
| ERS / final score | 63,82 | **62,16** | **-1,66** |
| TTFT p50 | 34 ms | **46 ms** | **+12 ms** |
| TTFT p95 | 61 ms | **63 ms** | **+2 ms** |
| Failed | 6/420 | **5/420** | **-1** |
| TBT median | 4 ms | **4 ms** | 0 ms |

Thông tin đầy đủ: `ers=62.16`, `final_score=62.16`, `total_count=420`,
`failed_count=5`, `warmup_count=0`, `f_delta=1`, `penalty=1`,
`accuracy_drop=0`, config hash
`sha256:603c84f67bd0fadaa6ea739f2d1aa564761ff94e00dc61da25fe7e1d13853881`.
Hash grader này trùng giá trị đã được ghi cho lượt warmup1, vì vậy không nên
dùng riêng trường `config_hash` để phân biệt hai artifact/lượt chấm.

So với baseline, cấu hình lấy lại một failed request nhưng TTFT p50 tăng 12 ms
và TTFT p95 tăng 2 ms; phần giảm latency score trên số đông request lớn hơn lợi
ích của một request không còn failed. Keep-alive không làm thay đổi accuracy,
nhưng cũng không loại được nhóm 5 failure trên grader H200 như hai lần proxy L4.
Vì vậy không nên diễn giải lỗi L4 `ServerDisconnectedError` là nguyên nhân duy
nhất của failed request chính thức.

## Kết quả H200 chính thức của phiên bản warmup1

Kết quả được chấm ngày 2026-07-22 cho phiên bản warmup1 trước đây; phiên bản này
không còn nằm trong `submission/docker-compose.yml` hiện tại:

| Chỉ số | Baseline trước đó | Warmup1 | Chênh lệch |
|---|---:|---:|---:|
| ERS / final score | 63,82 | **62,97** | **-0,85** |
| TTFT p50 | 34 ms | **44 ms** | **+10 ms** |
| TTFT p95 | 61 ms | **60 ms** | **-1 ms** |
| Failed | 6/420 | **5/420** | **-1** |
| TBT median | 4 ms | **4 ms** | 0 ms |

Thông tin còn lại: `total_count=420`, `f_delta=1`, `penalty=1`,
`accuracy_drop=0`, `warmup_count=0`, config hash
`sha256:603c84f67bd0fadaa6ea739f2d1aa564761ff94e00dc61da25fe7e1d13853881`.

Warmup1 giảm được một failed request và giữ p95/TBT, nhưng TTFT p50 tăng 10 ms;
phần mất điểm ở đa số request lớn hơn phần điểm lấy lại từ một request thành
công. Kết quả official này bác bỏ dự báo tăng khoảng 0,9 điểm từ proxy L4 và phải
được ưu tiên hơn kết quả synthetic khi chọn submission tiếp theo.

## Chẩn đoán chính

Một lượt cold stock trên Lightning trả đủ 420 HTTP response nhưng có đúng 5
request rơi ra ngoài ngưỡng TTFT và nhận điểm TTFT bằng 0. Mẫu này gần với 5–6
failed request quan sát trên các lần chấm H200. Sau warmup đồng thời từ client,
số này giảm từ 5 xuống 0. Vì grader khai báo `warmup_count=0`, warmup phải nằm
bên trong image và hoàn tất trước khi endpoint báo healthy.

Warmup mới:

- chạy 32 request token-level với độ dài prefill 128–2.175 token;
- phủ output length 8/16/24/32 để prime decode/CUDA graph;
- chạy đồng thời để prime các batch shape thay vì chỉ warm một request tuần tự;
- xóa prefix cache sau khi xong, nên không đưa dữ liệu synthetic vào workload;
- chạy trước `serve_http`, do đó request đầu tiên của grader không thể chen vào;
- có thể tắt bằng `AI_RACE_WARMUP=off`; submission pin `AI_RACE_WARMUP=32`.

Startup stock khoảng 110–112 giây. Image warmup public đạt health sau 114 giây ở
lần xác nhận cuối; lần cold trước đó là 122 giây. Cả hai đều thấp hơn timeout
khởi động 600 giây.

## Ma trận A/B trên Lightning L4

Các số dưới đây là proxy token-exact tự dựng, không phải điểm private grader.
Mỗi hàng giữ nguyên workload 420 request; `r4` dùng tốc độ đến gấp 4 để stress,
`r1` dùng tốc độ đến dự kiến. Control tốt dùng external concurrent warmup; hàng
`cold stock` và `public warmup` bắt đầu cold và không dùng warmup từ client.

| Cấu hình | Rate | Success | TTFT zero | TTFT mean (ms) | TPOT mean (ms) | ERS proxy | Quyết định |
|---|---:|---:|---:|---:|---:|---:|---|
| Cold stock | 4× | 420/420 | 5 | 78,95 | 11,214 | 0,36450 | Vấn đề cần sửa |
| Stock control | 4× | 420/420 | 0 | 65,38 | 11,318 | 0,37059 | Mốc so sánh |
| Batch tokens 12.288 | 4× | 420/420 | 0 | 70,70 | 11,402 | 0,35908 | Loại |
| N-gram GPU, 2 draft token | 4× | 401/420 | 3 | 103,33 | 16,154 | 0,28781 | Loại |
| Stream interval 2 | 4× | 420/420 | 0 | 67,16 | 11,367 | 0,36668 | Loại |
| Block size 32 | 4× | 420/420 | 0 | 67,30 | 11,311 | 0,36626 | Loại |
| Performance mode throughput | 4× | 420/420 | 0 | 70,96 | 11,381 | 0,35872 | Loại |
| 2 API server frontend | 4× | 420/420 | 0 | 65,54 | 11,339 | 0,37017 | Không hơn control |
| FlashInfer cưỡng bức | 4× | 420/420 | 0 | 65,63 | 11,325 | 0,37008 | Không hơn control |
| FP8 KV cache | 4× | 419/420 | 0 | 63,23 | 9,614 | 0,37730 | Cần kiểm tra tải chuẩn |
| Stock control | 1× | 420/420 | 0 | 50,16 | 8,044 | 0,42807 | Mốc sát workload |
| FP8 KV cache | 1× | 420/420 | 0 | 55,35 | 7,703 | 0,42431 | Loại: ERS giảm |
| **Public image + internal warmup** | **4×** | **420/420** | **0** | **63,90** | **11,224** | **0,37371** | **Chọn** |

Concurrent partial prefill không được LFM2.5 hybrid hỗ trợ và fail ngay khi
startup. Các cấu hình batch 4.096, max-seqs 24, performance mode interactivity
và custom Nexus scheduler đã được đo trên các lần H200 trước; tất cả thấp hơn
stock 8.192/32. Vì batch 12.288 tiếp tục giảm điểm nên không tăng lên 16.384.

## Accuracy guardrail

So sánh weight dtype trên toàn bộ 198 câu GPQA-Diamond mirror:

- BF16: 46/198 (23,23%), không request error;
- online FP8 weight hiện tại: 46/198 (23,23%), không request error;
- delta quan sát được: 0 điểm phần trăm; 23 câu chỉ BF16 đúng và 23 câu chỉ FP8
  đúng;
- 94/198 prediction thay đổi, vì vậy vẫn nên giữ BF16 fallback trong quota tối
  đa 5 bài. Mirror không thay thế lần `lm_eval` full chính thức của BTC.

Kết quả này không phát hiện accuracy drop do FP8 weight và ủng hộ giữ FP8 làm
submission ERS chính. FP8 KV không được chọn, nhưng vẫn được kiểm tra để tránh
bỏ qua một ứng viên có thể tăng decode:

- checkpoint không chứa K/V scale đã calibration; vLLM dùng scale mặc định 1,0;
- GPQA-Diamond mirror, cùng 100 câu và greedy decode: stock 17/100, FP8 KV
  24/100, không lỗi request ở cả hai;
- 58/100 prediction thay đổi, nên chỉ có thể kết luận “không phát hiện giảm
  accuracy trên slice này”, không được diễn giải là FP8 KV tăng chất lượng;
- do ERS ở tải chuẩn đã giảm và accuracy bất định hơn, FP8 KV bị loại.

Warmup được chạy ngoài traffic thật và reset prefix cache; nó không đổi weight,
sampling config hay KV dtype của request chấm, nên không tạo thêm accuracy risk
so với baseline đang nộp.

## Các kỹ thuật đã cân nhắc

- Giữ online weight FP8 hiện tại: trường `f_delta=1` trong lượt online là giá trị
  tạm vì BTC chỉ chấm GPQA hậu kỳ; tuy vậy FP8 đã cho lợi ích latency lớn và cần
  giữ một BF16 submission dự phòng cho bước chọn tối đa 5 bài sau vòng online.
- Giữ automatic prefix caching: workload có shared prefix và context tăng dần;
  đây là reuse pattern lý tưởng.
- Giữ chunked prefill, batch tokens 8.192, max sequences 32: đây là điểm thắng
  cả H200 A/B trước và L4 A/B hiện tại.
- Không dùng speculative n-gram/suffix: model chỉ 1,2B nên verify overhead lớn;
  n-gram GPU giảm mạnh cả TTFT và TPOT, đồng thời gây request lỗi.
- Không tăng số frontend: API layer không phải bottleneck.
- Không ép FlashInfer: bằng control trong nhiễu; backend mặc định đã phù hợp với
  kiến trúc hybrid LFM.
- Không dùng FP8 KV: chỉ thắng dưới stress 4×, thua ở tải chuẩn và dùng scale
  chưa calibration.
- Không dùng custom scheduler/deadline scheduler: các lần H200 trước giảm điểm;
  scheduler stock của vLLM 0.25.1 đã có async scheduling và chính sách ưu tiên
  phù hợp.
- Không giảm stream interval: tăng số lần frontend flush và làm ERS giảm.
- Không tăng block size: không có lợi cho trace context 2,1K–4,5K token.

## Artifact và xác minh

- Submission hiện tại: `submission/docker-compose.yml`, SHA-256 file
  `5a8b8bed9d66ae1593b7035c9592e3b0d372d0379ecf49d705a52a5008f32815`.
- BF16 hậu kiểm dự phòng: `submission/docker-compose.bf16-fallback.yml`,
  SHA-256 file
  `d99d7b71868ef07d9ec06eddbec1f003b86f10c2defff33bd57d725e176ffab9`.
- A/B runner: `scripts/lfm25_remote_ab.sh`; matrix:
  `scripts/lfm25_matrix_20260722.sh`.
- Tóm tắt và bootstrap theo conversation:
  `scripts/summarize_lfm25_ab.py`, `scripts/analyze_lfm25_ab.py`.
- GPQA guardrail runner: `scripts/lfm25_remote_gpqa.sh`.
- JSON/metrics/server logs: `results/lightning_20260722/lfm25_ab/`.
- Public pull từ Lightning xác nhận stock image digest
  `sha256:1d03088f685d6c8ddab5078d3b3374ba3e85ed557e5baa50aca9770b6cabdf18`.
- Warmup1 audit cũ vẫn nằm ở `submission/Dockerfile.warmup`,
  `submission/warmup/race_warmup.py` và
  `submission/warmup/api_server_warmup.patch`; không dùng trong compose hiện tại.

Proxy L4 từng dự báo tăng khoảng 0,507 điểm nhờ keep-alive loại cold failure.
Kết quả H200 chính thức của chính compose này là 62,16, giảm 1,66 điểm so với
baseline 63,82: vẫn có 5 failed request, TTFT p50 tăng từ 34 lên 46 ms và TTFT
p95 tăng từ 61 lên 63 ms. Kết quả H200 phải được ưu tiên hơn proxy L4; keep-alive
không được xem là cải thiện leaderboard đã xác nhận. Mốc tốt nhất trong các kết
quả đã biết vẫn là stock 8.192/32 đạt 63,82. Lần A/B tiếp theo cần quay về đúng
artifact/config của mốc 63,82 và chỉ thay một biến mỗi lượt.
