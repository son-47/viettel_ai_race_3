25/07/2026, 10:03 

AI Race 2026 - Cuộc đua AI cho kỹ sư Việt Nam 

NH 





Quay lại đề bài 

VÒNG 1 Đang mở 

Lịch sử nộp bài 

Nộp bài 

# **Vòng 1 - Sơ loại** 

02/07/2026 - 30/07/2026 

## **Đề bài & Quy định** 

### 1. Nhiệm vụ & Hạ tầng 

#### Nhiệm vụ (Cập nhật ngày 18/07/2026): 

Triển khai và tối ưu một LLM inference server cho mô hình LFM2.5-1.2B-Instruct xử lý một workload trace multi-turn mô phỏng traffic production. Trong vòng online, mục tiêu là tối đa hoá ERS (điểm độ trễ). Accuracy Gate chỉ chạy sau khi kết thúc vòng online, trên tối đa 5 submissions đội tự chọn. 

Mô tả các giá trị trong file trace mô tả workload: 

- `num_conversations` : Số hội thoại độc lập chạy đồng thời. 

- `user_turns_per_conversation` : Số lượt hỏi của user trên mỗi hội thoại. 

- `total_request` : tổng số request. 

- `shared_system_prefix_tokens` : System prefix, giống nhau trên các hội thoại. 

- `per_conversation_prefix_tokens` : ngữ cảnh riêng cho từng hội thoại (bổ sung input cho turn 1 của từng 

- hội thoại). 

`new_user_tokens_per_turn` : số lượng token prompt của user tại mỗi turn (turn 1 có thêm 2 khối prefix). 

- `output_tokens_per_turn_pinned` : số lượng token output tại mỗi turn. 

- `arrival` : nhịp đến của các request. 

Hạ tầng & Môi trường đánh giá: Toàn bộ quá trình chạy benchmark được thực hiện tự động trên hệ thống của BTC. Thí sinh sẽ serve endpoint trên 1 instance MiG và BTC sẽ thực hiện benchmark trực tiếp vào endpoint đó: 

- Hạ tầng Hardware: 1 instance MiG H200 (18GB VRAM, 3 Core CPU, 8GB RAM) được cấp phát tự động cho mỗi lượt chấm. 

- Hệ điều hành & Driver (host): Ubuntu 24.04 LTS, NVIDIA driver 590.x (hỗ trợ CUDA 13.x). 

- Model: LiquidAI/LFM2.5-1.2B-Instruct 

Weights: <u>https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct</u> 

### 2. Cách tính điểm 

https://competition.viettel.vn/contests/llm-2026/phases/019e649f-4e27-74db-82da-920f57b13786 

1/7 

25/07/2026, 10:03 

AI Race 2026 - Cuộc đua AI cho kỹ sư Việt Nam 

Effective Request Score được đánh giá dựa theo tốc độ trên 2 metrics TTFT và TPOT. Công thức cụ thể như sau: 

ERS = _N_ <u>1</u> ∑ _Ni_ =1 _S_ request, _i_ ∈ [0, 1] với _N_ là tổng số request. Trong đó: 



|Tham sốcấu hình:<br>_s_<br> =<br>tpot<br>_x_<br>=<br>(<br>tpot<sup>)</sup><sup>_γ_</sup><br>clamp<br> , 0, 1<br>[<br>(<br>_C_<br> −_F_<br>tpot<br>tpot<br>_C_<br> −TPOT<br>tpot<br>mean<br>)]<br>_γ_||
|---|---|
|Ký hiệu<br>Ý nghĩa|Giá trị|
|Floor của TTFT<br>_F_<br>ttft|10 ms|
|Ceiling của TTFT<br>_C_<br>ttft|400 ms|
|Floor của TPOT<br>_F_<br>tpot|1 ms|
|Ceiling của TPOT<br>_C_<br>tpot|10 ms|
|Hệsốlũy thừa<br>_γ_|2|
|Trọng sốcủa TTFT<br>_w_|0.5|



#### Accuracy Gate - sau vòng online: 

Không chấm GPQA trên từng lượt nộp online. Sau khi vòng online kết thúc, đội chọn thủ công tối đa 5 submissions tốt nhất. BTC lần lượt: (1) hậu kiểm tính hợp lệ phương án; (2) dựng endpoint và chạy GPQA full. Độ sụt giảm chất lượng ( (Δ ) so với baseline BF16 (mặc định 0.4): 

− Δ = Accuracy _baseline_ Accuracy _submission_ (Trong đó, Accuracy _baseline_ là accuracy tham chiếu của mô hình gốc chạy bằng trọng số BF16; Accuracy _submission_ là accuracy bài nộp của đội.) 



https://competition.viettel.vn/contests/llm-2026/phases/019e649f-4e27-74db-82da-920f57b13786 

2/7 

25/07/2026, 10:03 

AI Race 2026 - Cuộc đua AI cho kỹ sư Việt Nam 

Điểm cuối mỗi submission hợp lệ: _Score_ = 100 × _ERC_ × _f_ (Δ) (ERS lấy từ lần chấm online của đúng bài 

đó). Điểm đội = Score tốt nhất trong các bài còn hợp lệ. 

Trong đó: 

ERS: Điểm trung bình hiệu năng trên trace (đã mô tả ở phần ERS), chấm trong vòng online. 

- _f_ (Δ) : Hệ số phạt accuracy, chỉ có sau bước GPQA post-online. 

### 3. Không gian Tối ưu 

Thí sinh chỉ được phép sử dụng serving framework vLLM cho bài thi này. Các hướng tiếp cận bao gồm: 

- Quantization: Các kỹ thuật Online Quantization. 

- KV Cache & Memory: Tối đa hóa lượng request xử lý đồng thời bằng Paged Attention; KV cache quantization (FP8, INT8); Prefix caching và Semantic caching; Offloading xuống CPU/NVMe. 

- Serving & Scheduling: Ứng dụng Dynamic/Continuous batching; Speculative decoding; Memory-aware scheduling. 

- System & Runtime: Viết custom CUDA/Triton kernels; Tích hợp Fused attention kernels (FlashAttention, FlashInfer); Tối ưu hóa memory layout và CUDA Graphs. 

### 4. Nộp bài & Tài nguyên 

Quy trình thực hiện: 

1. Develop & Package: Thí sinh phát triển code giải pháp, tối ưu hệ thống và đóng gói toàn bộ thành một Docker Image. 

2. Push Image: Đẩy (Push) Docker Image hoàn chỉnh lên Docker Hub cá nhân hoặc tổ chức dưới dạng công khai (Public). 

3. Submit: Thí sinh truy cập hệ thống Portal của BTC, gửi file cấu hình `docker-compose.yml` (trong đó có khai báo chính xác đường dẫn Image trên Docker Hub và lệnh thực thi). 

4. Automated Evaluation: Hệ thống tự động pull Image, dựng container trên MiG H200, healthcheck và chạy benchmark ERS (không chạy GPQA trên mỗi lượt nộp). 

5. Leaderboard: Cập nhật theo ERS. 

6. Sau vòng online: Đội chọn tối đa 5 submissions → BTC hậu kiểm hợp lệ → chấm GPQA full ( `lm_eval` / `bench-gpqa-diamond.sh` ) → chốt Score. 

- - Docker image baseline: https://hub.docker.com/layers/vllm/vllm <u>openai/v0.22.1/images/sha256 55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5</u> 

File `docker-compose.yml` mẫu 

```
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
```

- `python3 #Don't change this to vllm-server` 

- `-m  #Don't change this to vllm-server` 

https://competition.viettel.vn/contests/llm-2026/phases/019e649f-4e27-74db-82da-920f57b13786 

3/7 

