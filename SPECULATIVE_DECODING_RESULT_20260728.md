# Báo cáo speculative decoding LFM2.5 — 2026-07-28

## Kết quả đã đạt được

Đã tạo và public image speculative decoding dùng target LFM2.5-1.2B FP8 và draft LiquidAI/LFM2.5-350M. Image đã tích hợp bản vá hybrid draft liên quan đến vLLM issue #49112 và compose nộp portal tại:

- `submission/docker-compose_speculative_draft_34.9.yml`
- Image: `misokaio/ghfjdk@sha256:573f9399dc288b5d873e6d389ede4c1445293fcc2e00462984f95e382f4fc241`

Image chạy được trên portal, không bị accuracy drop và đã xử lý 414/420 request thành công.

## Kết quả portal

| Chỉ số | Kết quả |
|---|---:|
| ERS / final score | 34.9000 |
| total_count | 420 |
| failed_count | 6 |
| TTFT p50 / p95 | 67 / 126 ms |
| TBT median | 14 ms |
| tokens_per_sec | 0.0448 |
| accuracy_drop | 0 |

Baseline tốt nhất đã có bằng chứng portal là `submission/docker-compose_silu_fp8_65.71.yml` với 65.71 điểm, TTFT p50 32 ms và TBT median 4 ms.

## Phân tích

Theo thể lệ, TPOT/TBT có trần 10 ms. TBT median 14 ms khiến phần điểm TPOT bị clamp về 0; vì vậy accuracy đúng không thể bù cho chi phí runtime. So với baseline, speculative làm TTFT p50 tăng 2.09 lần, TBT tăng 3.5 lần và throughput giảm khoảng 21%.

Các thử nghiệm Medusa trước đó cũng cho kết quả cùng chiều: K1/K2/K3 lần lượt có TPOT 12.125/13.276/11.774 ms, đều kém control. Điều này cho thấy vấn đề là chi phí draft/verify và kernel overhead của kiến trúc hybrid nhỏ, không phải riêng một lỗi correctness.

Speculative decoding có thể hiệu quả với target lớn, memory-bound, QPS thấp và drafter rất nhỏ hoặc được huấn luyện chuyên biệt như EAGLE/P-EAGLE. Tuy nhiên, với target LFM2.5 chỉ 1.2B đã có baseline decode khoảng 4 ms, draft 350M và hai token speculative không tạo đủ phần tiết kiệm để bù overhead. Chưa tìm thấy bằng chứng công khai đáng tin cậy cho speedup của draft/EAGLE chuyên biệt trên LFM2/LFM2.5.

Tham khảo upstream:

- [vLLM PR #33318](https://github.com/vllm-project/vllm/pull/33318): hybrid speculative trên Gemma 1B + draft 270M chậm hơn 2.2 lần dù acceptance khoảng 50.5%.
- [vLLM PR #24322](https://github.com/vllm-project/vllm/pull/24322): speedup rõ hơn trên target Qwen3-32B với drafter 1.7B, tức target lớn hơn rất nhiều.
- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/v0.20.0/features/speculative_decoding/): hiệu quả phụ thuộc mạnh vào model, hardware và traffic; phù hợp nhất với workload memory-bound/QPS thấp.

## Quyết định

Speculative draft 350M được đánh giá là đã đạt mục tiêu tích hợp/chạy đúng, nhưng không đạt mục tiêu tối ưu điểm. Không nên tiếp tục dùng image này để nộp. Baseline `docker-compose_silu_fp8_65.71.yml` được giữ lại làm phương án khôi phục.

## Dọn dẹp local

Đã xác định và xóa khỏi Docker local các image chỉ phục vụ thử nghiệm speculative:

- `misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8-draft350m2-hybridfix6` — image đã nộp, 19.5 GB.
- `misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8-draft350m2` — image thử nghiệm cũ, 19.5 GB.
- Image dangling `40737ac3ed5a` — intermediate layer của speculative build, 19.5 GB logical size.

Sau dọn dẹp, Docker local còn image baseline `misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8` (18.7 GB) và image hệ thống `alpine:3.20`. Các image baseline/fused, đặc biệt image được compose 65.71 tham chiếu, không bị xóa. Không phát hiện model directory hoặc Docker volume riêng của draft trong workspace; draft model đã được đóng gói bên trong các image speculative nói trên.
