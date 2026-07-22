# Tối ưu hóa LLM Inference cho mô hình MoE + Hybrid Attention (kiểu Qwen3.5/Qwen3-Next) trên NVIDIA H200: Báo cáo kỹ thuật chuyên sâu

## TL;DR
- **FP8 (W8A8 E4M3) + FlashAttention-3 + PagedAttention/continuous batching + MTP/EAGLE-3 speculative decoding** là "bộ tứ gần như miễn phí" (near-lossless) cho mô hình MoE + hybrid-attention trên H200: FA-3 đạt ~75% (FP16) đến 85% (BF16) utilization và ~1.2-1.3 PFLOPS ở FP8 với sai số thấp hơn 2.6× so với FP8 baseline; PagedAttention cho throughput 2-4× ở cùng latency; speculative decoding (MTP/EAGLE-3) cho 1.8-6.5× ở batch nhỏ — tất cả không (hoặc gần như không) đổi phân phối đầu ra.
- **Các đòn bẩy cần đánh đổi (tunable)**: quantization bit-width (W4A16 AWQ/GPTQ cho decode memory-bound, FP8 cho cân bằng), KV-cache quantization (4-bit gần lossless, 2-bit suy giảm rõ), pruning 2:4 (Hopper Sparse Tensor Core, mất điểm vừa phải), early exit/lossy speculative — chọn theo ngưỡng batch size và acceptance rate.
- **Các điểm nhạy cảm nhất (giữ high precision)**: MoE router/gating, các shared expert, và state hồi quy của Gated DeltaNet (linear attention) — quantize/prune ở đây gây "expert-shift cascade" và mất khả năng in-context; chỉ ~25% layer full-attention cần KV cache O(N) nên tối ưu KV nên tập trung vào đó.

## Key Findings (tóm tắt định lượng, có attribution)
1. **FlashAttention-3** (Shah et al., arXiv:2407.08608): FA-2 chỉ đạt 35% utilization trên H100; FA-3 nhanh hơn FA-2 1.5-2.0×, FP16 ~740 TFLOPS (75% util), BF16 lên tới ~840 TFLOPS (85% util, bản paper/OpenReview), FP8 ~1.2-1.3 PFLOPS, với sai số FP8 thấp hơn 2.6× so với baseline per-tensor FP8. Lossless với FP16/BF16 (softmax/accumulation giữ FP32).
2. **PagedAttention/vLLM** (Kwon et al., SOSP 2023, arXiv:2309.06180): hệ thống cũ lãng phí 60-80% KV cache; vLLM <4% lãng phí, throughput 2-4× so với FasterTransformer/Orca ở cùng latency. Lossless.
3. **Speculative decoding** (Leviathan et al., arXiv:2211.17192; Chen et al., arXiv:2302.01318): chứng minh lossless; kỳ vọng token/round = (1−α^(γ+1))/(1−α). **EAGLE-3** (Li et al., arXiv:2503.01840): 3.0-6.5× so với autoregressive, hơn EAGLE-2 20-40%, và trong SGLang đạt 1.38× throughput ở batch size 64. **DeepSeek-V3 MTP** (arXiv:2505.09343): acceptance 80-90% cho token thứ hai, +1.8× TPS.
4. **AWQ** (Lin et al., MLSys 2024, arXiv:2306.00978): bảo vệ ~1% salient weight bằng per-channel scaling từ activation; TinyChat >3× so với FP16 HF. **GPTQ** (Frantar et al., arXiv:2210.17323): nén 3-4 bit với Hessian H=2XX^T, Cholesky reformulation.
5. **KV quantization** (KIVI arXiv:2402.02750, KVQuant): 4-bit gần lossless, 2-bit suy giảm rõ (KIVI 2-bit drop tới ~8-22 điểm trên một số task LongBench).
6. **Pruning** (SparseGPT, Wanda arXiv:2306.11695): 50% unstructured gần như giữ nguyên; 2:4 structured mất điểm rõ hơn (Wanda LLaMA-2-7B 2:4 PPL 5.12→~11 trên Wikitext).
7. **DeepGEMM** (DeepSeek): FP8 GEMM cho Hopper, dense 1.4-2.7× so với CUTLASS, MoE grouped GEMM 1.1-1.3×, >1350 TFLOPS trên H800.
8. **Qwen3-Next-80B-A3B**: 512 routed expert + 1 shared, 10 expert active/token, 80B total/3B active (~3.75%), hybrid Gated DeltaNet:Gated Attention = 3:1, một MTP module (NextN), context native 262.144 token. Theo Qwen blog: prefill throughput ~7× Qwen3-32B ở 4K context và >10× ở >32K; decode ~4× ở 4K và vẫn >10× ở >32K.

---

## Phần 1 — Bối cảnh kiến trúc: tại sao MoE + hybrid attention thay đổi bài toán tối ưu

Mô hình mục tiêu (kiểu Qwen3-Next/Qwen3.5) có ba đặc điểm chi phối toàn bộ chiến lược tối ưu:

**(a) Hybrid attention 3:1.** 75% layer dùng Gated DeltaNet (linear attention, state hồi quy kích thước cố định, O(1) bộ nhớ khi decode, O(N) compute khi prefill), 25% layer dùng Gated full attention (cần KV cache O(N)). Theo Qwen, mix 3:1 này "consistently outperforms any monolithic architecture" và được chọn qua "systematic experiments". Hệ quả trực tiếp: **KV cache chỉ tồn tại ở ~1/4 số layer**. Báo cáo cộng đồng (llama.cpp trên Apple Silicon) cho thấy 1M token context chỉ tốn ~25 GB KV, ~4× ít hơn ước tính transformer thuần. Điều này làm dịu nút thắt memory-bandwidth ở decode và thay đổi thứ tự ưu tiên: tối ưu KV cache (paging, quantization, offload) chỉ áp dụng cho phần full-attention; phần linear-attention cần tối ưu kiểu scan/recurrent kernel (giống Mamba).

**(b) MoE siêu thưa.** 512 routed expert + 1 shared expert, top-10 routed/token, 3B/80B active (~3.75%). Điều này tách rời capacity (tổng tham số → chất lượng) khỏi compute (active params → tốc độ). Nhưng nó tạo ra nút thắt mới: (i) **memory** — phải giữ toàn bộ 80B tham số trên HBM (H200 141GB HBM3e đủ cho FP8 ~80GB nhưng chật ở BF16); (ii) **all-to-all communication** cho expert parallelism; (iii) **load imbalance** giữa các expert.

**(c) Multi-Token Prediction (MTP).** Qwen3-Next có một MTP module (NextN block) dùng làm draft head speculative-decoding bản địa. DeepSeek-V3 dùng cơ chế tương tự (một MTP module, dự đoán token thứ 2) với 80-90% acceptance cho token thứ hai → +1.8× TPS.

Trên **H200** (Hopper SM90a, 141GB HBM3e, ~4.8TB/s, FP8 Transformer Engine, TMA, WGMMA, thread block clusters + distributed shared memory, 227KB SMEM/SM, NVLink ~900GB/s), tất cả các tối ưu phải khai thác FP8 tensor core và async data movement.

---

## Phần 2 — Quantization (Topic 1)

### 2.1 Granularity: per-tensor vs per-channel vs per-group/per-block
- **Per-tensor**: một scale cho cả tensor — rẻ nhất, hiệu quả phần cứng cao nhất (một scale duy nhất nạp vào WGMMA), nhưng nhạy với outlier.
- **Per-channel**: một scale mỗi channel (mỗi output/input column) — cân bằng tốt cho weight.
- **Per-group/per-block** (ví dụ group=128): một scale mỗi nhóm 128 phần tử — chính xác nhất, là chuẩn của AWQ/GPTQ INT4. DeepGEMM trên H200 dùng **fine-grained scaling**: một scale mỗi tile 128 cột của K, mỗi tile được scale độc lập trước khi WGMMA accumulate.

Trên H200, FP8 tensor core hỗ trợ block/tile scaling hiệu quả; DeepGEMM bake hai sự thật kiến trúc: SM90 cần scale factor FP32 ở layout TMA-aligned, transposed cho LHS.

### 2.2 Weight vs Activation quantization
- **Weight-only (W4A16)**: chỉ nén weight, activation giữ FP16/BF16. Lý tưởng cho **decode memory-bound** (giảm lượng weight phải nạp từ HBM mỗi token) — đây là chế độ then chốt cho MoE vì mỗi token chỉ kích hoạt 10 expert nhưng vẫn phải nạp weight của chúng.
- **W8A8 / FP8 (E4M3 cho forward, E5M2 cho range lớn hơn)**: nén cả hai, tăng arithmetic intensity, tận dụng FP8 tensor core. Đây là **sweet-spot đầu tiên** trên H200.
- **INT4/INT8**: INT8 an toàn; INT4 (W4A4) khó vì **activation outliers** — vài channel có magnitude cực lớn (massive activations).

### 2.3 AWQ — Activation-aware Weight Quantization
Quan sát: "protecting only 1% of salient weights can greatly reduce quantization error" (Lin et al.). Salient channel được nhận diện qua **activation magnitude** (không phải weight magnitude), vì weight nhân với activation lớn đóng góp nhiều hơn vào output. AWQ tìm per-channel scale s để bảo vệ salient channel: W' = W·diag(s)^{-1}, X' = X·diag(s), giữ tương đương toán học X'W'=XW. Không cần backprop/reconstruction → bảo toàn generalization. TinyChat đạt >3× so với FP16 HF.

### 2.4 GPTQ — Hessian-based error compensation
Kế thừa OBD→OBS→OBQ. Hessian layer-wise H = 2XX^T. Quantize tuần tự, mỗi bước chọn weight tối thiểu hóa (Q(w_q)−w_q)²/[H^{-1}]_qq, rồi cập nhật weight còn lại: w_k ← w_k − ([H^{-1}]_kj/[H^{-1}]_jj)·Δw_j. GPTQ dùng **fixed-order** (chia sẻ Hessian giữa các row) + **Cholesky reformulation** H^{-1}=LL^T (damping λI) để ổn định số học và tính trước. Nén 3-4 bit với suy giảm tối thiểu.

### 2.5 FlexRound (Lee et al., ICML 2023, arXiv:2306.00317)
Đổi rounding từ element-wise addition sang **element-wise division**: học một grid size chung s1 và một scale riêng cho mỗi weight. Nhờ "reciprocal rule of derivatives", FlexRound khai thác được magnitude của pre-trained weight khi cập nhật scale — weight magnitude lớn cần khám phá range discrete rộng hơn.

### 2.6 Sensitivity Analysis & mixed-precision bit allocation
Phân bổ bit theo độ nhạy per-layer/per-module (Hessian-based). **Với MoE, router/gating cực kỳ nhạy**: lỗi nhỏ ở router làm token đi sai expert → "expert-shift cascade failure" (sai một bước routing kéo theo sai toàn bộ chuỗi expert). Khuyến nghị: giữ router, shared expert, và gate của Gated DeltaNet ở high precision (BF16/FP16); chỉ quantize mạnh các routed expert weight.

### 2.7 Các phương pháp liên quan
- **SmoothQuant** (Xiao et al.): migrate độ khó từ activation sang weight bằng per-channel scaling; nén KV xuống 8-bit ổn, nhưng drop rõ ở 4-bit.
- **QuaRot/SpinQuant** (Ashkboos et al.; Liu et al., ICLR 2025): dùng **Hadamard rotation** (QuaRot, fixed) hoặc learned rotation (SpinQuant) để "smear" outlier qua nhiều channel, đưa kurtosis từ >200 xuống ~3 (gần Gaussian), enable W4A4. SpinQuant W4A4KV4 chỉ cách full-precision 2.9 điểm trên LLaMA-2-7B; SpinQuant hơn QuaRot tới 16 điểm zero-shot ở một số cấu hình; QuaRot báo cáo mất ~3.5 điểm.
- **KIVI/KVQuant**: quantize KV cache. KIVI (arXiv:2402.02750) — bất đối xứng: **key per-channel, value per-token** (vì key có outlier theo channel), giữ residual window full precision. 4-bit gần lossless; 2-bit suy giảm rõ ở task khó (GSM8K, math). KVQuant đạt ~1.5× nhỏ hơn bit-width của KIVI ở cùng độ chính xác; 2-3 bit cho 1.2-1.7× latency saving.

---

## Phần 3 — Pruning & Sparsity (Topic 2)

- **Structured vs unstructured**: unstructured (zero ra bất kỳ weight nào) chất lượng cao nhưng không tăng tốc trên GPU vì vẫn nạp đủ weight; structured **2:4 / N:M** (đúng 2 zero trong 4 weight liên tiếp) được **Hopper Sparse Tensor Core** (H200) tăng tốc trực tiếp.
- **Magnitude pruning**: thất bại nặng trên LLM (Frantar & Alistarh).
- **SparseGPT** (Frantar & Alistarh): one-shot, layer-wise sparse regression dùng second-order info; 50-60% unstructured trên OPT-175B/BLOOM-176B trong <4.5h, PPL tăng tối thiểu.
- **Wanda** (Sun et al., arXiv:2306.11695): điểm = |weight| × ‖activation‖_2 per-output; không cần Hessian/update. 50% unstructured ~ngang SparseGPT; 2:4 mất rõ hơn (LLaMA-2-7B 2:4: 5.12→~11 PPL).
- **MoE-specific**: **expert pruning/merging** (bỏ/gộp expert ít dùng), **attention head pruning**, **layer/depth dropping**. Đây là vùng nhạy: pruning expert thay đổi hành vi routing → nguy cơ cao; cần monitor histogram expert utilization để tránh expert collapse.

---

## Phần 4 — Neural Architecture Search (Topic 3)
Hardware-aware NAS, **once-for-all networks** (train một supernet, trích nhiều sub-net không cần retrain), elastic models. Với MoE: search expert count/top-k tối ưu; với hybrid: search tỉ lệ linear:full attention (Qwen tìm 3:1 qua "systematic experiments"). NAS chủ yếu là chi phí một lần ở design-time, không phải đòn bẩy serving runtime, nhưng định hình trần hiệu quả.

---

## Phần 5 — CUDA Programming trên Hopper/H200 (Topic 4)

Các đặc trưng SM90/SM90a và cách map vào kernel LLM:
- **TMA (Tensor Memory Accelerator)**: async bulk copy global↔shared memory, offload khỏi SM, hide memory latency. DeepGEMM/FA-3 dùng TMA nạp tile trong khi tensor core tính. Ràng buộc: 16-byte alignment global, 128-byte shared (vấn đề cho MoE grouped GEMM với row động → TMA descriptor pool xử lý padding; báo cáo TMA-Adaptive FP8 Grouped GEMM cho 1.7-20.4% speedup, giảm tới 23.8% memory).
- **WGMMA (warp-group MMA, wgmma.mma_async)**: MMA bất đồng bộ trên cả warp-group (128 thread); FP8 16×8×32 mỗi clock per tensor core.
- **Thread block clusters + distributed shared memory (DSM)**: nhiều block cùng cluster chia sẻ SMEM qua SM-to-SM network — hữu ích cho reduction/attention tile lớn.
- **Async pipelines + warp-specialization (producer-consumer)**: FA-3 chia warp thành nhóm producer (TMA load) và consumer (WGMMA + softmax), dùng **ping-pong scheduling** để overlap GEMM với softmax, ẩn stage throughput thấp dưới GEMM throughput cao. `setmaxnreg` điều chỉnh register allocation per warp-group.
- **227KB SMEM/SM**: cho phép tile lớn hơn, ít vòng lặp HBM.
- **Kernel fusion / persistent kernels / megakernels**: gộp nhiều op (dequant+GEMM, layernorm+...) giảm launch overhead và HBM round-trip.
- **MoE grouped GEMM (DeepGEMM)**: nhóm theo M-axis (các expert cùng shape), layout **contiguous** (prefill) và **masked** (decode với CUDA graph khi CPU chưa biết số token/expert). Two-level accumulation giải quyết imprecise FP8 accumulation. Dense 1.4-2.7× CUTLASS, MoE 1.1-1.3×, >1350 TFLOPS trên H800.

---

## Phần 6 — KV Caching (Topic 5)
KV cache lưu key/value để tránh tính lại. Kích thước per-token = **2 × n_layers × n_kv_heads × head_dim × dtype_bytes**. Decode **memory-bandwidth-bound** vì mỗi token phải nạp toàn bộ weight + KV từ HBM cho rất ít FLOP. **GQA/MQA** giảm n_kv_heads → giảm KV.

**Với hybrid attention**: chỉ ~25% layer (full-attention) cần KV O(N); 75% layer Gated DeltaNet chỉ giữ state hồi quy kích thước cố định O(1). vLLM dùng **hybrid KV cache manager**: tự tune logical block size của full-attention layer để state của linear-attn và full-attn chiếm cùng lượng physical memory → paging thống nhất.

---

## Phần 7 — FlashAttention (Topic 6)
- **Online softmax**: tính softmax incremental theo block mà không materialize ma trận attention N×N, dùng running max m_i và running sum l_i, rescale khi gặp block mới → giảm HBM traffic từ O(N²) xuống O(N). **Exact/lossless** (không xấp xỉ).
- **v1** (giảm HBM read/write), **v2** (parallelism theo sequence, giảm non-matmul FLOP, ~72% util A100), **v3** (Hopper: TMA+WGMMA async, warp-specialization, block-quantization FP8 + incoherent processing). FA-3: 1.5-2.0× FA-2, FP16 740 TFLOPS/75%, BF16 lên tới 840/85%, FP8 1.2-1.3 PFLOPS, sai số FP8 thấp hơn 2.6×.

Áp dụng: chạy trên 25% layer full-attention của mô hình; linear-attention layer dùng kernel scan riêng.

---

## Phần 8 — PagedAttention (Topic 7)
Chia KV cache thành **block cố định** (ví dụ 16 token/block), map qua **block table** (như page table OS) tới physical block không liên tục → loại bỏ external fragmentation, giới hạn internal fragmentation ≤ 1 block. <4% lãng phí (so với 60-80% hệ cũ). **Copy-on-write prefix sharing**: nhiều request chia sẻ system prompt trỏ cùng physical block. Cho phép batch lớn hơn → concurrency cao hơn → throughput 2-4×. Là nền tảng cho **continuous batching**. Lossless.

---

## Phần 9 — Speculative Decoding (Topic 8)
**Cơ chế draft-and-verify**: draft model q đề xuất γ token; target p verify song song trong 1 forward pass. **Rejection sampling**: chấp nhận token x~q với xác suất min(1, p(x)/q(x)); nếu reject, resample từ phân phối residual norm(max(0, p−q)). **Chứng minh lossless**: phân phối đầu ra đồng nhất với sampling từ p (Leviathan; Chen). **Acceptance rate α**; **kỳ vọng token/round = (1−α^(γ+1))/(1−α)**; cải thiện walltime ~ tỉ lệ với số token/round chia chi phí tương đối c của draft.

**Medusa** (Cai et al., arXiv:2401.10774): thêm nhiều decoding head dự đoán token tương lai song song + tree attention; Medusa-1 ~2.2× lossless, Medusa-2 2.3-3.6×.

**Nhạy cảm batch size**: ở batch lớn, GPU đã bận, chi phí verify tăng → break-even acceptance rate tăng → speculative có thể net-negative. Với MoE, expert dispatch đắt khiến speculative ở concurrency cao cần acceptance rate cao hơn (báo cáo GLM-4.7-Flash: per-request latency 1.30× ở concurrency cao nhưng throughput hệ thống 1.70×).

---

## Phần 10 — EAGLE / EAGLE-2 / EAGLE-3 (Topic 9)
**EAGLE**: autoregression ở **feature level** (hidden state trước LM head) thay vì token level, acceptance cao hơn. **EAGLE-2**: dynamic draft tree (điều chỉnh cây draft theo độ tin cậy). **EAGLE-3** (arXiv:2503.01840): bỏ feature-prediction constraint, **training-time test** + **multi-layer feature fusion** → draft head robust với lỗi của chính nó; phát hiện scaling law (nhiều data → speedup tăng). **3.0-6.5×** so với autoregressive, hơn EAGLE-2 20-40%, và trong SGLang đạt 1.38× throughput ở batch size 64. Acceptance instruction-following tăng từ 0.72-0.78 (EAGLE-2) lên 0.80-0.88 (EAGLE-3) trên họ Llama/Qwen; AngelSlim báo cáo 1.8-2.0× end-to-end ổn định trên Qwen3/Qwen3-VL với accepted length 1.74-2.2. **Lossless** (strict acceptance, không đổi weight target).

**MTP compatibility**: MTP của Qwen3-Next/DeepSeek-V3 là dạng self-draft tương tự EAGLE; có thể dùng trực tiếp làm draft (vLLM: `--speculative-config '{"method":"qwen3_next_mtp",...}'`; SGLang: `--speculative-algo NEXTN`). Nhạy batch size như mọi speculative.

---

## Phần 11-13 — Batching & Parallel Decoding

**Batch Inference (10)**: batching khuếch tán chi phí nạp weight qua nhiều request, nâng arithmetic intensity → throughput tăng. Roofline: decode ở batch nhỏ memory-bound; batch lớn dịch về compute-bound. Đánh đổi: latency per-request tăng.

**Dynamic Batching (11)**: **continuous/in-flight batching** với iteration-level scheduling (**Orca**) — thêm request mới vào batch ngay sau mỗi forward pass thay vì chờ cả batch xong; kết hợp PagedAttention nâng GPU utilization. **Chunked prefill**: chia prefill dài thành chunk để xen kẽ với decode, giảm TTFT spike.

**Early Exit Decoding (12)**: thoát ở layer sớm khi đủ tin cậy. **LayerSkip** (Elhoushi et al., arXiv:2404.16710): layer dropout + early exit loss khi train → self-speculative decoding (draft = layer sớm, verify = layer còn lại), speedup 1.34-2.16× (2.16× summarization CNN/DM, 1.82× coding, 2.0× TOPv2), acceptance E=12: 97.2%, E=18: 98.9%. **CALM** (confidence-based). Thách thức: KV cache cho layer bị skip; với hybrid + MoE phức tạp hơn vì state linear-attn và routing.

**Parallel Decoding (13)**: **Lookahead decoding** (Fu et al., arXiv:2402.02057) — Jacobi iteration + n-gram cache, break sequential dependency không cần draft model; vanilla Jacobi hầu như không giảm decoding step nên lookahead giữ trajectory để sinh n-gram song song; **blockwise parallel**; Medusa-style heads. CLLM fine-tune consistency loss cho Jacobi đạt ≥2× giữ PPL thấp.

---

## Phần 14 — Streaming Generation (Topic 14)
Token streaming (SSE), phân biệt **TTFT** (time-to-first-token, chi phối bởi prefill) vs **ITL** (inter-token latency, chi phối bởi decode). **StreamingLLM** (Xiao et al., arXiv:2309.17453): giữ **attention sink** (4 token đầu) + sliding window → perplexity ổn định qua 4 triệu token không fine-tune; window-only collapse khi vượt cache size. Với hybrid model, linear-attn layer vốn O(1) state nên streaming tự nhiên; attention sink áp dụng cho full-attention/window layer.

---

## Phần 15-16 — Mixed Precision & Quantized Kernels

**Mixed Precision (15)**: FP16/BF16/FP8 mixed; **Transformer Engine** tự chọn precision. Ops giữ high precision: **softmax, layernorm, router/gating, accumulation (FP32)**. Automatic loss scaling cho ổn định số học. FP8 E4M3 (forward) / E5M2 (range).

**Quantized Kernels (16)**: INT4/INT8/FP8 GEMM (CUTLASS, **Marlin** cho W4A16, **Machete** Hopper-native, **DeepGEMM** FP8). **Dequant-fused kernel** (dequant trong shared memory ngay trước MMA). **Mixed-input GEMM** (FP16 activation × INT4 weight) cho decode memory-bound. **W4A16 vs W8A8 trên H200**: W4A16 thắng ở decode batch nhỏ (memory-bound, ít weight traffic); W8A8/FP8 thắng ở prefill/batch lớn (compute-bound, dùng FP8 tensor core).

---

## Phần 17 — Graph Optimization (Topic 17)
Operator/graph fusion, constant folding, layer/tensor fusion, kernel auto-tuning. **ONNX Runtime**, **TensorRT/TensorRT-LLM** build engine tối ưu cho shape cố định. **CUDA graphs**: capture chuỗi kernel launch thành graph, replay giảm CPU launch overhead (quan trọng cho decode nhiều kernel nhỏ; vLLM bật CUDA graph mặc định cho Qwen3-Next). Đánh đổi: static graph nhanh nhưng kém linh hoạt với dynamic shape (cần multiple captured graphs hoặc padding). Lossless.

---

## Phần 18 — Memory Offloading (Topic 18)
CPU/NVMe offload weight/KV/optimizer: **ZeRO-Offload, ZeRO-Inference, FlexGen** (throughput-oriented offload cho GPU đơn). KV cache offload to host (**LMCache**-style). **Expert offloading cho MoE**: giữ hot expert trên GPU, cold expert trên CPU, prefetch theo prediction routing (Expert Buffering ghép cặp expert high/low load). Đánh đổi: **PCIe bandwidth bottleneck** (~64GB/s PCIe5 vs 4.8TB/s HBM) → chỉ dùng khi không đủ HBM. Với 80B model trên H200 141GB ở FP8 (~80GB) thường không cần offload.

---

## Phần 19-24 — Parallelism & Frameworks phân tán

**Tensor Parallelism (19)**: Megatron-style chia column/row của MLP & attention; **2 all-reduce mỗi layer ở forward, 2 ở backward** (4 tổng/layer khi train; inference 2/layer). TP≤8 trong node trên NVLink (~900GB/s H200). Yêu cầu hidden/heads chia hết.

**Pipeline Parallelism (20)**: chia stage theo layer, micro-batching, **1F1B & interleaved 1F1B** giảm pipeline bubble; mất cân bằng memory giữa stage. P2P communication rẻ hơn TP.

**Sequence Parallelism (21)**: Megatron SP shard activation theo sequence dim, thay all-reduce bằng **reduce-scatter + all-gather** (không thêm chi phí comm, giảm activation memory xuống A/P). **Context parallelism / Ring Attention** cho long context: shard sequence, trao đổi KV qua ring P2P. Quan trọng cho context 256K-1M của Qwen3-Next.

**ZeRO (22)**: ZeRO-1 (shard optimizer state), ZeRO-2 (+gradient), ZeRO-3 (+parameter). Chủ yếu cho training memory; inference analogue = weight sharding + **ZeRO-Inference** offload.

**DeepSpeed (23)**: DeepSpeed-Inference, ZeRO-Inference, DeepSpeed-MII, kernel injection, tensor-slicing; DeepSpeed-MoE kết hợp expert parallelism + expert-slicing.

**Megatron-LM (24)**: triển khai TP+PP+SP; Megatron-Core. Với MoE: EP8×TP1 thường thắng EP4×TP2 (Mixtral). EP communication có thể vượt NVLink → 1F1B A2A overlap.

---

## Phần 25-28 — Họ mô hình & MoE

**LLM (25)**: dense transformer baseline, attention O(N²), tất cả params active, scaling laws.

**LVM (26)**: vision transformer, **patchification**; **vision encoder outliers** ảnh hưởng quantization (cần per-channel/rotation mạnh hơn); số image token lớn.

**LMM (27)**: vision encoder + connector/projector + LLM; cross-attention; **prefill cost của image token** cao (hàng nghìn token/ảnh) → TTFT lớn; KV implication: image token chiếm KV ở full-attention layer.

**MoE (28)**: sparse routing top-k gating; shared (luôn active, kiến thức chung) vs routed expert; **load balancing** qua aux loss hoặc bias-based (DeepSeek auxiliary-loss-free). Tách capacity khỏi compute. **Expert parallelism + all-to-all (DeepEP)**: kernel dispatch/combine NVLink+RDMA, FP8 dispatch, chỉ cần ~20 SM để saturate, hook-based comm-compute overlap. Lợi: throughput/chất lượng; hại: memory (giữ hết expert) + comm.

---

## Phần 29-30 — Linear & Window Attention

**Linear Attention (29)**: kernelized attention O(N) qua associativity φ(Q)(φ(K)^T V). **Gated DeltaNet** (DeltaNet + Mamba-style gating) — backbone của 75% layer Qwen3-Next; **RWKV**; **Mamba/SSM** (selective state space h'=Ah+Bx, **hardware-aware parallel scan** trên SRAM, recompute backward; scan 20-40× nhanh hơn naive, đạt O(N) và O(1) decode state). Đánh đổi chất lượng: **retrieval/in-context yếu hơn** ("recall tax" — kém recall chi tiết sâu trong context dài); chính lý do giữ 25% full-attention. **Quantize state hồi quy rất nhạy** (lỗi tích lũy qua thời gian) → giữ high precision.

**Window Attention (30)**: sliding window cố định, KV chỉ giữ window → tiết kiệm KV (sliding window có thể nhỏ tới 128 token với overhead KV không đáng kể). Kết hợp global layer (**Gemma3, GPT-OSS** interleave SWA + full). **StreamingLLM attention sink** giữ ổn định. Hybrid model trộn window/full/linear: ví dụ Qwen3-Next dùng Gated DeltaNet + Gated full attention; một số biến thể dùng SWA cho gated attention.

---

## BẢNG ĐÁNH ĐỔI TỐC ĐỘ vs CHẤT LƯỢNG (deliverable chính)

| Method | Tối ưu gì | Speedup (approx, nguồn) | Tác động chất lượng | Áp dụng MoE+hybrid trên H200 | Ghi chú/điều kiện |
|---|---|---|---|---|---|
| FlashAttention-3 | latency, throughput (attention) | 1.5-2.0× FA-2; FP8 ~1.2-1.3 PFLOPS (Shah 2024) | Lossless (FP16/BF16); FP8 sai số thấp hơn 2.6× | Chỉ 25% full-attn layer | Cần Hopper; linear layer dùng scan kernel |
| PagedAttention | concurrency, memory, throughput | 2-4× (Kwon 2023) | Lossless | KV chỉ cho full-attn; hybrid KV manager | Block size ↔ fragmentation |
| Continuous batching (Orca) | throughput, concurrency | tăng utilization mạnh | Lossless | Tốt cho serving nhiều request | Kết hợp chunked prefill |
| Speculative decoding (exact) | latency (batch nhỏ) | token/round=(1−α^(γ+1))/(1−α); 2-3× điển hình | Lossless | MTP bản địa | Net-negative ở batch lớn |
| EAGLE-3 | latency | 3.0-6.5× (Li 2025); 1.38× thr@batch64 SGLang; 1.8-2.0× e2e Qwen (AngelSlim) | Lossless | Tương thích MTP | Acceptance giảm ở concurrency cao |
| DeepSeek-V3 MTP | latency | +1.8× TPS, acceptance 80-90% (arXiv:2505.09343) | Lossless | Trực tiếp áp dụng | token thứ hai |
| Medusa | latency | 2.2-3.6× (Cai 2024) | Lossless (M-1)/gần (M-2) | Heads + tree attn | batch=1 tối ưu |
| FP8 (W8A8 E4M3) | throughput, memory | ~2× vs BF16 (FP8 tensor core) | Gần lossless | **Sweet-spot đầu tiên** | Giữ router/gate high precision |
| AWQ W4A16 | decode latency, memory | TinyChat >3× FP16 (Lin 2024) | Nhỏ (~lossless ở 4-bit g128) | Lý tưởng cho routed expert | Bảo vệ 1% salient |
| GPTQ W4/W3 | decode latency, memory | tương tự AWQ | Nhỏ ở 4-bit; rõ ở 3-bit | routed expert | Hessian/Cholesky |
| QuaRot/SpinQuant W4A4 | throughput, memory | INT4 GEMM | SpinQuant cách FP 2.9đ; QuaRot ~3.5đ | Cho phép W4A4 nhưng rủi ro MoE | Hadamard/learned rotation |
| KV quant 4-bit (KIVI) | memory (KV), concurrency | ~4× KV nén | Gần lossless | Chỉ full-attn KV | key per-channel, value per-token |
| KV quant 2-bit | memory (KV) | ~8× KV nén | **Suy giảm rõ** (math/long-context) | Rủi ro cho reasoning | Cần residual window FP |
| Pruning 2:4 (SparseGPT/Wanda) | throughput (Sparse TC) | ~2× GEMM lý thuyết | Mất điểm vừa phải (Wanda 2:4 PPL↑) | Rủi ro với expert | Hopper Sparse Tensor Core |
| Expert pruning/merging | memory, latency | giảm expert count | **Nhạy cao** (đổi routing) | Monitor utilization | Nguy cơ collapse |
| Early exit / LayerSkip | latency | 1.34-2.16× (Elhoushi 2024) | Tunable (tin cậy) | KV cho skipped layer phức tạp | Cần train recipe |
| Lookahead decoding | latency | ~1.5-2× (Fu 2024) | Lossless | Không cần draft | Jacobi+n-gram |
| Tensor Parallelism | latency (model lớn) | scale với GPU | Lossless | TP≤8 NVLink | 4 all-reduce/layer (train) |
| Expert Parallelism+DeepEP | throughput (MoE) | saturate BW, ~20 SM | Lossless | **Thiết yếu cho 512 expert** | all-to-all NVLink+RDMA |
| Sequence/Context Parallelism | long-context | giảm activation mem | Lossless | Cho 256K-1M context | reduce-scatter+all-gather |
| CUDA graphs + graph opt | latency (CPU overhead) | giảm launch overhead | Lossless | Bật mặc định Qwen3-Next vLLM | Static shape tradeoff |
| DeepGEMM grouped GEMM | throughput (MoE GEMM) | dense 1.4-2.7×, MoE 1.1-1.3× (vs CUTLASS) | Lossless (two-level accum) | **Cốt lõi MoE FP8** | TMA/WGMMA Hopper |
| Memory offload (FlexGen) | enable (memory) | throughput-oriented | Lossless | Chỉ khi thiếu HBM | PCIe bottleneck |
| StreamingLLM sinks | memory, infinite stream | window memory cố định | Gần lossless với sink | full/window layer | 4 sink token |

### Phân nhóm ba bucket

**Bucket 1 — Gần như miễn phí / lossless**: FlashAttention-3, PagedAttention, continuous batching (Orca), exact speculative decoding/EAGLE-3/MTP, lookahead decoding, LMCache-style KV reuse (prefix sharing), CUDA-graph/kernel fusion/DeepGEMM, graph optimization, TP/PP/SP/EP (không đổi output), streaming. **Bật tất cả mặc định.**

**Bucket 2 — Đánh đổi tinh chỉnh được**: quantization bit-width (FP8 → W8A8 → W4A16 → W4A4), KV cache 4-bit (an toàn) vs 2-bit (rủi ro), pruning ratio 2:4, early exit threshold, lossy/relaxed speculative. **Tinh chỉnh theo ngân sách chất lượng.**

**Bucket 3 — Nhạy cảm nhất, giữ high precision**: MoE router/gating quantization, expert pruning/merging, quantize state Gated DeltaNet/linear-attention, KV 2-bit cho reasoning task. **Tránh hoặc rất thận trọng.**

---

## Recommendations (theo giai đoạn, có ngưỡng)

**Giai đoạn 0 — Nền tảng lossless (luôn bật):**
1. FlashAttention-3 cho 25% full-attention layer; scan kernel tối ưu cho 75% Gated DeltaNet layer.
2. PagedAttention + continuous batching + chunked prefill; hybrid KV cache manager.
3. CUDA graphs + DeepGEMM FP8 grouped GEMM cho MoE; kernel fusion.
4. Expert parallelism với DeepEP (thiết yếu cho 512 expert); TP≤8 trong node nếu cần.

**Giai đoạn 1 — FP8 sweet-spot:**
5. Quantize routed expert weight + activation sang **FP8 E4M3** (W8A8). **Giữ router, shared expert, Gated DeltaNet gate ở BF16/FP16.** Đây cho ~2× throughput gần lossless; 80B FP8 ~80GB vừa HBM3e 141GB.

**Giai đoạn 2 — Speculative decoding theo batch:**
6. Bật **MTP/EAGLE-3** khi batch nhỏ/latency-critical (1.8-6.5×). **Ngưỡng tắt**: khi batch size lớn (GPU đã compute-bound) hoặc acceptance rate đo được < ~0.6 → tắt vì net-negative (đặc biệt với expert dispatch đắt của MoE).
7. **KV reuse/prefix sharing** ưu tiên cao khi nhiều request chia sẻ prefix (system prompt, RAG context).

**Giai đoạn 3 — Nén sâu hơn nếu cần memory/throughput:**
8. Decode memory-bound + cần thêm tốc độ → **W4A16 AWQ/GPTQ** cho routed expert (group=128). Ngưỡng: chấp nhận suy giảm nhỏ; benchmark trên task đích.
9. KV cache **4-bit (KIVI)** nếu KV là nút thắt (long context, batch lớn). **Không dùng 2-bit cho reasoning/math** trừ khi có dynamic precision boost (Kitty-style).
10. Long context 256K-1M → context/sequence parallelism + window attention/StreamingLLM sink cho full-attention layer.

**Tránh trừ khi có ngân sách validation lớn**: W4A4 cho toàn MoE, expert pruning/merging, quantize router, KV 2-bit. Các thay đổi này cần A/B test kỹ vì rủi ro expert-shift cascade và mất in-context.

**Ngưỡng quyết định chính:**
- batch nhỏ (≤4-8) → speculative ON, W4A16 ưu tiên (memory-bound).
- batch lớn (≥32) → speculative cân nhắc OFF, FP8 W8A8 ưu tiên (compute-bound), tăng EP.
- KV là nút thắt → KV 4-bit + window/sink.
- thiếu HBM → expert offload (chấp nhận PCIe), nhưng FP8 80B thường vừa H200.

## Caveats
- Phần lớn con số speedup (FA-3 1.5-2×, EAGLE-3 3-6.5×, PagedAttention 2-4×, DeepGEMM 2.7×) đến từ **benchmark trong điều kiện thuận lợi** (model/seq-len/batch cụ thể, GPU H100/H800). Trên H200 và mô hình MoE+hybrid cụ thể, kết quả thực tế thường thấp hơn.
- Speculative decoding speedup phụ thuộc mạnh **task** (code dễ draft hơn, acceptance cao) và **batch size**; có báo cáo cộng đồng MTP **net-negative** ở batch=1 trên một số cấu hình Qwen3-Next do CPU sync (`mamba_postprocess`) và giảm prefix-cache hit rate (~92%→~71%).
- Qwen **không công bố** acceptance rate/speedup tuyệt đối cho MTP của Qwen3-Next (chỉ định tính "accelerates inference" + các tỉ số throughput tương đối so với Qwen3-32B); con số 80-90% acceptance / 1.8× TPS là của **DeepSeek-V3**, dùng làm tham chiếu kiến trúc gần nhất.
- Con số quantization accuracy (SpinQuant 2.9đ, KIVI 2-bit drop) từ model LLaMA/Qwen dense; **chưa có benchmark công khai cho mô hình MoE 512-expert + Gated DeltaNet cụ thể** — cần tự validate, đặc biệt router và linear-attention state.
- Active params Qwen3-Next: vLLM/Qwen ghi 3B (~3.75%), NVIDIA NIM ghi 3.9B (khác biệt cách đếm shared/attention params).
- Không tìm thấy bảng throughput/KV-memory tuyệt đối cho H200 cụ thể; chỉ có claim tương đối của Qwen (prefill ~7× ở 4K, >10× ở >32K; decode ~4× ở 4K, >10× ở >32K so với Qwen3-32B) và mô tả cơ chế hybrid KV cache manager của vLLM.