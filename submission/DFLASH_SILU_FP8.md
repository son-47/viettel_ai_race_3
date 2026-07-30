# DFlash candidate for the 65.71 SiLU/FP8 baseline

## Artifacts

- `Dockerfile.silu-fp8-dflash`: starts from the published rollback-safe
  derivative of the 65.71 image and bakes the DFlash checkpoint at immutable
  Hugging Face revision `dee52da5ea67607ede76a906f6e30118f21aa036`.
- `apply_lfm25_dflash_aux.py`: adds the missing LFM2 auxiliary-hidden-state
  interface required by the native vLLM DFlash proposer.
- `docker-compose_silu_fp8_dflash_k3.yml`: recommended first portal candidate.
  The target stays online FP8; the bundled DFlash draft is BF16 and must not
  receive a separate `quantization` override.

The original `docker-compose_silu_fp8_65.71.yml` is unchanged and remains the
control/fallback.

## Build and pin

Run from the `submission` directory:

```powershell
docker build --pull --file Dockerfile.silu-fp8-dflash --tag misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8-dflash-v1 .
docker push misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8-dflash-v1
docker buildx imagetools inspect misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8-dflash-v1
```

The public image is pinned in the compose to
`sha256:1409fb0efd3405f67f6b3a07aea779943cc26df860a49d78be6a9528ae8cdb2e`.

## Required gates

1. Run exact greedy-token parity against the non-speculative 65.71 control,
   including rejection-heavy and multi-turn prompts:

   ```powershell
   # While the 65.71 control is running:
   python ..\harness\check_speculative_parity.py --mode record --reference ..\results\dflash_parity.json

   # After replacing it with the DFlash candidate:
   python ..\harness\check_speculative_parity.py --mode compare --reference ..\results\dflash_parity.json
   ```

2. Submit candidates in interleaved order to reduce portal noise:
   control, DFlash `k=2`, control, DFlash `k=3`, control, DFlash `k=4`, control.
3. Keep DFlash only if repeated runs preserve exact token parity, do not
   increase failed requests, keep median TBT at or below 3 ms, and beat the
   current best score. The supplied compose uses `k=3` because it is the best
   first trade-off for the small 1.2B target; the checkpoint's training block
   size of 16 is a maximum, not a requirement to verify 15 tokens per step.

For profiling only, temporarily remove `--disable-log-stats` and collect draft
tokens, accepted tokens, accepted length per verifier round, TTFT, TBT, failed
requests, and peak VRAM. Restore `--disable-log-stats` for final scoring.
