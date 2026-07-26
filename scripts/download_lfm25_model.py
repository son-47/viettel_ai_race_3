from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
    local_dir="/model",
    local_dir_use_symlinks=False,
)
