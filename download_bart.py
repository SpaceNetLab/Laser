from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="facebook/bart-base",
    local_dir="./facebook_bart_base",  # Local save directory
    local_dir_use_symlinks=False       # Avoid symlink issues
)
print(f"Model saved to: {local_dir}")