from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Kunbyte/ROSE-Dataset",
    repo_type="dataset",
    allow_patterns="Benchmark/*",   # 关键：只下载这个文件夹
    local_dir="./ROSE-Benchmark"
)