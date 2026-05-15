
CUDA_VISIBLE_DEVICES="0" python examples/remove_wan/infer_remove_wan.py \
    --fg_bg_path /hpc2hdd/home/mli861/cv_project/ROSE-Benchmark/Benchmark/common/Unedited/example-1.mp4 \
    --mask_path /hpc2hdd/home/mli861/cv_project/ROSE-Benchmark/Benchmark/common/Masked/example-1.mp4 \
    --output_path /hpc2hdd/home/mli861/cv_project/ROSE-Benchmark/Benchmark/common/predicted/effecterase/example-1.mp4 \
    --text_encoder_path Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth \
    --dit_path Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors \
    --image_encoder_path Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --pretrained_lora_path EffectErase.ckpt