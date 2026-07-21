#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_lora_to_base.py

用途：将 LoRA adapter 与 base 模型合并，并保存为 Hugging Face 格式的完整模型
使用方式：
    python merge_lora_to_base.py \
        --base_model ./path/to/base_model \
        --lora_weights ./path/to/lora_weights \
        --save_dir ./path/to/final_merged_model
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer,Qwen2_5_VLForConditionalGeneration,AutoProcessor
from peft import PeftModel

def merge_lora_and_save(base_model_path, lora_path, save_path):
    print(f"🔹 加载基础模型: {base_model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path,torch_dtype=torch.bfloat16).eval()
    # base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path)

    print(f"🔹 加载 LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path,torch_dtype=torch.bfloat16)

    print(f"✅ 合并 LoRA 权重到基础模型 ...")
    merged_model = model.merge_and_unload().to(torch.bfloat16)
    # merged_model = model.merge_and_unload()

    print(f"💾 保存合并后的模型到: {save_path}")
    merged_model.save_pretrained(save_path)

    print(f"💾 保存 tokenizer 到: {save_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.save_pretrained(save_path)

    print(f"💾 保存 processor 到: {save_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)
    processor.save_pretrained(save_path)

    print("🎉 成功合并并保存完整模型！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default='/mnt/data/yeq/2025_6_6_3090_project/MLLM/Model_Huggface_Download/Qwen2.5-VL-3B-Instruct/',
                        type=str, help="基础模型目录")
    parser.add_argument("--lora_weights", default='/mnt/data/yeq/MLLM/IAD_MMAD/checkpoints/rl/Qwen2.5-VL-3B-Instruct_RL_lora_5e-6_4_EMIT_and_MMAD/',
                        type=str, help="LoRA adapter 权重目录")
    parser.add_argument("--save_dir", default='/mnt/data/yeq/MLLM/IAD_MMAD/checkpoints/rl/Qwen2.5-VL-3B-Instruct_RL_merge_EMIT_and_MMAD_5e-6_1/', type=str,
                        help="最终模型保存目录")

    args = parser.parse_args()

    save_path = Path(args.save_dir)
    if save_path.exists():
        print(f"⚠️ 警告：保存目录 {save_path} 已存在，可能会被覆盖。")
    else:
        save_path.mkdir(parents=True)

    merge_lora_and_save(args.base_model, args.lora_weights, args.save_dir)

    print("🎉 成功保存合并模型和 tokenizer！")
