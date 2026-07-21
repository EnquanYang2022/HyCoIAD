import os

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

import random
from math import sqrt
from typing import Optional, Unpack, Union

import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast
)
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.cache_utils import Cache
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from qwen_vl_utils import process_vision_info
from quaternion.quaternion_layers import QuaternionLinear

from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback


class NewViT(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.my_module = nn.Linear(in_features=1280, out_features=1280, bias=False)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        hidden_states = self.my_module(hidden_states)
        return hidden_states


class NewVLModel(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = NewViT._from_config(config.vision_config)


class NewQwen(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = NewVLModel(config)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device, dtype = torch.device("cuda:6" if torch.cuda.is_available() else "cpu"), torch.bfloat16
    model_path = 'Qwen/Qwen2.5-VL-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=True,
                                              trust_remote_code=True)
    processor = Qwen2_5_VLProcessor.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=True,
                                                    trust_remote_code=True)
    cache_path = '/mnt/data/yeq/2025_6_6_3090_project/MLLM/Model_Huggface_Download/Qwen2.5-VL-3B-Instruct/'
    model = NewQwen.from_pretrained(model_path, cache_dir=cache_path, device_map='auto', torch_dtype=dtype)
    print(model)
