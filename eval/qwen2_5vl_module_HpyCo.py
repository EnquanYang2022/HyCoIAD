import torch
from torch import nn
import math
from torch.optim import AdamW
from torch.nn import functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionBlock,
)
from transformers import Qwen2_5_VLForConditionalGeneration

def truncated_normal_(tensor, mean=0, std=1):
	size = tensor.shape
	tmp = tensor.new_empty(size + (4,)).normal_()
	valid = (tmp < 2) & (tmp > -2)
	ind = valid.max(-1, keepdim=True)[1]
	tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
	tensor.data.mul_(std).add_(mean)
def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
		nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
		# nn.init.normal_(m.weight, std=0.001)
		# nn.init.normal_(m.bias, std=0.001)
		if m.bias is not None:
			truncated_normal_(m.bias, mean=0, std=0.001)
	if type(m) == nn.Linear:
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	if type(m) == nn.BatchNorm2d:
		nn.init.uniform_(m.weight)
		nn.init.constant_(m.bias, 0)

class GlobalFacT(nn.Module):
    """全局共享通信参数"""
    def __init__(self, rank, t_len):
        super().__init__()
        self.FacT_c = nn.Parameter(torch.ones(1, rank, t_len))
        self.FacT_p = nn.Parameter(torch.eye(t_len))

class IBC_QKV_Add(nn.Module):
    """DeepSpeed ZeRO 安全的 QKV 注入模块"""
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        get_FacT_c,      # <- callable, 返回 Parameter
        # FacT_d1: nn.Module,
        # FacT_d2: nn.Module,
        get_FacT_p,      # <- callable, 返回 Parameter
        idx: int,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.get_FacT_c = get_FacT_c
        self.get_FacT_p = get_FacT_p
        # self.FacT_d1 = FacT_d1

        # self.FacT_d2 = FacT_d2
        self.idx = idx
        rank = self.linear_a_q.out_features
        self.gamma = nn.Parameter(torch.ones(8), requires_grad=True)
        self.lora_dropout = nn.Dropout(0.05)

        self.k_cur = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.norm_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def padding_zero(self, x, normalize=True):
        num_dims = len(x.size())
        padding = [0] * (2 * num_dims)
        padding[-2 * num_dims] = 1
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True) * self.norm_scale.exp().clamp(max=10)  # normalization
        else:
            x = x * self.norm_scale.exp().clamp(max=10)
        x = F.pad(x, padding, "constant", value=0)  # padding with zero, x: d → d+1

        return x

    @staticmethod
    def lorentz_expmap0(u, k, dim=-1, min=1e-8):
        x = u.narrow(-1, 1, u.size(-1) - 1)
        sqrtK = torch.sqrt(torch.abs(k))
        x_norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min, max=math.asinh(2 ** 15))
        theta = x_norm / sqrtK

        l_v = sqrtK * torch.cosh(theta)
        r_v = sqrtK * torch.sinh(theta) * x / x_norm
        v = torch.cat((l_v, r_v), dim)
        return v

    @staticmethod
    def lorentz_logmap0(x, k, dim=-1, min=1e-7):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        sqrtK = torch.sqrt(torch.abs(k))
        y_norm = torch.norm(y, p=2, dim=dim, keepdim=True).clamp(min)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[..., 0:1] / sqrtK, min=1.0 + 1e-7)
        res[..., 1:] = sqrtK * torch.arccosh(theta) * y / y_norm
        return res


    def forward(self, x):
        # 获取全局参数，不再注册
        FacT_c = self.get_FacT_c()
        FacT_p = self.get_FacT_p()

        qkv = self.qkv(x)
        CP = (FacT_c @ FacT_p)[..., self.idx:self.idx + 2] #(1,r)
        # x = self.lora_dropout(x)
        x  = self.padding_zero(x)
        #(1) exponential map
        x = self.lorentz_expmap0(x,self.k_cur)


        #Q分支
        #(2) Transformation on the manifold
        x_space_q = self.linear_a_q(x)  #n+1->r
        x_space_q_addi = x_space_q*self.gamma[:4]

        x_space_q = x_space_q * CP[:,:,0]
        x_space_q = x_space_q + x_space_q_addi

        x_time_q = ((x_space_q ** 2).sum(dim=-1, keepdim=True) + self.k_cur.abs()).sqrt() #计算time

        x_space_q = torch.cat([x_time_q, x_space_q], dim=-1)

        x_q = self.linear_b_q(x_space_q)
        x_time_q = ((x_q ** 2).sum(dim=-1, keepdim=True) + self.k_cur.abs()).sqrt()
        x_q = torch.cat([x_time_q, x_q], dim=-1)

        #V分支
        x_space_v = self.linear_a_v(x)  # n+1->r
        x_space_v_addi = x_space_v*self.gamma[-4:]

        x_space_v = x_space_v * CP[:, :, 1]

        x_space_v = x_space_v + x_space_v_addi

        x_time_v = ((x_space_v ** 2).sum(dim=-1, keepdim=True) + self.k_cur.abs()).sqrt()  # 计算time

        x_space_v = torch.cat([x_time_v, x_space_v], dim=-1)

        x_v = self.linear_b_v(x_space_v)
        x_time_v = ((x_v ** 2).sum(dim=-1, keepdim=True) + self.k_cur.abs()).sqrt()
        x_v = torch.cat([x_time_v, x_v], dim=-1)

        # (3) Logarithmic map
        x_q = self.lorentz_logmap0(x_q, self.k_cur)[..., 1:]
        x_v = self.lorentz_logmap0(x_v, self.k_cur)[..., 1:]

        qkv[..., : self.dim] += x_q
        qkv[..., -self.dim:] += x_v
        return qkv




class Qwen2_5_VisionTransformerPretrainedModel2(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # self.blocks = nn.ModuleList(
        #     [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)] #对于Qwen2_5vl_3B,depth为32
        # )
        dim = self.blocks[0].attn.qkv.in_features
        rank = 4
        t_len = 64
        attention_len = 2
        # dtype = next(self.blocks[0].attn.qkv.parameters()).dtype
        # device = next(self.blocks[0].attn.qkv.parameters()).device
        self.global_FacT = GlobalFacT(rank, t_len)
        idx = 0
        for blk in self.blocks:
            w_a_linear_q = nn.Linear(dim+1, rank, bias=False)
            w_b_linear_q = nn.Linear(rank+1, dim, bias=False)
            w_a_linear_v = nn.Linear(dim+1, rank, bias=False)
            w_b_linear_v = nn.Linear(rank+1, dim, bias=False)
            # FacT_d1 = nn.Linear(rank+1, rank+1, bias=False)
            # FacT_d2 = nn.Linear(rank+1, rank+1, bias=False)

            w_a_linear_q.apply(init_weights)
            w_b_linear_q.apply(init_weights)
            w_a_linear_v.apply(init_weights)
            w_b_linear_v.apply(init_weights)
            # FacT_d1.apply(init_weights)
            # FacT_d2.apply(init_weights)


            w_qkv_linear = blk.attn.qkv
            blk.attn.qkv = IBC_QKV_Add(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                get_FacT_c=lambda: self.global_FacT.FacT_c,
                # FacT_d1=FacT_d1,
                # FacT_d2=FacT_d2,
                get_FacT_p=lambda: self.global_FacT.FacT_p,
                idx=idx,
            )
            idx += 2

    # def _init_weights(self, module):
    #     # 禁止复制我们自己的参数
    #     if hasattr(module, "FacT_c") or hasattr(module, "FacT_p"):
    #         return
    #     return super()._init_weights(module)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        # hidden_states.requires_grad = True #todo
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
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        # hidden_states.requires_grad = True
        # print('be', hidden_states.requires_grad, hidden_states.grad_fn)
        for layer_num, blk in enumerate(self.blocks):
            # print('mi_be', hidden_states.requires_grad, hidden_states.grad_fn)
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            # print('self.training',self.training,'checkpoint',self.gradient_checkpointing)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
                # print('mi', hidden_states.requires_grad, hidden_states.grad_fn)

        hidden_states = self.merger(hidden_states)
        # print('af', hidden_states.requires_grad, hidden_states.grad_fn)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

class Qwen2_5_VLModel2(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel2._from_config(config.vision_config)

class Qwen2_5_VLForConditionalGenerationHpyCo(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel2(config)


def init_weight(model):
    nn.init.ones_(model.visual.global_FacT.FacT_c)
    nn.init.eye_(model.visual.global_FacT.FacT_p)
    for blk in model.visual.blocks:
        blk.attn.qkv.linear_a_q.apply(init_weights)
        blk.attn.qkv.linear_b_q.apply(init_weights)
        blk.attn.qkv.linear_a_v.apply(init_weights)
        blk.attn.qkv.linear_b_v.apply(init_weights)
        # blk.attn.qkv.FacT_d1.apply(init_weights)
        # blk.attn.qkv.FacT_d2.apply(init_weights)
        if blk.attn.qkv.gamma is not None:
            nn.init.ones_(blk.attn.qkv.gamma)
        # ---- 曲率 k：新的稳定方式 ----
        if blk.attn.qkv.k_cur is not None:
            nn.init.constant_(blk.attn.qkv.k_cur, 0.5)
    
        if blk.attn.qkv.norm_scale is not None:
            nn.init.zeros_(blk.attn.qkv.norm_scale)


    return model


