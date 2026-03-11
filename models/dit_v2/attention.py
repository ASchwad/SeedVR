# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch
import torch.nn.functional as F

from torch import nn


def _varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """Drop-in replacement for flash_attn_varlen_func using PyTorch SDPA."""
    batch_size = len(cu_seqlens_q) - 1
    outputs = []
    for i in range(batch_size):
        sq_start, sq_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        sk_start, sk_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        qi = q[sq_start:sq_end].unsqueeze(0).transpose(1, 2)
        ki = k[sk_start:sk_end].unsqueeze(0).transpose(1, 2)
        vi = v[sk_start:sk_end].unsqueeze(0).transpose(1, 2)
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p, scale=softmax_scale, is_causal=causal)
        outputs.append(oi.transpose(1, 2).squeeze(0))
    return torch.cat(outputs, dim=0)


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        kwargs.pop("deterministic", None)
        return _varlen_attn(*args, **kwargs)
