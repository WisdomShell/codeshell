# coding=utf-8
# Copyright 2023 WisdomShell Inc. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit, Int8Params
except ImportError:
    pass
import torch

def Params4bitCuda(self, device):
    self.data = self.data.cuda(device)
    self.quant_state[0] = self.quant_state[0].cuda(device)
    self.quant_state[4][0] = self.quant_state[4][0].cuda(device)
    self.quant_state[4][1][0] = self.quant_state[4][1][0].cuda(device)
    self.quant_state[4][1][1] = self.quant_state[4][1][1].cuda(device)

    self.quant_state[6] = self.quant_state[6].cuda(device)
    return self

class Linear4bitOnline(torch.nn.Module):
    def __init__(self, weight, bias, quant_type):
        super().__init__()
        self.weight = Params4bit(
            weight.data, requires_grad=False, compress_statistics=True, quant_type=quant_type
        )
        self.compute_dtype = None
        #self.weight.cuda(weight.device)
        self.bias = bias

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, "quant_state", None) is None:
            print(
                "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first."
            )
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(
            x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state
        )

        out = out.to(inp_dtype)

        return out

class Linear8bitLtOnline(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out

def quantize_online(model, bits: int):
    def quant(weight, bias=None):
        if bits == 8:
            linear = Linear8bitLtOnline(
                weight,
                bias,
                has_fp16_weights=False,
                threshold=6.0,
            )
            if bias is not None:
                linear.bias = torch.nn.Parameter(bias)
        elif bits == 4:
            linear = Linear4bitOnline(
                weight,
                bias,
                quant_type="nf4", #fp4/nf4
            )
        else:
            raise ValueError("quantize only support 4/8 bit")
        return linear

    def auto_quant(layer):
        if hasattr(layer,"bias"):
            linear = quant(layer.weight,bias=layer.bias)
        else:
            linear = quant(layer.weight)
        return linear

    for i,layer in enumerate(model.transformer.h):
        layer.mlp.c_fc = auto_quant(layer.mlp.c_fc)
        layer.mlp.c_proj = auto_quant(layer.mlp.c_proj)

        layer.attn.c_attn=auto_quant(layer.attn.c_attn)
        layer.attn.c_proj=auto_quant(layer.attn.c_proj)

    return model