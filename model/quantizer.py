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
    if self.quant_state is not None:
        self.quant_state[0] = self.quant_state[0].cuda(device)
        self.quant_state[6] = self.quant_state[6].cuda(device)
    return self

def Params4bitTo(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    if (device is not None and device.type == "cuda" and self.data.device.type == "cpu"):
        return self.cuda(device)
    else:
        if self.quant_state is not None:
            # make sure the quantization state is on the right device
            self.quant_state[0] = self.quant_state[0].to(device)
            self.quant_state[6] = self.quant_state[6].to(device)
        new_param = Params4bit(self.to(device=device, dtype=dtype, non_blocking=non_blocking),
                                requires_grad=self.requires_grad, quant_state=self.quant_state,
                                blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                quant_type=self.quant_type)

    return new_param

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


general_weight_dict = {
    "transformer.wte.weight": False,
    "transformer.ln_f.weight": False,
    "transformer.ln_f.bias": False,
    "lm_head.weight": False,
}

layer_weight_dict = {
    "transformer.h.{i}.ln_1.weight": False,
    "transformer.h.{i}.ln_1.bias": False,
    "transformer.h.{i}.attn.c_attn.weight": True,
    "transformer.h.{i}.attn.c_attn.bias": False,
    "transformer.h.{i}.attn.c_proj.weight": True,
    "transformer.h.{i}.attn.c_proj.bias": False,
    "transformer.h.{i}.attn.rotary_emb.inv_freq": False,
    "transformer.h.{i}.ln_2.weight": False,
    "transformer.h.{i}.ln_2.bias": False,
    "transformer.h.{i}.mlp.c_fc.weight": True,
    "transformer.h.{i}.mlp.c_fc.bias": False,
    "transformer.h.{i}.mlp.c_proj.weight": True,
    "transformer.h.{i}.mlp.c_proj.bias": False,
}
num_dict = {str(i):i for i in range(100)}

def set_value(model, name, state_dict, is_4bit):
    keys = name.split('.')
    parent = model
    for key in keys[:-1]:
        if key in num_dict:
            parent = parent[num_dict[key]]
        else:
            parent = getattr(parent, key)
    if is_4bit:
        weight_data = state_dict[f'{name}.data']
        weight_quant_state = state_dict[f'{name}.quant_state']
        assert weight_data is not None, name
        assert weight_quant_state is not None, name
        setattr(parent, keys[-1], Params4bit(weight_data, requires_grad=False, quant_state=weight_quant_state))
    else:
        setattr(parent, keys[-1], state_dict[name])

def quantize_offline(model):
    for i, layer in enumerate(model.transformer.h):
        layer.mlp.c_fc = bnb.nn.Linear4bit(
                            layer.mlp.c_fc.weight.shape[1],
                            layer.mlp.c_fc.weight.shape[0],
                            False,
                            torch.bfloat16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.mlp.c_proj = bnb.nn.Linear4bit(
                            layer.mlp.c_proj.weight.shape[1],
                            layer.mlp.c_proj.weight.shape[0],
                            False,
                            torch.bfloat16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )

        layer.attn.c_attn = bnb.nn.Linear4bit(
                            layer.attn.c_attn.weight.shape[1],
                            layer.attn.c_attn.weight.shape[0],
                            False,
                            torch.bfloat16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
        layer.attn.c_proj = bnb.nn.Linear4bit(
                            layer.attn.c_proj.weight.shape[1],
                            layer.attn.c_proj.weight.shape[0],
                            False,
                            torch.bfloat16,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
    return model

def load_state_dict_for_qunantied_model(model, state_dict):
    #replace Params4bit.cuda with Params4bitCuda
    Params4bit.cuda = Params4bitCuda
    Params4bit.to = Params4bitTo

    for name, is_4bit in general_weight_dict.items():
        set_value(model, name, state_dict, is_4bit)
                
    for layer_i in range(len(model.transformer.h)):
        for name, is_4bit in layer_weight_dict.items():
            name = name.replace('{i}', str(layer_i))
            set_value(model, name, state_dict, is_4bit)
    return model