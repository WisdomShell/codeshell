
<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/6489a27bd0b2fd1f3297e5ca/3LQsqRzluBhBN2DipN6Ox.png" width="400"/>
<p>

<p align="center">
  🤗 <a href="https://huggingface.co/WisdomShell/CodeShell" target="_blank">Hugging Face</a> • 🌐 <a href="http://se.pku.edu.cn/kcl/" target="_blank">PKU-KCL</a> 
</p>

<div align="center">
    
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/WisdomShell/codeshell/blob/main/LICENSE)
<h4 align="center">
    <p><a href="https://github.com/WisdomShell/codeshell/blob/main/README.md"><b>中文</b></a>|<a href="https://github.com/WisdomShell/codeshell/blob/main/README_EN.md">English</a></p>
</h4>
</div>

## Introduction

CodeShell是[北京大学知识计算实验室](http://se.pku.edu.cn/kcl/)联合四川天府银行AI团队研发的多语言代码大模型基座。CodeShell具有70亿参数，在五千亿Tokens进行了训练，上下文窗口长度为8192。在权威的代码评估Benchmark（HumanEval与MBPP）上，CodeShell取得同等规模最好的性能。与此同时，我们提供了与CodeShell配套的部署方案与IDE插件，请参考代码库[CodeShell](https://github.com/WisdomShell/codeshell)。


本次开源的模型如下：

- <a href="https://huggingface.co/WisdomShell/CodeShell" target="_blank"><b>CodeShell Base</b></a>：CodelShell底座模型，具有强大的代码基础能力。
- <a href="https://huggingface.co/WisdomShell/CodeShell-Chat" target="_blank"><b>CodeShell Chat</b></a>：CodelShell对话模型，在代码问答、代码补全等下游任务重性能优异。
- <a href="https://huggingface.co/WisdomShell/CodeShell-Chat-int4" target="_blank"><b>CodeShell Chat 4bit</b></a>：CodelShell对话模型4bit量化版本，在保证模型性能的前提下内存消耗更小，速度更快。
- <a href="https://github.com/WisdomShell/llama_cpp_for_codeshell" target="_blank"><b>CodeShell CPP</b></a>：CodelShell对话模型CPP版本，支持开发者在没有GPU的个人电脑中使用。注意，CPP版本同样支持量化操作，用户可以在最小内存为8G的个人电脑中运行CodeShell。


## Main Characteristics of CodeShell

- **强大的性能**：CodelShell在HumanEval和MBPP上达到了7B代码基座大模型的最优性能
- **完整的体系**：除了代码大模型，同时开源IDE（VS Code与JetBrains）插件，形成开源的全栈技术体系
- **轻量化部署**：支持本地C++部署，提供轻量快速的本地化软件开发助手解决方案
- **全面的评测**：提供支持完整项目上下文、覆盖代码生成、代码缺陷检测与修复、测试用例生成等常见软件开发活动的多任务评测体系（即将开源）
- **高效的训练**：基于高效的数据治理体系，CodeShell在完全冷启动情况下，只训练了五千亿Token即获得了优异的性能

## Performance

我们选取了目前最流行的两个代码评测数据集（HumanEval与MBPP）对模型进行评估，与目前最先进的两个7b代码大模型CodeLllama与Starcoder相比，Codeshell 取得了最优的成绩。具体评测结果如下。

|   任务   |  CodeShell-7b | CodeLlama-7b | Starcoder-7b |
| ------- | --------- | --------- | --------- |
| humaneval	 | **34.32** | 29.44 | 27.80 |
| mbpp		 | **38.65** | 37.60 | 34.16 |
| multiple-js	 | **33.17** | 31.30 | 27.02 |
| multiple-java	 | **30.43** | 29.24 | 24.30 |
| multiple-cpp	 | **28.21** | 27.33 | 23.04 |
| multiple-swift | 24.30 | **25.32** | 15.70 |
| multiple-php	 | **30.87** | 25.96 | 22.11 |
| multiple-d	 | 8.85 | **11.60** | 8.08 |
| multiple-jl	 | 22.08 | **25.28** | 22.96 |
| multiple-lua	 | 22.39 | **30.50** | 22.92 |
| multiple-r	 | **20.52** | 18.57 | 14.29 |
| multiple-rkt	 | **17.20** | 12.55 | 10.43 |
| multiple-rs	 | 24.55 | **25.90** | 22.82 |

## Requirements

- python 3.8 and above
- pytorch 2.0 and above are recommended
- transformers 4.32 and above
- CUDA 11.8 and above are recommended (this is for GPU users, flash-attention users, etc.)

## Quickstart

CodeShell系列模型已经上传至 <a href="https://huggingface.co/WisdomShell/CodeShell" target="_blank">Hugging Face</a>，开发者可以通过Transformers快速调用CodeShell和CodeShell-Chat。

在开始之前，请确保已经正确设置了环境，并安装了必要的代码包，以及满足上一小节的环境要求。你可以通过下列代码快速安装相关依赖。

```
pip install -r requirements.txt
```

接下来你可以通过Transformers使用CodeShell。

### Code Generation

开发者可以使用CodeShell快速生成代码，加速开发效率。

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# use base model
# tokenizer = AutoTokenizer.from_pretrained("WisdomShell/
CodeShell-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("WisdomShell/
CodeShell-Chat", trust_remote_code=True).cuda()

tokenizer = AutoTokenizer.from_pretrained("WisdomShell/
CodeShell-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("WisdomShell/
CodeShell-Chat", trust_remote_code=True).cuda()
inputs = tokenizer('def merge_sort():', return_tensors='pt').cuda()
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

- Fill in the Moddle

CodeShell 支持Fill-in-the-Middle模式，从而更好的支持软件开发过程。

```
input_text = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
inputs = tokenizer(input_text, return_tensors='pt').cuda()
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

- Model Quantization

CodeShell 支持4 bit/8 bit量化，4 bit量化后，占用显存大小约6G，用户可以在显存较小的GPU上使用CodeShell。

```
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("WisdomShell/
CodeShell-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("WisdomShell/
CodeShell-Chat", trust_remote_code=True)
model = model.quantize(4).cuda()

inputs = tokenizer('def print_hello_world():', return_tensors='pt').cuda()
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

- CodeShell in c/c++

由于大部分个人电脑没有GPU，CodeShell提供了C/C++版本的推理支持，开发者可以根据本地环境进行编译，详见[CodeShell C/C++本地化版]()。编译完成后，可以通过下列命令启动Web API服务。

## Demo

我们提供了Web-UI、命令行、API、IDE四种形式的Demo。

### Web UI

开发者通过下列命令启动Web服务，服务启动后，可以通过`https://127.0.0.1:8000`进行访问。

```
python demos/web_demo.py
```

### CLI Demo

我们也提供了命令行交互的Demo版本，开发者可以通过下列命令运行。

```
python cli_demo.py
```

### API

CodeShell也提供了基于OpenAI API的部署方法。

```
python openai_api.py
```

### IDE

CodeShell最后提供了线上IDE，开发者可以通过IDE进行代码补全、代码问答等操作。同时，IDE插件也同时发布，开发者可以自行在本地进行安装使用。插件相关问题欢迎在[VS Code插件仓库]()中讨论。


## Model Details

Code Shell使用GPT-2作为基础架构，采用Grouped-Query Attention、RoPE相对位置编码等技术。

### Hyper-parameter

| Hyper-parameter | Value  |
|---|---|
| n_layer | 42 |
| n_embd | 4096 |
| n_inner | 16384 |
| n_head | 32 |
| num_query_groups | 8 |
| seq-length | 8192 |
| vocab_size | 70144 |


### Data

CodeShell基于Big Code开源的Stack数据集进行训练。在原始数据集的基础上，CodeShell采用基于Minihash对数据去重，基于KenLM以及高质量数据筛选模型对数据进行了过滤与筛选，最终得到高质量的预训练数据集。

### Tokenizer

CodeShell基于Starcoder词表进行了优化，去除了使用频率较低的词语，并添加了部分中文词表，显著提升了中文的压缩率，为Chat版本的训练提供了基础。


| Tokenizer | Size | Chinese  | English | Code | Total|
|---|---|---|---|---|---|
| Statcoder | 49152 | 1.22 | 3.47 | 3.30 | 2.66 |
| CodeShell | 70144 | 1.50 | 3.47 | 3.30 | 2.95 |


## License

本仓库开源的模型遵循[Apache 2.0 许可证](https://www.apache.org/licenses/LICENSE-2.0)，对学术研究完全开放，若需要商用，开发者可发送邮件进行申请，得到书面授权后即可使用。联系邮箱：[codeshell.opensource@gmail.com](mailto:codeshell.opensource@gmail.com)
