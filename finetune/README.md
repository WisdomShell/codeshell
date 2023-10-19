该文档为希望在特定领域任务中应用CodeShell模型的用户提供了官方微调示例。

开始前，您需要通过执行以下命令来配置必要的环境：
```bash
pip install peft deepspeed
```

您需要按照 JSON 格式整理训练数据，其中每个样本是一个包含 ID 和对话列表的字典。该对话列表是消息对象的数组，代表了用户和助手之间的交谈。如下所示为一个样例：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "您好，我是CodeShell，请问有什么可以帮助您的吗？"
      }
    ]
  }
]
```

当数据准备完毕后，导航至微调的目录并执行 `run_finetune.sh` 脚本，命令如下：

```bash
cd codeshell/finetune
./run_finetune.sh $model_name_or_path $dataset_path $save_path
```

按照这些步骤操作，您可以将预训练的模型微调，使其更加精确地适应您的特定任务。

该微调脚本基于qwen、fastchat 和 tatsu-lab/stanford_alpaca 的微调脚本。