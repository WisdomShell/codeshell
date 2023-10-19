In this guide, we present the official fine-tuning script for users who wish to adapt pretrained models for their domain-specific tasks.

To begin, please set up the required environment by executing the following command:
```bash
pip install peft deepspeed
```

The training data should be organized in JSON format, with each sample being a dictionary containing an ID and a conversation list. The conversation list is an array of message objects, representing the conversation between the user and the assistant. See the example below:

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

Once the data is prepared, navigate to the fine-tuning directory and execute the `run_finetune.sh` script using the following command:

```bash
cd codeshell/finetune
./run_finetune.sh $model_name_or_path $dataset_path $save_path
```

By following these instructions, you'll be able to fine-tune the pretrained model to better suit your specific downstream tasks.

This code is based on the revised code from qwen, fastchat and tatsu-lab/stanford_alpaca.