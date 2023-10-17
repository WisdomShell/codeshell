import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        user_string="## human:",
        copilot_string="## copilot:",
        assistant_string="## assistant:",
        end_string=" |<end>| ",
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        # self.data = self.data[:1000]
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_string = user_string
        self.assistant_string = assistant_string
        self.end_string = end_string
        self.user_tokens = self.tokenizer.encode(user_string)
        self.copilot_tokens = self.tokenizer.encode(copilot_string)
        self.assistant_tokens = self.tokenizer.encode(assistant_string)
        self.end_tokens = self.tokenizer.encode(end_string)
        self.ignore_index = -100
        
        self.preprocessed_data = self.preprocessing()
        item = self.preprocessed_data[0]
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.preprocessed_data)

    def preprocessing(self):
        preprocessed_data = []
        for example in tqdm(self.data, desc="Preprocessing"):
            preprocess_example = self.preprocess_one(example)
            if len(preprocess_example["input_ids"]) <= 16:
                continue
            preprocessed_data.append(preprocess_example)
        return preprocessed_data
    
    def preprocess_one(self, example):
        input_ids = []
        labels = []

        chat_mode = "human"
        if "copilot" in [message["from"] for message in example["conversations"]]:
            chat_mode = "copilot"
        
        if chat_mode == "human":
            for idx, message in enumerate(example["conversations"]):
                if idx == 0:
                    input_ids += [self.tokenizer.eos_token_id]
                    labels += [self.ignore_index]
                from_ = message["from"]
                value = message["value"]
                value_ids = self.tokenizer.encode(value)
                
                if len(input_ids) + len(self.user_tokens + value_ids + self.end_tokens) > self.model_max_length:
                    break
                
                if from_ == "human":
                    input_ids += self.user_tokens + value_ids + self.end_tokens
                    labels += [self.ignore_index] * len(
                        self.user_tokens + value_ids + self.end_tokens
                    )
                else:
                    input_ids += self.assistant_tokens + value_ids + self.end_tokens
                    labels += [self.ignore_index] * len(self.assistant_tokens) \
                                + value_ids + self.end_tokens
        elif chat_mode == "copilot":
            for idx, message in enumerate(example["conversations"]):
                from_ = message["from"]
                value = message["value"]
                value_ids = self.tokenizer.encode(value)

                if len(input_ids) + len(value_ids) > self.model_max_length:
                    break
                
                if from_ == "copilot":
                    input_ids += value_ids
                    labels += [self.ignore_index] * len(value_ids)
                else:
                    input_ids += value_ids + [self.tokenizer.eos_token_id]
                    labels += value_ids + [self.tokenizer.eos_token_id]
        else:
            raise ValueError("chat_mode should be human or copilot")

        input_ids = input_ids[-self.model_max_length:]
        labels = labels[-self.model_max_length:]
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.eos_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # print(self.preprocessed_data[idx]["input_ids"].shape)
        return self.preprocessed_data[idx]

    def print_dataset_example(self, num=3):
        for idx in range(num):
            example = self.preprocessed_data[idx]
            print("input_ids:\n{}".format(example["input_ids"]))
            print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
            print("label_ids:\n{}".format(example["labels"]))
            print("labels:\n{}".format(
                self.tokenizer.decode([d if d != self.ignore_index else self.tokenizer.eos_token_id for d in example["labels"]],
                                skip_special_tokens=False)
            ))
        

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
    )
    
    # tokenizer.eos_token_id = 70000
    # tokenizer.eos_token = "<|endoftext|>"
    
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.bos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    dataset.print_dataset_example()

    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()