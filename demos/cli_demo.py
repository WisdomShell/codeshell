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

# This code is based on Qwen's Cli Demo. It has been modified from
# its original forms to accommodate CodeShell.

# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
from transformers import StoppingCriteria, StoppingCriteriaList

DEFAULT_CKPT_PATH = 'WisdomShell/CodeShell-7B-Chat'

_WELCOME_MSG = '''\
Welcome to use CodeShell-Chat model, type text to start chat, type :h to show command help.
(欢迎使用 CodeShell-Chat 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of CodeShell.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
(注：本演示受CodeShell的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
'''
_HELP_MSG = '''\
Commands:
    :help / :h          Show this help message              显示帮助信息
    :exit / :quit / :q  Exit the demo                       退出Demo
    :clear / :cl        Clear screen                        清屏
    :clear-his / :clh   Clear history                       清除对话历史
    :history / :his     Show history                        显示对话历史
    :seed               Show current random seed            显示当前随机种子
    :seed <N>           Set random seed to <N>              设置随机种子
    :conf               Show current generation config      显示生成配置
    :conf <key>=<value> Change generation config            修改生成配置
'''


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=args.device,
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=torch.bfloat16
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f'History ({len(history)})'.center(terminal_width, '='))
    for index, (query, response) in enumerate(history):
        print(f'User[{index}]: {query}')
        print(f'CodeShell[{index}]: {response}')
    print('=' * terminal_width)


def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')

def main():
    parser = argparse.ArgumentParser(
        description='CodeShell-Chat command-line interactive chat demo.')
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0",help="Device name.")

    args = parser.parse_args()

    history, response = [], ''

    model, tokenizer, config = _load_model_tokenizer(args)
    orig_gen_config = deepcopy(model.generation_config)


    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed

    while True:
        query = _get_input()

        # Process commands.
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command in ['clear', 'cl']:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif command in ['clear-history', 'clh']:
                print(f'[INFO] All {len(history)} history cleared')
                history.clear()
                _gc()
                continue
            elif command in ['help', 'h']:
                print(_HELP_MSG)
                continue
            elif command in ['history', 'his']:
                _print_history(history)
                continue
            elif command in ['seed']:
                if len(command_words) == 1:
                    print(f'[INFO] Current random seed: {seed}')
                    continue
                else:
                    new_seed_s = command_words[1]
                    try:
                        new_seed = int(new_seed_s)
                    except ValueError:
                        print(f'[WARNING] Fail to change random seed: {new_seed_s!r} is not a valid number')
                    else:
                        print(f'[INFO] Random seed changed to {new_seed}')
                        seed = new_seed
                    continue
            elif command in ['conf']:
                if len(command_words) == 1:
                    print(model.generation_config)
                else:
                    for key_value_pairs_str in command_words[1:]:
                        eq_idx = key_value_pairs_str.find('=')
                        if eq_idx == -1:
                            print('[WARNING] format: <key>=<value>')
                            continue
                        conf_key, conf_value_str = key_value_pairs_str[:eq_idx], key_value_pairs_str[eq_idx + 1:]
                        try:
                            conf_value = eval(conf_value_str)
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            print(f'[INFO] Change config: model.generation_config.{conf_key} = {conf_value}')
                            setattr(model.generation_config, conf_key, conf_value)
                continue
            elif command in ['reset-conf']:
                print('[INFO] Reset generation config')
                model.generation_config = deepcopy(orig_gen_config)
                print(model.generation_config)
                continue
            else:
                # As normal query.
                pass

        # Run chat.
        set_seed(seed)
        try:
            for response in model.chat(query, history, tokenizer, generation_config=config, stream=True):
                response = response.replace('|end|', '')
                response = response.replace('|<end>|', '')
                _clear_screen()
                print(f"\nUser: {query}")
                print(f"\nQwen-Chat: {response}")
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue
        history.append((query, response))

if __name__ == "__main__":
    main()
