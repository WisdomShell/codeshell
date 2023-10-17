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

from transformers import AutoTokenizer

import jsonlines
import os

from fire import Fire


def evaluate(tokenizer_path: str, corpora_path: str):
    """
    Evaluate the compression ratio of a tokenizer on a corpora
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    corpora = [os.path.join(corpora_path, f) for f in os.listdir(corpora_path)
               if not os.path.isdir(os.path.join(corpora_path, f))]

    total_characters = 0
    total_tokens = 0

    for corpus in corpora:
        print(f"Processing {corpus}")
        texts = []
        partial_characters = 0
        partial_tokens = 0
        with open(corpus, "r", encoding="utf-8") as f_in:
            for item in jsonlines.Reader(f_in):
                texts.append(item["text"])
                partial_characters += len(item["text"])

        # noinspection PyUnboundLocalVariable
        tokens = tokenizer(texts)

        for seg in tokens["input_ids"]:
            partial_tokens += len(seg)

        total_characters += partial_characters
        total_tokens += partial_tokens
        print(f"Characters: {partial_characters}")
        print(f"Tokens: {partial_tokens}")
        print(f"Compression ratio: {partial_characters / partial_tokens}")

    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")
    print(f"Compression ratio: {total_characters / total_tokens}")


if __name__ == "__main__":
    Fire(evaluate)