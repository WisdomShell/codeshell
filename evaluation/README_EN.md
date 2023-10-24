# Evaluation
This guide introduces the evaluation process of codeshell. The evaluation script is based on [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness).

## Quick Start

Begin by cloning the bigcode-evaluation-harness repository and entering the directory:

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
```

Next, install PyTorch according to your device specifications, and then install the remaining packages using the command:

```bash
pip install -e .
```

To generate and evaluate tasks with the evaluation script, follow the example below. Ensure you are in the appropriate directory (`codeshell/evaluation`), then execute the two `run_eval.sh` commands:

```bash
cd codeshell/evaluation
./run_eval.sh local_gen humaneval $model_name_or_path $save_folder
./run_eval.sh eval humaneval $model_name_or_path $save_folder
```

By following this guide, you can now effectively utilize the bigcode-evaluation-harness to evaluate your model's performance on specific tasks.
