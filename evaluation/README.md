# 代码评估
本文档将向您完整地介绍codeshell的代码评估过程，该评估脚本都是基于[bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)。

## 开始步骤

首先，复制bigcode-evaluation-harness仓库并导航至所在的文件夹内：

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
```

接下来，依照您设备的规格，安装PyTorch，然后运行以下命令安装剩余的依赖：

```bash
pip install -e .
```

要使用评估脚本生成和评估任务，请按下述样例进行。确保您位于正确的目录中（`codeshell/evaluation`），然后依次执行两个 `run_eval.sh` 命令：

```bash
cd codeshell/evaluation
./run_eval.sh local_gen humaneval $model_name_or_path $save_folder
./run_eval.sh eval humaneval $model_name_or_path $save_folder
```

遵循本指南，您现在可以充分利用bigcode-evaluation-harness来对模型在特定任务上表现的评估。