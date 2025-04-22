---
license: apache-2.0
pipeline_tag: text-generation
frameworks:
  - MindSpore
hardwares:
  - NPU
  - Atlas 800T A2
language:
  - en
  - zh
---
# 基于MindSpore Transformers蒸馏Qwen2.5-Math-7B-Instruct

本项目参考OpenR1-Qwen-7B，基于MindSpore框架和MindSpore Transformers大模型套件，利用DeepSeek-R1蒸馏Qwen2.5-Math-7B。

> 本项目当前处在持续研发中，可能会有较大的变化，请关注我们的更新。如有问题，请及时在评论区与我们联系。

## 1. 目录结构

```text
OpenR1-Qwen-7B
  ├─ README.md                                  # 说明文档
  ├─ model.safetensors                          # 蒸馏后的权重
  ├─ configs                                    # SFT任务的配置文件目录
  |    └─ finetune_qwen_2_5_7b.yaml             #   SFT任务配置（Packing数据）
  ├─ datasets                                   # 数据集处理文件目录
  |    ├─ handling                              #   数据集处理文件目录
  |    |    ├─ data_process_handling.yaml       #     数据集预处理配置文件
  |    |    └─ processed_data_handling.arrow    #     预处理后的数据集文件
  |    └─ packing                               #   Packing数据集处理文件目录
  |         ├─ data_process_packing.yaml        #     数据集Packing处理配置文件
  |         └─ processed_data_packing.arrow     #     Packing处理后的数据集文件
  ├─ modules                                    # 模块代码目录
  |    ├─ qwen_2_5_tokenizer.py                 #   Qwen2.5分词器模块
  |    └─ openr1_data_handler.py                #   OpenR1数据集处理模块
  ├─ scripts                                    # 脚本目录
  |    ├─ reject_sampling.py                    #   数据集拒绝采样脚本
  |    └─ generate_reasoning.py                 #   生成蒸馏数据脚本
  └─ .gitattributes                             # GIT文件
```

## 2. 前提准备

### 2.1 环境

本项目基于以下环境依赖，在Atlas 800T A2服务器上进行验证。

| 软件依赖                   | 版本    |
|------------------------|-------|
| MindSpore Transformers | dev   |
| MindSpore              | 2.5.0 |
| CANN                   | 8.0.0 |

安装方式请参考[MindSpore Transformers安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/quick_start/install.html)

### 2.2 模型

本次微调使用的模型为Qwen2.5-Math-7B-Instruct，可以在[魔乐社区](https://modelers.cn/models/MindSpore-Lab/Qwen2.5-Math-7B-Instruct)下载。

### 2.3 数据集

我们提供了基于服务化推理生成蒸馏数据的流程，用户也可以直接使用开源蒸馏数据集。

#### 2.3.1 生成数据集

##### 2.3.1.1 安装依赖

执行以下命令安装所需依赖：

```shell
pip install datasets tqdm aiofiles aiohttp uvloop
```

##### 2.3.1.2 本地部署Deepseek-R1

参考[MindSpore-Lab/DeepSeek-R1 | 魔乐社区](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1)在本地部署DeepSeek-R1推理服务。

##### 2.3.1.3 生成数据

执行以下命令调用推理服务的接口，使用种子数据集中的问题，生成CoT数据：

```shell
python generate_reasoning.py \
    --model "DeekSeek-R1"
    --dataset-name "AI-MO/NuminaMath-1.5" \
    --output-file "numinamath_r1_generations.jsonl" \
    --prompt-column "problem" \
    --uuid-column "problem" \
    --api-addr "api.host.name" \
    --num-generations 2 \
    --max-tokens 16384 \
    --max-concurrent 100
```

参数说明：

- **model**: 推理服务的模型名，需要和服务化配置文件 `config.json` 中的 `modelName` 一致。
- **dataset-name**：种子数据集名称，配置为HuggingFace Datasets名称或本地的数据集路径。
- **output-file**：输出CoT数据文件的文件名。
- **prompt-column**：种子数据集中提示词的列名，使用此列的数据进行CoT数据生成。
- **uuid-column**：种子数据集中uuid的列名，使用此列计算哈希值去重数据。
- **api-addr**：推理服务api的地址，配置为 `ip:port` 。
- **num-generations**：对于种子数据集中每个问题生成CoT数据的数量。
- **max-tokens**：生成的CoT数据的最大Token数。
- **max-concurrent**：请求的最大并发数量。

##### 2.3.1.4 拒绝采样

执行以下命令，利用`math-verify`库过滤错误数据，确保生成的cot数据中回答都是正确的。

``` shell
pip install math_verify

python reject_sampling.py --src numinamath_r1_generations.jsonl --dst numinamath_r1_generations_filtered.jsonl
```

参数说明：

- **src**：源CoT数据集路径。
- **dst**：过滤后的数据集的保存路径。

#### 2.3.2 使用OpenR1-Math-220K数据集

如果使用OpenR1-Math-220K数据集（已经过DeepSeek-R1蒸馏）进行微调，我们提供详细步骤以及转换好的数据集格式。

##### 选项 1: 使用原始数据离线处理

在HuggingFace上下载OpenR1-Math-220K原始数据集，本项目已提供数据集处理所需要的脚本，放在modules目录下，数据集转换的配置文件放在/dataset/handling，其中，需要修改`data_files`的参数为各个parquet数据文件的路径，以及`vocab.json`和`merges.txt`的路径。

在MindSpore Transformers源码目录下执行以下脚本：

```shell
python toolkit/data_preprocess/huggingface/datasets_preprocess.py --config /path/to/data_process_handling.yaml --save_path /path/to/handled_data --register_path /path/to/modules
```

MindSpore Transformers已经支持数据集packing机制，减少微调所需要的时间。数据集packing的配置文件放在/dataset/packing目录下，其中，需要将`path`修改成`handled_data`的路径，并在MindSpore Transformers源码目录下执行如下脚本：
```shell
python toolkit/data_preprocess/huggingface/datasets_preprocess.py --config /path/to/data_process_packing.yaml --save_path /path/to/packed_data
```

更多数据集处理的教程请参考[MindSpore Transformers官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/function/dataset.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE-handler)。

##### 选项 2：使用已经处理好的数据

在/dataset目录下已经提供packing处理好的数据，格式为arrow。

### 2.4 YAML配置

微调配置文件finetune_qwen_2_5_7b.yaml，需要根据实际情况修改，具体如下：

```yaml
seed: 42
output_dir: './output' # 训练结果保存路径，根据实际情况修改
load_checkpoint: '/path/to/safetensors' # 权重加载路径，根据实际情况修改
load_ckpt_format: 'safetensors'
auto_trans_ckpt: True  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False # 断点续训需要开启
run_mode: 'finetune'
......
train_dataset: &train_dataset
  input_columns: &input_columns ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  divisor: 32
  remainder: 1
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 2
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  dynamic_batch: True
  pad_token_id: 151643
  data_loader:
    type: CommonDataLoader
    shuffle: True
    split: "train"
    load_func: "load_from_disk"
    path: "/path/to/save_data" # packing处理后的数据集路径
    input_columns: *input_columns
......
```
其余参数配置的解释可以参考[MindSpore Transformers官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/sft_tuning.html)。

## 3. 启动微调

设置如下环境变量防止OOM：

```shell
export ACLNN_CACHE_LIMIT=10 # CANN 缓存限制
export MS_DEV_RUNTIME_CONF="aclnn_cache_queue_length:128" # MS缓存队列长度建议设置成128，设置过大内存容易OOM，设置越小性能越差
```

在mindformers目录下执行如下命令行启动微调：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config /path/to/finetune_qwen_2_5_7b.yaml --run_mode finetune" 8
```

日志记录在output/msrun_log目录下，可以通过tail指令查看日志信息。
微调完成后，输出的权重文件在output/checkpoint目录下。

## 4. 执行推理

若想使用微调后的权重进行推理，可以复用[Qwen2.5-Math-7B-Instruct](https://modelers.cn/models/MindSpore-Lab/Qwen2.5-Math-7B-Instruct)中的推理部分，但需要修改`run_qwen2_5.py`脚本中的system的提示词:：

```python
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": input_prompt}
    ]
```

## 5. 评估结果

| Model                                   | MATH-500 |
|-----------------------------------------|:--------:|
| DeepSeek-Distill-Qwen-7B                | 91.6     |
| OpenR1-Qwen-7B (HuggingFace)            | 90.6     |
| OpenR1-Qwen-7B (MindSpore Transformers) | 90.0     |
| OpenThinker-7B                          | 89.6     |

> 注：上表第三行为本项目实验结果，该结果由本地实测得到。