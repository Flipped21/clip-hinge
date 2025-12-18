## 项目简介

本仓库是我的研究生毕业设计（Part 2）。  
模型的 backbone 采用 **CLIP**，在文本端加入类别特有的可学习提示（Prompt），并设计了一种 **Hinge 损失** 来促进新旧相邻类别的分离，缓解类增量学习中的遗忘问题。

## 方法概述

- **Backbone：CLIP**
  - 使用官方 CLIP 预训练权重（如 `ViT-B/16`），图像编码器与文本编码器均保持冻结或只对部分模块微调。
  - 文本端通过 `PromptLearner` 构建可学习的上下文向量（context tokens），为每个类别生成一条可训练的文本提示。

- **类增量学习设置**
  - 数据集：当前主要实验在 **CIFAR-100**（也支持 ImageNet-R）。
  - 增量方式：`init_class + increment * (total_tasks-1)` 的分组增量，例如：
    - 初始任务：10 类
    - 每个后续任务：新增 10 类
    - 共 10 个任务，最终覆盖 100 个类别。
  - 模式：**CIL（Class-Incremental Learning）**。

- **类别特有 Prompt**
  - 为每个任务维护一个 `text_prompt_pool[task_id]`，每个 prompt 对应该任务下所有类别。
  - 可配置是否在后续任务中继续更新旧任务的 prompt（`update_old_text_prompts`）。

- **旧类别特征建模与回放**
  - 在每个任务结束后，对该任务所有训练样本：
    - 提取图像编码器输出特征（不再经过 adapter）。
    - 按类别计算特征 **均值** 与 **协方差矩阵**，并存入 `class_means` 与 `class_covs`。
  - 在后续任务中：
    - 对每个旧类别，从对应的高斯分布中采样一定数量的特征（`samples_per_old_class` 为超参数）。
    - 在每个 epoch 开始时预采样所有旧类特征，并在整个 epoch 中 **均匀分配**到各个 batch，和新类别真实样本一起训练。

- **Hinge 损失设计**

  1. **选取 Hard Pairs（旧–新相邻类别对）**
     - 对当前任务中新类别集合 `new_class_ids` 与历史旧类别集合 `old_class_ids`：
       - 使用 **原始 CLIP 文本编码器** `get_raw_text_features` 为 `old + new` 所有类别生成文本特征。
       - 计算新–旧类别之间的余弦相似度矩阵。
       - 筛选相似度高于阈值 `hard_pair_threshold` 的 pair，并按相似度排序，取前 `top_hard_pairs` 作为 **hard pairs**。

  2. **计算 Hinge 损失**
     - 对于每个 hard pair `(old_class_id, new_class_id)`：
       - 从旧类别的高斯分布 `N(μ_old, Σ_old)` 中采样若干图像特征 `old_image_features` 并归一化。
       - 使用当前任务的 prompt 生成 **新类别文本特征** `new_text_feature`。
       - 使用对应任务的 prompt 生成 **旧类别文本特征** `old_text_feature`（可选择是否参与反向传播）。
       - 计算相似度：
         - `pos_sim = sim(old_image_features, old_text_feature)`
         - `neg_sim = sim(old_image_features, new_text_feature)`
       - 对每个样本计算：
         \[
         \ell_{\text{hinge}} = \max(0,\; \text{neg\_sim} - \text{pos\_sim} + \text{margin})
         \]
       - 对所有样本与所有 hard pairs 取均值，得到 `L_hinge`。
     - 总损失：
       \[
       L = L_{\text{CE}} + \lambda \cdot L_{\text{hinge}}
       \]
       其中 `hinge_margin` 控制间隔大小，`hinge_weight`（即 λ）控制 Hinge 损失权重。

## 代码结构

- `configs/`
  - `cifar100_split.json`：CIFAR-100 上的实验配置（增量划分、训练超参、hinge/回放相关超参）。
- `datasets/`
  - 数据读取与增量任务切分（`data_manager.py`）。
- `models/`
  - `clip/`：CLIP 官方模型与 tokenizer 的实现。
  - `tclip.py`：本项目封装的 CLIP + Prompt 模型（含 Gaussian 回放统计）。
- `methods/`
  - `base.py`：增量学习基类。
  - `tprompt.py`：本项目的主方法（训练流程、回放与 Hinge 损失均在此实现）。
- `results/`
  - 保存每次实验的指标与分任务准确率。
- `run.py` / `main.py`
  - 训练入口脚本。

## 运行方法（示例）

# 1. 创建并激活环境（示例）
conda create -n clip-hinge python=3.10 -y
conda activate clip-hinge
pip install -r requirements.txt

# 2. 准备数据集
# 将 CIFAR-100 或 ImageNet-R 下载/放置到 datasets/ 下，结构参考 data_manager.py

# 3. 运行增量训练（以 CIFAR-100 配置为例）
python main.py --config configs/cifar100_split.json## 关键配置说明（以 `configs/cifar100_split.json` 为例）

- **数据与任务划分**
  - `dataset`: `"cifar100"`
  - `init_class`: 初始任务类别数
  - `increment`: 后续每个任务新增的类别数
  - `total_tasks`: 任务总数

- **训练超参数**
  - `init_epochs`, `epochs`
  - `batch_size`
  - `lr`, `init_lr`, `weight_decay`
  - `seed`

- **Hinge 与 hard pair**
  - `use_hinge_loss`: 是否启用 Hinge 损失
  - `hinge_margin`: Hinge 间隔
  - `hinge_weight`: Hinge 损失权重
  - `hard_pair_threshold`: 选 hard pairs 的文本相似度阈值
  - `top_hard_pairs`: 每个任务最多使用的 hard pair 数

- **Prompt 更新与回放**
  - `update_old_text_prompts`: 是否允许旧任务的 prompt 参与更新
  - `samples_per_old_class`: 每个旧类别每个任务预采样的高斯特征数（用于 replay）
  - `use_image_adapter`: 是否使用图像 adapter（当前一般为 `false`，图像编码器冻结）

## 结果展示

实验结果会保存在 `results/` 目录下的 `.txt` 文件中，包括：

- 每个任务的 **Grouped 准确率**（总准确率 / 各任务段 / old / new）。
- 整体指标：
  - AIA（Average Incremental Accuracy）
  - AA（Average Accuracy）
  - Forgetting

