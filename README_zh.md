# TabEP — 表格数据平衡传播框架

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**TabEP** 是一个基于 **平衡传播 (Equilibrium Propagation, EP)** 的表格数据分类框架。它将物理启发的能量松弛动力学引入神经网络训练，为中小型表格分类任务提供了一种兼具生物合理性与竞争性能的替代方案。

> 📄 相关论文和实验报告见 [`papers/`](papers/) 目录。

---

## 目录

- [概述](#概述)
- [核心方法](#核心方法)
- [主要特性](#主要特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [命令行接口](#命令行接口)
- [基准测试结果](#基准测试结果)
- [复现实验](#复现实验)
- [项目结构](#项目结构)
- [引用](#引用)

---

## 概述

传统的表格数据深度学习方法（如 MLP）通常缺乏针对小样本表格任务的适当归纳偏置，且对数据尺度敏感。**TabEP** 提出将**动力松弛 (dynamical relaxation)** 作为结构性偏置：将分类建模为在一个可学习的能量景观 (energy landscape) 中，系统状态逐步达到平衡的过程。

该方法受 **FitzHugh-Nagumo (FHN)** 反应-扩散动力学启发，结合**中心化平衡传播 (Centered Equilibrium Propagation)** 进行训练，在 Drug200 基准测试中以 **macro-F1 0.8389** 的成绩超越了参数匹配的 MLP（0.8296）和经典方法（如 TileLang-KNN, 0.6813）。

---

## 核心方法

### 深度能量模型 (Deep Energy Model)

TabEP 的核心是一个多层连续状态网络，其中每一层在松弛过程中随时间演化。模型定义了一个可微分的能量函数，预测过程被建模为系统在该能量景观中达到平衡的过程。

### FHN 启发式动力学

状态更新采用 **FitzHugh-Nagumo** 反应项：

$$s_i(t+1) = s_i(t) + dt \cdot \left[-s_i + \delta \cdot \text{total}_i + \epsilon \cdot (s_i - s_i^3 - \alpha s_i - \beta)\right]$$

其中 $\delta, \epsilon, \alpha, \beta$ 为动力学超参数。激活函数默认使用 `hardtanh`，将状态约束在 $[0, 1]$。

### 三种训练模式

| 模式 | 描述 |
|------|------|
| **`ep`** | 纯平衡传播训练，使用中心化 nudging 计算梯度，无需标准反向传播 |
| **`gd`** | 沿展开的动力学时间轴进行 BPTT (Backpropagation Through Time)，监督最终状态的 logits |
| **`guided`** | 混合模式：将 EP 目标与轨迹引导损失相结合 |

### 轨迹引导监督 (Trajectory-Guided Supervision)

GD 模式支持对整个松弛轨迹进行监督，而非仅关注最终状态：

- **时间加权分类损失**：每个时间步的 logits 按时间权重贡献损失
- **一致性正则化**：中间状态的 logits 向最终状态进行 KL 散度正则化
- **类别间隔惩罚**：增大目标类 logit 与最大非目标类 logit 之间的间隔

### 原型 RBF 读出头 (Prototype RBF Readout)

采用基于 RBF 原型的快速读出层，结合能量状态门控机制，将原始输入特征与松弛后的状态特征进行融合分类。

---

## 主要特性

- **🧠 生物启发学习** — 基于平衡传播，提供反向传播之外的学习范式
- **⚡ TileLang 加速** — 利用 TileLang JIT 编译的 CUDA 内核加速迭代动力学过程
- **📊 完整的表格数据流程** — 自动处理数值/类别特征归一化、编码与嵌入
- **🔬 公平对比** — 自动参数匹配，确保与 MLP 等基线模型的可比性
- **📈 SwanLab 集成** — 原生支持 [SwanLab](https://swanlab.cn) 实验跟踪
- **📝 Hydra 配置管理** — 灵活的 Hydra 覆盖机制管理实验配置
- **🧪 完备的基准测试套件** — 支持 Drug200 及 13 个 UCI 数据集的系统性评估

---

## 安装

### 环境要求

- Python ≥ 3.13
- CUDA 12.8（用于 TileLang GPU 加速）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/ZiceWang/tabep.git
cd tabep

# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 验证安装
uv run tabep --help
```

### 通过 pip 安装（开发模式）

```bash
pip install -e .
```

---

## 快速开始

### MNIST 烟雾测试

```bash
uv run tabep train mnist \
  data.limit_train=256 \
  data.limit_test=128 \
  model.hidden_size=64 \
  model.hidden_layers=2 \
  eqprop.free_steps=5 \
  eqprop.nudge_steps=2 \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu
```

### Drug200 完整基准测试

```bash
uv run tabep-drug200
```

该命令从 Hugging Face 加载 `milotix/drug200`，在 TabEP、TileLang-KNN、TileLang-RBF 核分类器、决策树和逻辑回归之间进行比较，输出准确率、macro-F1、macro-精确率和 macro-召回率到 `outputs/tabep-drug200/`。

### 对比所有训练模式

```bash
uv run tabep-drug200 --all-training-modes
```

### 使用 TileLang GPU 加速

```bash
uv run tabep-drug200 --tilelang-dynamics --device cuda
```

---

## 命令行接口

### `tabep` — 通用训练入口

基于 Hydra + PyTorch Lightning 的通用训练命令：

```bash
tabep train [OVERRIDES...]
```

### `tabep-drug200` — Drug200 基准测试

Drug200 数据集专用基准测试脚本，支持以下选项：

| 选项 | 说明 |
|------|------|
| `--dataset` | 指定数据集源（Hugging Face repo 或本地 CSV） |
| `--cv` | 使用 5 折分层交叉验证 |
| `--output-dir` | 输出目录 |
| `--mlp-ablation` | 同时运行参数匹配的监督 MLP 基线 |
| `--training-mode` | 训练模式：`ep`, `gd`, `guided` |
| `--all-training-modes` | 一次运行所有三种训练模式 |
| `--free-steps` | 自由阶段步数（默认 2，GD 模式下更少步数） |
| `--tilelang-dynamics` | 使用 TileLang CUDA 内核加速 |
| `--device` | 运行设备（`cuda` 或 `cpu`） |
| `--no-trajectory-guidance` | 禁用轨迹引导损失 |
| `--trajectory-consistency` | 轨迹一致性权重 |
| `--trajectory-margin` | 轨迹间隔惩罚权重 |
| `--calibrate-readout` | 训练后逻辑校准 |

### `tabep-drug200-clustering` — Drug200 聚类分析

分析 Drug200 数据集特征对的聚类性能：

```bash
tabep-drug200-clustering --dataset data/raw/drug200/drug200.csv --output-dir outputs/paper2-drug200-clustering
```

实现了 K-means、模糊 C-means、DBSCAN、层次聚类和 NSGA-II 多目标聚类。

---

## 基准测试结果

### Drug200 分类

| 方法 | Macro-F1 |
|------|----------|
| **TabEP (Prototype RBF)** | **0.8389 ± 0.0618** |
| 参数匹配 MLP | 0.8296 |
| TileLang-KNN | 0.6813 |
| 决策树 | 0.7802 |
| 逻辑回归 | 0.8133 |

### UCI 基准测试套件

支持 13 个 UCI 数据集，包括：Adult、Breast Cancer、Iris、Wine、Letter Recognition、Optdigits、Pendigits、Satimage、Segment、Shuttle、Vehicle Silhouettes、Covertype、Vowel Recognition。

结果存储在 `outputs/uci-suite-*/` 目录中。

---

## 复现实验

完整的复现指南见 [`docs/reproduce.md`](docs/reproduce.md)。

### 论文 1：Drug200 分类基准测试

```bash
# 运行基准测试
uv run tabep-drug200 --dataset data/raw/drug200/drug200.csv --output-dir outputs/tabep-drug200-proto --mlp-ablation

# 种子稳定性研究
uv run python scripts/run_seed_study.py

# 效率分析
uv run python scripts/compute_efficiency.py

# 生成图表
uv run python scripts/make_paper1_figures.py

# 编译论文 PDF
cd papers/paper1
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper1 main.tex
```

### 论文 2：Drug200 聚类实验

```bash
uv run python -m tabep.reports.drug200_clustering \
  --dataset data/raw/drug200/drug200.csv \
  --output-dir outputs/paper2-drug200-clustering
```

---

## 项目结构

```
tabep/
├── main.py                         # 入口点
├── pyproject.toml                  # 项目配置与依赖
├── README.md                       # 英文文档
├── docs/
│   └── reproduce.md                # 复现指南
├── data/
│   └── raw/                        # 原始数据集
├── outputs/                        # 实验输出（Git 忽略）
├── scripts/                        # 可复现性脚本
│   ├── compute_efficiency.py       # 效率对比分析
│   ├── make_paper1_figures.py      # 论文 1 图表生成
│   └── run_seed_study.py           # 种子稳定性研究
├── papers/                         # 论文源文件与 PDFs
│   ├── paper1/                     # Paper 1: Drug200 TabEP 基准
│   └── paper2/                     # Paper 2: Drug200 聚类分析
└── src/tabep/
    ├── cli.py                      # CLI 定义（Typer）
    ├── train.py                    # Hydra + PL 训练入口
    ├── model.py                    # 深度能量模型（FHN 动力学 + 中心化 EP）
    ├── module.py                   # PyTorch Lightning 封装
    ├── tabep.py                    # TabEP / TabEnergyModel 别名
    ├── tabular.py                  # 表格数据加载与预处理
    ├── tabular_benchmark.py        # 基准测试引擎
    ├── data.py                     # 数据模块
    ├── experiment.py               # 实验管理
    ├── lit.py                      # 更多 Lightning 模块
    ├── tilelang_dynamics.py        # TileLang CUDA 动力学内核
    ├── tilelang_classifiers.py     # TileLang KNN / RBF 分类器
    ├── tilelang_utils.py           # TileLang 工具函数
    ├── conf/                       # Hydra 配置
    └── reports/                    # 报告生成代码
```

---

## 引用

如果您在研究中使用了 TabEP，请引用本仓库：

```bibtex
@software{tabep2025,
  author = {Wang, Zice},
  title = {TabEP: Tabular Equilibrium Propagation Framework},
  year = {2025},
  url = {https://github.com/ZiceWang/tabep}
}
```

---

## 许可证

本项目基于 MIT 许可证开源。
