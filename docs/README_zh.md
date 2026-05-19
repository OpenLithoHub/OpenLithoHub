<p align="center">
  <img src="assets/logo-full.png" alt="OpenLithoHub" width="280" />
</p>

# OpenLithoHub

**面向先进 EUV/曲线掩膜工艺的开源计算光刻评测与工作流工具包。**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/OpenLithoHub/OpenLithoHub/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenLithoHub/OpenLithoHub/actions)

[English Version](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/README.md)

---

## 项目简介

OpenLithoHub 为计算光刻研究提供统一的评测与工作流框架，打通从学术 Tensor 优化到工业掩膜制造的完整链路：

- **统一数据接入** — 通过单一接口加载 LithoBench、LithoSim、GAN-OPC、ICCAD'16 hotspot 等光刻数据集
- **零设置假数据** — `generate_dummy_layout` 在纯 NumPy/PyTorch 下生成确定性的、满足基本 DRC 的版图，可直接用于 CI 与 Colab
- **标准化评估指标** — EPE、PV Band、Shot Count、EUV 随机鲁棒性、Hotspot 检测 (recall / precision / F1)
- **制造合规检查** — MRC/DRC 规则检查作为一票否决指标
- **OASIS 工作流** — 从 Tensor 到 fab-ready 掩膜的端到端管线（Manhattan & Curvilinear）
- **EDA 桥接模板** — 导出 OASIS 时一并生成最小化的 Calibre nmDRC / IC Validator 规则脚本
- **论文级出图** — `openlithohub.vis` 提供 IEEE / SPIE 期刊规范的轮廓叠加图，矢量 PDF 导出
- **模型无关评测** — 任何 OPC/ILT 模型只需实现最小接口即可接入评测
- **公开排行榜** — 跨模型、数据集、工艺节点追踪 SOTA 结果

```text
┌─────────────────────────────────────────────────────────────────┐
│                       OpenLithoHub                              │
├─────────────┬──────────────┬──────────────┬───────────┬─────────┤
│   数据层    │   评测层     │   工作流层   │  可视化   │   CLI   │
│ LithoBench  │  EPE/PVBand  │ 切片/拼接    │ 论文出图  │ eval    │
│ LithoSim    │  MRC/DRC     │ 轮廓提取     │ Jupyter   │ optimize│
│ 变换工具    │  随机鲁棒性  │ OASIS 导出   │ EDA 桥接  │leaderbd │
│ 假数据生成  │  Shot Count  │ B 样条拟合   │           │         │
└─────────────┴──────────────┴──────────────┴───────────┴─────────┘
```

---

## 安装

```bash
# 核心功能（指标 + CLI）
pip install openlithohub

# 含数据集支持（HuggingFace、Parquet）
pip install openlithohub[data]

# 含完整工作流（KLayout、scipy B 样条）
pip install openlithohub[workflow]

# 含可视化与 Jupyter
pip install openlithohub[jupyter]

# 全部安装
pip install openlithohub[all]
```

**从源码安装（开发模式）：**

```bash
git clone https://github.com/OpenLithoHub/OpenLithoHub.git
cd OpenLithoHub
pip install -e ".[dev]"
```

---

## 快速开始

### 评测模型

```bash
openlithohub eval run \
  --model dummy-identity \
  --dataset lithobench \
  --data-root ./data/lithobench \
  --format table
```

输出：
```
┌──────────────────┬────────────────┐
│ Metric           │ Value          │
├──────────────────┼────────────────┤
│ epe_mean_nm      │ 0.0000         │
│ epe_max_nm       │ 0.0000         │
│ mrc_violation_rate│ 0.0000        │
│ mrc_passed       │ 1.0000         │
└──────────────────┴────────────────┘
```

### 端到端掩膜优化

```bash
openlithohub optimize run \
  --input design.oas \
  --model your-model \
  --writer mbmw \
  --node 3nm-euv \
  --drc-check \
  --output optimized.oas
```

### 作为 Python 库使用

```python
import torch
from openlithohub.benchmark.metrics import compute_epe, compute_pvband
from openlithohub.benchmark.compliance import check_mrc, check_drc

predicted = torch.load("predicted_mask.pt")
target = torch.load("target_mask.pt")

# 边缘放置误差
epe = compute_epe(predicted, target, pixel_size_nm=1.0)
print(f"EPE mean: {epe['epe_mean_nm']:.2f} nm")

# 工艺变化带
pvb = compute_pvband(predicted, defocus_range_nm=20.0)
print(f"PV Band: {pvb['pvband_mean_nm']:.2f} nm")

# 制造合规检查
mrc = check_mrc(predicted, min_width_nm=40.0, min_spacing_nm=40.0)
print(f"MRC 通过: {mrc.passed}（{mrc.violation_count} 个违规）")
```

### 注册自定义模型

```python
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

@registry.register
class MyOPCModel(LithographyModel):
    @property
    def name(self) -> str:
        return "my-opc"

    @property
    def supports_curvilinear(self) -> bool:
        return True

    def predict(self, design: torch.Tensor, **kwargs) -> PredictionResult:
        mask = my_optimization_algorithm(design)
        return PredictionResult(mask=mask)
```

---

## 架构

| 层 | 模块 | 说明 |
|----|------|------|
| **数据层** | `openlithohub.data` | 统一适配 LithoBench (.npy)、LithoSim (HuggingFace)、GAN-OPC (paired PNG)、ICCAD'16 hotspot (klayout 读 OASIS)、确定性假数据生成 |
| **评测层** | `openlithohub.benchmark` | EPE、PV Band、Shot Count、随机鲁棒性、Hotspot 检测、MRC/DRC 合规检查 |
| **模型层** | `openlithohub.models` | 抽象 `LithographyModel` 接口 + 装饰器注册机制 + 模型 Hub |
| **工作流层** | `openlithohub.workflow` | 版图解析、切片、轮廓提取、OASIS 导出、EDA 桥接模板 |
| **可视化层** | `openlithohub.vis` | 论文级 IEEE/SPIE 出图（轮廓叠加、PV Band 包络） |
| **Jupyter** | `openlithohub.jupyter` | IPython 显示助手与 `%load_ext` 魔法命令 |
| **前向模型** | `openlithohub._utils` | 可微分 Hopkins SOCS 成像、光刻胶仿真、形态学算子 |
| **CLI** | `openlithohub.cli` | `eval`、`optimize`、`leaderboard` 命令组（基于 Typer） |

---

## 评估指标

| 指标 | 说明 | 来源 |
|------|------|------|
| **EPE** | 边缘放置误差 — 预测轮廓与目标轮廓的边缘距离 | 通用标准 |
| **PV Band** | 工艺变化带 — 不同剂量/焦距下光刻胶轮廓的变化幅度 | 通用标准 |
| **Shot Count** | 掩膜写入时间代理指标（MBMW/VSB） | 工业标准 |
| **随机鲁棒性** | 蒙特卡洛光子噪声仿真，量化桥接/断线概率 | EUV 专用 |
| **MRC** | 最小线宽/间距规则检查（一票否决） | EasyMRC |
| **曲线 MRC** | 最小曲率半径 + 最小特征面积，针对 ILT 后曲线掩膜的 MBMW 可写性 | EUV 专用 |
| **DRC** | 设计规则检查：面积、缺口、线宽、间距 | OpenDRC |

---

## 支持的数据集

| 数据集 | 格式 | 工艺节点 | 任务 | 来源 |
|--------|------|----------|------|------|
| **LithoBench** | NumPy .npy | 45nm | 掩膜优化 | NeurIPS'23 |
| **LithoSim** | HuggingFace Parquet | Sub-28nm | 掩膜优化 | NeurIPS'25 |
| **GAN-OPC** | 配对 PNG | — | AI-OPC 训练 | TCAD'20 |
| **ICCAD'16 Problem C** | OASIS + CSV | N7 EUV | Hotspot 检测 | ICCAD'16 |

---

## 开发

```bash
# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/ tests/

# 类型检查
mypy src/

# 格式化
ruff format src/ tests/
```

---

## 路线图

- [x] Phase 1: 统一数据适配、EPE 指标、`eval` CLI
- [x] Phase 2: MRC 合规、Manhattan 轮廓提取、切片、Shot Count
- [x] Phase 3: OASIS 工作流、PV Band、随机鲁棒性、DRC、B 样条拟合、`optimize` CLI
- [x] Phase 4: 公开排行榜、MkDocs 文档站、文档 CI/CD
- [x] Phase 5: Web 演示平台（HuggingFace Spaces）
- [x] Phase 6: 真实 ILT 模型（LevelSet-ILT、Neural-ILT U-Net）、DTCO 工艺节点、光刻胶仿真、模型 Hub、Jupyter 集成、PyPI/Docker CI/CD
- [x] Phase 7: 论文级出图、假数据生成器、EDA 桥接模板、Colab 教程

---

## 相关项目

| 项目 | 发表 | 在生态中的角色 |
|------|------|----------------|
| LithoSim | NeurIPS'25 | Sub-28nm 工业级数据集 |
| LithoBench | NeurIPS'23 | 45nm 评测框架 |
| TorchLitho 2.0 | ASICON'25 | 可微分光刻仿真引擎 |
| [curvyILT](https://github.com/phdyang007/curvyILT) | NVIDIA arXiv'24 | GPU 加速曲线 ILT |
| EasyMRC | TODAES'25 | MRC 参考实现 |

---

## 参与贡献

参见 [CONTRIBUTING.md](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/CONTRIBUTING.md)。

---

## 免责声明

**OpenLithoHub is a purely academic, open-source project for fundamental research in computational physics and machine learning. It relies solely on publicly available datasets and published algorithms. It does not contain, nor does it seek to reverse-engineer, any proprietary commercial EDA tools or export-controlled manufacturing processes.**

OpenLithoHub 是一个纯学术性质的开源项目，专注于计算物理与机器学习的基础研究。本项目仅基于公开可用的数据集与已发表的算法，不包含、亦不试图逆向工程任何商业 EDA 工具或受出口管制的制造工艺。

## 许可证

[Apache 2.0](../LICENSE)
