# OpenLithoHub

**面向先进 EUV/曲线掩膜工艺的开源计算光刻评测与工作流工具包。**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/OpenLithoHub/OpenLithoHub/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenLithoHub/OpenLithoHub/actions)

[English Version](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/README.md)

---

## 项目简介

OpenLithoHub 为计算光刻研究提供统一的评测与工作流框架，打通从学术 Tensor 优化到工业掩膜制造的完整链路：

- **统一数据接入** — 通过单一接口加载 LithoBench、LithoSim 等光刻数据集
- **标准化评估指标** — EPE、PV Band、Shot Count、EUV 随机鲁棒性
- **制造合规检查** — MRC/DRC 规则检查作为一票否决指标
- **OASIS 工作流** — 从 Tensor 到 fab-ready 掩膜的端到端管线（Manhattan & Curvilinear）
- **模型无关评测** — 任何 OPC/ILT 模型只需实现最小接口即可接入评测

```text
┌─────────────────────────────────────────────────────────┐
│                    OpenLithoHub                          │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  数据层     │  评测层      │   工作流层   │    CLI      │
│ LithoBench  │  EPE/PVBand  │ 切片/拼接    │ eval        │
│ LithoSim    │  MRC/DRC     │ 轮廓提取     │ optimize    │
│ 变换工具    │  随机鲁棒性  │ OASIS 导出   │             │
│             │  Shot Count  │ B 样条拟合   │             │
└─────────────┴──────────────┴──────────────┴─────────────┘
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
openlithohub eval \
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
openlithohub optimize \
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
| **数据层** | `openlithohub.data` | 统一适配 LithoBench (.npy)、LithoSim (HuggingFace)，支持分辨率对齐 |
| **评测层** | `openlithohub.benchmark` | EPE、PV Band、Shot Count、随机鲁棒性、MRC/DRC 合规检查 |
| **模型层** | `openlithohub.models` | 抽象 `LithographyModel` 接口 + 装饰器注册机制 |
| **工作流层** | `openlithohub.workflow` | 版图解析、切片、轮廓提取（Manhattan/Curvilinear）、OASIS 导出 |
| **CLI** | `openlithohub.cli` | `eval` 与 `optimize` 命令（基于 Typer） |

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

| 数据集 | 格式 | 工艺节点 | 来源 |
|--------|------|----------|------|
| **LithoBench** | NumPy .npy | 45nm | NeurIPS'23 |
| **LithoSim** | HuggingFace Parquet | Sub-28nm | NeurIPS'25 |

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
- [ ] Phase 4: 公开排行榜、上游项目集成
- [ ] Phase 5: Web 演示平台（HuggingFace Spaces）

---

## 相关项目

| 项目 | 发表 | 在生态中的角色 |
|------|------|----------------|
| [LithoSim](https://github.com/) | NeurIPS'25 | Sub-28nm 工业级数据集 |
| [LithoBench](https://github.com/) | NeurIPS'23 | 45nm 评测框架 |
| [TorchLitho 2.0](https://github.com/) | ASICON'25 | 可微分光刻仿真引擎 |
| [curvyILT](https://github.com/) | NVIDIA arXiv'24 | GPU 加速曲线 ILT |
| [EasyMRC](https://github.com/) | TODAES'25 | MRC 参考实现 |

---

## 参与贡献

参见 [CONTRIBUTING.md](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/CONTRIBUTING.md)。

---

## 免责声明

**OpenLithoHub is a purely academic, open-source project for fundamental research in computational physics and machine learning. It relies solely on publicly available datasets and published algorithms. It does not contain, nor does it seek to reverse-engineer, any proprietary commercial EDA tools or export-controlled manufacturing processes.**

OpenLithoHub 是一个纯学术性质的开源项目，专注于计算物理与机器学习的基础研究。本项目仅基于公开可用的数据集与已发表的算法，不包含、亦不试图逆向工程任何商业 EDA 工具或受出口管制的制造工艺。

## 许可证

[Apache 2.0](../LICENSE)
