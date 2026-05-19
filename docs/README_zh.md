<p align="center">
  <img src="assets/logo-full.png" alt="OpenLithoHub" width="280" />
</p>

# OpenLithoHub

**面向先进 EUV/曲线掩膜工艺的开源计算光刻评测与工作流工具包。**

[![PyPI](https://img.shields.io/pypi/v/openlithohub?include_prereleases&label=PyPI)](https://pypi.org/project/openlithohub/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/OpenLithoHub/OpenLithoHub/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenLithoHub/OpenLithoHub/actions)

[English Version](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/README.md)

---

## 项目简介

OpenLithoHub 为计算光刻研究提供统一的评测与工作流框架，打通从学术 Tensor 优化到工业掩膜制造的完整链路：

- **统一数据接入** — 通过单一接口加载 LithoBench、LithoSim、GAN-OPC、ICCAD'16 hotspot、ASAP7、FreePDK45 + NanGate OCL、ORFS 布线后的 RISC-V 版图等光刻数据集
- **零设置假数据** — `generate_dummy_layout` 在纯 NumPy/PyTorch 下生成确定性的、满足基本 DRC 的版图，可直接用于 CI 与 Colab
- **PDK 感知合成版图生成器** — `openlithohub.synth` 在 FreePDK45 / ASAP7 上生成 SRAM、接触阵列与随机金属布线，按构造满足 MRC，含 `openlithohub synth` CLI 与 RFC 0001/0002 占位的扩散模型生成路径
- **标准化评估指标** — EPE、PV Band、Shot Count、EUV 随机鲁棒性、EUV 3D-mask 阴影代理、Hotspot 检测 (recall / precision / F1)
- **制造合规检查** — MRC/DRC 规则检查作为一票否决指标
- **OASIS 工作流** — 从 Tensor 到 fab-ready 掩膜的端到端管线（Manhattan & Curvilinear）
- **EDA 桥接模板** — 导出 OASIS 时一并生成最小化的 Calibre nmDRC / IC Validator 规则脚本
- **厂商中立的 Simulator Hook API** — `BaseSimulator` ABC，内置 Hopkins 参考实现，Calibre nmOPC / Tachyon 提供配置校验占位
- **论文级出图** — `openlithohub.vis` 提供 IEEE / SPIE 期刊规范的轮廓叠加图，矢量 PDF 导出
- **模型无关评测** — 任何 OPC/ILT 模型只需实现最小接口即可接入评测
- **公开排行榜** — 跨模型、数据集、工艺节点追踪 SOTA 结果，支持 `track` 字段区分 Hackathon 与常规提交

**OpenLithoHub 架构总览：**

| 数据层 | 评测层 | 工作流层 | 可视化 | CLI |
|--------|--------|----------|--------|-----|
| LithoBench | EPE / PVBand | 切片 / 拼接 | 论文出图 | `eval` |
| LithoSim | MRC / DRC | 轮廓提取 | Jupyter | `optimize` |
| 变换工具 | 随机鲁棒性 | OASIS 导出 | EDA 桥接 | `leaderboard` |
| 假数据生成 | Shot Count | B 样条拟合 | — | — |

---

## 安装

> OpenLithoHub 目前处于 **alpha 阶段**（PyPI 上是 `0.1.0a2`）。在
> 正式版 `0.1.0` 发布之前，请加 `--pre` 让 pip 安装预发布版本。

```bash
# 核心功能（指标 + CLI）
pip install --pre openlithohub

# 含数据集支持（HuggingFace、Parquet）
pip install --pre 'openlithohub[data]'

# 含完整工作流（KLayout、scipy B 样条）
pip install --pre 'openlithohub[workflow]'

# 含可视化与 Jupyter
pip install --pre 'openlithohub[jupyter]'

# 全部安装
pip install --pre 'openlithohub[all]'
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
| **合成层** | `openlithohub.synth` | PDK 感知规则生成器（FreePDK45 / ASAP7）+ 扩散模型生成路径占位 |
| **评测层** | `openlithohub.benchmark` | EPE、PV Band、Shot Count、随机鲁棒性、EUV 3D-mask 阴影代理、Monte Carlo 失效、Hotspot 检测、MRC/DRC 合规检查 |
| **模型层** | `openlithohub.models` | 抽象 `LithographyModel` 接口 + 装饰器注册机制 + 模型 Hub |
| **仿真器层** | `openlithohub.simulators` | 厂商中立 `BaseSimulator` ABC，内置 Hopkins 适配器，Calibre / Tachyon 占位适配器 |
| **工作流层** | `openlithohub.workflow` | 版图解析、切片、轮廓提取、OASIS 导出、EDA 桥接模板 |
| **可视化层** | `openlithohub.vis` | 论文级 IEEE/SPIE 出图（轮廓叠加、PV Band 包络） |
| **Jupyter** | `openlithohub.jupyter` | IPython 显示助手与 `%load_ext` 魔法命令 |
| **前向模型** | `openlithohub._utils` | 可微分 Hopkins SOCS 成像、光刻胶仿真、形态学算子 |
| **CLI** | `openlithohub.cli` | `eval`、`optimize`、`leaderboard`、`synth`、`simulate` 命令组（基于 Typer） |

---

## 评估指标

| 指标 | 说明 | 来源 |
|------|------|------|
| **EPE** | 边缘放置误差 — 预测轮廓与目标轮廓的边缘距离 | 通用标准 |
| **PV Band** | 工艺变化带 — 不同剂量/焦距下光刻胶轮廓的变化幅度 | 通用标准 |
| **Shot Count** | 掩膜写入时间代理指标（MBMW/VSB） | 工业标准 |
| **随机鲁棒性** | 蒙特卡洛光子噪声仿真，量化桥接/断线概率 | EUV 专用 |
| **EUV 3D-mask 阴影代理** | 各向异性卷积近似 EUV 反射式掩膜 3D 误差，绕开完整 Maxwell 求解 | EUV 专用 |
| **Monte Carlo 失效率** | 调用任意已注册仿真器后端，用蒙特卡洛抽样估算 bridge/break 概率 | EUV 专用 |
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
| **ASAP7 标准单元** | GDSII (klayout) | 7nm 预测 | PDK 感知 OPC | The-OpenROAD-Project/asap7 |
| **FreePDK45 + NanGate OCL** | GDSII (klayout) | 45nm 预测 | PDK 感知 OPC | mflowgen/freepdk-45nm |
| **ORFS 布线 ASAP7** | GDSII (klayout) | 7nm | RISC-V 切片热点 | OpenROAD-flow-scripts |

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

- [x] 里程碑 1：统一数据适配、EPE 指标、`eval` CLI
- [x] 里程碑 2：MRC 合规、Manhattan 轮廓提取、切片、Shot Count
- [x] 里程碑 3：OASIS 工作流、PV Band、随机鲁棒性、DRC、B 样条拟合、`optimize` CLI
- [x] 里程碑 4：公开排行榜、MkDocs 文档站、文档 CI/CD
- [x] 里程碑 5：Web 演示平台（HuggingFace Spaces）
- [x] 里程碑 6：真实 ILT 模型（LevelSet-ILT、Neural-ILT U-Net）、DTCO 工艺节点、光刻胶仿真、模型 Hub、Jupyter 集成、PyPI/Docker CI/CD
- [x] 里程碑 7：论文级出图、假数据生成器、EDA 桥接模板、Colab 教程
- [x] 里程碑 8：多阶段 KLayout Docker、面向 AI 工程师的术语指南、Auto-Leaderboard CI、社区章程（Discord）、v0.1 发布公告
- [x] 里程碑 9：PDK 感知合成版图生成器、厂商中立 Simulator Hook API、EUV 3D-mask 阴影代理、蒙特卡洛失效指标、Mini-Hackathon (2026-Q3)、RFC 0001 (Layout-MAE) + RFC 0002 (Layout Tokens)
- [x] 里程碑 10：真实 PDK 接入 — ASAP7 标准单元、FreePDK45 + NanGate OCL、ORFS 布线后的 RISC-V mock-alu（issue [#4](https://github.com/OpenLithoHub/OpenLithoHub/issues/4)）

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

## 社区

我们将在 **2026-Q3** 启动单一全球 **Discord** 服务器（English-first，欢迎多语言），频道涵盖模型讨论、物理仿真、帮助与展示。在邀请链接上线前，可以
[创建带 `community` 标签的 Issue](https://github.com/OpenLithoHub/OpenLithoHub/issues/new?labels=community&title=Community+launch+notification)
订阅通知，或 watch 本仓库。社区章程详见
[docs/community.md](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/community.md)。

🏆 **Mini-Hackathon (2026-Q3)** —
[章程与规则](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/hackathon.md)：
固定 EPE 目标、冻结测试集、MRC/DRC 硬性门槛，独立 leaderboard track。

📣 **v0.1 发布公告** —
[完整发布稿](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/announcements/2026-05-launch.md)，
含可粘贴的 X / LinkedIn / 知乎 / HuggingFace Forum 多平台文案。

📐 **路线图 RFC**：
- [RFC 0001 — Layout-MAE 基础模型](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/rfcs/0001-base-model.md)
- [RFC 0002 — Layout Tokens（无损多边形顶点 tokeniser）](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/rfcs/0002-layout-tokens.md)

---

## 免责声明

**OpenLithoHub is a purely academic, open-source project for fundamental research in computational physics and machine learning. It relies solely on publicly available datasets and published algorithms. It does not contain, nor does it seek to reverse-engineer, any proprietary commercial EDA tools or export-controlled manufacturing processes.**

OpenLithoHub 是一个纯学术性质的开源项目，专注于计算物理与机器学习的基础研究。本项目仅基于公开可用的数据集与已发表的算法，不包含、亦不试图逆向工程任何商业 EDA 工具或受出口管制的制造工艺。

## 许可证

[Apache 2.0](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/LICENSE)
