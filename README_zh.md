<p align="center">
  <img src="docs/assets/logo-full.png" alt="OpenLithoHub" width="280" />
</p>

# OpenLithoHub

> ⭐ **如果这个项目对你有帮助，请点一个 Star！** 这是早期开源项目最容易被社区发现的方式，也是你能为我们做的最有价值的事。

**面向先进 EUV / 曲线掩膜工艺的开源计算光刻评测与工作流工具包。**

[![PyPI](https://img.shields.io/pypi/v/openlithohub?include_prereleases&label=PyPI)](https://pypi.org/project/openlithohub/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/OpenLithoHub/OpenLithoHub/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenLithoHub/OpenLithoHub/actions)
[![codecov](https://codecov.io/gh/OpenLithoHub/OpenLithoHub/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenLithoHub/OpenLithoHub)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/colab_byom.ipynb)

> **官网：** [openlithohub.com](https://openlithohub.com) ｜ **文档：** [docs.openlithohub.com](https://docs.openlithohub.com) ｜ **在线 Demo：** [HuggingFace Space](https://huggingface.co/spaces/OpenLithoHub/playground)

[English version / 英文版](README.md) ｜ 当前为中文版（与英文版同步）

> **关于双语：** 英文版为主，中文版按章节 1:1 同步。如出现差异，请以英文 [README.md](README.md) 为准。

---

## 项目简介

OpenLithoHub 为计算光刻研究提供统一的评测与工作流框架，打通从学术 Tensor 优化到工业掩膜制造的完整链路：

- **统一数据接入** — 通过单一接口访问 LithoBench、LithoSim、GAN-OPC、ICCAD'16 hotspot、ASAP7、FreePDK45 + NanGate OCL，以及 ORFS 布线后的 RISC-V 版图
- **标准化评估指标** — EPE、PV Band、Shot Count、EUV 随机鲁棒性、Hotspot 检测（recall / precision / F1），以及可直接接入训练循环的可微损失（SRAF 非打印惩罚、曲线 MRC 损失）
- **制造合规检查** — MRC/DRC 规则检查作为一票否决门槛
- **OASIS 工作流** — 从 Tensor 到 fab-ready 掩膜的端到端管线（Manhattan 与 Curvilinear）
- **模型无关评测** — 任何 OPC/ILT 模型只需实现最小接口即可接入评测套件
- **JIT 加速前向模型** — Hopkins/SOCS 前向模型默认用 `torch.compile` 包装，在 PyTorch 2.x 上免费获得 kernel-fusion 加速（如需关闭可使用 `--no-compile`）

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          OpenLithoHub                                   │
├─────────────┬──────────────┬──────────────┬───────────┬─────────────────┤
│  Data Layer │  Benchmark   │   Workflow   │ Vis & UX  │      CLI        │
│ LithoBench  │  EPE/PVBand  │ Tiling/Stitch│ Paper figs│ eval / optimize │
│ LithoSim    │  MRC/DRC     │ Contour Ext. │ Jupyter   │ leaderboard     │
│ Transforms  │  Stochastic  │ OASIS Export │ EDA bridge│ simulate / synth│
│ Dummy gen.  │  Shot Count  │ B-spline Fit │           │ hackathon/export│
└─────────────┴──────────────┴──────────────┴───────────┴─────────────────┘
```

---

## 安装

> OpenLithoHub 目前处于 **alpha 阶段**（PyPI 上为 `0.1.0a2`）。
> 在正式版 `0.1.0` 发布之前，请加 `--pre` 让 pip 不要跳过预发布版本。

```bash
# 核心（指标 + CLI）
pip install --pre openlithohub

# 含数据集支持（HuggingFace、Parquet）
pip install --pre 'openlithohub[data]'

# 含完整工作流（KLayout、scipy B 样条）
pip install --pre 'openlithohub[workflow]'

# 全部安装
pip install --pre 'openlithohub[all]'
```

可用的 extras：`data`、`workflow`、`models`、`jupyter`、`export`、
`docs`、`dev`，以及聚合包 `all`。可使用逗号语法组合，例如
`'openlithohub[data,workflow,jupyter]'`。

**从源码安装（开发模式）：**

```bash
git clone https://github.com/OpenLithoHub/OpenLithoHub.git
cd OpenLithoHub
pip install -e ".[dev]"
```

**Docker（开箱即用，支持 GPU）：**

每次发版都会向 GitHub Container Registry 推送预构建镜像：

```bash
# CPU
docker run --rm -v "$PWD":/data ghcr.io/openlithohub/openlithohub:latest \
  eval run --model dummy-identity --dataset lithobench --data-root /data/lithobench

# GPU（需要主机已安装 nvidia-container-toolkit）
docker run --rm --gpus all -v "$PWD":/data ghcr.io/openlithohub/openlithohub:latest \
  optimize run --input /data/design.oas --output /data/optimized.oas
```

也提供按版本号打的标签（例如 `ghcr.io/openlithohub/openlithohub:0.1`）。

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

### 作为 HTTP 微服务运行

如果 fab 端的调度器（Slurm / LSF）或既有的 C++/Perl 流程无法直接嵌入
Python，可以启动 FastAPI 引擎，用 `curl` 发起请求：

```bash
pip install "openlithohub[server]"
openlithohub serve --port 8000 &

curl -X POST http://localhost:8000/v1/optimize \
     -F "layout=@design.oas" \
     -F "model=your-model" \
     -F "writer=mbmw" \
     -o optimized.oas
```

模型常驻进程内存，重复请求不会重新加载权重。
浏览器打开 `http://localhost:8000/docs` 即可看到自动生成的 Swagger UI：
每个端点都有 JSON Schema 文档，并支持直接上传文件交互调试，
无需先写客户端代码。

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

### 论文级出图

```python
from openlithohub.vis import plot_contours

# 矢量 PDF，IEEE 单栏宽度，色盲安全配色
plot_contours(target, predicted, save_path="fig.pdf", style="ieee")
```

### 确定性假版图（用于 CI / Colab）

```python
from openlithohub.data import generate_dummy_layout

mask = generate_dummy_layout(size=256, seed=0)  # 仅依赖 numpy + torch，无需 KLayout
```

### EDA 桥接（Calibre / IC Validator）

```python
from openlithohub.workflow import BridgeRules, emit_bridge_bundle

emit_bridge_bundle(
    "optimized.oas",
    BridgeRules(min_width_nm=40.0, min_spacing_nm=40.0),
)
# 会生成 optimized.svrf、optimized.rs、optimized.bridge.md
```

### 在 Colab 上试用

`notebooks/quickstart.ipynb` 教程在 Colab 默认环境上即可端到端跑通：
安装、生成版图、打分、产出论文级图，整个流程三分钟以内完成。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/quickstart.ipynb)

如果你想把自己的模型接入评测框架，使用 BYOM 教程 —— 它会演示如何继承
`LithographyModel`、跑完标准指标套件，并组织一份合格的 Leaderboard 提交。

[![Open BYOM In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/colab_byom.ipynb)

---

## 架构

| 层 | 模块 | 说明 |
|----|------|------|
| **数据层** | `openlithohub.data` | 统一适配 LithoBench (.npy)、LithoSim (HuggingFace)、GAN-OPC (paired PNG)、ICCAD'16 hotspot (klayout 读 OASIS) |
| **评测层** | `openlithohub.benchmark` | EPE、PV Band、Shot Count、随机鲁棒性、Hotspot 检测、MRC/DRC 合规检查 |
| **模型层** | `openlithohub.models` | 抽象 `LithographyModel` 接口 + 装饰器注册机制 |
| **工作流层** | `openlithohub.workflow` | 版图解析、切片、轮廓提取（Manhattan / Curvilinear）、OASIS 导出 |
| **CLI** | `openlithohub.cli` | `eval`、`optimize`、`leaderboard`、`simulate`、`synth`、`hackathon`、`export` 命令组（基于 Typer） |

---

## 评估指标

| 指标 | 说明 | 来源 |
|------|------|------|
| **EPE** | 边缘放置误差 — 预测轮廓与目标轮廓边缘的距离 | 通用标准 |
| **PV Band** | 工艺变化带 — 不同剂量/焦距窗口下光刻胶轮廓的变化 | 通用标准 |
| **Shot Count** | 掩膜写入时间代理（MBMW 与 VSB 写机） | 工业标准 |
| **随机鲁棒性** | 蒙特卡洛光子噪声仿真，估算桥接 / 断线概率 | EUV 专用 |
| **MRC** | 最小线宽 / 间距规则检查（一票否决） | EasyMRC |
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

## 基线参考

下表是内置模型在 8 个 64×64 合成版图（square、line、line/space、T、L、cross、
contacts、dense lines）上的参考成绩，由 `scripts/generate_baselines.py`
端到端生成并落盘到 `baselines/`。方法学、Hopkins 前向模型、复现指引详见
[`docs/benchmarks.md`](docs/benchmarks.md)。

| 模型 | EPE 平均 (nm) | EPE 最大 (nm) | PVB 平均 (nm) | MRC 通过率 |
|---|---|---|---|---|
| `dummy-identity` | 0.000 | 0.000 | 2.140 | 0% |
| `rule-based-opc`（解析式 OPC bias） | 0.530 | 1.414 | 2.487 | 0% |
| `levelset-ilt`（Gaussian PSF，200 次迭代） | 0.036 | 0.250 | 2.128 | 0% |
| `neural-ilt`（未训练的 U-Net） | 15.074 | 24.637 | 2.497 | 100% |

每个模式的细分结果见 [`baselines/results.md`](baselines/results.md)。

本地复现：

```bash
python scripts/generate_baselines.py --synthetic --limit 8 --output baselines/
```

---

## 光学前向模型

OpenLithoHub 提供两个可微分的前向模型，全部用纯 PyTorch 实现，所以整个 ILT
循环端到端可自动求导：

| 模型 | 模块 | 备注 |
|---|---|---|
| Gaussian PSF | `openlithohub._utils.forward_model.simulate_aerial_image` | 单 Gaussian 卷积；适合测试与小尺寸网格的廉价默认实现 |
| Hopkins SOCS | `openlithohub._utils.simulate_aerial_image_hopkins` | 通过 SVD 截断的 Sum-Of-Coherent-Systems 实现部分相干成像；支持环形 / 偶极 / 圆形照明 |

将 `LevelSetILTModel` 切换到 Hopkins：

```python
from openlithohub._utils import HopkinsParams
from openlithohub.models.levelset_ilt import LevelSetILTModel

model = LevelSetILTModel(
    iterations=200,
    forward_model="hopkins",
    hopkins_params=HopkinsParams(
        wavelength_nm=193.0, na=1.35, sigma=0.7, num_kernels=24, pixel_size_nm=2.0,
    ),
)
```

---

## 开发

```bash
# 运行测试
pytest tests/ -v

# Lint
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
- [x] 里程碑 4：公开 Leaderboard、MkDocs 文档站、文档 CI/CD
- [x] 里程碑 5：Web 演示平台（HuggingFace Spaces）
- [x] 里程碑 6：真实 ILT 模型（LevelSet-ILT、Neural-ILT U-Net）、DTCO 工艺节点、光刻胶仿真、模型 Hub、Jupyter 集成、PyPI/Docker CI/CD
- [x] 里程碑 7：论文级出图、假版图生成器、EDA 桥接模板、Colab quickstart
- [x] 里程碑 8：多阶段 KLayout Docker、面向 AI 工程师的术语指南、Auto-Leaderboard CI、社区章程（Discord）、v0.1 发布公告
- [x] 里程碑 9：PDK 感知合成版图生成器、厂商中立 Simulator Hook API、EUV 3D-mask 阴影代理、蒙特卡洛失效指标、Mini-Hackathon (2026-Q3)、RFC 0001（Layout-MAE）+ RFC 0002（Layout Tokens）
- [x] 里程碑 10：真实 PDK 接入 — ASAP7 标准单元、FreePDK45 + NanGate OCL、ORFS 布线后的 RISC-V mock-alu（issue [#4](https://github.com/OpenLithoHub/OpenLithoHub/issues/4)）
- [x] 里程碑 11：标准 MRC 规则集 schema（RFC 0003）、实测 source / Zernike-pupil I/O、Calibre/CSV gauge 解析器、`openlithohub export` CLI（ONNX / TorchScript / TensorRT-ready）、`--compile` 默认开启、首个 PyPI 版本（`openlithohub-0.1.0a2`）

---

## 相关项目

| 项目 | 发表 | 在生态中的角色 |
|------|------|----------------|
| LithoSim | NeurIPS'25 | Sub-28nm 工业级数据集 |
| LithoBench | NeurIPS'23 | 45nm 评测框架 |
| TorchLitho 2.0 | ASICON'25 | 可微分光刻仿真器 |
| [curvyILT](https://github.com/phdyang007/curvyILT) | NVIDIA arXiv'24 | GPU 加速曲线 ILT |
| EasyMRC | TODAES'25 | MRC 参考实现 |

---

## 参与贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 社区

![Status](https://img.shields.io/badge/Discord-launching%20soon-5865F2?logo=discord&logoColor=white)

OpenLithoHub 的 **Discord** 服务器将于 **2026-Q3** 启动 —— 频道涵盖
模型讨论、物理仿真、求助与展示，是讨论模型设计、可复现性与基准的
主要场所。

希望邀请链接上线后第一时间收到通知？请
**[创建带 `community` 标签的 Issue](https://github.com/OpenLithoHub/OpenLithoHub/issues/new?labels=community&title=Community+launch+notification)**
或 watch 本仓库。社区章程、频道结构、礼仪与 onboarding 流程详见
[docs/community.md](docs/community.md)。

📣 **阅读发布公告：**
[v0.1 发布稿](docs/announcements/2026-05-launch.md) — 含可粘贴的
X / LinkedIn / 知乎 / HuggingFace Forum 多平台文案。

🏆 **Mini-Hackathon 将于 2026-Q3 启动** ——
[章程与规则](docs/hackathon.md)。固定 EPE 目标、冻结测试集、
MRC/DRC 硬性门槛，独立 leaderboard track。

---

## 免责声明

**OpenLithoHub is a purely academic, open-source project for fundamental research in computational physics and machine learning. It relies solely on publicly available datasets and published algorithms. It does not contain, nor does it seek to reverse-engineer, any proprietary commercial EDA tools or export-controlled manufacturing processes.**

OpenLithoHub 是纯学术性质的开源项目，专注于计算物理与机器学习的基础研究。
本项目仅基于公开可用的数据集与已发表的算法，不包含、亦不试图逆向工程
任何商业 EDA 工具或受出口管制的制造工艺。

## 许可证

OpenLithoHub 采用分层许可模型：

- **代码** — [Apache License 2.0](LICENSE)
- **文档** — [CC-BY-SA 4.0](LICENSE-DOCS)
- **数据集** — 各数据集保留原始许可证；OpenLithoHub 仅提供适配器，
  不分发数据本身。详见 [DATA-LICENSES.md](DATA-LICENSES.md)。
- **第三方组件** — 参见 [NOTICE](NOTICE)。

你可以在开源许可证下自由地将 OpenLithoHub 用于商业用途
（仅需保留署名与 `NOTICE` 文件）。如果需要无署名的商业授权、专利保障
或带 SLA 的支持服务，参见 [COMMERCIAL-USE.md](COMMERCIAL-USE.md)。

学术引用方式见 [CITATION.cff](CITATION.cff)。
贡献者请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 以及
[贡献者许可协议](CLA-INDIVIDUAL.md)。安全问题：[SECURITY.md](SECURITY.md)。
