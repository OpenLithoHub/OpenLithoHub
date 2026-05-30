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

- **统一数据接入** — 通过单一接口访问 LithoBench、LithoSim、GAN-OPC、ICCAD'16 hotspot、ASAP7、FreePDK45 + NanGate OCL，以及 ORFS 布线后的 RISC-V 版图；`workflow.parse_layout` 支持 OASIS / GDSII / DEF / LEF 输入
- **标准化评估指标** — EPE（掩膜对掩膜或经前向仿真的 wafer 级）、L2 wafer error（Neural-ILT 标准）、PV Band、Shot Count、EUV 随机鲁棒性 + imec 风格的逐类缺陷率、Hotspot 检测（recall / precision / F1），以及可直接接入训练循环的可微损失（SRAF 非打印惩罚、曲线 MRC 损失）
- **制造合规检查** — MRC/DRC 规则检查作为一票否决门槛
- **OASIS / GDSII 工作流** — 从 Tensor 到 fab-ready 掩膜的端到端管线（Manhattan 与 Curvilinear）；ICCAD'13 contest gauge IO + Calibre `.gg` / CSV gauge 解析；ONNX / TorchScript 导出，CI 中含 onnxruntime 冒烟测试
- **模型无关评测** — 任何 OPC/ILT 模型只需实现最小接口即可接入评测套件
- **Opt-in 扩散光刻胶** — CAR（化学放大光刻胶）高斯酸扩散模型，通过 `--resist-diffusion-nm` 控制（默认 `0.0` = 传统 CTR）。提升 EPE/PVB 真实感但产生**不可比较**的数值；排行榜提交时禁用
- **Design→litho→DFM 闭环 CLI** — `openlithohub flow run` 接收 DEF/GDS/OAS 或 ORFS 产品目录，自动切片、运行光刻前向仿真，输出聚合可制造性报告（EPE、PV Band、DRC、MRC），支持可配置的按 PDK 层映射
- **可选物理插件** — DiffNano（严格 EM 求解器：RCWA / FDTD / FDFD + 可校准光刻胶模型）和 DiffCFD（Dill/Mack 光刻求解器 + 旋涂求解器 + 联合工艺优化）作为 opt-in `[diffnano]` / `[diffcfd]` extras 提供；核心安装不需要其中任何一个
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
`docs`、`dev`、`diffnano`、`diffcfd`、`plugins`（= 两者）以及聚合包 `all`。
可使用逗号语法组合，例如 `'openlithohub[data,workflow,jupyter]'`。

```bash
# 可选物理插件（早期研究阶段，未经第三方验证）
pip install --pre 'openlithohub[diffnano]'   # 纳米光子 EM 求解器 + 光刻胶
pip install --pre 'openlithohub[diffcfd]'    # 基于 CFD 的光刻 + 旋涂求解器
pip install --pre 'openlithohub[plugins]'    # 安装两者
```

> **注意：** DiffNano 和 DiffCFD 是可选插件，提供研究级物理后端。
> 两者均自述为早期个人研究项目，无外部用户、无第三方验证、不声称生产就绪。
> 不安装任何插件不影响核心功能。

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

### 闭环 design→litho→DFM 报告

```bash
openlithohub flow run design.gds \
  --pdk asap7 --layer metal1 \
  --node 45nm --tile-nm 2000 \
  --drc --mrc \
  --output report.json
```

支持独立的 GDS / OAS / DEF 文件或 ORFS 产品目录。
按 PDK 的层映射可配置（asap7、freepdk45、orfs_asap7、sky130 或自定义 JSON 文件）。
报告将切片级 EPE、PV Band、DRC、MRC 聚合为单个 JSON 摘要。

### 启用扩散光刻胶（opt-in）

```bash
# 默认：CTR（恒定阈值光刻胶），threshold=0.225 — 可比较的数值
openlithohub simulate run --input mask.npy --resist-diffusion-nm 0.0

# Opt-in：CAR 高斯酸扩散 — 更真实但不可比较
openlithohub simulate run --input mask.npy --resist-diffusion-nm 20.0
```

> 评分默认仍为 **CTR 无扩散、threshold = 0.225**。启用酸扩散（或任何插件 EM/光刻胶后端）
> 产生**不可比较**的数值，**排行榜提交时禁用**。

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

面向对象的门面 —— `Mask` / `LitheEngine` / `Report` —— 是从版图文件到打分结果最短的一条路径：

```python
from openlithohub import Mask, LitheEngine

mask      = Mask.from_oasis("design.oas", layer="1:0", pixel_size_nm=1.0)
engine    = LitheEngine(model="neural-ilt", node="3nm-euv")
optimized = engine.optimize(mask)
report    = engine.evaluate(optimized, target=mask)

print(report.epe_mean_nm, report.pvband_mean_nm, report.drc_violations)
optimized.to_oasis("optimized.oas")
```

需要更细粒度控制时，函数式 API 一直可用：

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
| **API 门面** | `openlithohub.api` | 面向对象的入口（`Mask`、`LitheEngine`、`Report`），同时在包根重新导出 |
| **数据层** | `openlithohub.data` | 统一适配 LithoBench (.npy)、LithoSim (HuggingFace)、GAN-OPC (paired PNG)、ICCAD'16 hotspot (klayout 读 OASIS) |
| **评测层** | `openlithohub.benchmark` | EPE（掩膜级与 wafer-sim 级）、L2 wafer error、PV Band、Shot Count、随机鲁棒性 + 逐类缺陷率、Hotspot 检测、MRC/DRC 合规检查 |
| **模型层** | `openlithohub.models` | 抽象 `LithographyModel` 接口 + 装饰器注册机制 |
| **工作流层** | `openlithohub.workflow` | 版图解析（OASIS / GDSII / DEF / LEF）、切片、轮廓提取（Manhattan / Curvilinear）、OASIS / GDSII 导出、OpenAccess layer-purpose 工具 |
| **仿真器** | `openlithohub.simulators` | 前向模型注册表（`register_simulator`）、Hopkins/Gaussian 内置、插件 EM 后端（RCWA/FDTD/FDFD） |
| **插件层** | `openlithohub.plugins` | 可选 DiffNano（EM + 光刻胶）和 DiffCFD（光刻 + 旋涂 + 联合优化）后端 |
| **CLI** | `openlithohub.cli` | `eval`、`optimize`、`leaderboard`、`simulate`、`flow`、`synth`、`hackathon`、`export` 命令组（基于 Typer） |

## 可选物理插件

OpenLithoHub 通过插件系统支持可选物理后端。核心安装不需要其中任何一个。

| 插件 | 提供的功能 | 安装 extra |
|------|-----------|------------|
| **DiffNano** | 严格 EM 求解器（RCWA / FDTD / FDFD）+ 可校准光刻胶模型（酸扩散、PEB、显影对比度）— 注册为 `diffnano_rcwa`、`diffnano_fdtd2d`、`diffnano_fdfd2d` 后端 | `[diffnano]` |
| **DiffCFD** | 可微稳态 CFD — Dill/Mack 光刻求解器、Meyerhofer 旋涂求解器、联合工艺优化（`optimize_joint_process`） | `[diffcfd]` |

```bash
pip install --pre 'openlithohub[plugins]'   # 安装两者
```

**注意事项：**
- 两个插件均为早期研究项目，无外部用户或第三方验证。请勿用于生产决策。
- 插件 EM/光刻胶后端产生**不可比较**的指标数值。内置 Hopkins + CTR（threshold `0.225`）仍是排行榜提交的唯一路径。
- 可选性基于未验证状态、安装占用空间和独立迭代节奏，而非依赖权重（两者均为 PyTorch 原生）。

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

> **扩散光刻胶：** EPE 和 PV Band 可选择通过 CAR 酸扩散模型运行（`--resist-diffusion-nm`）。
> 评分默认仍为 CTR、threshold `0.225`；启用扩散产生**不可比较**的数值，排行榜提交时禁用。
> 绝对晶圆预测仍需用户校准的代工厂保密参数——框架是基准相对的，非绝对预测性的。

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

## 性能与基准测试

> 所有数据均由内置基准测试脚本在真实硬件上运行获得。没有任何数据是通过估算、外推或"合理假设"得出的。
> 方法学、前向模型配置和逐模式细分见 [`docs/benchmarks.md`](docs/benchmarks.md)。

### 模型质量 — synthetic-8（表 1）

8 个手工构造的 64×64 版图（方形、水平线、线/间距、T 形、L 形、十字、
接触孔、密集线），8 nm/px 分辨率，使用共享的 `HopkinsSimulator` 评分
（波长 / NA / 阈值在每一行中完全一致）。

| 模型 | EPE 均值 (nm) | Wafer EPE (nm) | L2 (px) | PVB 均值 (nm) | MRC 通过 |
|---|---|---|---|---|---|
| `dummy-identity` | 0.000 | 4.529 | 299.9 | 18.340 | 88% |
| `rule-based-opc` | 4.242 | 7.786 | 356.4 | 16.000 | 88% |
| `levelset-ilt`（200 次迭代） | 0.322 | 4.482 | 294.9 | 18.516 | 75% |
| `openilt`（MOSAIC L2+PVB） | 0.000 | 4.529 | 299.9 | 18.340 | 88% |
| `neural-ilt`（v0.1 公开权重） | 0.000 | 4.529 | 299.9 | 18.340 | 88% |

- **`levelset-ilt`** 是唯一在 wafer L2 上优于 identity 的模型（294.9 vs 299.9），代价是 MRC 通过率较低（75%）——梯度下降生成的掩膜包含违反 `min_width_nm=40` 的窄特征。
- **`openilt`** 和 **`neural-ilt`** 在这些简单版图上收敛到 identity —— 它们内部的前向模型已能完整复现目标版图，无需修正。当输入具有非平凡倒角的真实版图时两者会产生差异。
- **`rule-based-opc`** 有意偏离目标掩膜（mask-EPE 升至 4.242 nm），但降低了 PVB（16.0 vs 18.3 nm）——这是 bias-OPC 预期中的权衡。
- **`dummy-identity`** 是*下界*，不是竞争者——mask-EPE 为零是结构性的（design == target），但 wafer-EPE 和 L2 因衍射而非零。

### 模型质量 — ICCAD16 testcase1（表 2）

真实 EUV 版图（1.9 µm × 1.5 µm，475×375 px，4 nm/px），来自
[Yang2016_ICCAD16Bench](https://github.com/phdyang007/ICCAD16-N7M2EUV)。
EPE/L2 列省略——该数据集不包含参考 OPC 掩膜。

| 模型 | PVB 均值 (nm) | PVB 最大 (nm) | MRC 违规率 |
|---|---|---|---|
| `dummy-identity` | 14.82 | 64.0 | 15.93% |
| `rule-based-opc` | 12.39 | 32.0 | 14.89% |
| `levelset-ilt` | 10.49 | 32.0 | 0.97% |
| `openilt` | 14.82 | 64.0 | 15.93% |
| `neural-ilt`（v0.1） | 0.00 | 0.0 | 0% |
| `gan-opc`（v0.1） | 10.97 | 48.0 | 8.48% |
| `gan-opc`（v0.2） | 11.76 | 64.0 | 5.99% |

- **`levelset-ilt`** 取得最低 PVB（10.49 nm），MRC 违规率接近零（0.97%）——排名与 synthetic-8 一致。
- **`neural-ilt` v0.1** 显示退化的结果（PVB 为零、零违规），因为仅在合成 64-px 样本上训练的权重在 475×375 网格上产生近似空白的掩膜——这是**分布外失效**，不是有效成绩。
- **`gan-opc` v0.2 vs v0.1**：MRC 违规率降低 29%（8.48%→5.99%），但 PVB 上升 7%（10.97→11.76 nm），体现了 Hopkins 前向模型参与训练循环的权衡。

### 与已发表论文的交叉对照（表 3）

将 OpenLithoHub 的复现结果与原论文报告的数据进行比较。
**非严格同条件对比，仅供参考**——测试版图、工艺节点和评测方法均有差异。
论文数据来自 ICCAD 2013 竞赛基准（10 个 clip，32 nm M1，1024 nm × 1024 nm，1 nm/px）；
OpenLithoHub 数据来自 ICCAD16 testcase1（7 nm EUV，475 × 375 px，4 nm/px）——属于本质上不同的基准。

| 方法 | 来源 | 论文报告（ICCAD13） | OpenLithoHub 复现（ICCAD16） | 注意事项 |
|---|---|---|---|---|
| MOSAIC (SGD, L2+PVB) | Gao et al., DAC 2014 (DOI [6881379](https://ieeexplore.ieee.org/document/6881379)) | PVB 均值 ≈ 56 890 nm²，TAT ≈ 1703 s | PVB 14.82 nm（identity） | OpenILT 在干净版图上收敛到 identity；ICCAD13 与 ICCAD16 指标不可直接比较 |
| Neural-ILT (U-Net) | Jiang et al., ICCAD 2020 (DOI [3415704](https://dl.acm.org/doi/10.1145/3400302.3415704)) | L2 均值 38 504 nm²，TAT ≈ 11 s（GPU） | N/A（ICCAD16 上退化） | v0.1 仅在合成数据上训练；论文使用 2048×2048 掩膜 + GPU |
| GAN-OPC (PGAN-OPC) | Yang et al., DAC 2018 / TCAD 2020 (DOI [3196056](https://dl.acm.org/doi/10.1145/3195970.3196056)) | L2 均值 39 949 nm²，TAT ≈ 371 s | PVB 10.97 nm，MRC 违规 8.48% | 论文报告 L2（nm²）；我们报告 PVB（nm）——指标与版图不同 |
| curvyILT | Yang & Ren, ISPD 2025 / arXiv [2411.07311](https://arxiv.org/abs/2411.07311) | MSE 均值 25 991 nm²，2.11 s/clip（RTX A6000） | —（尚未集成） | 外部 GPU 工具；ICCAD13 上已发表的学术 SOTA |

### 优化吞吐量（表 4）

所有计时使用 `perf_counter_ns` 测量，采样期间 `gc.disable()`。
前向模型/指标：100 个采样点；完整模型预测：20 个采样点。
报告中位数与 P99。仅 CPU（无 GPU）。

| 基准测试 | 网格 | 中位数 | P99 | 设备 |
|---|---|---|---|---|
| `forward_gaussian` | 64×64 | 238 µs | 549 µs | AMD 5600G CPU |
| `forward_gaussian` | 256×256 | 804 µs | 1.2 ms | AMD 5600G CPU |
| `forward_hopkins` | 64×64 | 2.1 ms | 2.7 ms | AMD 5600G CPU |
| `forward_hopkins` | 256×256 | 6.5 ms | 9.5 ms | AMD 5600G CPU |
| `metric_epe` | 64×64 | 541 µs | 941 µs | AMD 5600G CPU |
| `metric_pvband` | 64×64 | 1.4 ms | 3.6 ms | AMD 5600G CPU |
| `metric_epe` | 256×256 | 2.0 ms | 4.1 ms | AMD 5600G CPU |
| `metric_pvband` | 256×256 | 6.7 ms | 7.5 ms | AMD 5600G CPU |
| `model_dummy-identity` | 64×64 | 4 µs | 106 µs | AMD 5600G CPU |
| `model_rule-based-opc` | 64×64 | 632 µs | 1.3 ms | AMD 5600G CPU |
| `model_levelset-ilt`（10 次迭代） | 64×64 | 17.9 ms | 20.7 ms | AMD 5600G CPU |

- **Hopkins 比 Gaussian 慢约 8 倍**（64×64 上 2.1 ms vs 238 µs）——SOCS SVD 分解是瓶颈。
- **`levelset-ilt` 10 次迭代**在每张 64×64 tile 上耗时约 18 ms；200 次迭代线性扩展到约 360 ms。
- 未报告 GPU 计时——OpenLithoHub 的模型默认在 CPU 上运行。Neural-ILT（Jiang et al., ICCAD 2020）在 GPU 上报告约 11 s；硬件不匹配时直接对比无意义。

> **Surrogate-ILT** 使用在线训练的代理前向模型，相对于完整 Hopkins 物理前向模型报告 10–50× 加速——这是内部相对测量，不是与外部工具的 wall-clock 对比。

### 如何复现

**硬件：** AMD Ryzen 5 5600G（6C/12T），13 GB DDR4，SATA SSD，Ubuntu 24.04（内核 6.8.0）

**软件：** CPython 3.10.12，PyTorch 2.12.0+cpu，OpenLithoHub `4c3a699`（main）

```bash
# 模型质量（synthetic-8）：
python3 scripts/generate_baselines.py --synthetic --limit 8 --output baselines/

# 模型质量（ICCAD16 testcase1）：
openlithohub eval run --model levelset-ilt --dataset iccad16 \
  --data-root data/iccad16 --node 7nm --pixel-nm 4.0

# 性能计时：
python3 scripts/benchmark_performance.py --json results_timing.json

# 生成对比图表：
python3 scripts/plot_benchmarks.py --input baselines/results.json --output docs/images/
```

**方法学：** synthetic-8 数据为每模型 8 个版图的平均值，单次运行。ICCAD16 为单个 testcase，单次运行。无跨种子统计采样。计时基准使用 `perf_counter_ns`，采样期间 `gc.disable()`，报告中位数 / P95 / P99。前向模型和指标采样 100 次，完整模型预测采样 20 次。

> 所有测试数据均由上述命令在上述硬件上实际运行得出，不含任何主观推测。读者可通过上述命令自行复现。

### 可视化

```bash
python3 scripts/plot_benchmarks.py \
  --input baselines/results.json \
  --output docs/images/
```

![模型质量对比 — synthetic-8](docs/images/benchmark_models.svg)

图表使用透明背景 SVG，坐标轴文字使用中性灰（#888），确保在亮色和暗色 GitHub 主题下均可清晰阅读。

---

## 光学前向模型

OpenLithoHub 提供两个可微分的前向模型，全部用纯 PyTorch 实现，所以整个 ILT
循环端到端可自动求导：

| 模型 | 模块 | 备注 |
|---|---|---|
| Gaussian PSF | `openlithohub._utils.forward_model.simulate_aerial_image` | 单 Gaussian 卷积；适合测试与小尺寸网格的廉价默认实现 |
| Hopkins SOCS | `openlithohub._utils.simulate_aerial_image_hopkins` | 通过 SVD 截断的 Sum-Of-Coherent-Systems 实现部分相干成像；支持环形 / 偶极 / 圆形照明 |
| DiffNano RCWA/FDTD/FDFD | `openlithohub.plugins.diffnano_em`（opt-in） | 通过 DiffNano 插件提供的严格 EM 求解器；注册为 `diffnano_rcwa`、`diffnano_fdtd2d`、`diffnano_fdfd2d` 后端 |

内置 Hopkins 仍为默认路径，也是排行榜数值的唯一可比较路径。插件 EM 后端为 opt-in，产生不可比较的评分。

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
- [x] 里程碑 12：Opt-in 扩散光刻胶（`--resist-diffusion-nm`）、`openlithohub flow run` 闭环 CLI（design→litho→DFM）、可配置按 PDK 层映射、可选 DiffNano/DiffCFD 插件生态

---

## 相关项目

| 项目 | 发表 | 在生态中的角色 |
|------|------|----------------|
| LithoSim | NeurIPS'25 | Sub-28nm 工业级数据集 |
| LithoBench | NeurIPS'23 | 45nm 评测框架 |
| TorchLitho 2.0 | ASICON'25 | 可微分光刻仿真器 |
| [curvyILT](https://github.com/phdyang007/curvyILT) | NVIDIA arXiv'24 | GPU 加速曲线 ILT |
| EasyMRC | TODAES'25 | MRC 参考实现 |
| [DiffNano](https://github.com/OpenLithoHub/DiffNano) | — | 可选插件：PyTorch 原生纳米光子学（RCWA / FDTD / FDFD + 可校准光刻胶）。早期研究，未经第三方验证。 |
| [DiffCFD](https://github.com/OpenLithoHub/DiffCFD) | — | 可选插件：PyTorch 原生稳态 CFD 光刻（Dill/Mack 求解器、旋涂求解器、联合工艺优化）。早期研究，未经第三方验证。 |

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

**插件验证：** DiffNano 和 DiffCFD 是可选插件，两者均自述为早期个人研究项目，
无外部用户、无第三方验证。请勿将其用于生产决策。

**排行榜可比性：** 评分默认为 CTR（恒定阈值光刻胶）无扩散、threshold `0.225`。
启用酸扩散（`--resist-diffusion-nm > 0`）或切换到插件 EM/光刻胶后端产生
**不可比较**的指标数值，排行榜提交时禁用。

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
