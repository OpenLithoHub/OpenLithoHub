# OpenLithoHub 总体战略白皮书（2026 终极版）

**定位：面向先进工艺（EUV/曲线掩膜）的开源计算光刻工作流、合规评估基础设施与基础模型数据引擎**

> **项目愿景：** "我们不制造光刻机，也不编写底层物理引擎。我们为所有优化光刻机的人，铺设连接彼此的标准化公路，并确保这条公路同时通往曼哈顿的现在与曲线的未来。"

[English Version](../README.md)

---

## 一、 宏观产业背景与生态断层（2026 现状）

战略制定的前提是看清战场。当前计算光刻（Computational Lithography）领域正处于"算力爆炸"与"生态割裂"并存的矛盾期。

### 1.1 工业界：算力革命、曲线掩膜与 EUV 随机性

- **NVIDIA cuLitho 投产与算力过剩：** 台积电已将 cuLitho 引入生产环境，ILT 计算速度提升 40-60 倍。底层算力（CUDA/C++）的战争已经结束，竞争焦点已上移至 **AI 算法的编排、评估与工程化落地**。
- **MBMW 与 OASIS.MBW 成为标配：** 多束掩膜写入机（MBMW）让任意形状的曲线掩膜（Curvilinear Mask）成为 sub-28nm 的绝对主流。随之而来的是数据量的爆炸，传统的 GDSII 格式被彻底淘汰。工业界全面转向 **OASIS** 格式（可提供 10 倍的设计数据压缩和 4 倍以上的 OPC 后数据压缩）。特别是专为掩膜写入机制定的 **`OASIS.MBW 2.1` 标准**，原生支持曲线图元，成为先进工艺的生命线。
- **EUV 随机效应（Stochastics）成为核心痛点：** 在极紫外光刻中，光子散粒噪声导致的线边缘粗糙度（LER）和微桥接成为良率杀手。实验数据表明，在当前 EUV 剂量下，sub-20nm 特征的随机 LER 可超过临界尺寸的 **20%**，远超 ITRS 规定的 **8%** 安全红线。传统的确定性光学仿真已无法满足 2nm 时代的需求。

### 1.2 学术界：繁荣的孤岛与半衰期危机

以 `OpenOPC`（港中文）为代表的学术界贡献了大量顶尖开源项目（如 `TorchLitho 2.0`, `LithoSim`, `curvyILT`），但生态呈现严重碎片化：

- **格式与指标割裂：** 各大数据集（LithoBench vs LithoSim）格式互不兼容，EPE/PVB 评估指标定义各异，论文之间无法进行公平的横向对比。
- **工程管线断裂：** 学术界输入输出皆为 Tensor 矩阵，无任何开源工具能处理真实的 OASIS/GDSII 大版图端到端优化。
- **半衰期危机：** 论文发表即停止维护，环境依赖冲突（Python/PyTorch 版本地狱）导致工业界极难复现。

### 1.3 战略结论：OpenLithoHub 的精确生态位

```text
[底层算力层] NVIDIA cuLitho / GPU 集群（已有，不竞争）
[物理引擎层] TorchLitho 2.0 / OpenILT / curvyILT（已有，我们整合）
[数据集层]   LithoBench / LithoSim / ICCAD（已有，我们统一接入）
         ↑
[OpenLithoHub] 跨框架评估 + OASIS 工作流 + EUV/MRC 合规检查 + 数据引擎
         ↑
[用户层]  学术研究者 / 芯片工程师 / EDA 基础模型开发者
```

---

## 二、 核心架构设计（五层工业级模型）

OpenLithoHub 采用插件优先（Plugin-first）的模块化设计，彻底打通从学术研究到工业制造的壁垒。

### Layer 1：统一数据适配层（Data Adapter Layer）

屏蔽底层格式差异，提供统一的 PyTorch Tensor 输出接口。

- **向下兼容：** 一键加载 LithoBench（`.npy`）、LithoSim（HuggingFace Parquet）。
- **元数据对齐：** 自动对齐像素分辨率（Pixel Size）、工艺窗口（Process Window）和光源参数。

### Layer 2：制造合规与 EUV 评估层（Manufacturability & EUV Benchmark）—— *核心差异化*

打破学术界"唯 EPE 论"的局限，引入工业界真正关心的"可制造性"与"鲁棒性"指标：

- **EUV 随机鲁棒性（Stochastic Robustness）：** 针对 sub-20nm 节点 LER 飙升的痛点，独创性地引入随机噪声注入评估，量化 AI 生成的掩膜在对抗光子散粒噪声时，发生微桥接（Micro-bridging）或断线的概率。
- **MRC/DRC 违规率：** 集成 `EasyMRC` 与 `OpenDRC`，检查掩膜是否违反最小线宽/间距规则（一票否决指标）。
- **标准化精度与成本：** 统一 EPE、PV Band 计算标准；引入 Shot Count 估算掩膜制造成本。

### Layer 3：模型适配层（Model Integration Layer）

提供极简的 `LithographyModel` 接口。无论是传统的启发式 OPC、基于 U-Net 的深度学习模型，还是最新的曲线 ILT（如 `curvyILT`），只需实现 `predict()` 方法即可接入评测管线。

### Layer 4：OASIS.MBW 级工程工作流层（Workflow Engine）—— *最大工程护城河*

**彻底打通从 Tensor 到晶圆厂的"最后一公里"。**

- **大版图分布式处理：** 基于 `KLayout` Python API 实现原始设计版图（通常为曼哈顿图形）的快速解析与切片（Tiling）。
- **绕过 KLayout 局限的双轨轮廓输出：**
  - *Manhattan 模式（Legacy）：* 针对传统 VSB 写入机，输出阶梯状多边形，可继续使用 KLayout 引擎重组。
  - *Curvilinear 模式（Modern）：* **鉴于 KLayout 底层引擎会将曲线强制离散化为分段线性多边形**，本模式将绕过传统几何引擎，直接将 AI 生成的平滑轮廓拟合为 B 样条曲线，并**原生序列化导出为专为多束写入机设计的 `OASIS.MBW 2.1` 格式**。这确保了真正的数学曲线表达，避免了数据体积的二次爆炸。
- **目标 CLI 体验：**
  ```bash
  openlithohub optimize --input chip.oas --model diffusion-ilt --writer mbmw --node 3nm-euv --drc-check --output optimized.oas
  ```

### Layer 5：社区排行榜与基础模型数据引擎（Leaderboard & Data Engine）

- **公开 SOTA 追踪：** 建立计算光刻领域的 PapersWithCode，按工艺节点、掩膜形态分类排行。
- **EDA 基础模型（Foundation Models）的数据引擎：** 当前阻碍 EDA 视觉大模型（LVM）发展的最大瓶颈是**开源电路数据的极度稀缺**。OpenLithoHub 不仅是评测工具，更是**数据生成器**。通过内置的自动化管线，研究团队可以批量生成包含各种工艺条件、合规检查标签的"版图-掩膜-光刻胶轮廓"高质量配对数据集，为下一代 EDA 基础模型的 Pre-training 提供源源不断的燃料。

---

## 三、 执行路线图与里程碑（12 个月计划）

### Phase 1：最小可行基准 MVP（第 1-2 个月）

- **目标：** 建立第一把"跨数据集比较尺"。
- **行动：** 实现 LithoBench 与 LithoSim 的统一 DataLoader；实现基础 EPE 评估；封装 `openlithohub eval` 命令行工具。
- **交付：** 录制 30 秒终端运行 GIF，展示同一模型在两个数据集上的标准化评测报告。

### Phase 2：制造合规与曲线提取（第 3-4 个月）

- **目标：** 实现对现有学术 Benchmark 的降维打击。
- **行动：** 引入 MRC 检查模块；实现从像素矩阵到多边形/曲线的轮廓提取算法（参考 `EasyMRC` 与 `curvyILT`）。
- **交付：** 评测报告新增"MRC 违规率"指标。发布技术博客《为什么 90% 的 AI 光刻论文在晶圆厂里是一张废纸？》。

### Phase 3：OASIS 工作流与 Web Demo（第 5-6 个月）

- **目标：** 引爆工程师社区。
- **行动：** 实现 `.oas/.gds` 的端到端优化管线；在 HuggingFace Spaces 部署零配置的 Web 游乐场（拖拽版图，一键优化）。
- **交付：** 获得首批 200+ GitHub Stars，吸引工业界工程师试用。

### Phase 4：学术统战与排行榜上线（第 7-9 个月）

- **目标：** 确立行业标准地位。
- **行动：** 上线官方 Leaderboard 网站；主动为各大开源项目提交 PR，邀请原作者将 OpenLithoHub 作为官方推荐的评估工具。
- **交付：** 至少 3 支顶尖高校团队在排行榜上提交成绩。

### Phase 5：基金会孵化与商业化探索（第 10-12 个月）

- **目标：** 确立长期资金与官方背书。
- **行动：** 申请加入 **CHIPS Alliance**（Linux Foundation 旗下）成为孵化项目；推出面向 Fabless 公司的"私有化基准测试（Private Benchmark）"商业方案。

---

## 四、 商业模式与护城河分析

### 4.1 真正的护城河：标准惯性与生态依赖

OpenLithoHub 的终极壁垒不是代码的复杂度，而是**度量衡的垄断**。一旦学术界习惯用它发论文，工业界习惯用它做验证，大模型团队习惯用它生成数据，它就成为了不可替代的基础设施。

### 4.2 商业化变现路径（开源核心，商业增值）

1. **私有化基准测试（Private Benchmark Hosting）：** 帮助 Fabless（芯片设计公司）在不泄露机密 OASIS 数据的前提下，客观评估 Synopsys、Cadence 或初创 EDA 公司的 AI 算法能力（目前完全空白的付费市场）。
2. **企业级分布式编排（Enterprise Orchestration）：** 提供支持 Kubernetes 集群调度的商业版工具链，解决全芯片（Full-chip）规模的算力调度与内存溢出问题。
3. **云端掩膜优化 SaaS：** 面向中小设计公司，提供按需付费的云端 GPU 加速优化服务。

---

## 五、 Day 1 - Day 30 极简破冰指南（给创始人的 Action Items）

不要被宏大的架构吓倒，前 30 天只做最基础的工程：

- **Day 1-3（占领阵地）：** 注册 GitHub 组织 `OpenLithoHub`，将本白皮书作为 `README.md` 上传。注册 PyPI 包名。
- **Day 4-10（攻克数据）：** 下载 LithoSim 和 LithoBench 的极小样本数据（各 100 张图）。写一个 Python 脚本，把它们读进内存，用 `matplotlib` 画出来。
- **Day 11-20（攻克指标）：** 写一个简单的 `metrics.py`，算一下两张图片的像素差异（基础版 EPE）。
- **Day 21-30（封装 CLI）：** 用 `Typer` 库写一个命令行工具，把读取数据和算误差串起来。录制一个炫酷的终端运行 GIF，贴到 README 顶部。

**你的开源创业之旅，从敲下 `git init` 的那一刻正式开始。**

---

## 附录：关键开源项目与文献速查

| 项目 | 发表 | 描述 |
|------|------|------|
| **LithoSim** | NeurIPS'25 | Sub-28nm 工业级评测标准与数据集 |
| **LithoBench** | NeurIPS'23 | 45nm 基线评测框架 |
| **TorchLitho 2.0** | ASICON'25 | 顶尖可微分光刻仿真引擎 |
| **curvyILT** | NVIDIA arXiv'24 | GPU 加速曲线 ILT 与 B 样条轮廓提取参考 |
| **EasyMRC** | TODAES'25 | Manhattanization 和 MRC 的直接参考实现 |
| **IEEE DATC RDF-2025** | — | AI for EDA 可复现性危机的权威描述 |

---

## 许可证

[Apache 2.0](../LICENSE)
