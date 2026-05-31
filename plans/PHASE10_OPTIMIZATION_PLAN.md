# Phase 10 优化执行计划 — 文献驱动的下一步增量 (2026-05-31)

> 承接 `PHASE9_OPTIMIZATION_PLAN.md`。经对四仓 `main` 实测核对，**Phase 9 的全部核心任务已确认入库**，作为
> Phase 10 的基线（**不再列为任务**）。本文件只规划「未来提升」。
>
> Phase 9 已落地、不再重复的成果（实测确认）：
> - **diff-surrogate**：`codomain.py`（codomain-attention 可迁移骨干，S9.1）；`probabilistic.py`（PNO + proper
>   scoring rule 分布头，S9.2）；`decision.py`（`AcceptRejectGate` / CVaR 风险预算 / 覆盖率早停，S9.3）；
>   `pretraining.py`（多任务 + few-shot，adapter）。
> - **DiffNano**：`design/quantized.py`（STE + 二值/k-level 量化感知反设计，N9.1）；`solvers/fdtd3d.py`
>   `time_reversal` 伴随 + `validation` FDTDX 对拍脚手架（N9.2）；`design/robust_warm_start.py`（斜入射/工艺角
>   最坏分位暖启动，N9.3）；`solvers/backend_diagnostics.py`。
> - **OpenLithoHub**：`models/grpo_warm_start.py`（GRPO + ILT-引导模仿 + style-aware 后验微调，O9.1）；
>   `workflow/full_chip_tiling.py` + `inference/multiproc.py`（O9.2 拼接，GPU 待回填）；随机感知 loss + 共形区间（O9.3）。
> - **DiffCFD**：`solvers/learned_closure.py`（solver-in-the-loop / a-posteriori 学习闭合，C9.1）；
>   `validation/diff_flowfsi_crossval.py`（C9.2）；`envs/codomain_control.py`（跨雷诺数/几何迁移流控，C9.3）。
>
> 每条任务给出：动机文献 → 具体改动 → 验收标准（DoD）。
> 进度标记：`[ ]` 待办 · `🔬` 需先做文献复算 · `⚠️` 有许可/合规约束（clean-room，不 vendoring，CI 跑 `scancode`） ·
> `🖥️` 需 GPU/规模化资源 · `★` 高优先级

---

## 0. 跨仓总主线（Phase 10 的三条战略线）

Phase 9 把「**校准 UQ → 决策门**」「**solver-in-the-loop 单步/在线学习闭合**」「**codomain 可迁移骨干**」打通到
CPU 正确性级别。其共同特征是：**预测器与闭合仍以「确定性回归」为主**，UQ 来自事后共形或 PNO 的分布头贴片，
长程稳定性靠展开训练缓解、但尚未从架构层根治。2025Q4–2026 的主线已经明确转向 **生成式（diffusion /
flow-matching）算子建模**：把"下一状态/掩模/结构/子格场"建成**条件分布**而非点估计，天然给出
①不确定度感知的集成、②更强的多模态/逃逸非凸能力、③通过流匹配显著抑制长 rollout 漂移。Phase 10 在
Phase 9 的"可信原语 + 决策门"之上，把**生成式建模**作为贯穿四仓的新一等公民，并完成"GPU 实测 + 多保真度 +
第三方对拍"的可信成熟度收口。

1. **从「确定性预测 + 决策门」走向「生成式算子（diffusion/flow-matching）+ 不确定度感知集成」** —
   Phase 9 的 `codomain.py` 骨干与 `probabilistic.py` 分布头都是"一次前向出点估计/参数化分布"。2025–2026 的
   **Flow Marching**（arXiv:2509.18611）把神经算子学习与 flow-matching 桥接，用一个统一速度场把"带噪当前态→干净后继态"
   传输，**同时抑制长程 rollout 漂移并给出 uncertainty-aware 集成**，配套 P2VAE 潜空间 + Flow-Marching-Transformer
   （较全长视频扩散 ×15 提效）；**UniFluids**（arXiv:2603.22309, 2026-03）用条件 flow-matching 统一跨 PDE 族/维度，
   缓解自回归误差累积与过平滑/谱偏置。Phase 10 把 diff-surrogate 升级为**生成式算子骨干**，其"集成"直接喂 Phase 9
   的 `decision.py` 决策门——一处改动，DiffNano 反设计、OpenLithoHub 掩模合成、DiffCFD 子格闭合三域同时受益。

2. **从「单步/在线学习闭合」走向「长程稳定的生成式闭合 + 算子对齐时空预测」** — Phase 9 的 `learned_closure.py`
   是**确定性** a-posteriori 闭合，长程稳定靠展开步数曲线缓解。2024–2026 出现两条更根治的路线：①**生成式超分辨
   子格闭合**——粗网格可微求解 + Bayesian 条件扩散生成小尺度湍流（arXiv:2406.20047），保住高频/谱信息并自带 UQ；
   ②**算子对齐**——把网络架构直接映射到控制方程的微分/积分算子结构（Differential-Integral Neural Operator,
   arXiv:2509.21196），从架构层根治长程稳定。Phase 10 给 DiffCFD 补"生成式子格闭合 + 算子对齐"，并把"长 rollout
   漂移/谱保真"做成跨仓通用回归门（DiffCFD 主用，DiffNano 时域 FDTD rollout 与生成式骨干 rollout 复用同一门）。

3. **从「自测玩具 + GPU 待回填」走向「规模化 GPU 实测 + 多保真度 + 第三方/实验对拍的可信成熟度」** — 四仓 honesty
   boundaries 目前仍写着"无第三方实验验证、CPU-only、toy 规模"。Phase 9 §5 把 GPU 交叉对拍立为目标但多处"待回填"。
   Phase 10 ①把 N9.2/O9.2/C9.2 的 GPU 实测对拍**真正收口**（FDTDX / LithoBench·ICCAD13 / Diff-FlowFSI，仅数值参照、
   不 vendoring）；②引入**多保真度融合**（廉价 RCWA/粗 LES + 少量高保真），让 toy→规模化迁移有据可循；③建立
   **sim-to-real / 外部参照协议**并据此逐条升级 honesty boundaries（从"无验证"升级为"已对拍 + 残差量化"）。

---

## 1. DiffNano — 电磁求解器与反设计

**Phase 9 基线**：量化感知 STE 反设计（`quantized.py`）；FDTD3D 时间反演伴随 + FDTDX 对拍脚手架；斜入射/工艺角
最坏分位鲁棒暖启动（`robust_warm_start.py`）；RCWA 四后端 + 后端工况诊断。

### N10.1 — 物理引导潜空间扩散反设计：Maxwell-guided latent diffusion，RCWA-in-the-loop 🔬 ★
- **动机**：DiffNano 现有反设计走"连续松弛/STE + 暖启动 + 梯度优化"，**没有生成式直接映射**。2026 该方向已是 SOTA：
  *MxDiffusion*（Nano Lett. 2026, 26(14):4897）用**两段式扩散 + 显式 Maxwell 方程损失**把物理先验直接嵌进反设计，
  在分布外目标上显著优于纯数据驱动扩散；中科院 *AIGP*（2026-05）用**潜空间扩散**把全带透射/相位/偏振响应秒级
  直接映射到可制造结构，并在数据集构造阶段就过滤不可制造几何；*Diffusion-based EM inverse scattering*
  （arXiv:2511.05357）在 unseen 目标上中位 MPE < 19%、较 CMA-ES 把设计时间从小时压到秒级。
- **改动**：① 在 `design/` 新增 `latent_diffusion.py`：以 `codomain.py` 编码器构潜空间，训练条件扩散/flow（目标光学
  响应 → 结构），**Maxwell/RCWA 引导项**作为采样阶段的 classifier-guidance（复用现有 `rcwa.py` 可微前向）；② 与
  Phase 9 `quantized.py` STE 串联——扩散出连续结构 → 量化感知精修，保证可制造；③ 生成多候选 → `robust_warm_start.py`
  的最坏角分位打分 + `decision.py` 接受门选优；④ 对照"扩散直接映射 vs 现有暖启动+梯度优化"的 FoM 与分布外泛化。
- **⚠️ 合规**：机制 clean-room 复现，不取任何公开权重；module docstring 列 references-consulted。
- **DoD**：① Maxwell-guided 扩散在分布外目标族上 FoM/泛化优于纯数据驱动扩散与现有暖启动基线（≥ N 种子，Wilcoxon）；
  ② 生成结构经量化精修后通过制造约束检查；③ 候选打分/决策协议与 OpenLithoHub N10/diff-surrogate 一致；④ ≥ 10 新测试通过。

### N10.2 — 大孔径/3D 多尺度 metalens + FDTD GPU 实测收口（收口 N9.2/§5）🖥️🔬
- **动机**：N9.2 时间反演伴随的"质变显存收益"必须在 GPU 大网格兑现。2026 综述 *Inverse design for scalable photonic
  systems*（Nat. Rev. Mater., 2026-04）强调 GPU 显存带宽瓶颈与 systolic/分块更新，开源 **FDTDX**（JOSS 11:8912, 2026）
  为多 GPU 3D AD-FDTD 最佳数值参照。Phase 9 已具脚手架，Phase 10 回填实测并扩到 3D 多尺度孔径。
- **改动**：① 在 CUDA 机器回填 FDTD3D 前向 + time_reversal 伴随的 GPU 三方曲线（time_reversal / checkpoint / 纯 AD 的
  显存×速度）；② 把 LPA 大孔径前向扩到 3D 多尺度（粗-细分块），评估 systolic/分块对带宽的影响；③ 与 FDTDX 在 ≥ 1 个
  3D 算例前向相对误差 < 1e-3、梯度方向余弦 > 0.99（**仅数值参照，不 vendoring**，CI 跑 `scancode`）。
- **DoD**：① GPU 三方曲线 + 与 FDTDX 对拍达标（或诚实 CPU-only 标注 + CPU 正确性回归）；② 3D 多尺度大孔径跑通；
  ③ 升级 honesty boundary 中的"GPU 待回填"条目；④ ≥ 6 新测试通过。

### N10.3 — 多保真度反设计：RCWA↔FDTD 代价感知融合 + foundry-compatible 约束（对接 S10.3）🔬
- **动机**：§0 主线三。当前反设计在单一保真度（多为 RCWA）上做；2026 趋势是**廉价代理 + 少量高保真**融合，且
  *Foundry-Compatible Grating Couplers via Inverse Design*（Yale, 2026-05）把"代工可制造"作为反设计一等约束。
- **改动**：① 新增 `design/multifidelity.py`：以 RCWA 为低保真、FDTD 为高保真，按 diff-surrogate `multifidelity.py`/
  `budget.py` 做代价感知主动采样与融合；② 把 N10.1 扩散候选先用 RCWA 海选、再 FDTD 精验，量化"高保真调用次数 vs FoM"；
  ③ foundry-compatible 几何约束（最小线宽/间距）进 `quantized.py` 投影。
- **DoD**：① 多保真度反设计在固定高保真预算下 FoM 优于纯高保真/纯低保真；② foundry 约束达标且对 FoM 退化可量化；
  ③ ≥ 6 新测试通过。

---

## 2. OpenLithoHub — 计算光刻基准与工作流

**Phase 9 基线**：GRPO + ILT-引导模仿 + style-aware 后验微调（`grpo_warm_start.py`）；全芯片 Schwarz 拼接 + CPU 多
进程（GPU 待回填）；随机感知 CVaR/分位 loss + 共形区间。

### O10.1 — 扩散式掩模合成：条件/潜空间扩散 + 可制造性过滤（升级 O9.1 的生成器范式）🔬 ★
- **动机**：O9.1 用 **GRPO 微调 WGAN 后验**——强但单生成器、多样性受限。2026 计算光刻/反设计普遍转向**条件扩散**：
  扩散给出更高的候选多样性与逃逸非凸能力，并可在数据构造/采样阶段**内生过滤不可制造几何**（参照 AIGP latent
  diffusion 范式与 *Fast ILT via model-driven block-stacking CNN*, arXiv:2412.14599 的无标注/物理驱动思路）。
- **改动**：① 新增 `models/diffusion_mask.py`：以版图为条件训练潜空间扩散（潜空间复用 `models/layout_mae.py` 编码器），
  采样阶段以可微光刻前向（`models/surrogate_ilt.py` / 厚掩模代理）做 EPE/PVB classifier-guidance；② 与 O9.1 `grpo_warm_start`
  对照——"GRPO-WGAN 后验 vs 扩散后验"的最终 EPE 分布、多样性与多 hotspot 逃逸能力；③ 扩散候选 → `decision.py` MRC/EPE
  接受门 + 现有 `CandidateScorer` 选优。
- **⚠️ 合规**：clean-room；不取公开权重，标注"结构演示，非 SOTA 数字"。
- **DoD**：① 扩散后验在 ICCAD13/LithoBench 子集上候选多样性与最终 EPE 违例分布优于 O9.1（回归表）；② 内生可制造过滤
  使 MRC 违例率下降；③ 候选/决策协议与 DiffNano N10.1 一致；④ ≥ 10 新测试通过。

### O10.2 — High-NA EUV anamorphic SMO + 曲线掩模 shot-count 联合优化 🔬 ★
- **动机**：High-NA EUV 是 2026 计算光刻的主战场，带来**各向异性放大（anamorphic）、中心遮拦（central obscuration）、
  mask-3D 阴影**等新约束（Synopsys/imec 高 NA 综述；Science Tokyo *High-NA EUV STCC*, 2026-05）。同时**曲线 ILT**虽
  保真最高，但多束/VSB 写入的 **shot count** 直接决定可量产性（eBeam Initiative VSB shot-count 研究）。OpenLithoHub
  现有 SMO/ILT 未显式处理 anamorphic 与 shot-count。
- **改动**：① 在 `benchmark`/`workflow` 加 **anamorphic 成像 + 中心遮拦**的源/掩模建模（x/y 各向异性放大、mask-3D 阴影
  一阶修正）；② 把**曲线掩模 fracturing shot-count**做成**可微/可惩罚代价项**，与 EPE/PVB 组成多目标（"保真 vs 写入
  时间"Pareto）；③ 在 high-NA 设定下对照 anamorphic-aware vs isotropic SMO 的成像保真。
- **DoD**：① anamorphic + 中心遮拦 SMO 在 high-NA 设定下成像保真优于各向同性基线；② shot-count 代价项可调并给出
  "保真 vs shot-count"Pareto；③ mask-3D 一阶修正回归；④ ≥ 8 新测试通过。

### O10.3 — 3D 抗蚀剂随机模型 + 覆盖率门（升级 O9.3 的随机区间）🔬
- **动机**：O9.3 的随机区间作用在 2D 失效率上；high-NA 趋向**< 20nm 超薄抗蚀剂**，随机缺陷由**二次电子空间相关 +
  催化反应**主导（Fukuda 空间相关概率模型；imec/EUV Accelerator 2025–2026 抗蚀剂攻关）。需把随机模型升级到**3D 抗蚀剂
  剖面（线塌陷/纵向相关）**并配共形覆盖。
- **改动**：① 扩 `stochastic.py`：二次电子空间相关核 + 3D 抗蚀剂剖面（纵横比/线塌陷风险）；② 对 LCDU/3D 缺陷率的
  through-focus 分位用 diff-surrogate `conformal` 做覆盖率校准，输出"带覆盖保证的 3D 过程窗口"；③ 覆盖率达标作为
  随机感知 ILT 验收门（衔接 `decision.py`）。
- **DoD**：① 3D 随机缺陷/LCDU 区间达目标覆盖率（90/95%）；② 线塌陷风险指标随抗蚀剂厚度单调、定性符合文献；
  ③ ≥ 6 新测试通过。

---

## 3. DiffCFD — 可微流体与逆向设计 / RL

**Phase 9 基线**：solver-in-the-loop a-posteriori 学习闭合（`learned_closure.py`，确定性）；Diff-FlowFSI 对拍脚手架；
codomain 跨工况迁移流控（`codomain_control.py`）。

### C10.1 — 生成式超分辨子格闭合：Bayesian 条件扩散，保高频 + 自带 UQ 🔬 ★
- **动机**：C9.1 的学习闭合是**确定性回归**，对小尺度高频/谱信息有平滑倾向且无 UQ。*Neural Differentiable Modeling
  with Diffusion-Based Super-resolution*（arXiv:2406.20047, Jian-Xun Wang 组）在**粗网格可微求解**之上用 **Bayesian
  条件扩散生成小尺度湍流**，较物理 SGS 闭合与纯数据驱动求解器都更准且**自带不确定度**。这与 C9.1 的 tape/检查点
  基础设施直接复用。
- **改动**：① 新增 `solvers/generative_closure.py`：粗网格可微 NS rollout + 条件扩散/flow 生成子格修正（条件于粗解
  大尺度场，复用 C8.3/C9.1 检查点）；② 与 C9.1 确定性闭合对照——能谱/结构函数高频保真 + 长 rollout 发散率 + UQ 校准；
  ③ 生成集成 → diff-surrogate `decision.py` 风险预算/早停。
- **DoD**：① 生成式闭合在能谱高频段保真优于确定性闭合（谱回归）；② 长 rollout 发散率不劣于 C9.1 且给出校准 UQ；
  ③ 显存随展开步受控；④ ≥ 8 新测试通过。

### C10.2 — 算子对齐长程时空预测 + 与 Diff-FlowFSI GPU 实测收口（收口 C9.2/§5）🔬🖥️
- **动机**：§0 主线二。长程稳定除展开训练外，可从**架构层**根治：*Differential-Integral Neural Operator*
  （arXiv:2509.21196, 2025-09）把架构映射到控制方程的微分/积分算子结构，专攻长程稳定。同时 C9.2 与 Diff-FlowFSI
  （CMAME 2025, arXiv:2505.23940）的 GPU 实测对拍仍待收口。
- **改动**：① 在 `surrogates/` 新增算子对齐 backbone（differential + integral 分支），作为长程预测/代理选项；② 在
  ≥ 1 共有算例（顶盖驱动腔 / FSI 弹性边界）与 Diff-FlowFSI 做 GPU 前向 + 梯度对拍（**仅数值参照，不 vendoring**）；
  ③ "展开步数 / 算子对齐 / 生成式闭合"三者对长 rollout 漂移的对照，喂 §5 通用漂移门。
- **DoD**：① 算子对齐在长 rollout 上漂移低于纯 FNO/确定性闭合（漂移回归曲线）；② 与 Diff-FlowFSI 前向相对误差 < 2%、
  梯度方向余弦 > 0.99（或诚实 CPU-only）；③ ≥ 6 新测试通过。

### C10.3 — 流控的生成式策略迁移 + 多保真度训练（对接 S10.1/S10.3）🔬
- **动机**：C9.3 的 codomain 迁移流控是**确定性策略**。结合 S10.1 生成式骨干，可让策略/代理给出**多模态控制候选 + UQ**，
  并用多保真度（粗/细 LES）降训练成本。
- **改动**：① 让 `codomain_control` 的代理换成 S10.1 生成式骨干（多候选控制 + 集成 UQ）；② 多保真度 rollout（粗 LES
  海选、细 LES 精验，复用 diff-surrogate `multifidelity.py`）；③ 对照确定性 vs 生成式策略的最坏工况鲁棒性。
- **DoD**：① 生成式策略在最坏工况下鲁棒性优于确定性 + 与 C9.3 迁移可叠加；② 多保真度降高保真调用且不显著掉点；
  ③ ≥ 6 新测试通过。

---

## 4. diff-surrogate — 统一可微代理框架（共享库 / Phase 10 的脊柱）

**Phase 9 基线**：`codomain.py`（codomain-attention 骨干）；`probabilistic.py`（PNO + proper scoring）；`decision.py`
（决策门）；`pretraining.py`（多任务 + adapter）；`conformal.py` / `structure.py` / `active_sampling.py` / `multifidelity.py`。

### S10.1 — Flow-matching 生成式算子骨干：P2VAE 潜空间 + Flow-Marching Transformer 🔬🖥️ ★
- **动机**：§0 主线一。Phase 9 的 `codomain.py` + `probabilistic.py` 是"确定性骨干 + 分布头贴片"。**Flow Marching**
  （arXiv:2509.18611）用 location-scale 插值核把神经算子学习与 flow-matching 统一：桥参数 k=1 退化为确定性算子插值、
  k=0 为 flow-matching 生成核，**同时抑制长程漂移 + 给 uncertainty-aware 集成**；配 **P2VAE** 潜空间 + **Flow-Marching
  Transformer**（较全长视频扩散 ×15 提效）。**UniFluids**（arXiv:2603.22309, 2026-03）用条件 flow-matching 统一跨 PDE
  族/维度、缓解过平滑与谱偏置。这是把 Phase 9 骨干"生成化"的标准路径，一处改动三域复用。
- **改动**：① 新增 `flow_operator.py`：在 `codomain.py` 编码器之上实现 P2VAE 潜空间 + flow-matching 速度场（location-
  scale 核，可配桥参数 k 在确定性↔生成式间插值）；② `generate_ensemble()` 出集成 → 直接喂 `decision.py`；③ 与
  `probabilistic.py`（PNO）、纯 Ensemble 三方对照（同覆盖率下带宽 + 长 rollout 漂移 + 谱保真 + 采样多样性）；④ 公共 API
  导出，供 DiffNano N10.1 / OpenLithoHub O10.1 / DiffCFD C10.1·C10.3 复用。
- **DoD**：① flow 骨干在 toy 多物理 rollout 上长程漂移显著低于确定性 `codomain` 基线、谱高频保真更好；② 生成集成的
  覆盖率经 `conformal` 校准达标；③ 桥参数 k 扫描复现"确定性↔生成式"权衡；④ 被 ≥ 2 个下游仓实际调用；⑤ ≥ 12 新测试通过。

### S10.2 — In-context / 测试时自适应算子：免梯度 few-shot 迁移（升级 `pretraining.py` adapter）🔬
- **动机**：Phase 9 的 adapter 迁移仍需**梯度微调**。2026 PDE 基础模型主线（*PDE-FM / The Well*, AAAI 2026，IBM；
  ICON 类 in-context 算子）指向**in-context / 测试时自适应**——给若干 (输入,输出) 示例即可**免梯度**适配新工况，是
  "可迁移底座→基础模型雏形"的关键一跃。
- **改动**：① 新增 `in_context.py`：示例对作为 context token 输入 S10.1 骨干，免梯度预测新工况；② 与 `pretraining.py`
  的梯度 adapter 对照（同目标工况下 in-context vs few-shot 微调的样本/算力效率）；③ 在 The Well 子集（toy 规模）跑
  跨物理 in-context 迁移基准。
- **DoD**：① in-context 在 ≥ 2 个 held-out 物理上免梯度即达 adapter few-shot 可比精度；② 样本/算力效率回归表；
  ③ ≥ 8 新测试通过。

### S10.3 — 多保真度融合 + 主动实验设计闭环：可信成熟度脊柱（对接 §5 与 N10.3/C10.3）🔬
- **动机**：§0 主线三。四仓 honesty boundaries 仍是"自测玩具 + 无第三方验证"。把 `multifidelity.py` + `active_sampling.py`
  + `decision.py` 串成**代价感知 + 覆盖率驱动的主动实验设计闭环**，让"高保真/第三方对拍调用"花在最该花的地方，是把
  "无验证"升级为"按预算最优对拍 + 残差量化"的统一脊柱。
- **改动**：① 新增 `experiment_design.py`：以共形/生成式集成带宽 + 多保真度代价做 Bayesian 实验设计（下一个该跑哪个
  高保真/外部参照点）；② 暴露统一协议供 DiffNano N10.3（RCWA↔FDTD）、DiffCFD C10.3（粗↔细 LES）、§5 第三方对拍消费；
  ③ 对照"随机/均匀采样 vs 实验设计"在固定高保真预算下的对拍残差下降。
- **DoD**：① 实验设计在固定高保真预算下使对拍残差/最坏分位显著优于均匀采样；② 三仓各接入一处并跑通；③ ≥ 8 新测试通过。

---

## 5. 跨仓收尾（Phase 10 的统一门控）

- [ ] **生成式骨干跨仓复用**：S10.1 `flow_operator` 被 DiffNano N10.1（潜空间扩散反设计）、OpenLithoHub O10.1（扩散
      掩模）、DiffCFD C10.1·C10.3（生成式闭合/策略）复用；生成集成统一喂 Phase 9 `decision.py` 决策门。
- [ ] **长程漂移门（升级 Phase 9 的"展开步数→稳定性"门）**：把"长 rollout 漂移 + 谱高频保真"做成跨仓通用回归
      （DiffCFD C10.1/C10.2 主用；DiffNano 时域 FDTD rollout 与 S10.1 生成式 rollout 复用同一门）。
- [ ] **GPU 实测 + 第三方对拍全面收口（兑现 Phase 9 §5 的"待回填"）**：DiffNano↔FDTDX（N10.2）、OpenLithoHub↔
      LithoBench/ICCAD13（O10 端到端）、DiffCFD↔Diff-FlowFSI（C10.2）各至少一条 **GPU 实测**对拍（仅数值参照，
      **不 vendoring**，CI 跑 `scancode`）。据此**逐条升级 honesty boundaries**：从"无第三方验证 / CPU-only"改为
      "已对拍 + 残差量化"或诚实保留"仍待 GPU"。
- [ ] **多保真度 + 主动实验设计成为统一脊柱**：S10.3 协议被 N10.3 / C10.3 / §5 第三方对拍消费；"按预算最优对拍"
      回填各仓 README。
- [ ] **生成式 + 不确定度感知集成进决策门**：四仓各至少一条核心路径，把生成式集成（S10.1）→ `decision.py` 的
      接受/停机/风险预算决策门接通（衔接 Phase 9"带宽改变了决策"门 → 升级到"生成式集成改变了决策且更鲁棒"）。
- [ ] **全仓 CI 全绿 + clean-room 合规**：四仓 + diff-surrogate 联合 CI（含 `scancode`）一次性绿；所有生成式/扩散
      模块 docstring 列 references-consulted，不取任何公开权重；honesty boundaries 按本期实际进展逐条更新。

---

## 6. 关键参考文献（Phase 10 新增 / 重点，2024–2026）

**生成式算子 / flow-matching 基础模型（主线一）**
- *Flow Marching for a Generative PDE Foundation Model*, arXiv:2509.18611, 2025（神经算子 × flow-matching，
  location-scale 核，抑制长程漂移 + uncertainty-aware 集成；P2VAE 潜空间 + Flow-Marching-Transformer，×15 提效）。
- *UniFluids: Unified Neural Operator Learning with Conditional Flow-matching*, arXiv:2603.22309, 2026-03
  （条件 flow-matching 统一跨 PDE 族/维度，缓解自回归误差累积与过平滑/谱偏置）。
- *Towards a Foundation Model for PDEs Across Physics Domains* (PDE-FM / The Well), AAAI 2026, IBM Research
  （一次预训练→免改架构迁移，12 数据集 VRMSE −46%；in-context 基础模型基准）。

**生成式 / 算子对齐闭合与长程稳定（主线二）**
- *Neural Differentiable Modeling with Diffusion-Based Super-resolution for 2D Spatiotemporal Turbulence*,
  arXiv:2406.20047（粗网格可微求解 + Bayesian 条件扩散生成小尺度湍流，保高频 + 自带 UQ）。
- *Differential-Integral Neural Operator for Long-Term Turbulence Forecasting*, arXiv:2509.21196, 2025-09
  （算子对齐：架构映射到微分/积分算子结构，从架构层根治长程稳定）。
- Diff-FlowFSI, *GPU-optimized differentiable CFD/FSI*, CMAME 2025, arXiv:2505.23940（GPU 实测对拍参照）。

**光子反设计（生成式 / 扩散）**
- *MxDiffusion: Physics-Aware Maxwell's Law-Guided Diffusion for Inverse Metasurface Design*, Nano Lett. 2026,
  26(14):4897–4905（两段式扩散 + Maxwell 损失，分布外更优）。⚠️ clean-room。
- *AIGP: Map Optical Properties to Subwavelength Structures via Latent Diffusion*, 中科院, 2026-05（潜空间扩散
  秒级直接映射，数据构造内生过滤不可制造几何）。
- *Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media*, arXiv:2511.05357（unseen 目标
  中位 MPE < 19%，较 CMA-ES 小时→秒）。
- *Inverse Design in Nanophotonics via Representation Learning*, Adv. Opt. Mater. 14(1), 2026。
- *Inverse design for scalable photonic systems*, Nature Reviews Materials, 2026-04（GPU systolic FDTD 综述）。
- FDTDX, *High-performance open-source FDTD with AD*, J. Open Source Softw. 11:8912, 2026（多 GPU 3D AD-FDTD 参照）。
- *Foundry-Compatible Grating Couplers Using an Inverse Design Framework*, Yale, 2026-05（代工可制造一等约束）。

**计算光刻（High-NA EUV / 曲线掩模 / 扩散）**
- *The High-NA EUV Imperative* & *Enhancing High-NA EUV w/ Computational Lithography*, Synopsys, 2024–（anamorphic +
  中心遮拦 SMO、曲线 ILT、full-chip）。
- *High-NA EUV STCC formula*, Science Tokyo, 2026-05。
- *High-NA EUVL: the next step after EUVL*, imec（mask-3D、超薄抗蚀剂、随机失效）。
- *Optimizing VSB Shot Count for Curvilinear Masks*, eBeam Initiative（曲线掩模 shot-count vs 写入时间/保真）。
- *Fast Inverse Lithography via Model-Driven Block-Stacking CNN*, arXiv:2412.14599（物理驱动、无标注训练）。
- *Stochastic defect generation in EUV analyzed by spatially correlated probability model*, Fukuda, SPIE 11147
  （二次电子空间相关 / 反应-散射限）。
- LithoBench / ICCAD13（ILT 公开基准，端到端对拍用）。

---

### 优先级建议（若资源受限，先做这 5 条）
1. **S10.1 Flow-matching 生成式算子骨干** ★ — 一处改动、三域受益，把 Phase 9 的"确定性 codomain 骨干 + 分布头"
   升级为"生成式 + 不确定度感知集成 + 长程漂移抑制"，被 N10.1/O10.1/C10.1 直接复用（Flow Marching 2025 + UniFluids 2026）。
2. **C10.1 生成式超分辨子格闭合** ★ — 复用 C9.1 检查点/伴随，把确定性闭合升级为保高频 + 自带 UQ 的生成式闭合
   （arXiv:2406.20047），并把集成喂决策门。
3. **N10.1 物理引导潜空间扩散反设计** ★ — 复用 codomain 编码器 + RCWA 可微前向 + 量化精修，补上 2026 光子反设计
   唯一缺的"生成式直接映射"一环（MxDiffusion / AIGP）。
4. **O10.2 High-NA EUV anamorphic SMO + shot-count** ★ — 把 OpenLithoHub 从低 NA 各向同性推到 2026 主战场（high-NA
   anamorphic + 曲线掩模可量产性），是计算光刻领域相关性的硬指标。
5. **跨仓 §5「GPU 实测 + 第三方对拍全面收口」** — 兑现 Phase 9 留下的"待回填"，把四仓 honesty boundaries 从
   "无验证"升级为"已对拍 + 残差量化"。

> **资源前置依赖**：S10.1 / N10.2 / C10.2 / C10.3 标 🖥️ 的任务需要至少一台 CUDA GPU。若暂无 GPU，可先在 CPU 完成
> 正确性与接口（flow-matching 核、生成式闭合、潜空间扩散、in-context、实验设计、anamorphic 建模、shot-count 代价项
> 都可在 CPU 验证数值正确性与 toy 规模迁移），GPU 实测与第三方对拍留到资源到位，并在 README 继续诚实标注。
