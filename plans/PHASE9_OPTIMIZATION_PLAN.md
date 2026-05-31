# Phase 9 优化执行计划 — 文献驱动的下一步增量 (2026-05-31)

> 承接 `PHASE8_OPTIMIZATION_PLAN.md`。Phase 8 已落地的工作 **不再重复**，本文件只规划「未来提升」。
> 经对四仓 `main` 实测核对，Phase 8 的以下成果已确认入库，作为 Phase 9 的基线（不再列为任务）：
>
> - **DiffNano**：`fdtd3d.py` 已含 `backward="time_reversal"` 时间反演伴随（N8.1）；`workflows/lpa_metalens.py`
>   LPA 大孔径前向（N8.2）；`design/latent_warm_start.py` 潜空间多候选暖启动（N8.3）；`solvers/backend_diagnostics.py`
>   后端工况诊断（N8.4，meent 对拍仅在 README/plan 标注、GPU 待回填）。
> - **OpenLithoHub**：`benchmark/metrics/stochastic_loss.py` 已含 **可微 CVaR + pinball 分位数**（O8.1 完成度高）；
>   `models/posterior_warm_start.py` **VAE 后验采样**多候选（O8.2，但**无 RL 微调**）；`workflow/full_chip_tiling.py`
>   + `inference/multiproc.py`（O8.3 CPU 多进程，GPU 待回填）；`models/surrogate_ilt.py`/`_utils/resist_physics.py`（O8.4）。
> - **DiffCFD**：`solvers/transient_adjoint.py` 瞬态离散伴随（C8.3）；`props/sco2_uncertainty.py` 接共形 UQ（C8.4）；
>   `envs/hybrid_control.py` **AD warm-start PPO + AD-augmented advantage**（C8.2 完成）；`validation/external_crossval.py`（C8.1）。
> - **diff-surrogate**：`conformal.py`（split-CP + risk-controlling quantile，S8.1）；`structure.py`（散度/通量守恒投影，S8.2）；
>   `pretraining.py`（**MLP** 多任务预训练 + few-shot，S8.3）；`generative.py`（VAE/能量模型采样器 + 打分协议，S8.4）；
>   `active_sampling.py` **已把共形带宽接入采样排序**（S8.1 延伸）。
>
> 每条任务给出：动机文献 → 具体改动 → 验收标准（DoD）。
> 进度标记：`[ ]` 待办 · `🔬` 需先做文献复算 · `⚠️` 有许可/合规约束 · `🖥️` 需 GPU/规模化资源

---

## 0. 跨仓总主线（Phase 9 的三条战略线）

Phase 8 把「**校准覆盖率（calibrated coverage）**」与「**梯度保真度**」并列为一等公民，并打通了「时间反演伴随 /
LPA / 共形 / 结构保持」的 CPU 正确性。Phase 9 在此之上再推三步，核心是把 Phase 8 建好的「可信原语」**从"会报告"
变成"会决策、会在线学习、会跨域迁移"**：

1. **从「报告不确定度」走向「不确定度驱动决策（decision-grade UQ in the loop）」** — Phase 8 的共形带宽目前主要
   *被报告* 和用于主动采样排序，但下游反演/优化的**接受/拒绝、停机、风险预算**还没真正消费它。2025–2026 的
   *Probabilistic Neural Operators*（PNO，arXiv:2502.12902）把 UQ 用严格 proper scoring rule **直接进训练**而非事后
   贴片；与共形结合可得到「训练即校准 + 决策即风险控制」的闭环。Phase 9 把「共形/概率 UQ 作为优化回路的显式
   决策门」做成统一验收门（与 Phase 8 的覆盖率门衔接：覆盖率达标 → 进一步要求"带宽真的改变了决策"）。

2. **从「离线代理 + 单步梯度门」走向「在线/事后（a-posteriori）可微学习 + 长程稳定性」** — Phase 8 验证了单步/
   稳态梯度保真，但**长 rollout 的稳定性与分布漂移**还没被系统处理。2026 年可微物理主线已明确转向
   **solver-in-the-loop（在线 / a-posteriori）训练**：闭合模型嵌进可微求解器、按展开多步的 a-posteriori 误差反传，
   显著优于 a-priori 离线训练的稳定性（arXiv:2604.23874, 2026-04；arXiv:2504.03870, 2026-02；JFM 2022 学习闭合的
   "展开步数→稳定性"曲线）。Phase 9 给 DiffCFD 补「学习型闭合的 solver-in-the-loop 训练」，并把展开步数/检查点/
   稳定性的权衡曲线做成跨仓通用的「长程梯度」回归门。

3. **从「单任务 MLP 预训练 + 结构保持」走向「可迁移多物理基础底座（transferable multiphysics backbone）」** —
   Phase 8 的 `pretraining.py` 是 **MLP** 多任务，编码器对"物理场的数量/种类变化"不具适应性。2024–2026 的
   **codomain/通道注意力**路线（CoDA-NO, NeurIPS 2024, arXiv:2403.12553：在 NS 上预训练→**不改架构**迁移到 FSI，
   few-shot 误差降 ~36.8%）与 **adapter 式多物理预训练**（arXiv:2511.10829, 2025）、**PDE-FM/The Well 跨物理基准**
   （arXiv:2511.21861, 2025）给出了"变场数、变物理、低成本迁移"的范式。Phase 9 把共享底座升级为 codomain-attention
   + adapter 的可迁移骨干，让 EM↔litho↔CFD 三域共享一个能"加/减物理场"的编码器。

---

## 1. DiffNano — 电磁求解器与反设计

**Phase 8 基线**：RCWA 四后端 + 后端工况诊断；FDTD3D 已具 `backward="time_reversal"`（98% 显存路径，CPU 正确性已验）
+ 梯度检查点；LPA 大孔径前向 + `TwoStageOptimizer`；`latent_warm_start` 潜空间多候选暖启动。

### N9.1 — 可制造离散反设计：直通估计器（STE）+ 量化/二值化几何，护栏化 🔬
- **动机**：DiffNano 现有反设计走的是连续松弛 + 投影路线，**离散/工艺约束（最小线宽量化、二值化材料、
  非可微 shape 参数化）仍是事后投影**。*Quantized Inverse Design for Photonic Integrated Circuits*（ACS Omega，
  arXiv:2407.10273）用**直通梯度估计（straight-through estimator, STE）**让"不可微的离散/量化 shape 参数化"也能
  端到端反传，直接面向 2PP/多束写等真实工艺约束生成可制造结构。这与 OpenLithoHub 的 MRC 投影理念同源，但发生在
  EM 几何参数侧。
- **改动**：① 在 `design/` 新增 `quantized.py`：可配置量化层（k-level / 二值）+ STE 反传，含"量化误差 vs 量化步距"
  诊断；② 与现有 `projection.py`（连续投影）做对照——量化感知优化 vs 事后量化的 FoM 退化曲线；③ 提供"量化噪声护栏"
  （STE 在量化跳变处的梯度方向与有限差分一致性检查，复用 Phase 7 梯度保真门）。
- **DoD**：① 量化感知反设计 FoM 优于"连续优化 + 事后量化"（≥ N 种子，Wilcoxon）；② STE 梯度 vs 有限差分方向余弦
  > 0.95（量化跳变区允许放宽，需在测试中显式标注容差）；③ ≥ 8 新测试通过。

### N9.2 — FDTD GPU 实路径：systolic 更新 + 与 FDTDX 交叉对拍（收口 N8.1/§5）🖥️🔬
- **动机**：N8.1 时间反演伴随已在 CPU 验证数值正确，但"质变显存收益"必须在 GPU 大网格上才兑现。2026 综述
  *Inverse design for scalable photonic systems*（Nat. Rev. Mater., 2026-04）引用的 **GPU 显存带宽瓶颈 systolic
  更新方案**（Lu et al., IEEE Antennas Propag. Mag., 2026）与开源 **FDTDX**（JOSS 11:8912, 2026，多 GPU 3D
  AD-FDTD）是当前最可对拍的 GPU 参照。
- **改动**：① 在有 CUDA 的机器上回填 FDTD3D 前向 + 时间反演伴随的 GPU 基准（与 `use_checkpoint`、纯 AD 三方
  显存/速度曲线）；② 评估 systolic / 分块更新对显存带宽的影响（即便先做朴素 GPU 版，文档化带宽瓶颈）；③ 与
  FDTDX 在 ≥ 1 个 3D 算例上前向相对误差 < 1e-3、梯度方向余弦 > 0.99（**仅数值参照，不 vendoring**，CI 跑 `scancode`）。
- **⚠️ 合规**：systolic 思路从论文 clean-room；FDTDX 仅作外部数值对照。
- **DoD**：① GPU 基准三方曲线（time_reversal / checkpoint / AD）；② 与 FDTDX 交叉对拍达标；③ 无 GPU 时坦诚保留
  CPU-only 标注并给出 CPU 上的正确性回归；④ ≥ 5 新测试通过。

### N9.3 — metasurface 反设计的入射角/工艺鲁棒后验暖启动（对接 S9.3 概率 UQ）🔬
- **动机**：*Inverse Design of Nanophotonic Color Router Robust to Oblique Incidence*（Adv. Opt. Mater. 14(4),
  2026）等 2026 工作把**斜入射 / 工艺角**鲁棒性作为一等目标。DiffNano 已有 `latent_warm_start` 与 `robustness/`
  子空间鲁棒，但二者未联动：暖启动只给 nominal 候选，鲁棒优化再单独跑。
- **改动**：① 让 `latent_warm_start` 直接采"角度/工艺角扰动下表现稳健"的候选族（消费 N9.1 量化几何 + Phase 8
  `robustness/` 的 corner 评估作为打分）；② 与 S9.3 的概率/共形带宽联动，按"最坏角分位数"而非均值选优。
- **DoD**：① 鲁棒后验暖启动在斜入射/工艺角 sweep 上最坏分位 FoM 优于 nominal 暖启动；② 候选打分协议与
  OpenLithoHub/diff-surrogate 一致；③ ≥ 6 新测试通过。

---

## 2. OpenLithoHub — 计算光刻基准与工作流

**Phase 8 基线**：可微 CVaR + pinball 分位数随机感知 loss（O8.1）；VAE 后验暖启动多候选（O8.2，**无 RL 微调**）；
全芯片 Schwarz 拼接 + CPU 多进程（O8.3，GPU 待回填）；resist/ILT 代理真实化 + 梯度门（O8.4）。

### O9.1 — 生成式暖启动的 RL 微调：GRPO + ILT-引导模仿（升级 O8.2 后验采样）🔬 ★
- **动机**：O8.2 已做"后验掩模族 → 批量精修选优"，但**生成器本身没有按 ILT 目标被强化学习微调**——这是 2026
  SOTA 的关键。*Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning*（NVIDIA,
  arXiv:2602.19027, 2026-02）把掩模合成重述为**条件采样**：WGAN + 重建预训练，再用 **GRPO（Group Relative Policy
  Optimization）+ ILT-引导模仿 loss** 微调，使生成器学到"真正能逃离非凸陷阱"的后验，**首次在 LithoBench/ICCAD13
  大量样例上把 EPE 违例压到 3nm 以下、并取得 3× 加速**；同时提出按 **post-PnR 版图条件**的 style-aware 插件。
- **改动**：① 新增 `models/grpo_warm_start.py`：在 `PosteriorWarmStart` 之上加 GRPO 微调环（reward = 既有
  `CandidateScorer` 的 EPE/PVB/MRC 组合，group-relative 归一化优势，无需价值网络）；② style-aware 条件接口
  （消费 post-PnR / layer-purpose 上下文，复用 `workflow/layer_purpose.py`）；③ 对照"VAE 后验（O8.2）vs GRPO 微调"
  的逃逸非凸能力（多 hotspot 的最终 EPE 分布）。
- **⚠️ 合规**：仅从论文 clean-room 复现机制；NVIDIA 公开权重不可用时坦诚标注「结构演示，非 SOTA 数字」，module
  docstring 列 references-consulted，CI 跑 `scancode`。
- **DoD**：① GRPO 微调后的后验在 ICCAD13 子集上 EPE 违例分布优于 O8.2 VAE 后验（回归表）；② style-aware 条件
  可切换并影响候选分布；③ 候选协议与 DiffNano N9.3 一致；④ ≥ 10 新测试通过。

### O9.2 — 全芯片拼接 GPU batch 化 + LithoBench/ICCAD13 端到端对拍（收口 O8.3/§5）🖥️
- **动机**：O8.3 已做 CPU 多进程 Schwarz 拼接 + 残差量化，但 2026 综述（Light: Sci. Appl. 2025-07；
  EurekAlert 2025-08）反复指出 **full-chip ILT 必须 GPU 加速以消除子分区伪影**；NVIDIA GPU 曲线 ILT
  （arXiv:2411.07311）与 cuLitho 已进 TSMC 生产线（DCD, 2026-03）。本任务把 O8.3 的 CPU 路径升级为 GPU tile-batch，
  并补公开基准端到端数字。
- **改动**：① Schwarz 拼接前向/ILT 的 GPU tile-batch 化（保留 CPU 多进程退化路径）；② 在 LithoBench/ICCAD13
  子集跑端到端 EPE/PVB/MRC 并与公开数字交叉对拍（**仅数值参照，不 vendoring**）；③ 把"拼接残差随 tile 数 + GPU
  batch 大小"的回归曲线补进 README。
- **DoD**：① GPU tile-batch 可用（或诚实 CPU-only 标注）；② ICCAD13 子集端到端指标表 + 与公开结果讨论；
  ③ 拼接残差回归曲线；④ ≥ 8 新测试通过。

### O9.3 — 随机感知 ILT 的覆盖率门：把 EUV stochastic 进 O8.1 loss 并配共形区间（对接 S9.x）🔬
- **动机**：O8.1 的 CVaR/分位数 loss 目前作用在**确定性误差分布**上；EUV 随机性（光子散粒噪声、LCDU、
  through-focus）应进一步配**有覆盖率保证的区间**，而非仅经验分位。配合 §0 主线一，把"随机过程窗口"从点估计升级
  为"带共形覆盖的区间"。
- **改动**：① 用 `stochastic.py` 的随机采样 + diff-surrogate `conformal` 给 LCDU/随机 EPE 的 through-focus 分位
  做覆盖率校准；② PV-band × 随机区间联合输出"带覆盖保证的过程窗口"；③ 把覆盖率达标作为随机感知 ILT 的验收门。
- **DoD**：① 随机 EPE/LCDU 区间达目标覆盖率（90/95%）；② 过程窗口输出带覆盖率标注；③ ≥ 6 新测试通过。

---

## 3. DiffCFD — 可微流体与逆向设计 / RL

**Phase 8 基线**：瞬态离散伴随（C8.3）；sCO₂ 共形 UQ（C8.4）；`hybrid_control` AD warm-start PPO + AD-augmented
advantage（C8.2）；外部交叉对拍脚手架（C8.1）。`turbulence.py` 目前仅 **冻结/代数闭合**（mixing-length、frozen
eddy viscosity），**无学习型闭合**。

### C9.1 — Solver-in-the-loop 学习型闭合：a-posteriori 展开训练 + 稳定性曲线 🔬 ★
- **动机**：DiffCFD 已具可微 NS 求解器与瞬态伴随，但闭合仍是**离线/代数**。2026 可微物理主线明确：**a-posteriori
  （在线 / solver-in-the-loop）训练**——把 NN 闭合嵌进可微求解器、按**展开多步**的 a-posteriori 误差反传——比
  a-priori 离线训练**稳定性与泛化显著更好**（*Deep Learning of Solver-Aware Turbulence Closures from Nudged LES*,
  arXiv:2604.23874, 2026-04；*A posteriori closure: are symmetries preserved?* arXiv:2504.03870, 2026-02；JFM 2022
  给出"展开步数↑→稳定性↑"曲线）。这正好复用 C8.3 的检查点/伴随基础设施。
- **改动**：① 新增 `solvers/learned_closure.py`：可训练 SGS/eddy-viscosity 修正项，嵌入可微 NS rollout；
  ② solver-in-the-loop 训练环（按 N 步展开 a-posteriori loss，检查点控显存，复用 C8.3 tape）；③ 对照 a-priori
  离线训练 vs a-posteriori 在线训练的**部署稳定性**（长 rollout 发散率 + 高阶统计量）+ "展开步数→稳定性"曲线。
- **DoD**：① a-posteriori 闭合在长 rollout 上比 a-priori 更稳定（发散率 + 能谱/结构函数回归）；② 展开步数→稳定性
  曲线复现文献定性趋势；③ 长 rollout 显存随展开步受控（检查点）；④ ≥ 8 新测试通过。

### C9.2 — 与 Diff-FlowFSI 的 GPU 交叉对拍 + 长 rollout 显存策略文档化（收口 C8.1/§5）🖥️🔬
- **动机**：C8.1 已有外部对拍脚手架与解析解对拍，但缺与**第三方可微 CFD**在同一算例上的 GPU 实测对拍。
  *Diff-FlowFSI*（CMAME 2025, arXiv:2505.23940，JAX GPU，隐式函数定理穿压力 Poisson + 检查点控长 rollout）是当前
  最对位的开源参照。
- **改动**：① 在 ≥ 1 共有算例（顶盖驱动腔 / FSI 弹性边界）与 Diff-FlowFSI 做前向 + 梯度交叉对拍（**仅数值参照，
  不 vendoring**）；② 有 GPU 时回填 SIMPLE 前向 + 隐式/伴随反传的 GPU 基准；③ 把 C9.1 的展开-检查点策略对照
  Diff-FlowFSI 的 implicit+checkpoint 混合，文档化。
- **DoD**：① 与 Diff-FlowFSI 前向相对误差 < 2%、梯度方向余弦 > 0.99；② GPU 基准表（或诚实 CPU-only 标注）；
  ③ 长 rollout 显存策略说明 + 回归。

### C9.3 — 主动流控的 codomain 迁移：跨雷诺数/几何的策略与代理迁移（对接 S9.1）🔬🖥️
- **动机**：C8.2 的 AD-warm-start PPO 在**单工况**学习；*Discovering Flow Separation Control in 3D Wings via
  DRL*（arXiv:2509.10185, 2025）与多物理迁移线指向"跨雷诺数/几何复用策略与代理"。结合 S9.1 的 codomain/adapter
  迁移底座，可让流控代理"加减物理场、换工况"低成本迁移。
- **改动**：① 把 `hybrid_control` 的代理换成 S9.1 的 codomain-attention 骨干 + adapter；② 在源工况预训练、目标
  工况（变 Re / 变几何）few-shot 微调，对照从头训；③ 迁移增益回归化（达到目标控制性能所需样本比）。
- **DoD**：① 迁移微调样本效率优于从头训（≥ 2 个目标工况）；② AD warm-start 与迁移可叠加不互斥；③ ≥ 6 新测试通过。

---

## 4. diff-surrogate — 统一可微代理框架（共享库）

**Phase 8 基线**：`conformal.py`（split-CP + risk-controlling quantile）；`structure.py`（守恒投影）；
`pretraining.py`（**MLP** 多任务 + few-shot）；`generative.py`（VAE/能量采样 + 打分协议）；`active_sampling.py`
已把共形带宽接入采样排序。

### S9.1 — Codomain-attention + adapter 可迁移骨干：变场数多物理底座 🔬🖥️ ★
- **动机**：S8.3 的 `pretraining.py` 是 **MLP**，编码器对"物理场数量/种类"不可变。**CoDA-NO**（NeurIPS 2024,
  arXiv:2403.12553）把注意力放在 **codomain（通道/变量空间）**，在 NS（u,v,p）上预训练后**不改架构**即可迁到 FSI
  （加 dₓ,d_y 两个场），few-shot 误差平均降 ~36.8%；*Towards Universal Neural Operators through Multiphysics
  Pretraining*（arXiv:2511.10829, 2025）进一步用 **adapter** 做低成本下游迁移；*PDE-FM / The Well*
  （arXiv:2511.21861, 2025）给出跨物理基准。这是把"几何感知"升级为"可迁移基础模型雏形"的标准路径。
- **改动**：① 新增 `codomain.py`：codomain-attention 编码器（变量数可变，掩码重建自监督预训练目标）；
  ② adapter / 问题特定头，支持"加/减物理场"迁移；③ 与 S8.2 结构保持投影正交组合；④ 跨物理 few-shot 基准
  （toy Darcy/NS/反应扩散 → FSI/litho-代理）。
- **DoD**：① codomain 骨干在"加场"迁移（如 NS→FSI toy）few-shot 误差显著低于 S8.3 MLP 从头/多任务基线；
  ② adapter 微调成本 < 全量微调且不显著掉点；③ 公共 API 导出，DiffNano N9.3 / DiffCFD C9.3 可复用；④ ≥ 10 新测试通过。

### S9.2 — 概率神经算子（PNO）：把 UQ 用 proper scoring rule 进训练（与共形互补）🔬
- **动机**：S8.1 的共形是**事后校准**；*Probabilistic Neural Operators*（arXiv:2502.12902, 2025）用**严格 proper
  scoring rule** 把不确定度**直接进训练**学一个函数空间上的预测分布，对现有架构改动小、在多域上 UQ 质量更好。
  与共形结合 → "训练即校准（PNO）+ 事后有限样本保证（split-CP）"双层 UQ。
- **改动**：① 新增 `probabilistic.py`：proper-scoring-rule（如 energy score / CRPS）训练目标 + 分布采样头；
  ② 与 `conformal.py` 串联（PNO 出分布 → split-CP 校准其覆盖）；③ 与裸 Ensemble、纯共形三方对照（带宽 + 覆盖率 +
  采样多样性）。
- **DoD**：① PNO+共形的带宽在同覆盖率下优于裸 Ensemble+共形；② OOD 上覆盖率不灾难性崩塌；③ ≥ 8 新测试通过。

### S9.3 — 决策门化的 UQ：把覆盖率从"指标"变成优化回路的接受/停机/风险预算 🔬 ★
- **动机**：§0 主线一。Phase 8 已能"报告覆盖率"且 `active_sampling` 用共形带宽排序，但下游反演/选优的**显式决策**
  （何时接受候选、何时停机、风险预算如何分配）还没消费它。这是把"校准 UQ"真正变现为"决策可信"的一步。
- **改动**：① 新增 `decision.py`：基于共形/PNO 带宽的接受-拒绝规则、CVaR 风险预算分配、覆盖率触发的早停；
  ② 暴露统一协议供 DiffNano（N9.3 最坏角分位选优）、OpenLithoHub（O9.3 过程窗口）、DiffCFD（C8.4 物性区间）消费；
  ③ 对照"带宽真的改变了决策"——同覆盖率下不同决策规则的下游 FoM/风险曲线。
- **DoD**：① 决策门使下游选优在固定预算下风险（如最坏分位）显著下降；② 三仓各接入一处决策门并跑通；
  ③ 决策规则的覆盖率-FoM 权衡曲线；④ ≥ 8 新测试通过。

---

## 5. 跨仓收尾（Phase 9 的统一门控）

- [ ] **决策门化 UQ 成为统一验收门**：四仓各至少一条核心优化/反演路径，把 S9.3 的"共形/PNO 带宽 → 接受/停机/
      风险预算"决策门接入（与 Phase 8 的覆盖率门衔接：从"覆盖率达标"升级到"带宽改变了决策"）。
- [ ] **GPU 实路径全面收口**：DiffNano↔FDTDX（N9.2）、OpenLithoHub↔LithoBench/ICCAD13（O9.2）、DiffCFD↔Diff-FlowFSI
      （C9.2）各至少一条 **GPU 实测** 交叉对拍（仅数值参照，**不 vendoring**，CI 跑 `scancode`）。无 GPU 时坦诚保留
      CPU-only 标注 + CPU 正确性回归。
- [ ] **长程稳定性门**：C9.1 的"展开步数→稳定性"曲线做成跨仓"长 rollout 梯度"通用回归（DiffCFD 主用，DiffNano
      时域 FDTD rollout 可复用同一门）。
- [ ] **可迁移底座跨仓复用**：S9.1 codomain+adapter 骨干被 DiffCFD C9.3、DiffNano N9.3 复用；迁移增益（few-shot
      数据效率比）回填各仓 README 竞品/参考表。
- [ ] **生成式 RL 暖启动协议统一**：O9.1 的 GRPO 后验微调与 DiffNano N9.3 鲁棒后验暖启动共享 `CandidateScorer`
      与候选采样协议。
- [ ] **全仓 CI 全绿 + 诚实边界更新**：四仓 + diff-surrogate 联合 CI（含 `scancode`）一次性绿；把"GPU 基准 /
      第三方对拍 / 在线学习稳定性"按本期实际进展逐条更新 honesty boundaries。

---

## 6. 关键参考文献（Phase 9 新增 / 重点，2024–2026）

**不确定度 / 决策门化（主线一）**
- *Probabilistic Neural Operators for Functional Uncertainty Quantification*, arXiv:2502.12902, 2025（proper
  scoring rule 把 UQ 进训练，函数空间预测分布）。
- Ma, Azizzadenesheli, Anandkumar, *Calibrated UQ for Operator Learning via Conformal Prediction*,
  arXiv:2402.01960（risk-controlling quantile NO；Phase 8 已引，Phase 9 用于决策门衔接）。

**在线 / a-posteriori 可微学习（主线二）**
- *Deep Learning of Solver-Aware Turbulence Closures from Nudged LES Dynamics*, arXiv:2604.23874, 2026-04
  （solver-in-the-loop / a-posteriori 闭合，稀疏观测，部署稳定性）。
- *A posteriori closure of turbulence models: are symmetries preserved?*, arXiv:2504.03870, 2026-02
  （a-posteriori 展开训练缓解协变量漂移 + 对称性）。
- *Learned Turbulence Modelling with Differentiable Fluid Solvers*, JFM, 2022（"展开步数↑→稳定性↑"曲线）。
- *Differentiable Turbulence: Closure as PDE-constrained optimization*, arXiv:2307.03683（a-posteriori 优于 a-priori）。
- Diff-FlowFSI, *GPU-optimized differentiable CFD/FSI*, CMAME 2025, arXiv:2505.23940（JAX GPU 参照，隐式 + 检查点）。

**多物理基础模型 / 迁移（主线三）**
- CoDA-NO, *Pretraining Codomain Attention Neural Operators for Multiphysics PDEs*, NeurIPS 2024,
  arXiv:2403.12553（通道/codomain 注意力，NS→FSI 不改架构迁移，few-shot −36.8%）。
- *Towards Universal Neural Operators through Multiphysics Pretraining*, arXiv:2511.10829, 2025（adapter 式迁移）。
- *Towards a Foundation Model for PDEs Across Physics Domains* (PDE-FM / The Well), arXiv:2511.21861, 2025
  （跨物理 12 数据集，VRMSE −46%）。

**电磁 / 光子反设计**
- *Quantized Inverse Design for Photonic Integrated Circuits*, ACS Omega, arXiv:2407.10273（直通估计器，离散/工艺
  约束可微化）。⚠️ clean-room。
- *Inverse design for scalable photonic systems*, Nature Reviews Materials, 2026-04（GPU systolic FDTD、FDTDX 参照综述）。
- FDTDX, *High-performance open-source FDTD with AD*, J. Open Source Softw. 11:8912, 2026（多 GPU 3D AD-FDTD 参照）。
- *Inverse Design of Nanophotonic Color Router Robust to Oblique Incidence*, Adv. Optical Mater. 14(4), 2026
  （斜入射鲁棒反设计）。

**计算光刻**
- *Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning*, arXiv:2602.19027, 2026-02
  （WGAN 预训练 + GRPO + ILT-引导模仿；post-PnR style-aware；EPE < 3nm、3× 加速）。⚠️ clean-room。
- *GPU-Accelerated Inverse Lithography Towards High Quality Curvy Mask Generation*, arXiv:2411.07311（NVIDIA 曲线 ILT）。
- cuLitho 进 TSMC 生产（DataCenterDynamics, 2026-03；生成式 AI 再 2× 提速）。
- *Advancements and challenges in ILT: AI-based review*, Light: Sci. & Appl., 2025-07（full-chip 拼接伪影需 GPU）。
- LithoBench / ICCAD13（ILT 公开基准，交叉对拍用）。

---

### 优先级建议（若资源受限，先做这 5 条）
1. **S9.1 Codomain+adapter 可迁移骨干** ★ — 一处改动、三域受益，把 Phase 8 的"单任务 MLP 预训练"真正升级为
   "可迁移多物理底座"，并被 C9.3/N9.3 直接复用（CoDA-NO 2024 + 多物理 adapter 2025 背书）。
2. **O9.1 GRPO 生成式暖启动** ★ — 复用 Phase 8 VAE 后验 + `CandidateScorer`，补上 2026 SOTA 唯一缺的"RL 微调"
   一环（arXiv:2602.19027），逃逸非凸能力质变。
3. **C9.1 Solver-in-the-loop 学习闭合** ★ — 复用 C8.3 检查点/伴随基础设施，把可微 CFD 从"求解器"推向"在线学习
   闭合"，是 2026 可微物理主线（arXiv:2604.23874）。
4. **S9.3 决策门化 UQ** ★ — 低成本、四仓共享，把 Phase 8 已建好的共形覆盖率从"指标"变现为"决策可信"，兑现
   主线一的收口。
5. **跨仓 §5「GPU 实路径全面收口」** — N9.2/O9.2/C9.2 把四仓从"GPU 待回填"推到"GPU 实测 + 第三方对拍"，是
   Phase 9 工程门控的硬指标。

> **资源前置依赖**：N9.2 / O9.2 / C9.2 / S9.1 / C9.3 标 🖥️ 的任务需要至少一台 CUDA GPU。若暂无 GPU，可先完成
> CPU 正确性与接口（量化 STE、GRPO 环、solver-in-the-loop 闭合、codomain 骨干、决策门都可在 CPU 验证数值正确性
> 与 toy 规模迁移），GPU 基准与第三方实测对拍留到资源到位，并在 README 继续诚实标注。
