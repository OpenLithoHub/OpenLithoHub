# Lithography for AI Engineers

A bilingual cheat-sheet that maps computational-lithography terminology onto the AI / deep-learning vocabulary you already know. Each entry has the standard definition, the closest AI analogue, a canonical reference, and a one-line Chinese summary so you can use it in either community.

写给 AI 工程师的计算光刻术语对照表 — 每条都给出英文定义、AI 类比、参考文献，外加一句中文注释，方便跨语境讨论。

---

## Mask

**Definition.** The patterned reticle whose transmission profile is projected through the scanner onto the wafer. Digitally, a binary or grayscale 2-D image at sub-nanometer pixel pitch.

**AI analogue.** Input image / output tensor of an image-to-image network. A "predicted mask" is the network's output; a "target mask" is the design intent (label).

**Reference.** Wong, *Resolution Enhancement Techniques in Optical Lithography*, SPIE Press.

**中文.** 掩膜 — 在 AI 视角下就是一张二维图像张量；模型预测它，设计层提供监督。

---

## OPC (Optical Proximity Correction)

**Definition.** Pre-distorting the mask so that, after projection through a diffraction-limited lens and resist development, the printed wafer pattern matches the design. Industrially solved by rule-based or model-based iterative correction.

**AI analogue.** Image-to-image inverse generation. Given the desired output (`design`), produce the input (`mask`) such that a known forward operator maps `mask → design`. A discriminative U-Net trained on `(design, mask)` pairs is the simplest learned variant.

**Reference.** Yang et al., *GAN-OPC*, DAC 2018.

**中文.** 光学邻近校正 — 类似图像逆问题：已知期望输出，反推输入。

---

## ILT (Inverse Lithography Technology)

**Definition.** Frame OPC as a continuous optimization: parametrize the mask, define a differentiable forward model (`mask → aerial image → resist`), and minimize a loss against the target design.

**AI analogue.** Differentiable optimization with a physics-based loss — the same shape as score-matching or PINNs. Curvilinear ILT closely mirrors learned image priors regularized by a physics simulator.

**Reference.** Pang et al., *Inverse Lithography Technology Principles in Practice*, JM3 2021.

**中文.** 逆向光刻 — 把 OPC 写成可微优化问题，物理仿真器作为可微 loss。

---

## SRAF (Sub-Resolution Assist Feature)

**Definition.** Small auxiliary mask features placed near main features to improve process window. They print sub-resolution (i.e., do not appear on the wafer) but bias the local diffraction pattern.

**AI analogue.** Auxiliary input/output channels — tokens that are part of the prediction but not part of the evaluation target. Conceptually similar to side outputs / deep supervision in segmentation networks.

**Reference.** Liebmann et al., *SPIE Advanced Lithography*, 2003.

**中文.** 辅助特征 — 不直接成像但影响主特征的"辅助通道"。

---

## EPE (Edge Placement Error)

**Definition.** Distance, in nanometers, between the predicted contour and the target contour at sampled edge points. The single most important per-feature accuracy metric.

**AI analogue.** Edge-aligned regression loss / pixel-wise distance metric. Closer to chamfer distance than to L2 — only edge pixels contribute, and the metric is asymmetric in some formulations.

**Reference.** Mack, *Fundamental Principles of Optical Lithography*, Wiley.

**中文.** 边缘放置误差 — 你可以把它当作 chamfer distance 的工业版。

---

## MRC / DRC (Mask / Design Rule Check)

**Definition.** Hard-fail compliance gating: minimum width, minimum spacing, minimum area, minimum curvature radius. A mask that fails MRC cannot be manufactured regardless of optical performance.

**AI analogue.** Constraint satisfaction layer — a post-hoc validity filter, similar to projection back onto a feasible set in constrained optimization.

**Reference.** SEMI P39, EasyMRC docs.

**中文.** 掩膜 / 设计规则检查 — 硬约束，类似可行域投影；不满足直接拒绝。

---

## Resist Model

**Definition.** Model that maps the aerial image (light intensity at the wafer) to the developed resist contour. Industrial models are calibrated empirically; learned variants substitute a CNN.

**AI analogue.** Forward surrogate model — the SciML pattern of replacing a PDE solver with a neural network. The "true" forward model is computationally expensive; the surrogate is cheap and differentiable.

**Reference.** Mack, *Fundamental Principles of Optical Lithography*, Ch. 11.

**中文.** 光刻胶模型 — 物理仿真的"代理网络"，让你能微分通过它优化掩膜。

---

## PV-Band (Process Variation Band)

**Definition.** The envelope swept by resist contours over a dose/focus process window. Width of the band measures how stable the printed pattern is under realistic manufacturing variation.

**AI analogue.** Aleatoric uncertainty band — analogous to a confidence interval predicted by a probabilistic model under input perturbation.

**Reference.** Sturtevant et al., SPIE Advanced Lithography, 2010.

**中文.** 工艺变动带 — 对应贝叶斯模型中的不确定性区间。

---

## Aerial Image

**Definition.** Light intensity distribution at the wafer plane after the projection optics, before the resist responds. Computed via Hopkins / Abbe imaging from the mask plus illumination + lens kernel.

**AI analogue.** Intermediate latent representation — the activation in the middle of a forward pipeline. Many learned ILT models predict the aerial image as a byproduct.

**Reference.** Hopkins, *Proc. R. Soc. A* (1953).

**中文.** 空间像 — 光在晶圆平面的强度分布，在管线里相当于"中间层激活"。

---

## Hotspot Detection

**Definition.** Identify layout patches that are likely to fail under nominal lithography conditions (bridge, break, necking). Trained on (layout patch, hotspot label) pairs from prior tape-outs.

**AI analogue.** Pixel-level or patch-level anomaly classification — same task structure as defect inspection, semantic segmentation of defects, or out-of-distribution detection.

**Reference.** Yang et al., *DAC 2017* / Lin et al., *ICCAD 2016*.

**中文.** 热点检测 — 标准的分块分类 / 语义分割问题，标签来自历史 tape-out 数据。

---

## Stochastic Failure (EUV-specific)

**Definition.** EUV photons arrive Poisson-distributed; resist response is also stochastic. Below a critical dose, low-photon variance produces random bridges, breaks, and missing contacts.

**AI analogue.** Monte Carlo sampling under input noise — similar to evaluating a model's robustness to per-pixel Gaussian perturbation, but the noise here is physically grounded photon shot noise.

**Reference.** De Bisschop, *J. Micro/Nanolith. MEMS MOEMS*, 2017.

**中文.** EUV 随机失效 — 光子打靶的统计性涨落，可用蒙特卡洛仿真评估。

---

## Process Window

**Definition.** The set of (dose, focus) pairs over which all CD/EPE/MRC constraints are simultaneously satisfied. A wider window means the recipe is more robust to scanner drift.

**AI analogue.** Feasible region in the input-perturbation parameter space — analogous to the set of latent perturbations that preserve a classifier's decision.

**Reference.** Sturtevant et al., SPIE Advanced Lithography, 2010.

**中文.** 工艺窗口 — 各类约束同时满足的 (dose, focus) 集合，类似分类器的稳健决策区域。
