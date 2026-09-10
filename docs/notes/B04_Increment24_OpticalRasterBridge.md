# B04 Increment 24 — Optical Raster-to-Contour Bridge

Increment 23 proved only a finite-grid mask statement:

- exact-vector mask: 90032 occupied pixels;
- current dense loader: 96258 occupied pixels;
- all 6226 disagreements are dense false positives;
- mutual Chebyshev radius: 1 pixel.

That result must **not** be read as an 11.31 nm printed-contour EPE bound.

For SOCS

\[
F(m)=\sum_j w_j |E_j(m)|^2,
\]

write \(D_j=E_j(\widetilde m-m)\).  Then exactly

\[
F(\widetilde m)-F(m)
=
\sum_j w_j\left(
2\Re(E_j(m)\overline{D_j})+|D_j|^2
\right).
\]

Thus if \(|E_j(m)|\le A_j\) and \(|D_j|\le B_j\) on the verification core,

\[
\|F(\widetilde m)-F(m)\|_\infty
\le
\sum_j w_j(2A_jB_j+B_j^2)=:\delta_I.
\]

If the reference threshold band of width at least \(\delta_I\) has certified

\[
|\nabla F|\ge\kappa>0,
\]

then the sharp level-set transfer gives

\[
d_H(\Gamma_{\widetilde m},\Gamma_m)\le\delta_I/\kappa.
\]

Under unique oriented-slab edge pairing:

- continuous contour EPE is at most \(\delta_I/\kappa\);
- paired CD error is at most \(2\delta_I/\kappa\).

The local benchmark is intentionally diagnostic.  It chooses the 128x128 crop
of the Increment-23 fixture with the largest exact-vs-dense disagreement,
runs both masks through the same K24 Hopkins model over focus/dose nodes, and
reports:

- max aerial-intensity discrepancy;
- thresholded printed-raster EPE;
- a grid finite-difference gradient diagnostic;
- the *diagnostic* ratio `delta_I / kappa_grid`.

No continuous/certified contour claim is made until the gradient floor is
replaced by an interval-certified full-band transversality bound.
