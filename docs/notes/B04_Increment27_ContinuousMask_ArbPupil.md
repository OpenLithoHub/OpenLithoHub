# B04 Increment 27 — Continuous Square-Aperture Lift and Fixed-Source Arb Pupil

## Repository baseline

Increment 27 is rebased after Increment 26 plus the public `verify_layout()` /
`WorkAccounting` additions. The canonical readiness scoreboard still lists
pupil/source quadrature remainder as the first physics blocker. Increment 27
therefore remains an independent proof backend and is not wired into the public
verification API yet.

## Exact continuous mask lift

The theorem mask is promoted to the continuous square-pixel aperture

\[
m_c(X,Y)=\sum_{(x,y)\in\mathcal M}
1_{[xp,(x+1)p)\times[yp,(y+1)p)}(X,Y).
\]

With Fourier convention

\[
\widehat m_c(f)=\int m_c(X)e^{-2\pi i f\cdot X}\,dX,
\]

one has exactly

\[
\boxed{
\widehat m_c(f_x,f_y)=
p^2\,\operatorname{sinc}_\pi(pf_x)\,
\operatorname{sinc}_\pi(pf_y)\,
M_{\rm point}(f_x,f_y).
}
\]

Thus Increment-26 pixel-center moments are connected to a continuous mask by
an explicit square-aperture factor; no silent point-mass-to-mask substitution
is allowed.

Relative to the declared `FINITE_EXACT_VECTOR_SQUARE_APERTURE_MASK` semantics,

\[
E_{\rm mask\ lift}=0.
\]

The upstream GDS-to-grid quantization convention remains explicit provenance.

## Fixed-source hard-pupil integral

For one physical source point \(s\),

\[
E_s(X,z)=\int_{|u|\le f_p}
\widehat m_c(u-s)
 e^{\pi i z\lambda |u|^2}
 e^{2\pi i(u-s)\cdot X}\,du,
\qquad f_p=NA/\lambda.
\]

Under

\[
u=f_p\rho(\cos\theta,\sin\theta),
\]

the hard pupil becomes a fixed rectangle
\(\rho\in[0,1],\theta\in[0,2\pi]\) with Jacobian \(f_p^2\rho\).
Every factor is entire in the mapped variables. Arb's normalized `sinc_pi`
handles the removable frequency-axis zeros without interval division through
zero.

The optional backend uses python-flint 0.9.0 `acb.integral` to produce a
rigorous complex ball for the fixed-source coherent field. The package offers
macOS ARM64 wheels and Python 3.10--3.14 support, matching the local execution
environment.

## Firewall

Certified after a successful local gate:

```text
exact finite vector geometry
-> continuous square-aperture transform
-> fixed-source hard-pupil coherent field ball
```

Still OPEN:

```text
source-plane integration / source discretization
partial-coherence intensity enclosure
continuous-HOPKINS vs sampled/SOCS runtime bridge
real-layout interval transversality
continuous EPE / CD / process window
```

The next theorem is a source-plane enclosure that composes fixed-source Arb
balls without reopening a spatial halo.
