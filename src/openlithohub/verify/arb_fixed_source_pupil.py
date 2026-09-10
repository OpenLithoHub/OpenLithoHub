"""Optional Arb-certified fixed-source hard-pupil coherent integral.

For one physical illumination source point ``s=(sx,sy)`` the scalar coherent
field under the OpenLithoHub paraxial defocus convention is modeled as

    E_s(X,z) = integral_{|u|<=fp} Mhat(u-s)
               exp(pi*i*z*lambda*|u|^2)
               exp(2*pi*i*(u-s).X) du,

with ``fp=NA/lambda``. Mapping ``u=fp*rho*(cos(theta),sin(theta))`` removes the
hard pupil indicator and gives an analytic integral on
``rho in [0,1], theta in [0,2*pi]``.

python-flint/Arb ``acb.integral`` is loaded lazily. This module certifies only
the fixed-source pupil integral; source-plane integration remains a separate
proof obligation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from openlithohub.verify.interface_runs import HorizontalRun


class ArbRunSource(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def pixel_size_nm(self) -> float: ...

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[HorizontalRun]: ...


@dataclass(frozen=True)
class FixedSourcePupilParams:
    wavelength_nm: float
    na: float
    source_fx_per_nm: float
    source_fy_per_nm: float
    defocus_nm: float = 0.0

    def __post_init__(self) -> None:
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if self.na <= 0:
            raise ValueError("na must be positive")

    @property
    def pupil_radius_per_nm(self) -> float:
        return self.na / self.wavelength_nm


@dataclass(frozen=True)
class ArbIntegralCertificate:
    value_repr: str
    real_mid: float
    real_rad: float
    imag_mid: float
    imag_rad: float
    precision_bits: int
    abs_tol_decimal: str
    rel_tol_decimal: str
    finite: bool
    source_discretization_certified: bool = False
    pupil_integral_certified: bool = True
    semantics: str = "FIXED_SOURCE_HARD_PUPIL_ARB_BALL"

    @property
    def max_component_radius(self) -> float:
        return max(self.real_rad, self.imag_rad)


def _require_flint() -> tuple[Any, Any, Any]:
    try:
        from flint import acb, arb, ctx
    except ImportError as exc:
        raise ImportError("python-flint>=0.9.0 is required for Arb pupil certification") from exc
    return acb, arb, ctx


def _acb_run_rectangle_transform(
    run: HorizontalRun,
    *,
    pixel_nm: Any,
    fx: Any,
    fy: Any,
) -> Any:
    width = pixel_nm * (run.x1 - run.x0)
    height = pixel_nm
    cx = pixel_nm * (run.x0 + run.x1) / 2
    cy = pixel_nm * (2 * run.y + 1) / 2
    xterm = width * (fx * width).sinc_pi() * (-2 * fx * cx).exp_pi_i()
    yterm = height * (fy * height).sinc_pi() * (-2 * fy * cy).exp_pi_i()
    return xterm * yterm


def _acb_mask_transform(
    runs: tuple[HorizontalRun, ...],
    *,
    pixel_nm: Any,
    fx: Any,
    fy: Any,
) -> Any:
    total = fx * 0
    for run in runs:
        total += _acb_run_rectangle_transform(
            run,
            pixel_nm=pixel_nm,
            fx=fx,
            fy=fy,
        )
    return total


def certified_fixed_source_coherent_field(
    source: ArbRunSource,
    *,
    params: FixedSourcePupilParams,
    observation_x_nm: float,
    observation_y_nm: float,
    precision_bits: int = 96,
    abs_tol_decimal: str = "1e-10",
    rel_tol_decimal: str = "1e-10",
    eval_limit: int = 100000,
    depth_limit: int = 30,
) -> ArbIntegralCertificate:
    """Return a rigorous Arb ball for one fixed-source coherent field."""
    if precision_bits < 53:
        raise ValueError("precision_bits must be at least 53")
    h, w = source.shape
    runs = tuple(source.iter_runs_for_bbox((0, 0, w, h)))
    for run in runs:
        if not (0 <= run.y < h and 0 <= run.x0 < run.x1 <= w):
            raise ValueError("source returned a run outside the finite layout")

    acb, arb, ctx = _require_flint()
    old_prec = ctx.prec
    try:
        ctx.prec = precision_bits
        p = arb(str(source.pixel_size_nm))
        wavelength = arb(str(params.wavelength_nm))
        fp = arb(str(params.na)) / wavelength
        sx = arb(str(params.source_fx_per_nm))
        sy = arb(str(params.source_fy_per_nm))
        z = arb(str(params.defocus_nm))
        xobs = arb(str(observation_x_nm))
        yobs = arb(str(observation_y_nm))
        abs_tol = arb(abs_tol_decimal)
        rel_tol = arb(rel_tol_decimal)
        two_pi = 2 * arb.pi()

        def theta_integrand(theta: Any, analytic_theta: bool) -> Any:
            del analytic_theta
            c = theta.cos()
            s = theta.sin()

            def rho_integrand(rho: Any, analytic_rho: bool) -> Any:
                del analytic_rho
                ux = fp * rho * c
                uy = fp * rho * s
                fx = ux - sx
                fy = uy - sy
                mhat = _acb_mask_transform(runs, pixel_nm=p, fx=fx, fy=fy)
                defocus = (z * wavelength * (ux * ux + uy * uy)).exp_pi_i()
                image_phase = (2 * (fx * xobs + fy * yobs)).exp_pi_i()
                return mhat * defocus * image_phase * fp * fp * rho

            return acb.integral(
                rho_integrand,
                0,
                1,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                eval_limit=eval_limit,
                depth_limit=depth_limit,
            )

        value = acb.integral(
            theta_integrand,
            0,
            two_pi,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eval_limit=eval_limit,
            depth_limit=depth_limit,
        )
        return ArbIntegralCertificate(
            value_repr=str(value),
            real_mid=float(value.real.mid()),
            real_rad=float(value.real.rad()),
            imag_mid=float(value.imag.mid()),
            imag_rad=float(value.imag.rad()),
            precision_bits=precision_bits,
            abs_tol_decimal=abs_tol_decimal,
            rel_tol_decimal=rel_tol_decimal,
            finite=bool(value.is_finite()),
        )
    finally:
        ctx.prec = old_prec
