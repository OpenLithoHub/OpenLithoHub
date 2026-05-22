"""EUV stochastic robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub._utils.morphology import binary_dilation, connected_components, distance_transform
from openlithohub._utils.tensor_ops import ensure_2d


@dataclass(frozen=True)
class _NominalState:
    """Cached nominal-image quantities shared by stochastic metrics.

    Both ``compute_stochastic_robustness`` and ``compute_stochastic_defect_classes``
    need the same aerial image, resist threshold, FG/BG labels, and Poisson
    rate map for one input mask. Computing them once and passing the result
    avoids ~2× redundant FFTs and connected-component passes when the
    metric pack runs both functions on the same mask.
    """

    binary: torch.Tensor
    aerial_nominal: torch.Tensor
    resist_nominal: torch.Tensor
    fg_labels: torch.Tensor
    bg_labels: torch.Tensor
    lambda_map: torch.Tensor
    pixel_area_nm2: float
    dose_scale: float


def _nominal_state(
    mask: torch.Tensor,
    dose_photons_per_nm2: float,
    pixel_size_nm: float,
    sigma_px: float = 2.0,
) -> _NominalState:
    """Compute the per-mask nominal quantities reused across stochastic trials."""
    m = ensure_2d(mask)
    binary = (m > 0.5).float()
    aerial_nominal = simulate_aerial_image(binary, sigma_px=sigma_px, dose=1.0)
    resist_nominal = apply_resist_threshold(aerial_nominal, threshold=0.5)
    fg_labels, _ = connected_components(resist_nominal, connectivity=8)
    nominal_bg = 1.0 - resist_nominal
    bg_labels, _ = connected_components(nominal_bg, connectivity=8)
    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    dose_scale = dose_photons_per_nm2 * pixel_area_nm2
    lambda_map = aerial_nominal.clamp(min=0.0) * dose_scale
    return _NominalState(
        binary=binary,
        aerial_nominal=aerial_nominal,
        resist_nominal=resist_nominal,
        fg_labels=fg_labels,
        bg_labels=bg_labels,
        lambda_map=lambda_map,
        pixel_area_nm2=pixel_area_nm2,
        dose_scale=dose_scale,
    )


def compute_stochastic_robustness(
    mask: torch.Tensor,
    num_trials: int = 100,
    dose_photons_per_nm2: float = 30.0,
    pixel_size_nm: float = 1.0,
    seed: int | None = 0,
) -> dict[str, float]:
    """Evaluate mask robustness against EUV photon shot noise.

    Simulates stochastic resist exposure via Poisson photon noise to quantify
    probability of micro-bridging and line breaks.

    ``seed`` defaults to ``0`` so leaderboard runs are reproducible. Pass
    ``seed=None`` to draw from system entropy (intentional non-determinism,
    e.g. ensemble runs).
    """
    state = _nominal_state(mask, dose_photons_per_nm2, pixel_size_nm)
    resist_nominal = state.resist_nominal
    fg_labels = state.fg_labels

    nominal_fg_label_set: set[int] = {
        int(v) for v in torch.unique(fg_labels[fg_labels >= 0]).tolist()
    }

    generator = torch.Generator(device=mask.device)
    if seed is not None:
        generator.manual_seed(seed)

    bridge_count = 0
    break_count = 0
    edge_flip_values: list[float] = []

    nominal_edge_dist = distance_transform(resist_nominal)
    nominal_edges = (nominal_edge_dist > 0) & (nominal_edge_dist <= 1.5)

    for _ in range(num_trials):
        photons = torch.poisson(state.lambda_map, generator=generator)
        noisy_intensity = photons / max(state.dose_scale, 1e-12)
        noisy_resist = apply_resist_threshold(noisy_intensity, threshold=0.5)

        # Per-component matching: a trial may simultaneously merge some
        # nominal lines and break others. The previous net-component-count
        # heuristic made these events cancel; tracking them independently
        # lets each trial contribute to both bridge and break probability.
        noisy_fg_labels, _ = connected_components(noisy_resist, connectivity=8)

        bridge_in_trial = False
        if nominal_fg_label_set:
            unique_noisy = torch.unique(noisy_fg_labels[noisy_fg_labels >= 0]).tolist()
            for noisy_lbl in unique_noisy:
                noisy_component = noisy_fg_labels == int(noisy_lbl)
                overlapping_nominal = torch.unique(fg_labels[noisy_component])
                overlapping_nominal = overlapping_nominal[overlapping_nominal >= 0]
                if int(overlapping_nominal.numel()) >= 2:
                    bridge_in_trial = True
                    break

        break_in_trial = False
        for nominal_lbl in nominal_fg_label_set:
            component_mask = fg_labels == nominal_lbl
            sub = (noisy_resist > 0.5) & component_mask
            if not bool(sub.any()):
                continue
            _, n_pieces = connected_components(sub.float(), connectivity=8)
            if n_pieces >= 2:
                break_in_trial = True
                break

        if bridge_in_trial:
            bridge_count += 1
        if break_in_trial:
            break_count += 1

        # Fraction of nominal edge-band pixels whose binary state flipped under
        # photon noise. This is dimensionless (0–1), NOT a line-edge roughness
        # in nm — true LER would require sub-pixel chord-displacement along
        # each contour normal. Reported separately so users don't conflate it
        # with published EUV LER numbers (~1–5 nm).
        diff = (noisy_resist - resist_nominal).abs()
        if nominal_edges.any():
            edge_flip_values.append(diff[nominal_edges].mean().item())

    bridge_probability = bridge_count / max(num_trials, 1)
    break_probability = break_count / max(num_trials, 1)
    edge_flip_rate = (
        sum(edge_flip_values) / max(len(edge_flip_values), 1) if edge_flip_values else 0.0
    )
    robustness_score = max(0.0, 1.0 - (bridge_probability + break_probability) / 2.0)

    return {
        "bridge_probability": bridge_probability,
        "break_probability": break_probability,
        "edge_flip_rate": edge_flip_rate,
        "robustness_score": robustness_score,
    }


@dataclass(frozen=True)
class StochasticDefectRates:
    """Per-class stochastic defect rates in failures per cm^2.

    The four classes follow the imec EUV stochastic-defectivity convention
    (microbridge / broken line / missing contact / merged contact). Per-cm^2
    rates are the industry reporting unit and let users compare against
    published defectivity floors regardless of mask tile size.
    """

    microbridge_per_cm2: float
    broken_line_per_cm2: float
    missing_contact_per_cm2: float
    merged_contact_per_cm2: float
    total_per_cm2: float
    num_trials: int
    image_area_cm2: float


def _classify_components(
    labels: torch.Tensor,
    num_components: int,
    contact_aspect_max: float,
    contact_area_max: int,
) -> tuple[set[int], set[int]]:
    """Split component labels into contact-like and line-like sets.

    A component is "contact-like" if its area is small AND its bounding-box
    aspect ratio is close to 1; otherwise it is "line-like". Picking these
    cuts via area+aspect rather than label semantics keeps the classifier
    self-contained — callers do not have to tag components in advance.
    """
    contacts: set[int] = set()
    lines: set[int] = set()
    if num_components == 0:
        return contacts, lines

    fg_labels = labels[labels >= 0]
    unique = torch.unique(fg_labels).tolist()
    for lbl in unique:
        ys, xs = torch.where(labels == lbl)
        area = int(ys.numel())
        if area == 0:
            continue
        h_ext = int(ys.max().item() - ys.min().item() + 1)
        w_ext = int(xs.max().item() - xs.min().item() + 1)
        long_side = max(h_ext, w_ext)
        short_side = max(1, min(h_ext, w_ext))
        aspect = long_side / short_side
        if area <= contact_area_max and aspect <= contact_aspect_max:
            contacts.add(int(lbl))
        else:
            lines.add(int(lbl))
    return contacts, lines


def _trial_defect_classes(
    nominal_resist: torch.Tensor,
    noisy_resist: torch.Tensor,
    nominal_fg_labels: torch.Tensor,
    nominal_bg_labels: torch.Tensor,
    nominal_contacts: set[int],
    nominal_lines: set[int],
    nominal_bg_holes: set[int],
) -> tuple[int, int, int, int]:
    """Count microbridge / broken-line / missing-contact / merged-contact in one trial.

    - Microbridge: a *line-like* nominal foreground component fuses with another
      foreground component in the noisy image (drop in FG component count caused
      by line-shaped components that share their label region in the noisy image).
    - Broken line: a *line-like* nominal foreground component splits into two or
      more pieces in the noisy image (extra FG components inside its support).
    - Missing contact: a *contact-like* nominal foreground component disappears
      entirely from the noisy resist.
    - Merged contact: two distinct nominal background holes (the spaces between
      contact pads) coalesce in the noisy image.
    """
    microbridge = 0
    broken_line = 0
    missing_contact = 0
    merged_contact = 0

    nominal_fg = nominal_resist > 0.5
    noisy_fg = noisy_resist > 0.5

    for lbl in nominal_contacts:
        component_mask = nominal_fg_labels == lbl
        if not bool((component_mask & noisy_fg).any()):
            missing_contact += 1

    for lbl in nominal_lines:
        component_mask = nominal_fg_labels == lbl
        sub = noisy_fg & component_mask
        if not bool(sub.any()):
            continue
        _, n_pieces = connected_components(sub.float(), connectivity=8)
        if n_pieces >= 2:
            broken_line += 1

    if nominal_lines:
        nominal_lines_mask = torch.zeros_like(nominal_fg)
        for lbl in nominal_lines:
            nominal_lines_mask = nominal_lines_mask | (nominal_fg_labels == lbl)
        # Only count bridge components that actually touch a nominal line —
        # photon-noise blobs in the far field are not microbridges. Label the
        # extra-foreground region, then keep components whose support overlaps
        # the dilated nominal-lines mask (1-pixel tolerance for adjacency).
        bridges_only = noisy_fg & ~nominal_fg
        if bool(bridges_only.any()):
            line_neighbourhood = binary_dilation(nominal_lines_mask.float(), radius=1) > 0.5
            bridge_labels, _n_bridge = connected_components(bridges_only.float(), connectivity=8)
            unique_bridge_labels = torch.unique(bridge_labels[bridge_labels >= 0]).tolist()
            for blbl in unique_bridge_labels:
                comp = bridge_labels == blbl
                if bool((comp & line_neighbourhood).any()):
                    microbridge += 1

    for hole_lbl in nominal_bg_holes:
        hole_mask = nominal_bg_labels == hole_lbl
        if not bool((hole_mask & ~noisy_fg).any()):
            merged_contact += 1

    return microbridge, broken_line, missing_contact, merged_contact


def compute_stochastic_defect_classes(
    mask: torch.Tensor,
    num_trials: int = 100,
    dose_photons_per_nm2: float = 30.0,
    pixel_size_nm: float = 1.0,
    seed: int | None = 0,
    contact_aspect_max: float = 1.5,
    contact_area_max: int = 64,
) -> StochasticDefectRates:
    """Per-class EUV stochastic defect rates in failures/cm^2.

    Extends :func:`compute_stochastic_robustness` (which returns aggregate
    bridge/break probabilities) with the four imec-style defect classes
    reported by the EUV stochastic-defectivity literature: microbridges,
    broken lines, missing contacts, and merged contacts. Output is
    normalised to failures per cm^2 so results are comparable across
    different mask tile sizes.

    Args:
        mask: Real-valued mask tensor (H, W) or 4D, values in [0, 1].
        num_trials: Number of Poisson trials. More trials → tighter rate
            estimates; 100 is a reasonable benchmarking default.
        dose_photons_per_nm2: Exposure dose in photons / nm^2 at the wafer.
            Scales the Poisson rate map.
        pixel_size_nm: Mask pixel size in nm; used both for the Poisson
            rate scaling and for converting failure counts to per-cm^2.
        seed: Optional RNG seed.
        contact_aspect_max: Maximum bounding-box long/short ratio for a
            component to count as contact-like. Lines are everything else.
        contact_area_max: Maximum pixel area for a component to count as
            contact-like. Tune for the contact size on your process node.

    Returns:
        StochasticDefectRates with per-class and total failure rates.
    """
    state = _nominal_state(mask, dose_photons_per_nm2, pixel_size_nm)
    resist_nominal = state.resist_nominal
    nominal_fg_labels = state.fg_labels
    nominal_bg_labels = state.bg_labels

    nominal_contacts, nominal_lines = _classify_components(
        nominal_fg_labels,
        num_components=int(torch.unique(nominal_fg_labels[nominal_fg_labels >= 0]).numel()),
        contact_aspect_max=contact_aspect_max,
        contact_area_max=contact_area_max,
    )
    bg_hole_labels = torch.unique(nominal_bg_labels[nominal_bg_labels >= 0]).tolist()
    h, w = resist_nominal.shape
    nominal_bg_holes: set[int] = set()
    for lbl in bg_hole_labels:
        ys, xs = torch.where(nominal_bg_labels == lbl)
        if ys.numel() == 0:
            continue
        touches_border = bool(
            (ys == 0).any() or (ys == h - 1).any() or (xs == 0).any() or (xs == w - 1).any()
        )
        if touches_border:
            continue
        nominal_bg_holes.add(int(lbl))

    generator = torch.Generator(device=mask.device)
    if seed is not None:
        generator.manual_seed(seed)

    microbridges = 0
    broken_lines = 0
    missing_contacts = 0
    merged_contacts = 0

    for _ in range(num_trials):
        photons = torch.poisson(state.lambda_map, generator=generator)
        noisy_intensity = photons / max(state.dose_scale, 1e-12)
        noisy_resist = apply_resist_threshold(noisy_intensity, threshold=0.5)

        mb, bl, mc, mr = _trial_defect_classes(
            nominal_resist=resist_nominal,
            noisy_resist=noisy_resist,
            nominal_fg_labels=nominal_fg_labels,
            nominal_bg_labels=nominal_bg_labels,
            nominal_contacts=nominal_contacts,
            nominal_lines=nominal_lines,
            nominal_bg_holes=nominal_bg_holes,
        )
        microbridges += mb
        broken_lines += bl
        missing_contacts += mc
        merged_contacts += mr

    image_area_nm2 = float(h * w * state.pixel_area_nm2)
    image_area_cm2 = image_area_nm2 * 1e-14
    norm_per_cm2 = 1.0 / max(num_trials * image_area_cm2, 1e-30)

    rates = StochasticDefectRates(
        microbridge_per_cm2=microbridges * norm_per_cm2,
        broken_line_per_cm2=broken_lines * norm_per_cm2,
        missing_contact_per_cm2=missing_contacts * norm_per_cm2,
        merged_contact_per_cm2=merged_contacts * norm_per_cm2,
        total_per_cm2=(microbridges + broken_lines + missing_contacts + merged_contacts)
        * norm_per_cm2,
        num_trials=num_trials,
        image_area_cm2=image_area_cm2,
    )
    return rates
