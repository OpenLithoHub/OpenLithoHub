"""EUV stochastic robustness evaluation."""

from __future__ import annotations

import torch

from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub._utils.morphology import distance_transform
from openlithohub._utils.tensor_ops import ensure_2d


def _count_connected_components(binary: torch.Tensor) -> int:
    """Count connected components using iterative erosion-based labeling."""
    remaining = binary.clone()
    count = 0
    while remaining.sum() > 0:
        seed = remaining.nonzero(as_tuple=False)[0]
        region = torch.zeros_like(remaining)
        region[seed[0], seed[1]] = 1.0
        prev_sum = 0.0
        while True:
            dilated = torch.nn.functional.max_pool2d(
                region.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
            ).squeeze(0).squeeze(0)
            region = dilated * remaining
            curr_sum = region.sum().item()
            if curr_sum == prev_sum:
                break
            prev_sum = curr_sum
        remaining = remaining - region
        remaining = remaining.clamp(min=0.0)
        count += 1
    return count


def compute_stochastic_robustness(
    mask: torch.Tensor,
    num_trials: int = 100,
    dose_photons_per_nm2: float = 30.0,
    pixel_size_nm: float = 1.0,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate mask robustness against EUV photon shot noise.

    Simulates stochastic resist exposure via Poisson photon noise to quantify
    probability of micro-bridging and line breaks.
    """
    m = ensure_2d(mask)
    binary = (m > 0.5).float()

    sigma_px = 2.0
    aerial_nominal = simulate_aerial_image(binary, sigma_px=sigma_px, dose=1.0)
    resist_nominal = apply_resist_threshold(aerial_nominal, threshold=0.5)

    nominal_fg_components = _count_connected_components(resist_nominal)
    nominal_bg_components = _count_connected_components((resist_nominal < 0.5).float())

    pixel_area_nm2 = pixel_size_nm * pixel_size_nm
    lambda_map = aerial_nominal.clamp(min=0.0) * dose_photons_per_nm2 * pixel_area_nm2

    generator = torch.Generator(device=mask.device)
    if seed is not None:
        generator.manual_seed(seed)

    bridge_count = 0
    break_count = 0
    ler_values: list[float] = []

    nominal_edge_dist = distance_transform(resist_nominal)
    nominal_edges = (nominal_edge_dist > 0) & (nominal_edge_dist <= 1.5)

    batch_size = min(10, num_trials)
    trials_done = 0

    while trials_done < num_trials:
        current_batch = min(batch_size, num_trials - trials_done)

        for _ in range(current_batch):
            if seed is not None:
                trial_seed = seed + trials_done
                generator.manual_seed(trial_seed)

            photons = torch.poisson(lambda_map, generator=generator)
            noisy_intensity = photons / max(dose_photons_per_nm2 * pixel_area_nm2, 1e-12)
            noisy_resist = apply_resist_threshold(noisy_intensity, threshold=0.5)

            noisy_fg_components = _count_connected_components(noisy_resist)
            noisy_bg_components = _count_connected_components((noisy_resist < 0.5).float())

            if noisy_fg_components < nominal_fg_components:
                bridge_count += 1
            if noisy_bg_components < nominal_bg_components:
                break_count += 1

            diff = (noisy_resist - resist_nominal).abs()
            if nominal_edges.any():
                edge_displacement = diff[nominal_edges].mean().item() * pixel_size_nm
                ler_values.append(edge_displacement)

            trials_done += 1

    bridge_probability = bridge_count / max(num_trials, 1)
    break_probability = break_count / max(num_trials, 1)
    ler_mean_nm = sum(ler_values) / max(len(ler_values), 1) if ler_values else 0.0
    robustness_score = max(0.0, 1.0 - (bridge_probability + break_probability) / 2.0)

    return {
        "bridge_probability": bridge_probability,
        "break_probability": break_probability,
        "ler_mean_nm": ler_mean_nm,
        "robustness_score": robustness_score,
    }
