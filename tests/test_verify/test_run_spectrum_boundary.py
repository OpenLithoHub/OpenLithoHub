import numpy as np
import pytest

from openlithohub.verify.run_spectrum_boundary import (
    KernelSupport,
    ZeroPaddingContract,
    stable_geometric_run_sum,
)


def test_geometric_sum_zero_frequency_exact():
    assert stable_geometric_run_sum(frequency_index=0, x0=2, x1=7, period=8) == 5 + 0j


def test_geometric_sum_matches_direct_exponential():
    for k in range(8):
        got = stable_geometric_run_sum(frequency_index=k, x0=1, x1=7, period=8)
        want = sum(np.exp(-2j * np.pi * k * x / 8) for x in range(1, 7))
        assert abs(got - want) < 2e-14


def test_zero_padding_contract_rejects_one_missing_side():
    with pytest.raises(ValueError):
        ZeroPaddingContract(
            physical_shape=(24, 32),
            pad_left=0,
            pad_top=1,
            pad_right=1,
            pad_bottom=1,
            kernel_support=KernelSupport(left=1, top=1, right=1, bottom=1),
        )


def test_zero_padding_contract_accepts_sufficient_support():
    z = ZeroPaddingContract(
        physical_shape=(24, 32),
        pad_left=2,
        pad_top=3,
        pad_right=2,
        pad_bottom=3,
        kernel_support=KernelSupport(left=2, top=3, right=2, bottom=3),
    )
    assert z.padded_shape == (30, 36)
