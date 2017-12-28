"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ..csmf_resampling import target_source_bin_indices

fixed_seed = 43


def test_target_source_bin_indices1():
    num_source, num_target = int(4e4), int(6e4)
    nhalo_min = 1

    source_galaxy_host_mass = np.repeat((10, 11, 14, 15), int(num_source/4))
    target_galaxy_host_mass = np.repeat((10, 11, 12, 13, 14, 15), int(num_target/6))

    source_galaxy_is_central = np.tile((True, False), int(num_source/2))
    target_galaxy_is_central = np.tile((False, False, True, False), int(num_target/4))

    host_mass_bin_edges = np.arange(10, 17)-0.01

    source_bin_numbers, target_bin_numbers = target_source_bin_indices(
        source_galaxy_host_mass, target_galaxy_host_mass,
        source_galaxy_is_central, target_galaxy_is_central, host_mass_bin_edges)

    assert np.count_nonzero(source_bin_numbers == 0) > nhalo_min
    assert np.count_nonzero(source_bin_numbers == 1) > nhalo_min
    assert np.count_nonzero(source_bin_numbers == 2) == 0
    assert np.count_nonzero(source_bin_numbers == 3) == 0
    assert np.count_nonzero(source_bin_numbers == 4) > nhalo_min
    assert np.count_nonzero(source_bin_numbers == 5) > nhalo_min

    assert set(source_bin_numbers) == set((0, 1, 4, 5))

    mask = target_galaxy_is_central * (target_galaxy_host_mass == 10)
    assert np.all(target_bin_numbers[mask] == 0)
    mask = target_galaxy_is_central * (target_galaxy_host_mass == 11)
    assert np.all(target_bin_numbers[mask] == 1)
    mask = target_galaxy_is_central * (target_galaxy_host_mass == 12)
    assert np.all(target_bin_numbers[mask] == 1)

    assert set(target_bin_numbers) == set((0, 1, 4, 5))

