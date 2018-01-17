"""
"""
import numpy as np


__all__ = ('source_halo_selection_indices', )


def my_randint(low, high):
    return np.floor(np.random.uniform(low, high)).astype(int)


def source_halo_selection_indices(source_log10_cumulative_nd, target_log10_cumulative_nd):
    idx_nearest_mass = np.searchsorted(
        source_log10_cumulative_nd, target_log10_cumulative_nd)

    window_half_length = 2
    low = idx_nearest_mass - window_half_length
    low = np.where(low < 0, 0, low)

    high = idx_nearest_mass + window_half_length
    high = np.where(high > len(idx_nearest_mass)-1, len(idx_nearest_mass)-1, high)

    return my_randint(low, high)
