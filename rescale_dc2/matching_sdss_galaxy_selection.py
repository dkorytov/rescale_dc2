"""
"""
import numpy as np
from scipy.spatial import cKDTree


__all__ = ('sdss_kd_tree_indices', )


def sdss_kd_tree_indices(mstar_mock, sfr_percentile_mock, logsm_sdss, sfr_percentile_sdss):
    """
    """
    sdss_tree = cKDTree(np.vstack((logsm_sdss, sfr_percentile_sdss)).T)

    nn_distinces, nn_indices = sdss_tree.query(
        np.vstack((np.log10(mstar_mock), sfr_percentile_mock)).T, k=1)

    return nn_distinces, nn_indices
