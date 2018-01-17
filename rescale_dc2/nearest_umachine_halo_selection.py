"""
"""
import numpy as np
from galsampler.cython_kernels import galaxy_selection_kernel
from astropy.table import Table
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box


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


def value_add_matched_target_halos(source_halos, target_halos, indices):
    """
    """
    target_halos['source_halo_id'] = source_halos['halo_id'][indices]
    target_halos['matching_mvir'] = source_halos['mvir'][indices]
    target_halos['matching_log10_nd'] = source_halos['log10_cumulative_nd_mvir'][indices]
    target_halos['richness'] = source_halos['richness'][indices]
    target_halos['first_galaxy_index'] = source_halos['first_galaxy_index'][indices]
    return target_halos


def source_galaxy_selection_indices(target_halos, indices):
    ngal_tot = int(np.sum(target_halos['richness'][source_halo_selection_indices]))

    first_indices = target_halos['first_galaxy_index'][source_halo_selection_indices].astype(long)
    richness = target_halos['richness'][source_halo_selection_indices].astype('i4')

    return np.array(galaxy_selection_kernel(first_indices, richness, ngal_tot))


def create_galsampled_dc2(umachine, target_halos, halo_indices, galaxy_indices):
    """
    """
    dc2 = Table()
    dc2['source_halo_id'] = umachine['hostid'][galaxy_indices]
    dc2['target_halo_id'] = np.repeat(
        target_halos['halo_id'][halo_indices], target_halos['richness'][halo_indices])

    idxA, idxB = crossmatch(dc2['target_halo_id'], target_halos['halo_id'])

    msg = "target IDs do not match!"
    assert np.all(dc2['source_halo_id'][idxA] == target_halos['source_halo_id'][idxB]), msg

    target_halo_keys = ('x', 'y', 'z', 'vx', 'vy', 'vz')
    for key in target_halo_keys:
        dc2['target_halo_'+key] = 0.
        dc2['target_halo_'+key][idxA] = target_halos[key][idxB]
    dc2['target_halo_mass'] = 0.
    dc2['target_halo_mass'][idxA] = target_halos['fof_halo_mass'][idxB]

    source_galaxy_keys = ('host_halo_mvir', 'upid',
                       'host_centric_x', 'host_centric_y', 'host_centric_z',
                       'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
                          'obs_sm', 'obs_sfr')
    for key in source_galaxy_keys:
        dc2[key] = umachine[key][galaxy_indices]

    dc2['x'] = enforce_periodicity_of_box(dc2['target_halo_x'] + dc2['host_centric_x'], 256.)
    dc2['y'] = enforce_periodicity_of_box(dc2['target_halo_y'] + dc2['host_centric_y'], 256.)
    dc2['z'] = enforce_periodicity_of_box(dc2['target_halo_z'] + dc2['host_centric_z'], 256.)

    return dc2
