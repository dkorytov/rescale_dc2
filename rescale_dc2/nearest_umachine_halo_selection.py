"""
"""
import numpy as np
from galsampler.cython_kernels import galaxy_selection_kernel
from astropy.table import Table
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box


__all__ = ('source_halo_selection_indices', 'value_add_matched_target_halos',
    'source_galaxy_selection_indices', 'create_galsampled_dc2')


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
    ngal_tot = int(np.sum(target_halos['richness'][indices]))

    first_indices = target_halos['first_galaxy_index'][indices].astype(long)
    richness = target_halos['richness'][indices].astype('i4')

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
                          'obs_sm', 'obs_sfr', 'sfr_percentile_fixed_sm')
    for key in source_galaxy_keys:
        dc2[key] = umachine[key][galaxy_indices]

    x_init = dc2['target_halo_x'] + dc2['host_centric_x']
    vx_init = dc2['target_halo_vx'] + dc2['host_centric_vx']
    dc2['x'], dc2['vx'] = enforce_periodicity_of_box(x_init, 256., velocity=vx_init)

    y_init = dc2['target_halo_y'] + dc2['host_centric_y']
    vy_init = dc2['target_halo_vy'] + dc2['host_centric_vy']
    dc2['y'], dc2['vy'] = enforce_periodicity_of_box(y_init, 256., velocity=vy_init)

    z_init = dc2['target_halo_z'] + dc2['host_centric_z']
    vz_init = dc2['target_halo_vz'] + dc2['host_centric_vz']
    dc2['z'], dc2['vz'] = enforce_periodicity_of_box(z_init, 256., velocity=vz_init)

    return dc2
