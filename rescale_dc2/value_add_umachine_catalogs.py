"""
"""
import numpy as np
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch
from halotools.mock_observables import relative_positions_and_velocities as rel_posvel
from halotools.empirical_models import NFWPhaseSpace


pos_keys = ('x', 'y', 'z')
vel_keys = ('vx', 'vy', 'vz')
posvel_keys = np.concatenate((pos_keys, vel_keys))


def apply_pbcs(catalog, Lbox=250.):
    catalog['x'] = enforce_periodicity_of_box(catalog['x'], Lbox)
    catalog['y'] = enforce_periodicity_of_box(catalog['y'], Lbox)
    catalog['z'] = enforce_periodicity_of_box(catalog['z'], Lbox)
    return catalog


def add_host_keys(catalog, host_keys_to_add=('mvir', 'vmax'), Lbox=250.):
    """
    """
    idxA, idxB = crossmatch(catalog['hostid'], catalog['id'])
    catalog['host_halo_is_in_catalog'] = False
    catalog['host_halo_is_in_catalog'][idxA] = True

    for key in np.concatenate((host_keys_to_add, posvel_keys)):
        catalog['host_halo_'+key] = catalog[key]
        catalog['host_halo_'+key][idxA] = catalog[key][idxB]

    return catalog


def add_ssfr(catalog, quenched_sequence_center=-13.5):
    ssfr = catalog['obs_sfr']/catalog['obs_sm']
    zero_mask = ssfr == 0.
    nzeros = np.count_nonzero(zero_mask)
    ssfr[zero_mask] = 10**np.random.normal(loc=quenched_sequence_center, scale=0.2, size=nzeros)
    catalog['obs_ssfr'] = np.log10(ssfr)
    return catalog


def add_host_centric_posvel(catalog, redshift, Lbox=250.):
    """
    """
    mask = ~catalog['host_halo_is_in_catalog']
    ngals_nomatch = np.count_nonzero(mask)
    nfw = NFWPhaseSpace(redshift=redshift)
    posvel_table = nfw.mc_generate_nfw_phase_space_points(
        Ngals=ngals_nomatch, mass=catalog['host_halo_mvir'][mask])

    for key in posvel_keys:
        catalog[key][mask] = catalog['host_halo_'+key][mask] + posvel_table[key]

    xrel, vxrel = rel_posvel(catalog['x'], catalog['host_halo_x'],
        v1=catalog['vx'], v2=catalog['host_halo_vx'], period=Lbox)
    yrel, vyrel = rel_posvel(catalog['y'], catalog['host_halo_y'],
        v1=catalog['vy'], v2=catalog['host_halo_vy'], period=Lbox)
    zrel, vzrel = rel_posvel(catalog['z'], catalog['host_halo_z'],
        v1=catalog['vz'], v2=catalog['host_halo_vz'], period=Lbox)

    catalog['host_centric_x'] = xrel
    catalog['host_centric_y'] = yrel
    catalog['host_centric_z'] = zrel
    catalog['host_centric_vx'] = vxrel
    catalog['host_centric_vy'] = vyrel
    catalog['host_centric_vz'] = vzrel

    return catalog
