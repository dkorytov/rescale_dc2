"""
"""
import numpy as np
import os
import fnmatch
from astropy.table import Table
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch
from galsampler.utils import compute_richness
from galsampler.source_galaxy_selection import _galaxy_table_indices
from halotools.mock_observables import relative_positions_and_velocities as rel_posvel


umachine_dirname = "/Users/aphearin/Dropbox/protoDC2/umachine/mstar_1e8_cut"


__all__ = ('load_umachine_and_value_added_halos', )


def load_umachine_and_value_added_halos(fname, halos):
    umachine = add_hostid(apply_pbcs(Table.read(fname, path='data')), halos)
    umachine = umachine[umachine['has_matching_host']]
    umachine.sort(('hostid', 'upid'))

    halos['richness'] = compute_richness(halos['halo_id'], umachine['hostid'])

    halos['first_galaxy_index'] = _galaxy_table_indices(halos['halo_id'], umachine['hostid'])

    umachine = add_hostpos(umachine, halos)
    umachine = add_host_centric_pos(umachine, halos)
    umachine = add_ssfr(umachine)
    return umachine, halos


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield os.path.join(path, filename)


def list_available_umachine_fnames():
    return list(fname_generator(umachine_dirname, '*.hdf5'))


def apply_pbcs(catalog, Lbox=250.):
    catalog['x'] = enforce_periodicity_of_box(catalog['x'], Lbox)
    catalog['y'] = enforce_periodicity_of_box(catalog['y'], Lbox)
    catalog['z'] = enforce_periodicity_of_box(catalog['z'], Lbox)
    return catalog


def add_hostid(catalog, halos):
    catalog['hostid'] = catalog['upid']
    cenmask = catalog['upid'] == -1
    catalog['hostid'][cenmask] = catalog['id'][cenmask]

    idxA, idxB = crossmatch(catalog['hostid'], halos['halo_id'])
    catalog['has_matching_host'] = False
    catalog['has_matching_host'][idxA] = True
    return catalog


def add_hostpos(umachine, halos):
    """
    """
    idxA, idxB = crossmatch(umachine['hostid'], halos['halo_id'])
    umachine['host_halo_x'] = np.nan
    umachine['host_halo_y'] = np.nan
    umachine['host_halo_z'] = np.nan
    umachine['host_halo_vx'] = np.nan
    umachine['host_halo_vy'] = np.nan
    umachine['host_halo_vz'] = np.nan
    umachine['host_halo_mvir'] = np.nan

    umachine['host_halo_x'][idxA] = halos['x'][idxB]
    umachine['host_halo_y'][idxA] = halos['y'][idxB]
    umachine['host_halo_z'][idxA] = halos['z'][idxB]

    umachine['host_halo_vx'][idxA] = halos['vx'][idxB]
    umachine['host_halo_vy'][idxA] = halos['vy'][idxB]
    umachine['host_halo_vz'][idxA] = halos['vz'][idxB]

    umachine['host_halo_mvir'][idxA] = halos['mvir'][idxB]

    return umachine


def add_host_centric_pos(umachine, halos):
    """
    """
    Lbox = 250.
    xrel, vxrel = rel_posvel(umachine['x'], umachine['host_halo_x'],
        v1=umachine['vx'], v2=umachine['host_halo_vx'], period=Lbox)
    yrel, vyrel = rel_posvel(umachine['y'], umachine['host_halo_y'],
        v1=umachine['vy'], v2=umachine['host_halo_vy'], period=Lbox)
    zrel, vzrel = rel_posvel(umachine['z'], umachine['host_halo_z'],
        v1=umachine['vz'], v2=umachine['host_halo_vz'], period=Lbox)

    umachine['host_centric_x'] = xrel
    umachine['host_centric_y'] = yrel
    umachine['host_centric_z'] = zrel
    umachine['host_centric_vx'] = vxrel
    umachine['host_centric_vy'] = vyrel
    umachine['host_centric_vz'] = vzrel
    return umachine


def add_ssfr(catalog, quenched_sequence_center=-13.):
    ssfr = catalog['obs_sfr']/catalog['obs_sm']
    zero_mask = ssfr == 0.
    nzeros = np.count_nonzero(zero_mask)
    ssfr[zero_mask] = 10**np.random.normal(loc=quenched_sequence_center, scale=0.2, size=nzeros)
    catalog['obs_ssfr'] = np.log10(ssfr)
    return catalog
