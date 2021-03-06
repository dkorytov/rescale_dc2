"""
"""
import os
import numpy as np
import fnmatch
from astropy.table import Table


__all__ = ('load_protoDC2_fof_halos', 'list_available_protoDC2_fof_fnames', 'load_bolshoi_planck_halos')

fof_dirname = "/Users/aphearin/Dropbox/protoDC2/fof_halos"
bpl_dirname = "/Users/aphearin/Dropbox/protoDC2/umachine/host_halos"


def load_bolshoi_planck_halos(fname):
    halos = Table.read(fname, path='data')

    Vbox_bpl = 250.**3
    halos.sort('mvir')
    halos['log10_cumulative_nd_mvir'] = np.log10(
        np.arange(len(halos), 0, -1)/Vbox_bpl)
    return halos[::-1]


def load_protoDC2_fof_halos(fname):
    """
    """
    pdc2_halos = Table.read(fname, path='data')

    pdc2_halos.rename_column('fof_halo_center_x', 'x')
    pdc2_halos.rename_column('fof_halo_center_y', 'y')
    pdc2_halos.rename_column('fof_halo_center_z', 'z')

    pdc2_halos.rename_column('fof_halo_mean_vx', 'vx')
    pdc2_halos.rename_column('fof_halo_mean_vy', 'vy')
    pdc2_halos.rename_column('fof_halo_mean_vz', 'vz')

    pdc2_halos.rename_column('fof_halo_vel_disp', 'vel_disp')
    pdc2_halos.rename_column('fof_halo_tag', 'halo_id')

    Vbox_aq = 256.**3
    pdc2_halos.sort('fof_halo_mass')
    pdc2_halos['log10_cumulative_nd_mvir'] = np.log10(
        np.arange(len(pdc2_halos), 0, -1)/Vbox_aq)
    return pdc2_halos[::-1]


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield os.path.join(path, filename)


def list_available_protoDC2_fof_fnames():
    return list(fname_generator(fof_dirname, '*.hdf5'))


def list_available_bpl_halo_fnames():
    return list(fname_generator(bpl_dirname, '*.hdf5'))
