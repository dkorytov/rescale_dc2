"""
"""
import os
import numpy as np
from astropy.table import Table
import fnmatch
from .value_add_umachine_catalogs import apply_pbcs, add_host_keys


dropbox_dirname = "/Users/aphearin/Dropbox/protoDC2/umachine"


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield os.path.join(path, filename)


def _parse_scale_factor_from_umachine_sfr_catalog_fname(fname):
    """
    """
    basename = os.path.basename(fname)
    ifirst = len(basename) - basename[::-1].find('_')
    ilast = len(basename) - basename[::-1].find('.') - 1
    return float(basename[ifirst:ilast])


def find_closest_available_umachine_snapshot(z, dirname=dropbox_dirname):
    """
    """
    available_fnames = list(fname_generator(dirname, 'sfr_catalog*.hdf5'))

    f = _parse_scale_factor_from_umachine_sfr_catalog_fname
    available_snaps = np.array([1./f(fname) - 1. for fname in available_fnames])

    return available_fnames[np.argmin(np.abs(z - available_snaps))]


def load_closest_available_umachine_catalog(z, dirname=dropbox_dirname):
    """
    """
    fname = find_closest_available_umachine_snapshot(z, dirname=dirname)
    return apply_pbcs(add_host_keys(Table.read(fname, path='data')))
