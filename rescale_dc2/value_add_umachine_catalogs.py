import numpy as np
from slidingpercentile import python_sliding_rank
import os
from astropy.table import Table
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch


def apply_pbcs(catalog, Lbox=250.):
    catalog['x'] = enforce_periodicity_of_box(catalog['x'], Lbox)
    catalog['y'] = enforce_periodicity_of_box(catalog['y'], Lbox)
    catalog['z'] = enforce_periodicity_of_box(catalog['z'], Lbox)
    return catalog


def add_host_keys(catalog, host_keys_to_add=('mvir', 'vmax')):
    """
    """
    idxA, idxB = crossmatch(catalog['hostid'], catalog['id'])
    catalog['host_halo_is_in_catalog'] = False
    catalog['host_halo_is_in_catalog'][idxA] = True

    for key in host_keys_to_add:
        catalog['host_halo_'+key] = catalog[key]
        catalog['host_halo_'+key][idxA] = catalog[key][idxB]

    return catalog
