"""
"""
import os
import numpy as np

from ..load_catalogs import find_closest_available_umachine_snapshot, load_closest_available_umachine_catalog


__all__ = ('test_identify_correct_snapname', 'test_load_snapshots')


def test_identify_correct_snapname():
    """
    """
    fname1 = find_closest_available_umachine_snapshot(0.0)
    assert os.path.basename(fname1) == "umachine_sfr_catalog_1.002310.hdf5"

    fname2 = find_closest_available_umachine_snapshot(1.1)
    assert os.path.basename(fname2) == "umachine_sfr_catalog_0.501122.hdf5"

    fname3 = find_closest_available_umachine_snapshot(1.6)
    assert os.path.basename(fname3) == "umachine_sfr_catalog_0.399872.hdf5"

    fname4 = find_closest_available_umachine_snapshot(0.4)
    assert os.path.basename(fname4) == "umachine_sfr_catalog_0.744123.hdf5"


def test_load_snapshots():
    """
    """
    zlist = np.random.uniform(-1, 2, 5)
    for z in zlist:
        catalog = load_closest_available_umachine_catalog(z)
