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
    assert os.path.basename(fname1) == "sfr_catalog_1.002310.hdf5"

    fname2 = find_closest_available_umachine_snapshot(1.1)
    assert os.path.basename(fname2) == "sfr_catalog_0.501122.hdf5"

    fname3 = find_closest_available_umachine_snapshot(1.6)
    assert os.path.basename(fname3) == "sfr_catalog_0.399872.hdf5"

    fname4 = find_closest_available_umachine_snapshot(0.4)
    assert os.path.basename(fname4) == "sfr_catalog_0.744123.hdf5"


def test_load_snapshots():
    """
    """
    zlist = np.random.uniform(-1, 2, 5)
    for z in zlist:
        fname = find_closest_available_umachine_snapshot(z)
        print("Loading z = {0:.2f} catalog with fname = {1}".format(z, fname))
        catalog = load_closest_available_umachine_catalog(z)
        assert 'hostid' in list(catalog.keys())

        for poskey in ('x', 'y', 'z'):
            assert np.all(catalog[poskey] >= 0.)
            assert np.all(catalog[poskey] <= 250.)
