"""
"""
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from .nearest_umachine_halo_selection import source_halo_selection_indices
from .nearest_umachine_halo_selection import source_galaxy_selection_indices


def load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname):
    """
    """
    return Table.read(umachine_z0p1_color_mock_fname, path='data')


def load_umachine_mstar_ssfr_mock(umachine_mstar_ssfr_mock_fname):
    """
    """
    return Table.read(umachine_mstar_ssfr_mock_fname, path='data')


def load_protoDC2_fof_halo_catalog(protoDC2_fof_halo_catalog_fname):
    """
    """
    return Table.read(protoDC2_fof_halo_catalog_fname, path='data')


def load_bolshoi_planck_halo_catalog(bolshoi_planck_halo_catalog_fname):
    """
    """
    return Table.read(bolshoi_planck_halo_catalog_fname, path='data')


def transfer_colors_to_umachine_mstar_ssfr_mock(umachine_mstar_ssfr_mock, umachine_z0p1_color_mock):
    """
    """
    X = np.vstack((np.log10(umachine_z0p1_color_mock['obs_sm']),
        umachine_z0p1_color_mock['sfr_percentile_fixed_sm'])).T

    Y = np.vstack((np.log10(umachine_mstar_ssfr_mock['obs_sm']),
        umachine_mstar_ssfr_mock['sfr_percentile_fixed_sm'])).T

    tree = cKDTree(X)
    nn_distinces, nn_indices = tree.query(Y, k=1)

    keys_to_inherit = ('rmag', 'sdss_petrosian_gr', 'sdss_petrosian_ri', 'size_kpc')
    for key in keys_to_inherit:
        umachine_mstar_ssfr_mock[key] = umachine_z0p1_color_mock[key][nn_indices]

    return umachine_mstar_ssfr_mock


def build_output_snapshot_mock(umachine_mstar_ssfr_mock_with_colors, protoDC2_fof_halo_catalog,
            source_halo_selection_indices, source_galaxy_selection_indices):
    """
    """
    raise NotImplementedError()


def add_log10_cumulative_nd_mvir_column(halos, key, Lbox):
    """
    """
    Vbox = float(Lbox**3.)
    halos.sort(key)
    halos['log10_cumulative_nd_mvir'] = np.log10(
        np.arange(len(halos), 0, -1)/Vbox)
    return halos[::-1]


def write_sdss_restframe_color_snapshot_mocks_to_disk(
            umachine_z0p1_color_mock_fname, protoDC2_fof_halo_catalog_fname_list,
            umachine_mstar_ssfr_mock_fname_list, bolshoi_planck_halo_catalog_fname_list,
            output_color_mock_fname_list, overwrite=False,
            source_halo_catalog_Lbox=250., target_halo_catalog_Lbox=256.):
    """
    """
    umachine_z0p1_color_mock = load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname)

    gen = zip(protoDC2_fof_halo_catalog_fname_list, umachine_mstar_ssfr_mock_fname_list,
            bolshoi_planck_halo_catalog_fname_list, output_color_mock_fname_list)

    for fname1, fname2, fname3, output_color_mock_fname in gen:

        #  Load all three catalogs into memory
        umachine_mstar_ssfr_mock = load_umachine_mstar_ssfr_mock(fname1)
        protoDC2_fof_halo_catalog = load_protoDC2_fof_halo_catalog(fname2)
        bolshoi_planck_halo_catalog = load_bolshoi_planck_halo_catalog(fname3)

        #  Add the number density columns needed to find matching pairs of source/target halos
        protoDC2_fof_halo_catalog = add_log10_cumulative_nd_mvir_column(
            protoDC2_fof_halo_catalog, 'fof_halo_mass', target_halo_catalog_Lbox)
        bolshoi_planck_halo_catalog = add_log10_cumulative_nd_mvir_column(
            bolshoi_planck_halo_catalog, 'mvir', source_halo_catalog_Lbox)

        #  Transfer the colors from the z=0.1 UniverseMachine mock to the other UniverseMachine mock
        umachine_mstar_ssfr_mock_with_colors = transfer_colors_to_umachine_mstar_ssfr_mock(
            umachine_mstar_ssfr_mock, umachine_z0p1_color_mock)

        #  For every host halo in the halo catalog hosting the protoDC2 galaxies,
        #  find a matching halo in halo catalog hosting the UniverseMachine galaxies
        source_halo_indx = source_halo_selection_indices(
            bolshoi_planck_halo_catalog['log10_cumulative_nd_mvir'],
            protoDC2_fof_halo_catalog['log10_cumulative_nd_mvir'])

        #  Calculate the indices of the UniverseMachine galaxies that will be selected
        #  find a matching halo in halo catalog hosting the UniverseMachine galaxies
        source_galaxy_indx = source_galaxy_selection_indices(
            protoDC2_fof_halo_catalog, source_halo_indx)

        #  Assemble the output protoDC2 mock
        output_snapshot_mock = build_output_snapshot_mock(
            umachine_mstar_ssfr_mock_with_colors, protoDC2_fof_halo_catalog,
            source_halo_indx, source_galaxy_indx)

        #  Write the output protoDC2 mock to disk
        output_snapshot_mock.write(output_color_mock_fname, path='data', overwrite=overwrite)
