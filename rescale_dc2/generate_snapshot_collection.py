"""
"""
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box

from .nearest_umachine_halo_selection import source_halo_selection_indices
from .nearest_umachine_halo_selection import source_galaxy_selection_indices
from .nearest_umachine_halo_selection import value_add_matched_target_halos


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


def build_output_snapshot_mock(umachine, target_halos, halo_indices, galaxy_indices,
            Lbox_target=256.):
    """
    """
    dc2 = Table()
    dc2['source_halo_id'] = umachine['hostid'][galaxy_indices]
    dc2['target_halo_id'] = np.repeat(
        target_halos['fof_halo_tag'][halo_indices], target_halos['richness'][halo_indices])

    idxA, idxB = crossmatch(dc2['target_halo_id'], target_halos['fof_halo_tag'])

    msg = "target IDs do not match!"
    assert np.all(dc2['source_halo_id'][idxA] == target_halos['source_halo_id'][idxB]), msg

    target_halo_keys = ('x', 'y', 'z', 'vx', 'vy', 'vz')

    dc2['target_halo_x'] = 0.
    dc2['target_halo_y'] = 0.
    dc2['target_halo_z'] = 0.
    dc2['target_halo_vx'] = 0.
    dc2['target_halo_vy'] = 0.
    dc2['target_halo_vz'] = 0.

    dc2['target_halo_x'][idxA] = target_halos['fof_halo_center_x'][idxB]
    dc2['target_halo_y'][idxA] = target_halos['fof_halo_center_y'][idxB]
    dc2['target_halo_z'][idxA] = target_halos['fof_halo_center_z'][idxB]

    dc2['target_halo_vx'][idxA] = target_halos['fof_halo_mean_vx'][idxB]
    dc2['target_halo_vy'][idxA] = target_halos['fof_halo_mean_vy'][idxB]
    dc2['target_halo_vz'][idxA] = target_halos['fof_halo_mean_vz'][idxB]

    dc2['target_halo_mass'] = 0.
    dc2['target_halo_mass'][idxA] = target_halos['fof_halo_mass'][idxB]

    source_galaxy_keys = ('host_halo_mvir', 'upid',
            'host_centric_x', 'host_centric_y', 'host_centric_z',
            'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
            'obs_sm', 'obs_sfr', 'sfr_percentile_fixed_sm',
            'rmag', 'sdss_petrosian_gr', 'sdss_petrosian_ri', 'size_kpc')
    for key in source_galaxy_keys:
        dc2[key] = umachine[key][galaxy_indices]

    x_init = dc2['target_halo_x'] + dc2['host_centric_x']
    vx_init = dc2['target_halo_vx'] + dc2['host_centric_vx']
    dc2['x'], dc2['vx'] = enforce_periodicity_of_box(x_init, Lbox_target, velocity=vx_init)

    y_init = dc2['target_halo_y'] + dc2['host_centric_y']
    vy_init = dc2['target_halo_vy'] + dc2['host_centric_vy']
    dc2['y'], dc2['vy'] = enforce_periodicity_of_box(y_init, Lbox_target, velocity=vy_init)

    z_init = dc2['target_halo_z'] + dc2['host_centric_z']
    vz_init = dc2['target_halo_vz'] + dc2['host_centric_vz']
    dc2['z'], dc2['vz'] = enforce_periodicity_of_box(z_init, Lbox_target, velocity=vz_init)
    return dc2


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
        print("...working on creating {0}".format(output_color_mock_fname))

        #  Load all three catalogs into memory
        protoDC2_fof_halo_catalog = load_protoDC2_fof_halo_catalog(fname1)
        umachine_mstar_ssfr_mock = load_umachine_mstar_ssfr_mock(fname2)
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

        protoDC2_fof_halo_catalog = value_add_matched_target_halos(
            bolshoi_planck_halo_catalog, protoDC2_fof_halo_catalog, source_halo_indx)

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
