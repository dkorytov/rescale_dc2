"""
"""
from astropy.table import Table


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
    raise NotImplementedError()
    return umachine_mstar_ssfr_mock


def calculate_source_halo_selection_indices(bolshoi_planck_halo_catalog, protoDC2_fof_halo_catalog):
    """
    """
    raise NotImplementedError()


def calculate_source_galaxy_selection_indices(protoDC2_fof_halo_catalog, source_halo_selection_indices):
    """
    """
    raise NotImplementedError()


def build_output_snapshot_mock(umachine_mstar_ssfr_mock_with_colors, protoDC2_fof_halo_catalog,
            source_halo_selection_indices, source_galaxy_selection_indices):
    """
    """
    raise NotImplementedError()


def write_sdss_restframe_color_snapshot_mocks_to_disk(
            umachine_z0p1_color_mock_fname, protoDC2_fof_halo_catalog_fname_list,
            umachine_mstar_ssfr_mock_fname_list, bolshoi_planck_halo_catalog_fname_list,
            output_color_mock_fname_list, overwrite=False):
    """
    """
    umachine_z0p1_color_mock = load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname)

    gen = zip(protoDC2_fof_halo_catalog_fname_list, umachine_mstar_ssfr_mock_fname_list,
                bolshoi_planck_halo_catalog_fname_list, output_color_mock_fname_list)

    for fname1, fname2, fname3, output_color_mock_fname in gen:
        umachine_mstar_ssfr_mock = load_umachine_mstar_ssfr_mock(fname1)
        protoDC2_fof_halo_catalog = load_protoDC2_fof_halo_catalog(fname2)
        bolshoi_planck_halo_catalog = load_bolshoi_planck_halo_catalog(fname3)

        umachine_mstar_ssfr_mock_with_colors = transfer_colors_to_umachine_mstar_ssfr_mock(
            umachine_mstar_ssfr_mock, umachine_z0p1_color_mock)

        source_halo_selection_indices = calculate_source_halo_selection_indices(
            bolshoi_planck_halo_catalog, protoDC2_fof_halo_catalog)

        source_galaxy_selection_indices = calculate_source_galaxy_selection_indices(
            bolshoi_planck_halo_catalog, protoDC2_fof_halo_catalog)

        output_snapshot_mock = build_output_snapshot_mock(
            umachine_mstar_ssfr_mock_with_colors, protoDC2_fof_halo_catalog,
            source_halo_selection_indices, source_galaxy_selection_indices)

        output_snapshot_mock.write(output_color_mock_fname, path='data', overwrite=overwrite)
