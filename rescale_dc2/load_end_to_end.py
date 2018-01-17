"""
"""
from .load_halos import load_protoDC2_fof_halos, load_bolshoi_planck_halos
from .load_umachine_mocks import load_umachine_and_value_added_halos
from .nearest_umachine_halo_selection import source_halo_selection_indices
from .nearest_umachine_halo_selection import value_add_matched_target_halos
from .nearest_umachine_halo_selection import source_galaxy_selection_indices
from .nearest_umachine_halo_selection import create_galsampled_dc2
from .value_add_umachine_catalogs import add_ssfr
from .load_catalogs import load_dc2_sdss
from .rescale_snapshot import assign_sdss_restframe_absolute_ugriz


def dc2_with_sdss_restframe_ugriz(umachine_fname, bolshoi_halos_fname, pdc2_fof_halos_fname):
    """
    """
    pdc2_halos = load_protoDC2_fof_halos(pdc2_fof_halos_fname)
    umachine, bpl_halos = load_umachine_and_value_added_halos(
        umachine_fname, load_bolshoi_planck_halos(bolshoi_halos_fname))

    halo_selection_indices = source_halo_selection_indices(
        bpl_halos['log10_cumulative_nd_mvir'],
        pdc2_halos['log10_cumulative_nd_mvir'])

    pdc2_halos = value_add_matched_target_halos(
        bpl_halos, pdc2_halos, halo_selection_indices)

    galaxy_selection_indices = source_galaxy_selection_indices(
        pdc2_halos, halo_selection_indices)

    dc2 = create_galsampled_dc2(umachine, pdc2_halos,
            halo_selection_indices, galaxy_selection_indices)
    dc2 = add_ssfr(dc2)

    sdss = load_dc2_sdss()

    dc2 = assign_sdss_restframe_absolute_ugriz(dc2, sdss)
    return dc2, umachine, sdss
