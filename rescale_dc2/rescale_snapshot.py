"""
"""
import numpy as np
from scipy.spatial import cKDTree

from halotools.empirical_models import conditional_abunmatch

from .csmf_resampling import source_target_bin_indices
from .csmf_resampling import rescale_stellar_mass_to_match_unnormalized_source_csmf
from .csmf_resampling import renormalize_csmf


__all__ = ('rescale_stellar_mass', 'rescale_ssfr', 'assign_sdss_restframe_absolute_ugriz')


def nearest_neighbor_mask(x, xc):
    xdist = np.abs(x - xc)
    return np.argsort(xdist)


def rescale_stellar_mass(protoDC2, universe_machine,
            host_mass_bin_edges=10**np.arange(10., 14.75, 0.01)):
    """
    """

    #  Bin galaxies by host halo mass
    source_galaxy_host_mass = universe_machine['host_halo_mvir']

    target_galaxy_host_mass = protoDC2['hostHaloMass']
    dlogM = 0.1
    target_galaxy_host_mass = 10**(
        np.log10(protoDC2['hostHaloMass']) + np.random.uniform(-dlogM, dlogM, len(protoDC2)))


    source_galaxy_is_central = universe_machine['upid'] == -1
    target_galaxy_is_central = protoDC2['isCentral'].astype(bool)

    universe_machine['host_mass_bin'], protoDC2['host_mass_bin'] = source_target_bin_indices(
        source_galaxy_host_mass, target_galaxy_host_mass,
        source_galaxy_is_central, target_galaxy_is_central, host_mass_bin_edges)

    #  Rescale M* label to match the M* PDF predicted by UniverseMachine
    source_galaxy_stellar_mass = universe_machine['obs_sm']
    target_galaxy_property = protoDC2['totalMassStellar']
    target_galaxy_property = protoDC2['infallHaloMass']

    source_bin_numbers = universe_machine['host_mass_bin']
    target_bin_numbers = protoDC2['host_mass_bin']

    protoDC2['rescaled_mstar'] = rescale_stellar_mass_to_match_unnormalized_source_csmf(
                source_galaxy_stellar_mass, target_galaxy_property,
                source_galaxy_is_central, target_galaxy_is_central,
                source_bin_numbers, target_bin_numbers)

    host_mass_bin_edges2 = 10**np.arange(10., 14.75, 0.5)
    source_bin_numbers2, target_bin_numbers2 = source_target_bin_indices(
        source_galaxy_host_mass, target_galaxy_host_mass,
        source_galaxy_is_central, target_galaxy_is_central, host_mass_bin_edges2)
    #  Resample protoDC2 so that the normalized CSMF matches UniverseMachine
    source_host_halo_id = universe_machine['hostid']
    target_host_halo_id = protoDC2['hostIndex']

    indices = renormalize_csmf(source_galaxy_is_central, target_galaxy_is_central,
                source_bin_numbers2, target_bin_numbers2,
                source_host_halo_id, target_host_halo_id)

    return protoDC2[indices]


def rescale_ssfr(protoDC2, universe_machine, logsm_bins=np.linspace(9, 12, 40)):
    """
    """
    #  Add sSFR column to protoDC2
    quenched_sequence_center = -13.5
    ssfr = protoDC2['totalStarFormationRate']/protoDC2['totalMassStellar']
    zero_mask = ssfr == 0.
    nzeros = np.count_nonzero(zero_mask)
    ssfr[zero_mask] = 10**np.random.normal(loc=quenched_sequence_center, scale=0.2, size=nzeros)
    protoDC2['ssfr'] = np.log10(ssfr)-9.

    protoDC2['remapped_ssfr'] = protoDC2['ssfr']
    protoDC2['remapped_ssfr_no_scatter'] = protoDC2['ssfr']

    cenmask_um = universe_machine['upid'] == -1
    satmask_um = ~cenmask_um

    cenmask_dc2 = protoDC2['isCentral']
    satmask_dc2 = ~protoDC2['isCentral']

    for low_sm, high_sm in zip(logsm_bins[:-1], logsm_bins[1:]):
        mid_sm = 0.5*(low_sm + high_sm)

        sm_mask_dc2 = protoDC2['totalMassStellar'] > 10**low_sm
        sm_mask_dc2 *= protoDC2['totalMassStellar'] < 10**high_sm

        cenmask_dc2 = sm_mask_dc2 * (protoDC2['isCentral'] == True)
        satmask_dc2 = sm_mask_dc2 * (protoDC2['isCentral'] == False)

        num_dc2_cens = np.count_nonzero(cenmask_dc2)
        if num_dc2_cens > 1:
            idx_censelect = nearest_neighbor_mask(universe_machine['obs_sm'][cenmask_um], 10**mid_sm)
            um_ssfr_cens = universe_machine['obs_ssfr'][cenmask_um][idx_censelect[:1000]]
            ssfr_cens = conditional_abunmatch(protoDC2['ssfr'][cenmask_dc2], um_ssfr_cens)
            protoDC2['remapped_ssfr'][cenmask_dc2] = ssfr_cens

        num_dc2_sats = np.count_nonzero(satmask_dc2)
        if num_dc2_sats > 1:
            idx_satselect = nearest_neighbor_mask(universe_machine['obs_sm'][satmask_um], 10**mid_sm)
            um_ssfr_sats = universe_machine['obs_ssfr'][satmask_um][idx_satselect[:1000]]
            ssfr_sats = conditional_abunmatch(protoDC2['ssfr'][satmask_dc2], um_ssfr_sats, sigma=1)
            protoDC2['remapped_ssfr'][satmask_dc2] = ssfr_sats

    return protoDC2


def assign_sdss_restframe_absolute_ugriz(protoDC2, sdss):
    """
    """
    tree = cKDTree(np.vstack((sdss['sm'], sdss['ssfr'])).T)

    nn_distinces, nn_indices = tree.query(
        np.vstack((np.log10(protoDC2['obs_sm']), protoDC2['obs_ssfr'])).T, k=1)

    protoDC2['obs_sm'] = 10**sdss['sm'][nn_indices]

    absmag_keys = ('AbsMagu', 'AbsMagg', 'AbsMagr', 'AbsMagi', 'AbsMagz')
    for key in absmag_keys:
        protoDC2[key] = sdss[key][nn_indices]

    return protoDC2
