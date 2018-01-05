"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from galsampler import halo_bin_indices, matching_bin_dictionary
from halotools.empirical_models import conditional_abunmatch


__all__ = ('source_target_bin_indices', 'rescale_stellar_mass_to_match_unnormalized_source_csmf')


def renormalize_csmf(source_galaxy_is_central, target_galaxy_is_central,
            source_bin_numbers, target_bin_numbers,
            source_host_halo_id, target_host_halo_id):
    """
    """
    selection_indices = np.arange(0, len(target_galaxy_is_central))

    unique_target_bin_numbers = np.unique(target_bin_numbers)
    accumulator = []
    for ibin in unique_target_bin_numbers:

        target_ibin_mask = target_bin_numbers == ibin

        target_cenmask = target_galaxy_is_central & target_ibin_mask
        accumulator.append(selection_indices[target_cenmask])

        source_ibin_mask = source_bin_numbers == ibin
        num_sats_source = np.count_nonzero(~source_galaxy_is_central & source_ibin_mask)
        num_unique_source_host_halos = len(np.unique(source_host_halo_id[source_ibin_mask]))
        mean_nsat_source = num_sats_source/float(num_unique_source_host_halos)
        num_unique_target_host_halos = len(np.unique(target_host_halo_id[target_ibin_mask]))
        num_sats_to_select = int(round(mean_nsat_source*num_unique_target_host_halos))

        target_satmask = ~target_galaxy_is_central & target_ibin_mask
        accumulator.append(np.random.choice(
            selection_indices[target_satmask], num_sats_to_select, replace=True))
    return np.concatenate(accumulator)


def rescale_stellar_mass_to_match_unnormalized_source_csmf(
            source_galaxy_stellar_mass, target_galaxy_property,
            source_galaxy_is_central, target_galaxy_is_central,
            source_bin_numbers, target_bin_numbers):
    """
    """
    result = np.copy(target_galaxy_property)

    unique_target_bin_numbers = np.unique(target_bin_numbers)
    for ibin in unique_target_bin_numbers:

        source_ibin_mask = source_bin_numbers == ibin
        target_ibin_mask = target_bin_numbers == ibin

        num_cens_target = np.count_nonzero(target_galaxy_is_central & target_ibin_mask)
        num_sats_target = np.count_nonzero(~target_galaxy_is_central & target_ibin_mask)

        num_cens_source = np.count_nonzero(source_galaxy_is_central & source_ibin_mask)
        num_sats_source = np.count_nonzero(~source_galaxy_is_central & source_ibin_mask)

        if num_cens_target > 0:
            result[target_galaxy_is_central & target_ibin_mask] = conditional_abunmatch(
                target_galaxy_property[target_galaxy_is_central & target_ibin_mask],
                source_galaxy_stellar_mass[source_galaxy_is_central & source_ibin_mask],
                npts_lookup_table=min(num_cens_source, num_cens_target))

        if num_sats_target > 0:
            result[~target_galaxy_is_central & target_ibin_mask] = conditional_abunmatch(
                target_galaxy_property[~target_galaxy_is_central & target_ibin_mask],
                source_galaxy_stellar_mass[~source_galaxy_is_central & source_ibin_mask],
                npts_lookup_table=min(num_sats_source, num_sats_target))

    return result


def source_target_bin_indices(source_galaxy_host_mass, target_galaxy_host_mass,
            source_galaxy_is_central, target_galaxy_is_central, host_mass_bin_edges, nhalo_min=50):
    """
    """
    source_bin_numbers = halo_bin_indices(mass=(source_galaxy_host_mass, host_mass_bin_edges))
    target_bin_numbers = halo_bin_indices(mass=(target_galaxy_host_mass, host_mass_bin_edges))

    bin_shapes = np.atleast_1d(host_mass_bin_edges).shape
    central_bin_dict = matching_bin_dictionary(
        source_bin_numbers[source_galaxy_is_central], nhalo_min, bin_shapes)

    satellite_bin_dict = matching_bin_dictionary(
        source_bin_numbers[~source_galaxy_is_central], nhalo_min, bin_shapes)

    for source_bin_number, target_bin_number in central_bin_dict.items():
        target_cenmask = target_galaxy_is_central*(target_bin_numbers == source_bin_number)
        target_bin_numbers[target_cenmask] = target_bin_number

    for source_bin_number, target_bin_number in satellite_bin_dict.items():
        target_satmask = (~target_galaxy_is_central)*(target_bin_numbers == source_bin_number)
        target_bin_numbers[target_satmask] = target_bin_number

    return source_bin_numbers, target_bin_numbers


def nearest_neighbor_rescale_stellar_mass(protoDC2_host_halo_mass, universe_machine_host_halo_mass,
            universe_machine_stellar_mass, universe_machine_is_central, protoDC2_is_central):
    """
    """
    pdc2_cens_mhost = protoDC2_host_halo_mass[protoDC2_is_central]
    pdc2_sats_mhost = protoDC2_host_halo_mass[~protoDC2_is_central]

    um_cens_mhost = universe_machine_host_halo_mass[universe_machine_is_central]
    um_sats_mhost = universe_machine_host_halo_mass[~universe_machine_is_central]

    pdc2_cens_rescaled_mhost = conditional_abunmatch(pdc2_cens_mhost, um_cens_mhost)
    x_table = np.log10(np.sort(pdc2_cens_mhost))
    y_table = np.log10(np.sort(pdc2_cens_rescaled_mhost))
    pdc2_sats_rescaled_mhost = 10**np.interp(np.log10(pdc2_sats_mhost), x_table, y_table)

    um_cens_mstar = universe_machine_stellar_mass[universe_machine_is_central]
    um_sats_mstar = universe_machine_stellar_mass[~universe_machine_is_central]

    idx_um_cens_sorted = np.argsort(um_cens_mhost)
    idx_um_sats_sorted = np.argsort(um_sats_mhost)

    um_sorted_cens_mstar = um_cens_mstar[idx_um_cens_sorted]
    um_sorted_sats_mstar = um_sats_mstar[idx_um_sats_sorted]

    um_sorted_cens_mhost = um_cens_mhost[idx_um_cens_sorted]
    um_sorted_sats_mhost = um_sats_mhost[idx_um_sats_sorted]

    idx_censelect = np.searchsorted(um_sorted_cens_mhost, pdc2_cens_rescaled_mhost)
    idx_satselect = np.searchsorted(um_sorted_sats_mhost, pdc2_sats_rescaled_mhost)

    idx_censelect = np.where(idx_censelect >= len(um_cens_mstar), idx_censelect-1, idx_censelect)
    idx_satselect = np.where(idx_satselect >= len(um_sats_mstar), idx_satselect-1, idx_satselect)

    rescaled_cens_mstar = um_sorted_cens_mstar[idx_censelect]
    rescaled_sats_mstar = um_sorted_sats_mstar[idx_satselect]

    result = np.zeros_like(protoDC2_host_halo_mass)
    result[protoDC2_is_central] = rescaled_cens_mstar
    result[~protoDC2_is_central] = rescaled_sats_mstar

    return result
