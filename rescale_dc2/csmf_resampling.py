"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from galsampler import halo_bin_indices, matching_bin_dictionary
from halotools.empirical_models import conditional_abunmatch


__all__ = ('source_target_bin_indices', 'rescale_stellar_mass_to_match_unnormalized_source_csmf')


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
