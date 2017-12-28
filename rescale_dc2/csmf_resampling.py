"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from galsampler import halo_bin_indices, matching_bin_dictionary


__all__ = ('source_target_bin_indices', )


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
