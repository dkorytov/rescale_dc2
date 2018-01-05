"""
"""
import numpy as np
from .csmf_resampling import source_target_bin_indices
from .csmf_resampling import rescale_stellar_mass_to_match_unnormalized_source_csmf
from .csmf_resampling import renormalize_csmf


__all__ = ('rescale_stellar_mass', )


def rescale_stellar_mass(protoDC2, universe_machine,
            host_mass_bin_edges=np.logspace(10, 14.75, 25)):
    """
    """

    #  Bin galaxies by host halo mass
    source_galaxy_host_mass = universe_machine['host_halo_mvir']
    target_galaxy_host_mass = protoDC2['hostHaloMass']

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

    #  Resample protoDC2 so that the normalized CSMF matches UniverseMachine

    source_host_halo_id = universe_machine['hostid']
    target_host_halo_id = protoDC2['hostIndex']

    indices = renormalize_csmf(source_galaxy_is_central, target_galaxy_is_central,
                source_bin_numbers, target_bin_numbers,
                source_host_halo_id, target_host_halo_id)

    return protoDC2[indices]
