"""
"""
import numpy as np
from scipy.stats import binned_statistic


def powerlaw_extrapolate_AbsMagr_from_mstar(log10_mstar, data_log10_mstar, data_AbsMagr,
            logsm_fitting_range, npts_lookup_table=10):
    """
    Parameters
    ----------
    log10_mstar : ndarray
        Array of shape (npts_output, ) storing the values of stellar mass
        at which to estimate the median AbsMagr.

    data_log10_mstar : ndarray
        Array of shape (npts_data, ) storing the stellar mass of the data used to build the model

    data_AbsMagr : ndarray
        Array of shape (npts_data, ) storing the AbsMagr of the data used to build the model

    logsm_fitting_range : tuple
        Two element tuple storing the range in log10_mstar that will be used to fit a power law

    Returns
    -------
    AbsMagr : ndarray
        Array of shape (npts_output, ) storing the estimated median AbsMagr
        evaluated at each input stellar mass
    """
    logsm_bins = np.linspace(logsm_fitting_range[0], logsm_fitting_range[1], npts_lookup_table)
    median_AbsMagr, __, __ = binned_statistic(data_log10_mstar, data_AbsMagr,
        bins=logsm_bins, statistic='median')

    logsm_mids = 0.5*(logsm_bins[:-1] + logsm_bins[1:])
    p1, p0 = np.polyfit(logsm_mids, median_AbsMagr, deg=1)
    return p0 + p1*log10_mstar