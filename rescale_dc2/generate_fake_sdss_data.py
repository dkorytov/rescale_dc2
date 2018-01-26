"""
"""
import numpy as np
from scipy.stats import binned_statistic
from scipy.special import erf
from astropy.table import Table
from halotools.empirical_models import conditional_abunmatch


def polyfit_median_quantity_vs_mstar(log10_mstar, data_log10_mstar, data_quantity,
            logsm_fitting_range, npts_lookup_table=10):
    """
    Parameters
    ----------
    log10_mstar : ndarray
        Array of shape (npts_output, ) storing the values of stellar mass
        at which to estimate the median AbsMagr.

    data_log10_mstar : ndarray
        Array of shape (npts_data, ) storing the stellar mass of the data used to build the model

    data_quantity : ndarray
        Array of shape (npts_data, ) storing the quantity of the data being modeled

    logsm_fitting_range : tuple
        Two element tuple storing the range in log10_mstar that will be used to fit a power law

    Returns
    -------
    quantity : ndarray
        Array of shape (npts_output, ) storing the estimated median of the input quantity
        evaluated at each input stellar mass
    """
    logsm_bins = np.linspace(logsm_fitting_range[0], logsm_fitting_range[1], npts_lookup_table)
    median_AbsMagr, __, __ = binned_statistic(data_log10_mstar, data_quantity,
        bins=logsm_bins, statistic='median')

    logsm_mids = 0.5*(logsm_bins[:-1] + logsm_bins[1:])
    p1, p0 = np.polyfit(logsm_mids, median_AbsMagr, deg=1)
    return p0 + p1*log10_mstar


def monte_carlo_magr_mock_sample(log10_mstar_mock_sample, data_log10_mstar, data_AbsMagr,
            logsm_fitting_range, magr_scatter=0.4, npts_lookup_table=10):
    """
    """
    median_magr = polyfit_median_quantity_vs_mstar(
        log10_mstar_mock_sample, data_log10_mstar, data_AbsMagr, logsm_fitting_range)

    return np.random.normal(loc=median_magr, scale=magr_scatter)


def monte_carlo_mock_sdss_data_faint_end(ngals_mock, data_gr, data_ri, data_log10_mstar,
            data_AbsMagr, logsm_fitting_range=(8.5, 10)):
    """
    """
    mock_log10_mstar = np.random.normal(loc=8, scale=0.75, size=ngals_mock)

    magr_scatter = np.interp(mock_log10_mstar, [6, 8.5], [0.4, 0.3])
    mock_magr = monte_carlo_magr_mock_sample(mock_log10_mstar, data_log10_mstar, data_AbsMagr,
            logsm_fitting_range, magr_scatter)

    ngals_mock = len(mock_log10_mstar)
    X = np.vstack((data_gr, data_ri))
    cov = np.cov(X)/8.

    gr_center = np.median(data_gr)-0.1
    ri_center = np.median(data_ri)-1.
    median_gr = np.interp(mock_log10_mstar, [6, 9], [gr_center-0.2, gr_center])
    median_ri = np.interp(mock_log10_mstar, [6, 9], [ri_center-0.3, ri_center])
    median_array = np.vstack((median_gr, median_ri)).T

    Z = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=ngals_mock) + median_array
    mock_gr, mock_ri = Z[:, 0], Z[:, 1]

    mock = Table()
    mock['gr'] = conditional_abunmatch(mock_gr, data_gr)
    mock['ri'] = conditional_abunmatch(mock_ri, data_ri)
    mock['sm'] = mock_log10_mstar
    mock['magr'] = mock_magr

    gr_std = np.std(mock['gr'])
    gr_med = np.median(mock['gr'])
    ri_std = np.std(mock['ri'])
    ri_med = np.median(mock['ri'])

    gr_x = mock['gr'] - gr_med
    ri_x = mock['ri'] - ri_med

    mu_x = gr_x + ri_x
    joint_sigma = np.sqrt(gr_std**1 + ri_std**2)
    joint_zscore = mu_x/joint_sigma
    mock['color_percentile'] = 1. - 0.5*(1 + erf(joint_zscore/np.sqrt(2)))

    return mock
