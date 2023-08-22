import numpy as np
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf


def acf_error(fakes, real, nlags):
    """
    ACF (auto-correlation function) error is the MSE between an ACF of real data and
    an average ACF of fake data.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    :param nlags: int
        Number of lags for calculation of ACF.
    
    :return: list
        Float value of ACF error for each segment.
    """
    mses = []
    for col in real.columns:
        fake_acf = []
        for fake in fakes:
            fake_acf.append(acf(fake[col], nlags=nlags)[1:])
        fake_acf = np.mean(fake_acf, axis=0)
        real_acf = acf(fake[col], nlags=nlags)[1:]
        error = (real_acf - fake_acf)**2
        mses.append(error.mean())
    return mses


def ks_test(fakes, real):
    """
    Kolmogorov-Smirnov test checks that the real and the fake data are 
    sampled from the same distribution.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    
    :return: list
        Dictionary with keys 'statistic' and 'pvalue' for each segment.
    """
    res = []
    for col in real.columns:
        fake_values = np.concatenate([fake[col].values for fake in fakes])
        ks_res = ks_2samp(fake_values, real[col].values)
        res.append({'statistic': ks_res.statistic, 'pvalue': ks_res.pvalue})
    return res