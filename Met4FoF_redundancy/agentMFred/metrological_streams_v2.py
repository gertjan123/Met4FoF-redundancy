"""
This module contains the definition of a metrological datastream that can be used to generate data for usage in the agentMET4FOF framework.

"""


import numpy as np
from agentMET4FOF.streams import DataStreamMET4FOF
from scipy.stats import norm


class MetrologicalDataStreamMET4FOF_v2(DataStreamMET4FOF):
    """
    Class to request time-series data points of a signal including uncertainty information.
    """
    batch_size1 = 1

    def init_parameters(self, batch_size1):
        self.batch_size1 = batch_size1

    def __init__(self):
        super().__init__()

    def _next_sample_generator(self, batch_size=10):  # =1
        """
        Internal method for generating a batch of samples from the generator function. Overrides
        _next_sample_generator() from DataStreamMET4FOF. Includes time uncertainty ut and measurement uncertainty
        uv to sample
        """
        time_arr = np.arange(self.sample_idx, self.sample_idx + self.batch_size1, 1) / self.sfreq
        self.sample_idx += batch_size
        time_arr = time_arr.reshape((len(time_arr), 1))
        time_unc_arr = 0.005 * np.ones_like(time_arr)
        value_arr, value_unc_arr = self.generator_function(time_arr, **self.generator_parameters)
        data_arr2d = np.concatenate((time_arr, time_unc_arr, value_arr, value_unc_arr), 1)
        return data_arr2d


class MetrologicalMultiWaveGenerator(MetrologicalDataStreamMET4FOF_v2):
    """
    Class to generate data as a sum of cosine wave and additional Gaussian noise.
    Values with associated uncertainty are returned.

    Parameters
    ----------
    sfreq:     float
                sampling frequency which determines the time step when next_sample is called.
    intercept: float
                constant intercept of the signal
    freq_arr:  np.ndarray of float
              array with frequencies of components included in the signal
    ampl_arr:  np.ndarray of float
              array with amplitudes of components included in the signal
    phase_ini_arr:  np.ndarray of float
              array with initial phases of components included in the signal
    exp_unc_abs: float
                absolute expanded uncertainty of each data point of the signal
    """

    def __init__(self, sfreq= 500, intercept=0, freq_arr=np.array([50]), ampl_arr=np.array([1]),
                             phase_ini_arr=np.array([0]), expunc_abs=0.1):
        super().__init__()
        self.set_metadata("DataGenerator","time","s", ("m"), ("kg"), "data generator")
        self.set_generator_function(generator_function=self.multi_wave_function, sfreq=sfreq,
                                    intercept=intercept, freq_arr=freq_arr, ampl_arr=ampl_arr,
                                    phase_ini_arr=phase_ini_arr, expunc_abs=expunc_abs)

    def multi_wave_function(self, time_arr, intercept=0, freq_arr=np.array([50]), ampl_arr=np.array([1]),
                             phase_ini_arr=np.array([0]), expunc_abs=0.1):
        value_arr = intercept + expunc_abs/2 * norm.rvs(size=time_arr.shape)
        n_comps = len(freq_arr)
        for i_comp in range(n_comps):
            value_arr = value_arr + ampl_arr[i_comp] * np.cos(2 * np.pi * freq_arr[i_comp] * time_arr +
                                                              phase_ini_arr[i_comp])

        value_expunc_arr = expunc_abs * np.ones_like(value_arr)
        # print('value_arr: ', value_arr)
        return value_arr, value_expunc_arr



