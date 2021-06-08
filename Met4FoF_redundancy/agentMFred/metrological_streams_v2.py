"""
This module contains the definition of a metrological datastream that can be used to
generate data for usage in the agentMET4FOF framework.
"""


import numpy as np
from agentMET4FOF.metrological_streams import MetrologicalDataStreamMET4FOF
from scipy.stats import norm

from typing import Any, Iterable, Optional, Tuple, Union


class MetrologicalMultiWaveGenerator(MetrologicalDataStreamMET4FOF):
    """
    Class to generate data as a sum of cosine waves and additional Gaussian noise.
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
    """

    def __init__(
                 self,
                 sfreq: int = 500,
                 freq_arr: np.array = np.array([50]),
                 ampl_arr: np.array = np.array([1]),
                 phase_ini_arr: np.array = np.array([0]),
                 intercept: float = 0,
                 device_id: str = "DataGenerator",
                 time_name: str = "time",
                 time_unit: str = "s",
                 quantity_names: Union[str, Tuple[str, ...]] = ("Length", "Mass"),
                 quantity_units: Union[str, Tuple[str, ...]] = ("m", "kg"),
                 misc: Optional[Any] = " Generator for a linear sum of cosines",
                 value_unc: Union[float, Iterable[float]] = 0.1,
                 time_unc: Union[float, Iterable[float]] = 0,
                 noisy: bool = True
                 ):
        super(MetrologicalMultiWaveGenerator, self).__init__(
            value_unc=value_unc, time_unc=time_unc
        )
        self.set_metadata(
            device_id=device_id,
            time_name=time_name,
            time_unit=time_unit,
            quantity_names=quantity_names,
            quantity_units=quantity_units,
            misc=misc
        )
        self.value_unc = value_unc
        self.time_unc = time_unc
        self.set_generator_function(
            generator_function=self._multi_wave_function,
            sfreq=sfreq,
            intercept=intercept,
            freq_arr=freq_arr,
            ampl_arr=ampl_arr,
            phase_ini_arr=phase_ini_arr,
            noisy=noisy
        )

    def _multi_wave_function(self, time, intercept, freq_arr, ampl_arr,
                             phase_ini_arr, noisy):

        value_arr = intercept
        if noisy:
            value_arr += self.value_unc * norm.rvs(size=time.shape)

        for freq, ampl, phase_ini in zip(freq_arr, ampl_arr, phase_ini_arr):
            value_arr = value_arr + ampl * np.cos(2 * np.pi * freq * time + phase_ini)

        return value_arr
