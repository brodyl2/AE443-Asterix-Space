
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore")

@dataclass
class RiskTable:
    # Name of the Risk Table
    name: str = field(init=True)
    units: str = field(init=True)

    # Severity level breakpoints. Each breakpoint corresponds to the point at which the risk table assigns a new utility level.
    # Defined in ascending order of impact. e.g. [0., 6., 12.] for months of schedule delay or [1., 5., 10.] for millions
    # of euros in cost impact. Defaults to five levels capturing a range of 0-10+
    utility_breakpoints: list[float] = field(init=True,
                                             default_factory=lambda: [0.,
                                                                      2.5,
                                                                      5.,
                                                                      7.5,
                                                                      10.])

    # Utility associated with each level of impact. Should be the same length as `utility_breakpoints`. Each entry
    # indicates the utility associated with exceeding that breakpoint. e.g. [-1., -3., -5.]. Defaults to a range of 0 to -7.
    utilities: list[float] = field(init=True,
                                   default_factory=lambda: [0.,
                                                            -1.,
                                                            -3.,
                                                            -5.,
                                                            -7.])

    utility_names: list[str] = field(init=True,
                                     default_factory=lambda: ['None',
                                                              'Negligible',
                                                              'Moderate',
                                                              'Critical',
                                                              'Catastrophic'])

    # Discrete probability levels to be associated with each risk level.
    # Defaults to the geometric means of the probability ranges in ICD 203.
    probability_levels: list = field(init=True,
                                     default_factory=lambda: [0.022,
                                                              0.100,
                                                              0.300,
                                                              0.497,
                                                              0.663,
                                                              0.87,
                                                              0.97])

    # Plain language description of each level in the probability range specification.
    probability_names: list[str] = field(init=True,
                                         default_factory=lambda: ['Remote',
                                                                  'Very Unlikely',
                                                                  'Unlikely',
                                                                  'Roughly Even Chance',
                                                                  'Likely',
                                                                  'Very Likely',
                                                                  'Almost Certain'])

    # Logistic parameters for each level of impact
    L: float = field(init=True, default=10.0)
    k: float = field(init=True, default=1.0)
    x0: float = field(init=True, default=0.0)

    def __post_init__(self):

        # Check that utility specification is well-formed
        assert len(self.utility_breakpoints) == len(self.utilities)
        assert len(self.utility_breakpoints) == len(self.utility_names)

        # Check that probability specification is well-formed
        assert len(self.probability_levels) == len(self.probability_names)
        assert np.array(self.probability_levels).min() >= 0.0
        assert (np.array(self.probability_levels).max() <= 1.0)
        assert np.all(np.diff(np.array(self.probability_levels)))

        logistic_params = self.fit_utilities()
        self.L = logistic_params[0][0]
        self.k = logistic_params[0][1]
        self.x0 = logistic_params[0][2]

    def utility(self,
                 impact,
                 L=10.,
                 k=1.,
                 x0=0.,
                 mode='logistic',
                 u_func=lambda x: x):

        if mode == 'discrete':
            bps = np.atleast_1d(self.utility_breakpoints)
            i_arr = np.atleast_2d(impact)
            return np.array([self.utilities[i] for i in
                             np.sum(
                                 np.tile(bps, i_arr.shape[1]).reshape((i_arr.shape[1], bps.shape[0])) < i_arr.T,
                                 axis=1) - 1
                             ])
        if mode == 'logistic':
            return np.array(L / (1 + np.exp(-k * (impact - x0))))
        if mode == 'custom':
            return u_func(impact)

    # Fit a logistics function to the utility levels. Runs by default when the risk table is created.
    # Can be run again if the utility levels are redefined.
    def fit_utilities(self):

        u_func = lambda impact, L, k, x0: self.utility(impact,
                                                       mode='logistic',
                                                       L=L,
                                                       k=k,
                                                       x0=x0)
        params = curve_fit(u_func,
                           xdata=np.array(self.utility_breakpoints),
                           ydata=np.array(self.utilities),
                           p0=[self.utilities[-1],
                               -np.sign(self.utilities[-1])*\
                               (self.utilities[-1] - self.utilities[0])/\
                               (self.utility_breakpoints[-1] - self.utility_breakpoints[0]),
                               np.mean(self.utility_breakpoints)])

        return params

    def plot_utilities(self):
        xrange = np.linspace(self.utility_breakpoints[0],
                             self.utility_breakpoints[-1] * 1.1,
                             1000)
        plt.plot(xrange, self.discrete_utility(xrange), '-', label='Discrete')
        plt.plot(xrange, self.logistic_utility(xrange), '--', label='Logistic')
        plt.axis((self.utility_breakpoints[0],
                  self.utility_breakpoints[-1]*1.1,
                  min(self.utilities),
                  max(self.utilities)))
        plt.legend()
        plt.show()

    # Evaluate the utility of a given impact level assuming discrete breakpoints and utility levels.
    def discrete_utility(self,
                         impact):

        return self.utility(impact,
                            mode='discrete')

    # Evaluate the utility of a given impact level assuming a logistic utility function.
    def logistic_utility(self,
                         impact):

        return self.utility(impact,
                            mode='logistic',
                            L=self.L,
                            k=self.k,
                            x0=self.x0)

    # Evaluate the utility of a given impact level using a custom utility function.
    def custom_utility(self,
                       impact,
                       utility_function):

        return self.utility(impact,
                            mode='custom',
                            u_func=utility_function)