from itertools import permutations, product, combinations_with_replacement, combinations
import numpy as np
from types import FunctionType, MethodType
from NLOFC_wrapper import get_pol2_total


class NonLinearResponse2ndOrder:
    """
    Class to calculate non-linear 2nd order response with two high-frequency modulated Optical Frequency Combs
    """

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)
        self.frequency = None

    def get_frequency_nonuniform(self):
        """
        Generate non-uniform grid of frequencies for polarization and OFC fields
        :return: None
        """
        omega_M1 = self.omega_M1
        omega_M2 = self.omega_M2
        N = self.comb_size

        w0_pol3 = (omega_M1 + omega_M2) + self.central_freq
        w0_field1 = 2 * omega_M1 + self.central_freq
        w0_field2 = 2 * omega_M2 + self.central_freq

        self.frequency = w0_pol3 + np.linspace(-N * self.delta_freq, N * self.delta_freq, 2 * N + 1)
        self.field_freq1 = w0_field1 + np.linspace(-N * self.delta_freq, N * self.delta_freq, 2 * N + 1)
        self.field_freq2 = w0_field2 + np.linspace(-N * self.delta_freq, N * self.delta_freq, 2 * N + 1)

    def get_polarization_2nd_order(self, modulations):
        """
        Returns the second order polarization for a specific molecule
        :return:
        """


if __name__ == "__main__":
    from collections import namedtuple
    import time

    GammaMu = namedtuple("CTransition", ["g", "mu"])
    energies = np.cumsum([0.0, 2.01e5, 2.01e5])
    gamma = np.zeros((len(energies), len(energies)))
    gamma[1, 0] = 2.25e3
    gamma[0, 1] = 2.25e3
    gamma[2, 0] = 2.50e3
    gamma[0, 2] = 2.50e3
    gamma[2, 1] = 2.75e3
    gamma[1, 2] = 2.75e3

    mu = np.ones_like(gamma)
    np.fill_diagonal(mu, 0.)

    system = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
        central_freq=energies[2],
        comb_size=500,
        omega_M1=3,
        omega_M2=7,
        comb_lw=5e-4,
        delta_freq=10,
        N_terms=8
    )
