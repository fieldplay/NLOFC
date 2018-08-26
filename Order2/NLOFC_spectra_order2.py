from itertools import permutations, product, combinations_with_replacement, combinations
import numpy as np
from types import FunctionType, MethodType
from NLOFC_wrapper import *


# noinspection PyTypeChecker
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
        self.field_freq1 = None
        self.field_freq2 = None
        self.pol2 = None

    def get_frequency_nonuniform(self):
        """
        Generate non-uniform grid of frequencies for polarization and OFC fields
        :return: None
        """
        omega_M1 = self.omega_M1
        omega_M2 = self.omega_M2
        N_comb = self.comb_size
        N_res = self.resolution_size

        w0_pol3 = (omega_M1 + omega_M2) + self.central_freq
        w0_field1 = 2 * omega_M1 + self.central_freq
        w0_field2 = 2 * omega_M2 + self.central_freq

        self.frequency = w0_pol3 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2 * N_comb + 1)[:, np.newaxis] + np.linspace(-0.3 * self.delta_freq, 0.3 * self.delta_freq, N_res)
        self.field_freq1 = w0_field1 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2 * N_comb + 1)[:, np.newaxis] + np.linspace(-0.3 * self.delta_freq, 0.3 * self.delta_freq, N_res)
        self.field_freq2 = w0_field2 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2 * N_comb + 1)[:, np.newaxis] + np.linspace(-0.3 * self.delta_freq, 0.3 * self.delta_freq, N_res)

        self.frequency = self.frequency.flatten()
        self.field_freq1 = self.field_freq1.flatten()
        self.field_freq2 = self.field_freq2.flatten()
        self.pol2 = np.empty(len(self.frequency), dtype=np.complex)

    def create_molecule(self, mol):
        mol.nDIM = len(self.energies)
        mol.energies = self.energies.ctypes.data_as(POINTER(c_double))
        mol.gamma = self.gamma.ctypes.data_as(POINTER(c_double))
        mol.mu = self.mu.ctypes.data_as(POINTER(c_double))
        mol.pol2 = self.pol2.ctypes.data_as(POINTER(c_complex))

    def create_parameters(self, params):
        params.central_freq = self.central_freq
        params.comb_size = self.comb_size
        params.omega_M1 = self.omega_M1
        params.omega_M2 = self.omega_M2
        params.comb_lw = self.comb_lw
        params.delta_freq = self.delta_freq
        params.N_terms = self.N_terms
        params.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        params.N_freq = len(self.frequency)

    def get_polarization_2nd_order(self):
        """
        Returns the second order polarization for a specific molecule
        :return:
        """
        self.get_frequency_nonuniform()
        mol = Molecule()
        params = Parameters()
        self.create_molecule(mol)
        self.create_parameters(params)
        get_pol2_total(mol, params)


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

    parameters = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
        central_freq=energies[2],
        comb_size=500,
        resolution_size=11,
        omega_M1=3,
        omega_M2=7,
        comb_lw=5e-4,
        delta_freq=10,
        N_terms=8
    )

    System = NonLinearResponse2ndOrder(**parameters)
    System.get_polarization_2nd_order()
