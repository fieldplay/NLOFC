from itertools import permutations, product, combinations_with_replacement, combinations
import numpy as np
from types import FunctionType, MethodType
from NLOFC_wrapper import *
from itertools import permutations
import matplotlib.pyplot as plt


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
        self.chi_iterator = np.zeros((8, 4))
        self.iter = 0
        # for mod_p, mod_q in product(*(2 * [[self.omega_M1, self.omega_M2]])):
        for mod_p, mod_q in permutations([self.omega_M1, self.omega_M2], 2):
            for m, n in permutations(range(1, len(self.energies)), 2):
                print(mod_p, mod_q, m, n)
                self.chi_iterator[self.iter] = np.array([mod_p, mod_q, m, n])
                self.iter += 1
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

        omega1 = self.field_freq1[:, np.newaxis]
        omega2 = self.field_freq2[:, np.newaxis]
        comb_omega = (self.delta_freq * np.arange(-self.comb_size, self.comb_size))[np.newaxis, :]

        self.field1 = (self.comb_lw / ((omega1 - self.central_freq - 2 * (self.omega_M1 - comb_omega)) ** 2 + self.comb_lw ** 2)).sum(axis=1)
        self.field2 = (self.comb_lw / ((omega2 - self.central_freq - 2 * (self.omega_M2 - comb_omega)) ** 2 + self.comb_lw ** 2)).sum(axis=1)

    def create_molecule(self, mol):
        mol.nDIM = len(self.energies)
        mol.energies = self.energies.ctypes.data_as(POINTER(c_double))
        mol.gamma = self.gamma.ctypes.data_as(POINTER(c_double))
        self.pol2 = np.zeros_like(self.frequency, dtype=np.complex)
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
        params.chi_iterator = self.chi_iterator.ctypes.data_as(POINTER(c_double))
        params.N_iter = self.iter

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

    energies = np.cumsum([0.0, 2.01e5, 2.01e5])
    gamma = np.zeros((len(energies), len(energies)))
    gamma[1, 0] = 2.25e1
    gamma[0, 1] = 2.25e1
    gamma[2, 0] = 2.50e1
    gamma[0, 2] = 2.50e1
    gamma[2, 1] = 2.75e1
    gamma[1, 2] = 2.75e1

    mu = np.ones_like(gamma)
    np.fill_diagonal(mu, 0.)

    parameters = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
        central_freq=energies[2],
        comb_size=100,
        resolution_size=11,
        omega_M1=2.5,
        omega_M2=7.5,
        comb_lw=5e-6,
        delta_freq=10,
        N_terms=5
    )

    System = NonLinearResponse2ndOrder(**parameters)
    System.get_polarization_2nd_order()

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot((System.frequency - System.central_freq) / System.delta_freq, System.pol2.real, 'r-')
    axes[1].plot((System.frequency - System.central_freq) / System.delta_freq, System.pol2.imag, 'r-')
    axes_field = axes[0].twinx()
    axes_field.plot((System.field_freq1 - System.central_freq) / System.delta_freq, System.field1, 'b-', linewidth=1)
    axes_field.plot((System.field_freq2 - System.central_freq) / System.delta_freq, System.field2, 'k-', linewidth=1)

    axes[0].get_xaxis().set_tick_params(which='both', direction='in', width=1)
    axes[0].get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='large')
    axes_field.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='b', labelsize='large')
    axes[0].get_xaxis().set_ticks_position('both')
    axes[0].get_yaxis().set_ticks_position('both')
    axes_field.get_yaxis().set_ticks_position('both')
    axes[0].grid(color='b', linestyle=':', linewidth=0.5)
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)

    axes_field = axes[1].twinx()
    axes_field.plot((System.field_freq1 - System.central_freq) / System.delta_freq, System.field1, 'b-', linewidth=1)
    axes_field.plot((System.field_freq2 - System.central_freq) / System.delta_freq, System.field2, 'k-', linewidth=1)

    axes[1].get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=90, labelsize='large')
    axes[1].get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='large')
    axes_field.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='b', labelsize='large')
    axes[1].get_xaxis().set_ticks_position('both')
    axes[1].get_yaxis().set_ticks_position('both')
    axes_field.get_yaxis().set_ticks_position('both')
    axes[1].grid(color='b', linestyle=':', linewidth=0.5)
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)
    plt.show()
