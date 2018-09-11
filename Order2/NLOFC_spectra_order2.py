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
        self.env1 = None
        self.env2 = None
        self.field1 = None
        self.field2 = None

        self.chi_iterator = np.zeros((8, 4))
        self.iter = 0
        mods = list(product(*(2 * [[self.omega_M1, self.omega_M2]])))
        # del mods[0]
        # del mods[-1]

        # for mod_p, mod_q in permutations([self.omega_M1, self.omega_M2], 2):
        for mod_p, mod_q in mods:
            for m, n in permutations(range(1, len(self.energies)), 2):
                if (self.mu[0, m] * self.mu[m, n] * self.mu[n, 0]) > 0.0:
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

        w0_pol3 = (omega_M2 + omega_M2) + self.central_freq
        w0_field1 = 2 * omega_M1 + self.central_freq
        w0_field2 = 2 * omega_M2 + self.central_freq

        self.frequency = w0_pol3 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, N_comb + 1)[:, np.newaxis] + np.linspace(-0.3 * self.delta_freq, 0.3 * self.delta_freq, N_res)
        self.field_freq1 = w0_field1 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, N_comb + 1)[:, np.newaxis] + np.linspace(-0.3 * self.delta_freq, 0.3 * self.delta_freq, N_res)
        self.field_freq2 = w0_field2 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, N_comb + 1)[:, np.newaxis] + np.linspace(-0.3 * self.delta_freq, 0.3 * self.delta_freq, N_res)

        self.frequency = self.frequency.flatten()
        self.field_freq1 = self.field_freq1.flatten()
        self.field_freq2 = self.field_freq2.flatten()

        # self.env1 = np.cos((self.field_freq1 - w0_field1) * np.pi / (self.field_freq1.max() - self.field_freq1.min()))**2
        self.env1 = np.sin(.05 * (self.field_freq1 - w0_field1)) * np.cos((self.field_freq1 - w0_field1) * np.pi / (self.field_freq1.max() - self.field_freq1.min()))**2
        # self.env2 = np.cos((self.field_freq2 - w0_field2) * np.pi / (self.field_freq2.max() - self.field_freq2.min()))**2
        self.env2 = np.sin(.05 * (self.field_freq1 - w0_field1)) * np.cos((self.field_freq2 - w0_field2) * np.pi / (self.field_freq2.max() - self.field_freq2.min()))**2

        omega1 = self.field_freq1[:, np.newaxis]
        omega2 = self.field_freq2[:, np.newaxis]
        comb_omega = (self.delta_freq * np.arange(-self.comb_size, self.comb_size))[np.newaxis, :]

        self.field1 = (self.comb_lw / ((omega1 - self.central_freq - 2 * (self.omega_M1 - comb_omega)) ** 2 + self.comb_lw ** 2)).sum(axis=1)
        self.field2 = (self.comb_lw / ((omega2 - self.central_freq - 2 * (self.omega_M2 - comb_omega)) ** 2 + self.comb_lw ** 2)).sum(axis=1)

        self.field1 *= self.env1
        self.field2 *= self.env2

    def create_molecule(self, mol):
        mol.nDIM = len(self.energies)
        mol.energies = self.energies.ctypes.data_as(POINTER(c_double))
        mol.gamma = self.gamma.ctypes.data_as(POINTER(c_double))
        self.pol2 = np.zeros((self.iter, len(self.frequency)), dtype=np.complex)
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
        params.field_env1 = self.env1.ctypes.data_as(POINTER(c_double))
        params.field_env2 = self.env2.ctypes.data_as(POINTER(c_double))

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
        count = 0
        for m, n in permutations(range(1, len(self.energies)), 2):
            if (self.mu[0, m] * self.mu[m, n] * self.mu[n, 0]) > 0.0:
                self.pol2[count, :] *= self.mu[0, m] * self.mu[m, n] * self.mu[n, 0]
                count += 1
        self.pol2 = self.pol2.sum(axis=0)


if __name__ == "__main__":

    energies_A = np.cumsum([0.0, 2.415e5, 2.415e5])
    energies_B = np.cumsum([0.0, 2.416e5, 2.415e5])
    gamma = np.zeros((len(energies_A), len(energies_A)))
    gamma[1, 0] = 2.25e1
    gamma[0, 1] = 2.25e1
    gamma[2, 0] = 4.50e1
    gamma[0, 2] = 4.50e1
    gamma[2, 1] = 7.75e1
    gamma[1, 2] = 7.75e1

    mu = np.ones_like(gamma)
    np.fill_diagonal(mu, 0.)

    parameters = dict(
        energies=energies_A,
        gamma=gamma,
        mu=mu,
        central_freq=energies_A[2],
        comb_size=2000,
        resolution_size=11,
        omega_M1=.25*0.5,
        omega_M2=.75*0.5,
        comb_lw=5e-10,
        delta_freq=.5,
        N_terms=3
    )

    def render_ticks(axes, axes_field):
        axes.get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=0, labelsize='large')
        axes.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='large')
        axes_field.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='b', labelsize='large')
        axes.get_xaxis().set_ticks_position('both')
        axes.get_yaxis().set_ticks_position('both')
        axes_field.get_yaxis().set_ticks_position('both')

    System = NonLinearResponse2ndOrder(**parameters)
    System.get_polarization_2nd_order()
    real_max = np.abs(System.pol2.real).max()
    imag_max = np.abs(System.pol2.imag).max()
    # System1 = NonLinearResponse2ndOrder(**parameters)
    # System1.energies = energies_B
    # System1.get_polarization_2nd_order()
    # real_max = max(np.abs(System.pol2.real).max(), np.abs(System1.pol2.real).max())
    # imag_max = max(np.abs(System.pol2.imag).max(), np.abs(System1.pol2.imag).max())
    z_indx = [int(System.frequency.size*(1/2 + 1 /128)), int(System.frequency.size*(33/64))]
    print(z_indx[0], z_indx[1])
    # ---------------------------------------------------------------------------------------------------------------- #
    #                               PLOTS OF SECOND ORDER NON-LINEAR RESPONSE                                          #
    # ---------------------------------------------------------------------------------------------------------------- #

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].set_title("Real part of $2^{nd}$ order non-linear response")
    axes[0, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System.pol2.real, 'k')
    # axes[0, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System1.pol2.real, 'b')
    axes[0, 0].set_ylim(-1.1 * real_max, 1.1 * real_max)
    axes[0, 0].grid(color='b', linestyle=':', linewidth=0.5)
    axes[0, 0].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    axes_field = axes[0, 0].twinx()
    axes_field.plot((System.field_freq1 - System.central_freq) / System.delta_freq, System.field1, 'r', alpha=0.5,
                    linewidth=2.)
    axes_field.plot((System.field_freq2 - System.central_freq) / System.delta_freq, System.field2, 'b', alpha=0.5,
                    linewidth=2.)
    render_ticks(axes[0, 0])
    axes_field.set_ylim(-1.1 * np.abs(System.field1).max(), 1.1 * np.abs(System.field1).max())
    axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)

    axes[1, 0].set_title("Imaginary part of $2^{nd}$ order non-linear response")
    axes[1, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System.pol2.imag, 'k')
    # axes[1, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System1.pol2.imag, 'b')
    axes[1, 0].set_ylim(-1.1 * imag_max, 1.1 * imag_max)
    axes[1, 0].grid(color='b', linestyle=':', linewidth=0.5)
    axes[1, 0].set_xlabel("$(\\omega - \\omega_{20}^A) / {\\Delta \\omega}$", size="large")
    axes[1, 0].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    axes_field = axes[1, 0].twinx()
    axes_field.plot((System.field_freq1 - System.central_freq) / System.delta_freq, System.field1, 'r', alpha=0.5,
                    linewidth=2.)
    axes_field.plot((System.field_freq2 - System.central_freq) / System.delta_freq, System.field2, 'b', alpha=0.5,
                    linewidth=2.)
    render_ticks(axes[1, 0], axes_field)
    axes_field.set_ylim(-1.1 * np.abs(System.field1).max(), 1.1 * np.abs(System.field1).max())
    axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)

    axes[0, 1].set_title("Real part of $2^{nd}$ order non-linear response (Zoomed-in)")
    axes[0, 1].plot((System.frequency[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.pol2.real[z_indx[0]:z_indx[1]], 'k')
    # axes[0, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System1.pol2.real, 'b')
    # axes[0, 1].set_ylim(-1.1 * real_max, 1.1 * real_max)
    axes[0, 1].grid(color='b', linestyle=':', linewidth=0.5)
    axes[0, 1].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    axes_field = axes[0, 1].twinx()
    axes_field.plot((System.field_freq1[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field1[z_indx[0]:z_indx[1]], 'r', alpha=0.5,
                    linewidth=2.)
    axes_field.plot((System.field_freq2[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field2[z_indx[0]:z_indx[1]], 'b', alpha=0.5,
                    linewidth=2.)
    render_ticks(axes[0, 1])
    # axes_field.set_ylim(-1.1 * np.abs(System.field1).max(), 1.1 * np.abs(System.field1).max())
    axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)

    axes[1, 1].set_title("Imaginary part of $2^{nd}$ order non-linear response")
    axes[1, 1].plot((System.frequency[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.pol2.imag[z_indx[0]:z_indx[1]], 'k')
    # axes[1, 1].plot((System.frequency - System.central_freq) / System.delta_freq, System1.pol2.imag, 'b')
    # axes[1, 1].set_ylim(-1.1 * imag_max, 1.1 * imag_max)
    axes[1, 1].grid(color='b', linestyle=':', linewidth=0.5)
    axes[1, 1].set_xlabel("$(\\omega - \\omega_{20}^A) / {\\Delta \\omega}$", size="large")
    axes[1, 1].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    axes_field = axes[1, 1].twinx()
    axes_field.plot((System.field_freq1[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field1[z_indx[0]:z_indx[1]], 'r', alpha=0.5,
                    linewidth=2.)
    axes_field.plot((System.field_freq2[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field2[z_indx[0]:z_indx[1]], 'b', alpha=0.5,
                    linewidth=2.)
    render_ticks(axes[1, 1], axes_field)
    # axes_field.set_ylim(-1.1 * np.abs(System.field1).max(), 1.1 * np.abs(System.field1).max())
    axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)
    plt.show()
