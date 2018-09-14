from itertools import permutations, product, combinations_with_replacement, combinations
import numpy as np
from types import FunctionType, MethodType
from NLOFC_wrapper import *
from itertools import permutations
import matplotlib.pyplot as plt


# noinspection PyTypeChecker
class NonLinearResponse3rdOrder:
    """
    Class to calculate non-linear 3rd order response with two high-frequency modulated Optical Frequency Combs
    """

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)
        self.frequency = None
        self.frequency_12 = None
        self.frequency_21 = None
        self.field_freq1 = None
        self.field_freq2 = None
        self.env1 = None
        self.env2 = None
        self.field1 = None
        self.field2 = None

        self.chi_iterator = np.zeros((12, 6))
        self.iter = 0
        mods = list(product(*(3 * [[self.omega_M1, self.omega_M2]])))
        del mods[2:]
        # del mods[-1]
        del mods[0]

        # print(mods)

        for mod_p, mod_q, mod_r in mods:
            for m, n, v in permutations(range(1, len(self.energies)), 3):
                if (self.mu[0, m] * self.mu[m, n] * self.mu[n, v] * self.mu[v, 0]) > 0.0:
                    # print(mod_p, mod_q, mod_r, m, n, v)
                    self.chi_iterator[self.iter] = np.array([mod_p, mod_q, mod_r, m, n, v])
                    self.iter += 1
        self.pol3 = None

    def get_frequency_nonuniform(self):
        """
        Generate non-uniform grid of frequencies for polarization and OFC fields
        :return: None
        """
        omega_M1 = self.omega_M1
        omega_M2 = self.omega_M2
        N_comb = self.comb_size
        N_res = self.resolution_size

        w0_pol3_12 = (2 * omega_M1 - omega_M2) + self.central_freq
        w0_pol3_21 = (2 * omega_M2 - omega_M1) + self.central_freq
        w0_field1 = omega_M1 + self.central_freq
        w0_field2 = omega_M2 + self.central_freq

        self.frequency_12 = w0_pol3_12 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2*N_comb + 1)[:, np.newaxis] + np.linspace(-0.5 * self.delta_freq, 0.5 * self.delta_freq, N_res)
        # self.frequency_21 = w0_pol3_21 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2*N_comb + 1)[:, np.newaxis] + np.linspace(-0.5 * self.delta_freq, 0.3 * self.delta_freq, N_res)
        self.field_freq1 = w0_field1 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2*N_comb + 1)[:, np.newaxis] + np.linspace(-0.5 * self.delta_freq, 0.5 * self.delta_freq, N_res)
        self.field_freq2 = w0_field2 + np.linspace(-N_comb * self.delta_freq, N_comb * self.delta_freq, 2*N_comb + 1)[:, np.newaxis] + np.linspace(-0.5 * self.delta_freq, 0.5 * self.delta_freq, N_res)

        self.frequency = self.frequency_12.flatten()
        self.field_freq1 = self.field_freq1.flatten()
        self.field_freq2 = self.field_freq2.flatten()

        # self.env1 = np.cos((self.field_freq1 - w0_field1) * np.pi / (self.field_freq1.max() - self.field_freq1.min()))**2
        self.env1 = np.sin(.0001 * (self.field_freq1 - w0_field1)) * np.cos((self.field_freq1 - w0_field1) * np.pi / (self.field_freq1.max() - self.field_freq1.min()))**2
        # self.env2 = np.cos((self.field_freq2 - w0_field2) * np.pi / (self.field_freq2.max() - self.field_freq2.min()))**2
        self.env2 = np.sin(.0001 * (self.field_freq1 - w0_field1)) * np.cos((self.field_freq2 - w0_field2) * np.pi / (self.field_freq2.max() - self.field_freq2.min()))**2

        # self.env1 = np.ones_like(self.field_freq1)
        # self.env2 = np.ones_like(self.field_freq1)
        omega1 = self.field_freq1[:, np.newaxis]
        omega2 = self.field_freq2[:, np.newaxis]
        comb_omega = (self.delta_freq * np.arange(-self.comb_size, self.comb_size))[np.newaxis, :]

        self.field1 = (self.comb_lw / ((omega1 - self.omega_M1 - comb_omega) ** 2 + self.comb_lw ** 2)).sum(axis=1)
        self.field2 = (self.comb_lw / ((omega2 - self.omega_M2 - comb_omega) ** 2 + self.comb_lw ** 2)).sum(axis=1)

        self.field1 *= self.env1
        self.field2 *= self.env2

    def create_molecule(self, mol):
        mol.nDIM = len(self.energies)
        mol.energies = self.energies.ctypes.data_as(POINTER(c_double))
        mol.gamma = self.gamma.ctypes.data_as(POINTER(c_double))
        self.pol3 = np.zeros((self.iter, len(self.frequency)), dtype=np.complex)
        mol.pol3 = self.pol3.ctypes.data_as(POINTER(c_complex))

    def create_parameters(self, params):
        params.central_freq = self.central_freq
        params.comb_size = self.comb_size
        params.comb_lw = self.comb_lw
        params.delta_freq = self.delta_freq
        params.N_terms = self.N_terms
        params.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        params.N_freq = len(self.frequency)
        params.chi_iterator = self.chi_iterator.ctypes.data_as(POINTER(c_double))
        params.N_iter = self.iter
        params.field_env1 = self.env1.ctypes.data_as(POINTER(c_double))
        params.field_env2 = self.env2.ctypes.data_as(POINTER(c_double))

    def get_polarization_3rd_order(self):
        """
        Returns the second order polarization for a specific molecule
        :return:
        """
        self.get_frequency_nonuniform()
        mol = Molecule()
        params = Parameters()
        self.create_molecule(mol)
        self.create_parameters(params)
        # print(self.pol3.shape)
        # print(self.frequency.shape)
        get_pol3_total(mol, params)

        # count = 0
        # for m, n, v in permutations(range(1, len(self.energies)), 3):
        #     if (self.mu[0, m] * self.mu[m, n] * self.mu[n, v] * self.mu[v, 0]) > 0.0:
        #         self.pol3[count, :] *= self.mu[0, m] * self.mu[m, n] * self.mu[n, v] * self.mu[v, 0]
        #         count += 1
        self.pol3 = self.pol3.sum(axis=0)


if __name__ == "__main__":

    energies_A = np.cumsum([0, 113, 2844, 113]) * 1e2
    energies_B = np.cumsum([0, 113, 2844, 113]) * 1e2

    gamma_decay = np.ones((4, 4)) * 1e2  # All population relaxation times equal 1 GHz (1 ns inverse)
    np.fill_diagonal(gamma_decay, 0.0)  # Diagonal elements zero; no decay to self
    gamma_decay = np.tril(gamma_decay)  # Relaxation only to lower energy states
    # dephasing rates (T_ij = T_ji for dephasing)
    gamma_dephasing = np.ones((4, 4)) * 1.e6  # All electronic dephasing rates are 10 THz (100 fs inverse)
    np.fill_diagonal(gamma_dephasing, 0.0)
    gamma_dephasing[0, 1] = 5.88e4  # All vibrational dephasing rates are 0.59 THz (1.7 ps inverse)
    gamma_dephasing[1, 0] = 5.88e4
    gamma_dephasing[2, 3] = 5.88e4
    gamma_dephasing[3, 2] = 5.88e4

    # Net damping rates given by Boyd pg. 156, G_nm = (1/2) * \sum_i (g_decay_ni + g_decay_mi) + g_dephasing_nm

    gamma = np.zeros_like(gamma_decay)
    for n in range(4):
        for m in range(4):
            for i in range(4):
                gamma[n, m] += 0.5 * (gamma_decay[n, i] + gamma_decay[m, i])
            gamma[n, m] += gamma_dephasing[n, m]

    mu = np.ones_like(gamma)
    np.fill_diagonal(mu, 0.)

    parameters = dict(
        energies=energies_A,
        gamma=gamma,
        mu=mu,
        central_freq=energies_A[0],
        comb_size=5000,
        resolution_size=11,
        omega_M1=170,
        omega_M2=30,
        comb_lw=1e1,
        delta_freq=200,
        N_terms=3
    )

    def render_ticks(axes, axes_field):
        axes.get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=0, labelsize='large')
        axes.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='large')
        axes_field.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='b', labelsize='large')
        axes.get_xaxis().set_ticks_position('both')
        axes.get_yaxis().set_ticks_position('both')
        axes_field.get_yaxis().set_ticks_position('both')

    System = NonLinearResponse3rdOrder(**parameters)
    System.get_polarization_3rd_order()
    System.pol3 *= -1e8
    z_indx = [int(System.frequency.size * (1 / 2 + 1 / 64)), int(System.frequency.size * (25 / 48))]
    print(z_indx[0], z_indx[1])

    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].set_title("Real part of $3^{rd}$ order non-linear response")
    axes[0, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System.pol3.real, 'k')
    axes_field = axes[0, 0].twinx()
    axes_field.plot((System.field_freq1 - System.central_freq) / System.delta_freq, System.field1, 'r', alpha=.5, linewidth=2.)
    axes_field.plot((System.field_freq2 - System.central_freq) / System.delta_freq, System.field2, 'b', alpha=.5, linewidth=2.)
    axes[0, 0].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    render_ticks(axes[0, 0], axes_field)
    axes[0, 0].grid(color='b', linestyle=':', linewidth=0.5)
    axes_field.grid(color='b', linestyle=':', linewidth=0.5)
    axes[0, 0].set_ylim(-1.1 * np.abs(System.pol3.real).max(), 1.1 * np.abs(System.pol3.real).max())

    # axes[1, 0].set_title("Imaginary part of $3^{rd}$ order non-linear response")
    # axes[1, 0].set_xlabel("$(\\omega - \\omega_{20}^A) / {\\Delta \\omega}$", size="large")
    # axes[1, 0].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    # axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    # axes_field.set_ylim(-1.1*np.abs(System.field2).max(), 1.1*np.abs(System.field2).max())
    # axes_field = axes[1, 0].twinx()
    # axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    # axes_field.plot((System.field_freq1 - System.central_freq) / System.delta_freq, System.field1, 'r', alpha=0.5, linewidth=2.)
    # axes_field.plot((System.field_freq2 - System.central_freq) / System.delta_freq, System.field2, 'b', alpha=0.5, linewidth=2.)
    # axes[1, 0].plot((System.frequency - System.central_freq) / System.delta_freq, System.pol3.imag, 'k')
    # axes[1, 0].set_ylim(-1.1 * np.abs(System.pol3.imag).max(), 1.1 * np.abs(System.pol3.imag).max())
    # render_ticks(axes[1, 0], axes_field)
    # axes_field.set_ylim(-1.1*np.abs(System.field2).max(), 1.1*np.abs(System.field2).max())
    # axes[1, 0].grid(color='b', linestyle=':', linewidth=0.5)
    # axes_field.grid(color='b', linestyle=':', linewidth=0.5)
    #
    # axes[0, 1].set_title("Real part of $3^{rd}$ order non-linear response")
    # axes[0, 1].plot((System.frequency[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.pol3[z_indx[0]:z_indx[1]].real, 'k')
    # axes_field = axes[0, 1].twinx()
    # axes_field.plot((System.field_freq1[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field1[z_indx[0]:z_indx[1]], 'r', alpha=.5,
    #                 linewidth=2.)
    # axes_field.plot((System.field_freq2[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field2[z_indx[0]:z_indx[1]], 'b', alpha=.5,
    #                 linewidth=2.)
    # axes[0, 1].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    # render_ticks(axes[0, 1], axes_field)
    # axes[0, 1].grid(color='b', linestyle=':', linewidth=0.5)
    # axes_field.grid(color='b', linestyle=':', linewidth=0.5)
    #
    # axes[1, 1].set_title("Imaginary part of $3^{rd}$ order non-linear response")
    # axes[1, 1].set_xlabel("$(\\omega - \\omega_{20}^A) / {\\Delta \\omega}$", size="large")
    # axes[1, 1].set_ylabel("$2^{nd}$-order polarization $P^{(2)}(\\omega)$ (arb. units)")
    # axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    # axes_field = axes[1, 1].twinx()
    # axes_field.set_ylabel("Electric field $E(\\omega)$ (arb. units)")
    # axes_field.plot((System.field_freq1[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field1[z_indx[0]:z_indx[1]], 'r', alpha=0.5,
    #                 linewidth=2.)
    # axes_field.plot((System.field_freq2[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.field2[z_indx[0]:z_indx[1]], 'b', alpha=0.5,
    #                 linewidth=2.)
    # axes[1, 1].plot((System.frequency[z_indx[0]:z_indx[1]] - System.central_freq) / System.delta_freq, System.pol3[z_indx[0]:z_indx[1]].imag, 'k')
    # render_ticks(axes[1, 1], axes_field)
    # axes[1, 1].grid(color='b', linestyle=':', linewidth=0.5)
    # axes_field.grid(color='b', linestyle=':', linewidth=0.5)

    width = 10**(-np.linspace(-4, 7, 48))
    pol3_lw = np.empty_like(width)
    # for i, w_i in enumerate(width):
    #     System = NonLinearResponse3rdOrder(**parameters)
    #     System.comb_lw = w_i
    #     System.get_polarization_3rd_order()
    #     pol3_lw[i] = np.abs(System.pol3.real).max()
    #     print(i, w_i, pol3_lw[i])

    from multiprocessing import Pool

    def f(x):
        System = NonLinearResponse3rdOrder(**parameters)
        System.comb_lw = x
        System.get_polarization_3rd_order()
        return np.abs(System.pol3.real).max()

    p = Pool(12)
    res = (p.map(f, list(width)))

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(-np.log10(width), -np.log10(res), 'r*-')
    plt.show()
