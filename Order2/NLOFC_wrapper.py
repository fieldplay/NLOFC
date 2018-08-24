__doc__ = """
Python wrapper for NLOFC_Spectra_order2.c

Note: You must compile the C shared library
       gcc -O3 -shared -o NLOFC_Spectra_order2.so NLOFC_Spectra_order2.c -lm -fopenmp -fPIC
"""
import os
import ctypes
from ctypes import c_double, c_int, POINTER, Structure


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [('real', c_double), ('imag', c_double)]


class Molecule(Structure):
    """
    Molecule structure ctypes
    """
    _fields_ = [
        ('energies', POINTER(c_double)),
        ('gamma', POINTER(c_double)),
        ('mu', POINTER(c_double)),
        ('pol2', POINTER(c_complex))
    ]


class Parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('central_freq', c_double),
        ('comb_size', c_int),
        ('omega_M1', c_double),
        ('omega_M2', c_double),
        ('comb_lw', c_double),
        ('delta_freq', c_double),
        ('N_terms', c_int),
        ('frequency', POINTER(c_double))
    ]

try:
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/eval_pol2_full.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o NLOFC_Spectra_order2.so NLOFC_Spectra_order2.c -lm -fopenmp -fPIC
        """
    )

############################################################################################
#
#   Declaring the function pol2_a2
#
############################################################################################

lib.calculate_pol2_total.argtypes = (
    POINTER(Molecule),
    POINTER(Parameters)
)
lib.calculate_pol2_total.restype = None


def get_pol2_total(mol, params):
    return lib.calculate_pol2_total(
        mol,
        params
    )
