__doc__ = """
Python wrapper for NLOFC_spectra_order3.c

Note: You must compile the C shared library
       gcc -O3 -shared -o NLOFC_spectra_order3.so NLOFC_spectra_order3.c -lm -fopenmp -fPIC
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
        ('nDIM', c_int),
        ('energies', POINTER(c_double)),
        ('gamma', POINTER(c_double)),
        ('pol3', POINTER(c_complex))
    ]


class Parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('central_freq', c_double),
        ('comb_size', c_int),
        ('comb_lw', c_double),
        ('delta_freq', c_double),
        ('N_terms', c_int),
        ('frequency', POINTER(c_double)),
        ('N_freq', c_int),
        ('chi_iterator', POINTER(c_double)),
        ('N_iter', c_int),
        ('field_env1', POINTER(c_double)),
        ('field_env2', POINTER(c_double))
    ]

try:
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/NLOFC_spectra_order3.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o NLOFC_spectra_order3.so NLOFC_spectra_order3.c -lm -fopenmp -fPIC
        """
    )

############################################################################################
#
#   Declaring the function pol3
#
############################################################################################

lib.calculate_pol3_total.argtypes = (
    POINTER(Molecule),
    POINTER(Parameters)
)
lib.calculate_pol3_total.restype = None


def get_pol3_total(mol, params):
    return lib.calculate_pol3_total(
        mol,
        params
    )
