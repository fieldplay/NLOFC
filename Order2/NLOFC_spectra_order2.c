#include <complex.h>
#include <math.h>
#include <stdio.h>

typedef double complex cmplx;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* gamma;
    double* mu;
    cmplx* pol2;
} molecule;

typedef struct parameters{
    double central_freq;
    int comb_size;
    double omega_M1;
    double omega_M2;
    double comb_lw;
    double delta_freq;
    int N_terms;
    double* frequency;
    int N_freq;
} parameters;


//===========================================================================//
//                                                                           //
//                           MAIN FUNCTIONS                                  //
//                                                                           //
//===========================================================================//
void calculate_pol2_total(molecule* mol, parameters* params)
{

}