#include <complex.h>
#include <math.h>
#include <stdio.h>

typedef double complex cmplx;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* gamma;
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
    double* chi_iterator;
    int N_iter;
    double* field_env1;
    double* field_env2;
} parameters;

void print_double_mat(double *A, int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void pol2_XY(molecule* mol, parameters* params, const double M_field_p, const double M_field_q, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    int m_p0 = (crealf(wg_1) - M_field_p)/params->delta_freq;

    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];
            int m_q0 = omega - M_field_q - (crealf(wg_1))/params->delta_freq;

            cmplx result = 0. + 0. * I;

            for(int m_p = m_p0 - params->N_terms; m_p < m_p0 + params->N_terms; m_p++)
            {
                const cmplx term_X = M_field_p + m_p * params->delta_freq - wg_1 + params->comb_lw * I;
                for(int m_q = m_q0 - params->N_terms; m_q < m_q0 + params->N_terms; m_q++)
                {
                    const cmplx term_Y = omega - (M_field_q + m_q * params->delta_freq - params->comb_lw * I) - wg_1 ;
                    result += 1./(term_X * term_Y);
                }
            }

            mol->pol2[iter_indx*params->N_freq + out_i] += result*sign/(omega - wg_2);
        }

}

void pol2_XZ(molecule* mol, parameters* params, const double M_field_p, const double M_field_q, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    int m_p0 = (crealf(wg_1) - M_field_p)/params->delta_freq;

    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];

            cmplx result = 0. + 0. * I;
            int m_q0 = (omega - M_field_q - M_field_p)/params->delta_freq - m_p0;

            for(int m_p = m_p0 - params->N_terms; m_p < m_p0 + params->N_terms; m_p++)
            {
                const cmplx term_X = M_field_p + m_p * params->delta_freq - wg_1 + params->comb_lw * I;
                for(int m_q = m_q0 - params->N_terms; m_q < m_q0 + params->N_terms; m_q++)
                {
                    const cmplx term_Z = M_field_p + M_field_q - omega + (m_p + m_q) * params->delta_freq + 2 * I * params->comb_lw;
                    result += 1./(term_X * term_Z);
                }
            }

            mol->pol2[iter_indx*params->N_freq + out_i] += result*sign/(omega - wg_2);
        }

}

void pol2_YZstar(molecule* mol, parameters* params, const double M_field_p, const double M_field_q, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];
            int m_q0 = (omega - M_field_q - crealf(wg_1))/params->delta_freq;
            int m_p0 = (omega - M_field_q - M_field_p)/params->delta_freq - m_q0;

            cmplx result = 0. + 0. * I;

            for(int m_q = m_q0 - params->N_terms; m_q < m_q0 + params->N_terms; m_q++)
            {
                for(int m_p = m_p0 - params->N_terms; m_p < m_p0 + params->N_terms; m_p++)
                {
                    const cmplx term_Zstar = omega - (M_field_p + M_field_q  + (m_p + m_q) * params->delta_freq) + 2 * I * params->comb_lw;
                    const cmplx term_X = M_field_p + m_p * params->delta_freq - wg_1 + params->comb_lw * I;
                    result -= 1./(term_X * term_Zstar);
                }
            }

            mol->pol2[iter_indx*params->N_freq + out_i] += result*sign/(omega - wg_2);
        }

}

void pol2(molecule* mol, parameters* params, const double M_field_p, const double M_field_q, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    pol2_XY(mol, params, M_field_p, M_field_q, wg_2, wg_1, sign, iter_indx);
    pol2_XZ(mol, params, M_field_p, M_field_q, wg_2, wg_1, sign, iter_indx);
    pol2_YZstar(mol, params, M_field_p, M_field_q, wg_2, wg_1, sign, iter_indx);

    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            mol->pol2[iter_indx*params->N_freq + out_i] *= params->field_env1[out_i] * params->field_env2[out_i];
        }
}

//===========================================================================//
//                                                                           //
//                           MAIN FUNCTIONS                                  //
//                                                                           //
//===========================================================================//
void calculate_pol2_total(molecule* mol, parameters* params)
{
    int l, m, n;
    double M_field_p, M_field_q;
    cmplx wg_ml, wg_nl, wg_mn, wg_nm;

    for(int i=0; i<params->N_iter; i++)
    {
        M_field_p = params->chi_iterator[i*4 + 0];
        M_field_q = params->chi_iterator[i*4 + 1];
        m = (int) params->chi_iterator[i*4 + 2];
        n = (int) params->chi_iterator[i*4 + 3];
        l = 0;

        wg_ml = mol->energies[m] - mol->energies[l] + I * mol->gamma[m * mol->nDIM + l];
        wg_nl = mol->energies[n] - mol->energies[l] + I * mol->gamma[n * mol->nDIM + l];
        wg_mn = mol->energies[m] - mol->energies[n] + I * mol->gamma[m * mol->nDIM + n];
        wg_nm = mol->energies[n] - mol->energies[m] + I * mol->gamma[n * mol->nDIM + m];

        pol2(mol, params, M_field_p, M_field_q, conj(wg_nl), conj(wg_ml), 1, i);
        pol2(mol, params, M_field_p, M_field_q, conj(wg_mn), -wg_nl, -1, i);
        pol2(mol, params, M_field_p, M_field_q, -wg_nm, conj(wg_ml), -1, i);
        pol2(mol, params, M_field_p, M_field_q, -wg_ml, -wg_nl, 1, i);
    }
}