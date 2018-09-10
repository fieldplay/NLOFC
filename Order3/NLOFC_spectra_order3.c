#include <complex.h>
#include <math.h>
#include <stdio.h>

typedef double complex cmplx;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* gamma;
    cmplx* pol3;
} molecule;

typedef struct parameters{
    double central_freq;
    int comb_size;
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

void pol3_XYR(molecule* mol, parameters* params, const double M_field_h, const double M_field_i, const double M_field_j,
    const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    int p0 = ceil((crealf(wg_1) - M_field_h)/params->delta_freq);
    int q0 = ceil((crealf(wg_2) - M_field_h - M_field_i)/params->delta_freq) - p0;

    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - omega - M_field_j)/params->delta_freq);

            for(int p = p0 - params->N_terms; p < p0 + params->N_terms; p++)
            {
                const cmplx term_R = M_field_h + p * params->delta_freq - wg_1 + params->comb_lw * I;
                for(int q = q0 - params->N_terms; q < q0 + params->N_terms; q++)
                {
                    const cmplx term_Y = M_field_h + M_field_i + (p + q) * params->delta_freq - wg_2 + 2. * params->comb_lw * I;
                    for(int r = r0 - params->N_terms; r < r0 + params->N_terms; r++)
                    {
                        const cmplx term_X = omega + M_field_j + r * params->delta_freq - wg_2 + params->comb_lw * I;
                        result += 1./(term_X * term_Y * term_R);
                    }

                }
            }

            mol->pol3[iter_indx*params->N_freq + out_i] += sign*result/(omega - wg_3);
        }

}

void pol3_XSR(molecule* mol, parameters* params, const double M_field_h, const double M_field_i, const double M_field_j,
    const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    int p0 = ceil((crealf(wg_1) - M_field_h)/params->delta_freq);

    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - omega - M_field_j)/params->delta_freq);
            int q0 = r0 - ceil((crealf(wg_2) - M_field_h - M_field_i)/params->delta_freq);

            for(int p = p0 - params->N_terms; p < p0 + params->N_terms; p++)
            {
                const cmplx term_R = M_field_h + p * params->delta_freq - wg_1 + params->comb_lw * I;
                for(int q = q0 - params->N_terms; q < q0 + params->N_terms; q++)
                {
                    for(int r = r0 - params->N_terms; r < r0 + params->N_terms; r++)
                    {
                        const cmplx term_X = omega + M_field_j + r * params->delta_freq - wg_2 + params->comb_lw * I;
                        const cmplx term_S = omega + M_field_j - M_field_i + (r - q) * params->delta_freq - wg_2 + 2. * params->comb_lw * I;
                        result += 1./(term_X * term_S * term_R);
                    }

                }
            }

            mol->pol3[iter_indx*params->N_freq + out_i] += sign*result/(omega - wg_3);
        }

}

void pol3_XSZ(molecule* mol, parameters* params, const double M_field_h, const double M_field_i, const double M_field_j,
    const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - omega - M_field_j)/params->delta_freq);
            int q0 = r0 - ceil((crealf(wg_1) - omega - M_field_j + M_field_i)/params->delta_freq);
            int p0 = ceil((omega - (M_field_h + M_field_i - M_field_j))/params->delta_freq)
                            + ceil((crealf(wg_1) - omega - M_field_j + M_field_i)/params->delta_freq);

            for(int p = p0 - params->N_terms; p < p0 + params->N_terms; p++)
            {
                for(int q = q0 - params->N_terms; q < q0 + params->N_terms; q++)
                {
                    for(int r = r0 - params->N_terms; r < r0 + params->N_terms; r++)
                    {
                        const cmplx term_X = omega + M_field_j + r * params->delta_freq - wg_2 + params->comb_lw * I;
                        const cmplx term_S = omega + M_field_j - M_field_i + (r - q) * params->delta_freq - wg_2 + 2. * params->comb_lw * I;
                        const cmplx term_Z = omega - (M_field_h + M_field_i - M_field_j) - (p + q - r) * params->delta_freq + 3. * params->comb_lw * I;
                        result += 1./(term_X * term_S * term_Z);
                    }

                }
            }

            mol->pol3[iter_indx*params->N_freq + out_i] += sign*result/(omega - wg_3);
        }

}

void pol3_YRZstar(molecule* mol, parameters* params, const double M_field_h, const double M_field_i, const double M_field_j,
    const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{

    int p0 = ceil((crealf(wg_1) - M_field_h)/params->delta_freq);
    int q0 = ceil((crealf(wg_2) - M_field_h - M_field_i)/params->delta_freq) - p0;

    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - M_field_h - M_field_i)/params->delta_freq)
                            - ceil((omega - (M_field_h + M_field_i - M_field_j))/params->delta_freq);

            for(int p = p0 - params->N_terms; p < p0 + params->N_terms; p++)
            {
                const cmplx term_R = M_field_h + p * params->delta_freq - wg_1 + params->comb_lw * I;
                for(int q = q0 - params->N_terms; q < q0 + params->N_terms; q++)
                {
                    const cmplx term_Y = M_field_h + M_field_i + (p + q) * params->delta_freq - wg_2 + 2. * params->comb_lw * I;
                    for(int r = r0 - params->N_terms; r < r0 + params->N_terms; r++)
                    {
                        const cmplx term_Zstar = omega - (M_field_h + M_field_i - M_field_j) - (p + q - r) * params->delta_freq - 3. * params->comb_lw * I;
                        result += 1./(term_Y * term_R * term_Zstar);
                    }

                }
            }

            mol->pol3[iter_indx*params->N_freq + out_i] -= sign*result/(omega - wg_3);
        }

}


void pol3(molecule* mol, parameters* params, const double M_field_h, const double M_field_i, const double M_field_j,
 const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign, int iter_indx)
{
    pol3_XYR(mol, params, M_field_h, M_field_i, M_field_j, wg_3, wg_2, wg_1, sign, iter_indx);
    pol3_XYR(mol, params, M_field_h, M_field_i, M_field_j, wg_3, wg_2, wg_1, sign, iter_indx);
    pol3_XSZ(mol, params, M_field_h, M_field_i, M_field_j, wg_3, wg_2, wg_1, sign, iter_indx);
    pol3_YRZstar(mol, params, M_field_h, M_field_i, M_field_j, wg_3, wg_2, wg_1, sign, iter_indx);

    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            mol->pol3[iter_indx * params->N_freq + out_i] *= params->field_env1[out_i] * params->field_env2[out_i] * params->field_env2[out_i];
        }
}

//===========================================================================//
//                                                                           //
//                           MAIN FUNCTIONS                                  //
//                                                                           //
//===========================================================================//
void calculate_pol3_total(molecule* mol, parameters* params)
{
    int l, m, n, v;
    double M_field_h, M_field_i, M_field_j;
    cmplx wg_ml, wg_nl, wg_mn, wg_nm, wg_vl, wg_mv, wg_nv, wg_vm, wg_vn;

    for(int i=0; i<1; i++)
    {
        M_field_h = params->chi_iterator[i*6 + 0];
        M_field_i = params->chi_iterator[i*6 + 1];
        M_field_j = params->chi_iterator[i*6 + 2];
        m = (int) params->chi_iterator[i*6 + 3];
        n = (int) params->chi_iterator[i*6 + 4];
        v = (int) params->chi_iterator[i*6 + 5];
        l = 0;

        wg_ml = mol->energies[m] - mol->energies[l] + I * mol->gamma[m * mol->nDIM + l];
        wg_nl = mol->energies[n] - mol->energies[l] + I * mol->gamma[n * mol->nDIM + l];
        wg_mn = mol->energies[m] - mol->energies[n] + I * mol->gamma[m * mol->nDIM + n];
        wg_nm = mol->energies[n] - mol->energies[m] + I * mol->gamma[n * mol->nDIM + m];
        wg_vl = mol->energies[v] - mol->energies[l] + I * mol->gamma[v * mol->nDIM + l];
        wg_mv = mol->energies[m] - mol->energies[v] + I * mol->gamma[m * mol->nDIM + v];
        wg_nv = mol->energies[n] - mol->energies[v] + I * mol->gamma[n * mol->nDIM + v];
        wg_vn = mol->energies[v] - mol->energies[n] + I * mol->gamma[v * mol->nDIM + n];
        wg_vm = mol->energies[n] - mol->energies[m] + I * mol->gamma[v * mol->nDIM + m];

        pol3(mol, params, M_field_h, M_field_i, M_field_j, conj(wg_vl), conj(wg_nl), -conj(wg_vl), -1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, conj(wg_nv), conj(wg_mv), wg_vl, 1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, conj(wg_nv), -wg_vm, -conj(wg_ml), 1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, conj(wg_mn), -wg_nl, wg_vl, -1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, -wg_vn, conj(wg_nl), -conj(wg_ml), 1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, -wg_nm, conj(wg_mv), wg_vl, -1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, -wg_nm, -wg_mv, -conj(wg_ml), -1, i);
        pol3(mol, params, M_field_h, M_field_i, M_field_j, -wg_ml, -wg_nl, wg_vl, 1, i);
    }
}