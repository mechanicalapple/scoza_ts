/*
    SCOZA TS - thermodynamic software based on self-consistent Ornstein-Zernike application

    Copyright (C) 2017 Artem A Anikeev

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <linux/limits.h>	// PATH_MAX
//#include <sys/types.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define DEBUG 1
#define DEBUG2 1
//#define DEBUG3 1
#define PARALLEL 1
#define STRICT 1 //for strict numerical compatibility between parallel and non-parallel
//#define FFTW 1
//#define OUTRESFILES 1
#define CALC_RDF 1
//#define AUTO_SCALE //for automatic basis potential scaling

#ifdef PARALLEL
#include <omp.h>
#endif

#include <fftw3.h>

#include <stdarg.h>	// va_*()

#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multiroots.h>


/* ===== Global Stuff ===== */

#define MAX_PAR_COUNT (1<<3) //max count of potential parameters
#define MAX_INITIALS (1<<3) //max count of initial parameters
#define MAX_CAL_PARMS (1<<4) //max count of calorical parameters
#define BLIND_ITER 100

double kb = 1.38064852e-23; //Boltzmann constant, J/K
double Na = 6.02214082e23; //Avogadro constant, mole^-1
double KILO = 1000.0;
double TENKILO = 10000.0;
double IUPAC1982_P = 10000.0; //IUPAC1982 standard state pressure in Pa for NIST JANAF and IVTANTERMO gas thermochemistry DBs both. Do not be confused with NIST not ISO10780 nor NTP nor STP nor SATP.

#ifdef M_PI
#	define pi M_PI
#else
	double pi = 3.141592653589793;
#endif

struct external_tdc_data
{
	int problem_type,**mole_compos,*calor_style, n_atom_type, Nnewt, itermaxnewt, *frozen;
	double *initials, *atom_weights, **cal_parms, deltanewt, dnewt;

	char thermo_path[PATH_MAX];
}exter_tdc;

struct internal_tdc_data
{
	double *mu_thermal_id, *mu_thermal_ex, *mu_caloric, *mu, e_scale, r_scale, *mol_wv, **a;
}inter_tdc;

struct external_data
{
	int n_molec_type,***itermax,itermaxsmall,itermaxbig,itermaxbiggest,nh,N,***poten_type,inp__type,Na,Nrho,***Nphi,***r_type,***cl_type,***sw_type;
	double Rhicut,*a0,dK,delta,Te,rho,drho,***p,***rfe,***d,deltam,***lmb,*mol_fr,da,maxabsdlg;
}external;

struct internal_data
{
	int pair_count, ***nl, ***nrm;
	double dr, ***Rlocut, ***rm ,dq, b, *r, *q, ***fi0, ***fi1, *Simp, *Simp0, *rmul, dlambda;
	//double **si;
	fftw_plan plan;
}internal;

struct parameters
{
	gsl_vector *y;
	FILE *file, *file1, *file2;
};

typedef double (*funct_t)(int i, int j, double r);
typedef int (*funct_cl)(double *T, double *C, double *fsw, int k, int kk, FILE *file); //closure
typedef double (*funct_b)(double *T, double *C, double *fsw, int k, int kk, int i, double lambda, FILE *file); //bridge
typedef double (*funct_sw)(int i, int j, double r, double a); //switch function
typedef double (*funct_cp0)(int i); //heat capacity
typedef double (*funct_h0)(int i); //standard enthalpy
typedef double (*funct_s0)(int i); //standard entropy

#define MAX_N_MOLEC_TYPE (1<<8) //max count of molecular types

funct_t	_f[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_df[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_ddf[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_d3f[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_d4f[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_d5f[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_I1[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_t	_I2[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_cl	_closure[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_b		_bridge[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_sw	_swfu[MAX_N_MOLEC_TYPE][MAX_N_MOLEC_TYPE];
funct_cp0	_cp0[MAX_N_MOLEC_TYPE];
funct_h0	_h0[MAX_N_MOLEC_TYPE];
funct_s0	_s0[MAX_N_MOLEC_TYPE];

static inline int fscanf_safe(FILE *stream, const char *format, ...) {
	va_list args;
	int r;
	va_start(args, format);
	errno = 0;
	r = vfscanf(stream, format, args);
	va_end(args);

	if (r < 1||errno)
	{
		fprintf(stderr, "fscanf_safe(): Cannot parse any value (\"%s\") :%s\n", format,strerror(errno));
		exit(-1);
	}

	return r;
}

double f(int i, int j,  double r)
{
	return _f[i][j](i,j,r);
}
double df(int i, int j, double r)
{
	return _df[i][j](i,j,r);
}
double ddf(int i, int j, double r)
{
	return _ddf[i][j](i,j,r);
}
double d3f(int i, int j, double r)
{
	return _d3f[i][j](i,j,r);
}
double d4f(int i, int j, double r)
{
	return _d4f[i][j](i,j,r);
}
double d5f(int i, int j, double r)
{
	return _d5f[i][j](i,j,r);
}
double I1(int i, int j, double r)
{
	return _I1[i][j](i,j,r);
}
double I2(int i, int j, double r)
{
	return _I2[i][j](i,j,r);
}
int closure(double *T, double *C, double *fsw, int k, int kk, FILE *file)
{
	if (k > kk)
	{
		return _closure[kk][k](T,C,fsw,kk,k,file);
	}
	else
	{
		return _closure[k][kk](T,C,fsw,k,kk,file);
	}
}
double bridge(double *T, double *C, double *fsw, int k, int kk, int i, double lambda, FILE *file)
{
	if (k > kk)
	{
		return _bridge[kk][k](T,C,fsw,kk,k,i,lambda,file);
	}
	else
	{
		return _bridge[k][kk](T,C,fsw,k,kk,i,lambda,file);
	}
}
double swfu(int i, int j, double r, double a)
{
	return _swfu[i][j](i,j,r,a);
}

double cp0(int i)
{
	return _cp0[i](i);
}

double h0(int i)
{
	return _h0[i](i);
}

double s0(int i)
{
	return _s0[i](i);
}

/* ===== Mathematics ===== */  

int createv(int n, double **a)
{
	double *m;
	m = (double*)malloc(sizeof(double) * n);
	*a = m;
	return 0;
}

int createa(int n, int **a)
{
	int *m;
	m = (int*)malloc(sizeof(int) * n);
	*a = m;
	return 0;
}

int createa_tri(int n, int ****a)
{
	int i,j,***b;
	b = (int***)malloc(sizeof(int**) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (int**)malloc(sizeof(int*) * n);
		for (j = 0; j <= i; j++)
		{
			b[i][j] = (int*)malloc(sizeof(int));
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			b[i][j] = b[j][i];
		}
	}
	*a=b;
	return 0;
}

int freea_tri(int n, int ***a)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j <= i; j++)
		{
			free(a[i][j]);
		}
		free(a[i]);
	}
	free(a);
	return 0;
}

int outv(int n, double *a, FILE *file)
{
	int i;
	double x;
	//flockfile(file); //Should be locked from parent process
	for (i = 0; i < n; i++)
	{
		x = a[i];
		fprintf(file,"\na[%i]=%.9e",i,x);
	}
	fprintf(file,"\n");
	//funlockfile(file);
	return 0;
}

int createm(int n, int m, double ***a)
{
	int i;
	double **b;
	b = (double**)malloc(sizeof(double*) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (double*)malloc(sizeof(double) * m);
	}
	*a=b;
	return 0;
}

int freem(int n, double **a)
{
	int i;
	for (i = 0; i < n; i++)
	{
		free(a[i]);
	}
	free(a);
	return 0;
}

int outm(int n, double **a, FILE *file)
{
	int i,j;
	double x;
	//flockfile(file); //Should be locked from parent process
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			x = a[i][j];
			fprintf(file,"\na[%i][%i]=%.9e ",i,j,x);	
		}
		fprintf(file,"\n");
	}
	fprintf(file,"\n");
	//funlockfile(file);
	return 0;
}

int gete(int n, double **a)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i == j)
			{
				a[i][j]=1.0;
			}
			
			if ((i<j)|(i>j))
			{
				a[i][j]=0.0;
			}
		}
	}
	return 0;
}

int createma(int n, int m, int ***a)
{
	int i,**b;
	b = (int**)malloc(sizeof(int*) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (int*)malloc(sizeof(int) * m);
	}
	*a=b;
	return 0;
}

int freema(int n, int **a)
{
	int i;
	for (i = 0; i < n; i++)
	{
		free(a[i]);
	}
	free(a);
	return 0;
}


int createm3(int n, int m, int o, double ****a)
{
	int i,j;
	double ***b;
	b = (double***)malloc(sizeof(double**) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (double**)malloc(sizeof(double*) * m);
		for (j = 0; j < m; j++)
		{
			b[i][j] = (double*)malloc(sizeof(double) * o);
		}
	}
	*a=b;
	return 0;
}

int freem3(int n, int m, double ***a)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			free(a[i][j]);
		}
		free(a[i]);
	}
	free(a);
	return 0;
}

/*
int createm_tri(int n, int m, double ****a)
{
	int i,j;
	double ***b;
	b = (double***)malloc(sizeof(double**) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (double**)malloc(sizeof(double*) * n);
		for (j = 0; j <= i; j++)
		{
			b[i][j] = (double*)malloc(sizeof(double) * m);
//			memset(b[i][j], 0x7f, sizeof(double) * m);	// For debugging only
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			b[i][j] = b[j][i];
		}
	}
	*a=b;
	return 0;
}

int freem_tri(int n, double ***a)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j <= i; j++)
		{
			free(a[i][j]);
		}
		free(a[i]);
	}
	free(a);
	return 0;
}
*/

int createm_tri(const int n, const int m, double ****a)
{
	int i,j,tmpsize;
	double ***b;
	double *current;

	tmpsize = 0;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j <= i; j++)
		{
			tmpsize++;
		}
	}

	// The internal one
	current = fftw_alloc_real(tmpsize * m);

	// The external one
	b = (double***)malloc(sizeof(double**) * n);
	for (i = 0; i < n; i++)
	{
		// The middle one
		b[i] = (double**)malloc(sizeof(double*) * n);
		for (j = 0; j <= i; j++)
		{
			/*
			if (!((a == 0)&&(b == 0)))
			{
				current += sizeof(double) * m;
			}
			*/
			b[i][j] = current;
			current += m;
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			b[i][j] = b[j][i];
		}
	}
	*a=b;
	return 0;
}

int freem_tri(int n, double ***a)
{
	int i;
	fftw_free(a[0][0]);
	for (i = 0; i < n; i++)
	{
		free(a[i]);
	}
	free(a);
	return 0;
}

int copym_tri(int n, int m, double ***a, double ***b)
{
	memcpy(b[0][0], a[0][0], sizeof(double) * m * internal.pair_count);
	return 0;
}

int createma_tri(int n, int m, int ****a)
{
	int i,j,***b;
	b = (int***)malloc(sizeof(int**) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (int**)malloc(sizeof(int*) * n);
		for (j = 0; j <= i; j++)
		{
			b[i][j] = (int*)malloc(sizeof(int) * m);
//			memset(b[i][j], 0x7f, sizeof(double) * m);	// For debugging only
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			b[i][j] = b[j][i];
		}
	}
	*a=b;
	return 0;
}

int freema_tri(int n, int ***a)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j <= i; j++)
		{
			free(a[i][j]);
		}
		free(a[i]);
	}
	free(a);
	return 0;
}

int createv_tri(int n, double ****a)
{
	int i,j;
	double ***b;
	b = (double***)malloc(sizeof(double**) * n);
	for (i = 0; i < n; i++)
	{
		b[i] = (double**)malloc(sizeof(double*) * n);
		for (j = 0; j <= i; j++)
		{
			b[i][j] = (double*)malloc(sizeof(double));
		}
	}
	for (i = 0; i < n; i++)
	{
		for (j = i + 1; j < n; j++)
		{
			b[i][j] = b[j][i];
		}
	}
	*a=b;
	return 0;
}

int freev_tri(int n, double ***a)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j <= i; j++)
		{
			free(a[i][j]);
		}
		free(a[i]);
	}
	free(a);
	return 0;
}

int createmsin(int n, double ***a)//if j>=i+1 a[i][j] is not correct. use a[j][i]
{
	int i,j;
	double **m;
	m = (double**)malloc(sizeof(double*) * n);
	for (i = 0; i < n; i++)
	{
		m[i] = (double*)malloc(sizeof(double) * (i + 1));
		for (j = 0; j <= i; j++)
		{
			m[i][j] = sin(pi / ((double)n + 1.0)*((double)i + 1.0) * ((double)j + 1.0));//n+1 is correct. take a look in the theory. they use N-1 for n nodes
		}
	}
	*a=m;
	return 0;
}

int outmsin(int n,double **a,FILE *file)//if j>=i+1 a[i][j] is not correct. use a[j][i]
{
	int i,j;
	double x;
	//flockfile(file); //Should be locked from parent process
	for (i = 0; i < n; i++)
	{
		for (j = 0 ; j < n ; j++)
		{
			if (j <= i)
			{
				x = a[i][j];
			}
			else
			{
				x = a[j][i];
			}
			fprintf(file,"\na[%i][%i]=%9e ",i,j,x);
		}
		fprintf(file,"\n");
	}
	fprintf(file,"\n");
	//funlockfile(file);
	return 0;
}

int fourier(int n, double *a, double *aa, double x, double **si)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		aa[i] = 0.0;
		for (j = 0 ; j < n ; j++)
		{
			if (j <= i)
			{
				aa[i] = aa[i] + x * a[j] * si[i][j];
			}
			else
			{
				aa[i] = aa[i] + x * a[j] * si[j][i];
			}
		}
	}
	return 0;
}

int copym(int n,double **a, double **b)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = 0 ; j < n ; j++)
		{
			b[i][j] = a[i][j];
		}
	}
	return 0;
}

/*
int copym_tri(int n, int m, double ***a, double ***b)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
		for (j = i; j < n; j++)
		{
			memcpy(b[i][j], a[i][j], sizeof(*b[i][j])*m);
		}
	}
	return 0;
}
*/

int getlu(int M, double **LU, double **cjk,double *CC, double *q, double ro)
{
	int i,j;
	double x;
	for (i = 0; i < M; i++)
	{
		x = ro *CC[i] / (q[i] - ro * CC[i]);
		for (j = 0 ; j < M ; j++)
		{
			if (i == j)
			{
				LU[i][j] = 1.0 - cjk[i][j] * x * (2.0 + x);/////////////////////////////////WARNING!!!!
			}
			else
			{
				LU[i][j] = - cjk[i][j] * x * (2.0 + x);////////////////////////////////WARNING!!!!
			}
		}
	}
	return 0;
}

int getv(int M, double *v, double *TT,double *CC, double *q, double ro)
{
	int i;
	for (i = 0; i < M; i++)
	{
		v[i] = ro * CC[i] * CC[i] / (q[i] - ro * CC[i]) - TT[i];
	}
	return 0;
}

int lu(int n,double **a, double **l,double **u,FILE *file)
{
	int i,j,k;
	double x;
	copym(n,a,u);
	gete(n,l);
	for (k = 1; k < n; k++)
	{
		for (i = k; i < n; i++)
		{
			//printf("\nk=%i,i=%i",k,i);
			if (u[k-1][k-1] == 0)
			{
				flockfile(file);
				fprintf(file,"\ndivision by zero!");
				funlockfile(file);
			}
			else
			{
				l[i][k-1] = (u[i][k-1] / u[k-1][k-1]);
				u[i][k-1] = 0;
			}
		}

		for (j = k; j < n; j++)
		{
			x = u[k-1][j];
			for (i = k ; i < n ; i++)
			{
				u[i][j] = (u[i][j] - l[i][k-1] * x);
			}
		}
	}
	return 0;
}

int solveu(int n,double **a,double *x,double *y,FILE *file)
{
	int i,j;
	double s;

	///////WARNING!///////////////
	for (i = 0; i < n; i++)
	{
		x[i] = 0;
	}
	////////////////////////////

	for (i = n - 1; i >= 0; i--)
	{
		s = 0.0;
		for (j = i; j < n; j++)
		{
			s += a[i][j] * (x[j]);
		}

		if (fabs(a[i][i]) < 1.0e-100)//Warning! ==0.0?
		{
			flockfile(file);
			fprintf(file,"\ndivision by zero!");
			funlockfile(file);
		}
		else
		{
			x[i] = (y[i] - s) / a[i][i];
		}
	}
	return 0;
}

int solvel(int n,double **a,double *x,double *y,FILE *file)
{
	int i,j;
	double s;

	///////WARNING!///////////////
	for (i = 0; i < n; i++)
	{
		x[i] = 0;
	}
	////////////////////////////

	for (i = 0; i < n; i++)
	{
		s = 0.0;
		for (j = 0; j < i; j++)
		{
			s += a[i][j] * x[j];
		}

		if (fabs(a[i][i]) < 1.0e-100)//Warning! ==0.0?
		{
			flockfile(file);
			fprintf(file,"\ndivision by zero!");
			funlockfile(file);
		}
		else
		{
		x[i] = (y[i] - s) / a[i][i];
		}
	}
	return 0;
}

int findrev(int n, double **x, double **l, double **u)
{
	int i,j,k;
	double sum;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			x[i][j] = 0.0;
		}
	}

	for (i = n - 1; i >= 0; i--)
	{
		for (j = n - 1; j >= 0; j--)
		{

			if (i==j)
			{
				sum = 0.0;
				for (k = i + 1 ; k < n ; k++)
				{
					sum += u[i][k] * x[k][j];
				}
				x[i][i] = 1.0 / u[i][i] * (1.0 - sum);
			}
			else
			{
				if(i<j)
				{
					sum = 0.0;
					for (k = i + 1 ; k < n ; k++)
					{
						sum += u[i][k] * x[k][j];
					}
					x[i][j] = -sum / u[i][i];
				}
				else
				{
					sum = 0.0;
					for (k = j + 1 ; k < n ; k++)
					{
						sum += l[k][j] * x[i][k];
					}
					x[i][j] = -sum;
				}
			}
		}
	}

	return 0;
}
double normm(int n, double **m)
{
	int i,j;
	double tmp;

	tmp = 0.0;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			tmp += m[i][j] * m[i][j];
		}
	}
	return sqrt(tmp);
}

int newtonlogmul(int i, int j)
{
	double a,b,c,h,dlg;
	int k;
	a = *external.rfe[i][j];
	c = *external.rfe[i][j];
	h = a / 2.0;
	k = 0;
	while (((h - c > *external.d[i][j])|(h - c < - *external.d[i][j]))&&(k < *external.itermax[i][j]))
	{
		k++;
		h = c;
		dlg = - df(i,j,c) / (ddf(i,j,c) * c);
		b = fabs(2.0 / dlg);
		if (b > 1.0)
		{
			b = 1.0;
		}
		c = c * exp(b * dlg);

		
	}

	if (k < *external.itermax[i][j])
	{
		*internal.Rlocut[i][j] = c;
	}
	else
	{
		printf("Newtonlogmul: error!");
		*internal.Rlocut[i][j] = 0.0;
	}
	
	return 0;
}

int getjacob(int n, double *x, double **LU, FILE *file, FILE *file1, FILE *file2);

int newtonv(int n, double *x, double *y, FILE *file, FILE *file1, FILE *file2)
{
	double *last,delta,*v,**LU,**L,**U;
	int i,k;
	createv(n,&last);
	createv(n,&v);
	createm(n,n,&LU);
	createm(n,n,&L);
	createm(n,n,&U);

	k = 0;
	delta = exter_tdc.deltanewt + 1.0;
	while ((delta > exter_tdc.deltanewt)&&(k < exter_tdc.itermaxnewt))
	{
		fprintf(file,"\ntdc step %i\n",k);
		k++;
		for (i = 0; i < n; i++)
		{
			last[i] = x[i];
		}

		getjacob(n,x,LU,file,file1,file2);
		//lu(external.n_molec_type,LU,L,U,file);
		//solvel(external.n_molec_type,L,y,v,file);
		//solveu(external.n_molec_type,U,v,x,file);
		lu(n,LU,L,U,file);
		solvel(n,L,v,y,file);
		solveu(n,U,x,v,file);

		//
		outv(n,x,file);
		outm(n,LU,file);
		outm(n,L,file);
		outm(n,U,file);
		//

		delta = 0.0;
		for (i = 0; i < n; i++)
		{
			double tmp = ((last[i] - x[i]) / last[i]);
			//
			fprintf(file,"\ntdc debug last %le x %le tmp %le",last[i],x[i],tmp);
			//
			delta += tmp * tmp;
		}
		delta = sqrt(delta);
		//
		fprintf(file,"\ntdc debug delta = %le",delta);
		//
	}

	freem(n,U);
	freem(n,L);
	freem(n,LU);
	free(v);
	free(last);

	return 0;
}

double inner_product(int n, double *a, double *r, double dr)
{
	int i;
	double b;
	b = 0.0;
	for (i = 0; i < n; i++)
	{
		b += a[i] * a[i] * r[i] * r[i];
	}
	b = b * dr;
	return b;
}

int getsimp(int n, double *Simp)//With zero point shifting: 4,2,4,2...
{
	int i, b0dd;
	b0dd = 0;
	for (i = 0; i < n; i++)
	{
		if ( b0dd == 0 )
		{
			Simp[i] = 4.0;
			b0dd = 1;
		}
		else
		{
			Simp[i] = 2.0;
			b0dd = 0;
		}
	}
	return 0;
}

int getsimp0(int n, double *Simp)//Without zero point shifting: 1,4,2,4,2...4,1
{
	int i, b0dd;
	b0dd = 0;
	for (i = 0; i < n; i++)
	{
		if ((i == 0)^(i == n - 1))
		{
			Simp[i] = 1.0;
		}
		else
		{
			if ( b0dd == 0 )
			{
				Simp[i] = 4.0;
				b0dd = 1;
			}
			else
			{
				Simp[i] = 2.0;
				b0dd = 0;
			}
		}
	}
	return 0;
}

int vmul(int n, double *a, double *b, double *c)
{
	int i;
	for (i = 0; i < n; i++)
	{
		c[i] = a[i] * b[i];
	}
	return 0;
}

/* ===== Physics ===== */

double sw_exp(int i, int j, double r, double a)//switch function //TODO: wether coupling coefficient lambda is needed?
{
	return 1.0 - exp( - a * r);
}

int hmsa(double *T, double *C, double *fsw, int k, int kk, FILE *file) //TODO: rewrite universal closure with variable Bridge if possible for linear and non-linear closures
{
#ifdef DEBUG3
#ifdef PARALLEL
	flockfile(file);
#endif
#endif
	int i;
	for (i = 0; i < *internal.nl[k][kk]; i++)
	{
		C[i] = - internal.r[i] - T[i];
	#ifdef DEBUG3
		fprintf(file,"\ni:%i C:%e T:%e r:%e",i,C[i],T[i],internal.r[i]);
	#endif
	}
	for (i = *internal.nl[k][kk]; i < external.nh; i++)
	{
		C[i] = - internal.r[i] - T[i]
             + internal.r[i] * exp(- internal.b * internal.fi0[k][kk][i]) * 
                 ( 1.0 + ( exp(fsw[i] * (T[i] / internal.r[i] - internal.b * internal.fi1[k][kk][i])) - 1.0 ) / fsw[i] );
	#ifdef DEBUG3
//		fprintf(file,);
	#endif
	}
#ifdef DEBUG3
#ifdef PARALLEL
	funlockfile(file);
#endif
#endif
	return 0;
}

double Bhmsa(double *T, double *C, double *fsw, int k, int kk, int i, double lambda, FILE *file)
{
#ifdef DEBUG3
#ifdef PARALLEL
	flockfile(file);
#endif
#endif
	//s=t-bfi1
	//B=-s*lmb+ln(1+(exp(f*s*lmb)-1)/f)

	double sl;

	#ifdef DEBUG3
		fprintf(file,"\nk %i kk %i i %i T[i] %p r[i] %p fi1[k][kk][i] %p fsw[i] %p",k,kk,i,&T[i],&internal.r[i],&internal.fi1[k][kk][i],&fsw[i]);
	#endif

	if (i < *internal.nl[k][kk])
	{
		//sl = (T[i] / internal.r[i] - internal.b * 0.0 ) * lambda; //TODO:f(rm) or 0.0 or polynomic expansion?
		sl = (T[i] / internal.r[i] - internal.b * f(k,kk,*internal.rm[k][kk])) * lambda; //TODO:f(rm) or 0.0 or polynomic expansion?
	}
	else
	{
		sl = (T[i] / internal.r[i] - internal.b * internal.fi1[k][kk][i]) * lambda;
	}

#ifdef DEBUG3
#ifdef PARALLEL
	funlockfile(file);
#endif
#endif

	return - sl + log(1.0 + (exp(fsw[i] * sl) - 1.0) / fsw[i]);
}

int ms(double *T, double *C, double *fsw, int k, int kk, FILE *file) //TODO: rewrite universal closure with variable Bridge if possible for linear and non-linear closures
{
#ifdef DEBUG3
#ifdef PARALLEL
        flockfile(file);
#endif
#endif
        int i;
        for (i = 0; i < *internal.nl[k][kk]; i++)
        {
                C[i] = - internal.r[i] - T[i];
        #ifdef DEBUG3
                fprintf(file,"\ni:%i C:"my_e" T:"my_e" r:"my_e"",i,C[i],T[i],internal.r[i]);
        #endif
        }
        for (i = *internal.nl[k][kk]; i < external.nh; i++)
        {
                C[i] = - internal.r[i] - T[i]
             + internal.r[i] * exp(- internal.b * (internal.fi0[k][kk][i] + internal.fi1[k][kk][i]) * pow(1.0 + 2.0 * T[i] / internal.r[i],0.5) - 1.0);

        #ifdef DEBUG3
//              fprintf(file,);
        #endif
        }
#ifdef DEBUG3
#ifdef PARALLEL
        funlockfile(file);
#endif
#endif
        return 0;
}

double Bms(double *T, double *C, double *fsw, int k, int kk, int i, double lambda, FILE *file)
{
#ifdef DEBUG3
#ifdef PARALLEL
        flockfile(file);
#endif
#endif
        //s=t-bfi1
        //B=-s*lmb+ln(1+(exp(f*s*lmb)-1)/f)

        double sl;

        #ifdef DEBUG3
                fprintf(file,"\nk %i kk %i i %i T[i] %p r[i] %p fi1[k][kk][i] %p fsw[i] %p",k,kk,i,&T[i],&internal.r[i],&internal.fi1[k][kk][i],&fsw[i]);
        #endif

        sl = T[i] / internal.r[i] * lambda; //TODO:f(rm) or 0.0 or polynomic expansion?

#ifdef DEBUG3
#ifdef PARALLEL
        funlockfile(file);
#endif
#endif

        return pow(1.0 + 2.0 * sl, 0.5) - sl - 1.0;
}

/* ===== Soft Spheres ===== */

double ssf(int i, int j, double r)//potential for soft spheres
{
	//e=external.p[0]	sgm=external.p[1]	c=external.p[2]	n=external.p[3]

	// c*e*(sgm/r)^n

	return external.p[i][j][2]*external.p[i][j][0]*pow((external.p[i][j][1]/r),external.p[i][j][3]);
}

double ssdf(int i, int j, double r)
{
	// c*e*(-n)*(sgm^n)*(r^(-n-1))

	return external.p[i][j][2]*external.p[i][j][0]*(-external.p[i][j][3])*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-1.0);
}

double ssddf(int i, int j, double r)
{
	// c*e*n*(n+1)*(sgm^n)*(r^(-n-2))

	return external.p[i][j][2]*external.p[i][j][0]*external.p[i][j][3]*(external.p[i][j][3]+1.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-2.0);
}

double ssd3f(int i, int j, double r)
{
	// c*e*n*(n+1)*(-n-2)*(sgm^n)*(r^(-n-3))

	return external.p[i][j][2]*external.p[i][j][0]*external.p[i][j][3]*(external.p[i][j][3]+1.0)*(-external.p[i][j][3]-2.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-3.0);
}

double ssd4f(int i, int j, double r)
{
	// c*e*n*(n+1)*(n+2)*(n+3)*(sgm^n)*(r^(-n-4))

	return external.p[i][j][2]*external.p[i][j][0]*external.p[i][j][3]*(external.p[i][j][3]+1.0)*(external.p[i][j][3]+2.0)*(external.p[i][j][3]+3.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-4.0);
}

double ssd5f(int i, int j, double r)
{
	// c*e*n*(n+1)*(n+2)*(n+3)*(-n-4)*(sgm^n)*(r^(-n-5))

	return external.p[i][j][2]*external.p[i][j][0]*external.p[i][j][3]*(external.p[i][j][3]+1.0)*(external.p[i][j][3]+2.0)*(external.p[i][j][3]+3.0)*(-external.p[i][j][3]-4.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-5.0);
}

double ssI1(int i, int j, double r)
{
	// c*e*(sgm^n)*(1/(n-3))*r^(-n+3)

	return external.p[i][j][2]*external.p[i][j][0]*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]+3.0)/(external.p[i][j][3]-3.0);
}

double ssI2(int i, int j, double r)
{
	// c*e*(sgm^n)*(1/(3-n))*r^(-n+3)*n

	return external.p[i][j][2]*external.p[i][j][0]*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]+3.0)/(3.0-external.p[i][j][3])*external.p[i][j][3];
}

int softspheres(int *flag/*flag<>0 is condition of fast exit*/, int i, int j, FILE *file)
{

	*internal.Rlocut[i][j] = 0.0;//Rlocut
	*internal.rm[i][j]=external.p[i][j][1];//this function have no minimum. sgm is a radius scale

	_f[i][j] = ssf;
	_df[i][j] = ssdf;
	_ddf[i][j] = ssddf;
	_d3f[i][j] = ssd3f;
	_d4f[i][j] = ssd4f;
	_d5f[i][j] = ssd5f;
	_I1[i][j] = ssI1;
	_I2[i][j] = ssI2;

	_f[j][i] = ssf;
	_df[j][i] = ssdf;
	_ddf[j][i] = ssddf;
	_d3f[j][i] = ssd3f;
	_d4f[j][i] = ssd4f;
	_d5f[j][i] = ssd5f;
	_I1[j][i] = ssI1;
	_I2[j][i] = ssI2;

	fprintf(file,"soft spheres potential\n pair=%i %i\n e=%e\n sgm=%e\n c=%e\n n=%e\n\n",i,j,external.p[i][j][0],external.p[i][j][1],external.p[i][j][2],external.p[i][j][3]);

	return 0;
}


/* ===== LJ ===== */

double ljf(int i, int j, double r)//Lennard Jones potential
{
	//e=external.p[0]	sgm=external.p[1]	c=external.p[2]	n=external.p[3]	m=external.p[4]

	// c*e*(((sgm/r)^n)-((sgm/r)^m))

	return external.p[i][j][2]*external.p[i][j][0]*(pow((external.p[i][j][1]/r),external.p[i][j][3])-pow((external.p[i][j][1]/r),external.p[i][j][4]));
}

double ljdf(int i, int j, double r)
{
	// c*e*((-n)*(sgm^n)*(r^(-n-1))-(-m)*(sgm^m)*(r^(-m-1)))

	return external.p[i][j][2]*external.p[i][j][0]*((-external.p[i][j][3])*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-1.0)+external.p[i][j][4]*pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]-1.0));
}

double ljddf(int i, int j, double r)//d2f/dr2 for Newton algorithm
{
	//c*e*[n*(n+1)*sgm^n*r^(-n-2)-m*(m+1)*sgm^m*r^(-m-2)]

	return external.p[i][j][2]*external.p[i][j][0]*(external.p[i][j][3]*(external.p[i][j][3]+1.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-2.0)-external.p[i][j][4]*(external.p[i][j][4]+1.0)*pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]-2.0));
}

double ljd3f(int i, int j, double r)
{
	// c*e*(n*(n+1)*(-n-2)*(sgm^n)*(r^(-n-3))-m*(m+1)*(-m-2)*(sgm^m)*(r^(-m-3)))

	return external.p[i][j][2]*external.p[i][j][0]*(external.p[i][j][3]*(external.p[i][j][3]+1.0)*(-external.p[i][j][3]-2.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-3.0)-external.p[i][j][4]*(external.p[i][j][4]+1.0)*(-external.p[i][j][4]-2.0)*pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]-3.0));
}

double ljd4f(int i, int j, double r)
{
	// c*e*(n*(n+1)*(n+2)*(n+3)*(sgm^n)*(r^(-n-4))-m*(m+1)*(m+2)*(m+3)*(sgm^m)*(r^(-m-4)))

	return external.p[i][j][2]*external.p[i][j][0]*(external.p[i][j][3]*(external.p[i][j][3]+1.0)*(external.p[i][j][3]+2.0)*(external.p[i][j][3]+3.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-4.0)-external.p[i][j][4]*(external.p[i][j][4]+1.0)*(external.p[i][j][4]+2.0)*(external.p[i][j][4]+3.0)*pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]-4.0));
}

double ljd5f(int i, int j, double r)
{
	// c*e*(n*(n+1)*(n+2)*(n+3)*(-n-4)*(sgm^n)*(r^(-n-5))-m*(m+1)*(m+2)*(m+3)*(-m-4)*(sgm^m)*(r^(-m-5)))

	return external.p[i][j][2]*external.p[i][j][0]*(external.p[i][j][3]*(external.p[i][j][3]+1.0)*(external.p[i][j][3]+2.0)*(external.p[i][j][3]+3.0)*(-external.p[i][j][3]-4.0)*pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]-5.0)-external.p[i][j][4]*(external.p[i][j][4]+1.0)*(external.p[i][j][4]+2.0)*(external.p[i][j][4]+3.0)*(-external.p[i][j][4]-4.0)*pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]-5.0));
}

double ljI1(int i, int j, double r)
{
	// c*e*((sgm^n)*(1/(n-3))*r^(-n+3)-(sgm^m)*(1/(m-3))*r^(-m+3))

	return external.p[i][j][2]*external.p[i][j][0]*(pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]+3.0)/(external.p[i][j][3]-3.0)-pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]+3)/(external.p[i][j][4]-3.0));
}

double ljI2(int i, int j, double r)
{
	// c*e*((sgm^n)*(1/(3-n))*r^(-n+3)*n-(sgm^m)*(1/(3-m))*r^(-m+3)*m)

	return external.p[i][j][2]*external.p[i][j][0]*(pow(external.p[i][j][1],external.p[i][j][3])*pow(r,-external.p[i][j][3]+3.0)/(3.0-external.p[i][j][3])*external.p[i][j][3]-pow(external.p[i][j][1],external.p[i][j][4])*pow(r,-external.p[i][j][4]+3.0)/(3.0-external.p[i][j][4])*external.p[i][j][4]);
}

int lj(int *flag/*flag<>0 is condition of fast exit*/, int i, int j, FILE *file)
{

	_f[i][j] = ljf;
	_df[i][j] = ljdf;
	_ddf[i][j] = ljddf;
	_d3f[i][j] = ljd3f;
	_d4f[i][j] = ljd4f;
	_d5f[i][j] = ljd5f;
	_I1[i][j] = ljI1;
	_I2[i][j] = ljI2;

	_f[j][i] = ljf;
	_df[j][i] = ljdf;
	_ddf[j][i] = ljddf;
	_d3f[j][i] = ljd3f;
	_d4f[j][i] = ljd4f;
	_d5f[j][i] = ljd5f;
	_I1[j][i] = ljI1;
	_I2[j][i] = ljI2;

	newtonlogmul(i,j);
	*internal.rm[i][j] = *internal.Rlocut[i][j];
	*internal.Rlocut[i][j] = 0.0;

	fprintf(file,"LJ potential\n pair=%i %i\n e=%e\n sgm=%e\n c=%e\n n=%e\n m=%e\n\n",i,j,external.p[i][j][0],external.p[i][j][1],external.p[i][j][2],external.p[i][j][3],external.p[i][j][4]);

	return 0;
}


/* ===== Morse ===== */

double morsef(int i, int j, double r)//Morse potential
{
	//e=external.p[0]	sgm=external.p[1]	a=external.p[2]

	//e*(exp(-2a(r-sgm))-2exp(-a(r-sgm)))

	return external.p[i][j][0]*(exp(-2.0*external.p[i][j][2]*(r-external.p[i][j][1]))-2.0*exp(-external.p[i][j][2]*(r-external.p[i][j][1])));
}

double morsedf(int i, int j, double r)
{
	//e*(2a*exp(-a(r-sgm))-2a*exp(-2a(r-sgm)))

	return external.p[i][j][0]*(2.0*external.p[i][j][2]*exp(-external.p[i][j][2]*(r-external.p[i][j][1]))-2.0*external.p[i][j][2]*exp(-2.0*external.p[i][j][2]*(r-external.p[i][j][1])));
}

double morseddf(int i, int j, double r)//d2f/dr2 for Newton algorithm
{
	//e*(4a^2*exp(-2a(r-sgm))-2a^2*exp(-a(r-sgm)))

	return external.p[i][j][0]*(4.0*pow(external.p[i][j][2],2.0)*exp(-2.0*external.p[i][j][2]*(r-external.p[i][j][1]))-2.0*pow(external.p[i][j][2],2.0)*exp(-external.p[i][j][2]*(r-external.p[i][j][1])));
}

double morsed3f(int i, int j, double r)
{
	//e*(2a^3*exp(-a(r-sgm))-8a^3exp(-2a(r-sgm)))

	return external.p[i][j][0]*(2.0*pow(external.p[i][j][2],3.0)*exp(-external.p[i][j][2]*(r-external.p[i][j][1]))-8.0*pow(external.p[i][j][2],3.0)*exp(-2.0*external.p[i][j][2]*(r-external.p[i][j][1])));
}

double morsed4f(int i, int j, double r)
{
	//e*(16a^4*exp(-2a(r-sgm))-2a^4*exp(-a(r-sgm)))

	return external.p[i][j][0]*(16.0*pow(external.p[i][j][2],4.0)*exp(-2.0*external.p[i][j][2]*(r-external.p[i][j][1]))-2.0*pow(external.p[i][j][2],4.0)*exp(-external.p[i][j][2]*(r-external.p[i][j][1])));
}

double morsed5f(int i, int j, double r)
{
	//e*(2a^5*exp(-a(r-sgm))-32a^5*exp(-2a(r-sgm)))

	return external.p[i][j][0]*(2.0*pow(external.p[i][j][2],5.0)*exp(-external.p[i][j][2]*(r-external.p[i][j][1]))-32.0*pow(external.p[i][j][2],5.0)*exp(-2.0*external.p[i][j][2]*(r-external.p[i][j][1])));
}

double morseI1(int i, int j, double r)
{
	//

	return external.p[i][j][0]*exp(-2.0*external.p[i][j][2]*r)*(((-8.0*pow(external.p[i][j][2],2.0)*pow(r,2.0))-16.0*external.p[i][j][2]*r-16.0)*exp(external.p[i][j][2]*external.p[i][j][1]+external.p[i][j][2]*r)+(2.0*pow(external.p[i][j][2],2.0)*pow(r,2.0)+2.0*external.p[i][j][2]*r+1.0)*exp(2.0*external.p[i][j][2]*external.p[i][j][1]))/4.0/pow(external.p[i][j][2],3.0);
}

double morseI2(int i, int j, double r)
{
	//

	return -external.p[i][j][0]*exp(-2.0*external.p[i][j][2]*r)*(((-8.0*pow(external.p[i][j][2],3.0)*pow(r,3.0))-24.0*pow(external.p[i][j][2],2.0)*pow(r,2.0)-48.0*external.p[i][j][2]*r-48.0)*exp(external.p[i][j][2]*external.p[i][j][1]+external.p[i][j][2]*r)+(4.0*pow(external.p[i][j][2],3.0)*pow(r,3.0)+6.0*pow(external.p[i][j][2],2.0)*pow(r,2.0)+6.0*external.p[i][j][2]*r+3.0)*exp(2.0*external.p[i][j][2]*external.p[i][j][1]))/4.0/pow(external.p[i][j][2],3.0);
}

int morse(int *flag/*flag<>0 is condition of fast exit*/, int i, int j, FILE *file)
{
	*internal.rm[i][j] = external.p[i][j][1];//analitic minimum point
	*internal.Rlocut[i][j] = 0.0;

	_f[i][j] = morsef;
	_df[i][j] = morsedf;
	_ddf[i][j] = morseddf;
	_d3f[i][j] = morsed3f;
	_d4f[i][j] = morsed4f;
	_d5f[i][j] = morsed5f;
	_I1[i][j] = morseI1;
	_I2[i][j] = morseI2;

	_f[j][i] = morsef;
	_df[j][i] = morsedf;
	_ddf[j][i] = morseddf;
	_d3f[j][i] = morsed3f;
	_d4f[j][i] = morsed4f;
	_d5f[j][i] = morsed5f;
	_I1[j][i] = morseI1;
	_I2[j][i] = morseI2;

	fprintf(file,"Morse potential\n pair=%i %i\n e=%e\n sgm=%e\n a=%e\n\n",i,j,external.p[i][j][0],external.p[i][j][1],external.p[i][j][2]);

	return 0;
}


/* ===== Exp6 ====== */

double e6f(int i, int j, double r)//EXP-6 form of Buckingham potential
{
	//e=external.p[0]	rm=external.p[1]	a=external.p[2]

	// e/(a-6)*(6*exp(a*(1-r/rm))-a*(rm/r)^6)

	return external.p[i][j][0]/(external.p[i][j][2]-6.0)*(6.0*exp(external.p[i][j][2]*(1.0-(r/external.p[i][j][1])))-external.p[i][j][2]*pow((external.p[i][j][1]/r),6.0));
}

double e6df(int i, int j, double r)
{
	// e*a/(a-6)*6(*exp(a*(1-r/rm)*(-1/rm)+rm^6/r^7)

	return external.p[i][j][0]/(external.p[i][j][2]-6.0)*external.p[i][j][2]*6.0*(exp(external.p[i][j][2]*(1.0-r/external.p[i][j][1]))*(-1.0/external.p[i][j][1])+pow(external.p[i][j][1],6.0)*pow(r,-7.0));
}

double e6ddf(int i, int j, double r)//d2f/dr2 for Newton algorithm
{
	// e/(a-6)*a*6*(a/rm/rm*exp(a*(1-r/rm))-rm^6*7/r^8)

	return external.p[i][j][0]/(external.p[i][j][2]-6.0)*external.p[i][j][2]*6.0*(external.p[i][j][2]/external.p[i][j][1]/external.p[i][j][1]*exp(external.p[i][j][2]*(1.0-r/external.p[i][j][1]))-7.0*pow(external.p[i][j][1],6.0)*pow(r,-8.0));
}

double e6d3f(int i, int j, double r)
{
	// e/(a-6)*a*6(a*a/rm/rm/rm*exp(a*(1-r/rm)-rm^6*56/r^9)

	return external.p[i][j][0]/(external.p[i][j][2]-6.0)*external.p[i][j][2]*6.0*(external.p[i][j][2]*external.p[i][j][2]/external.p[i][j][1]/external.p[i][j][1]/external.p[i][j][1]*exp(external.p[i][j][2]*(1.0-r/external.p[i][j][1]))-56.0*pow(external.p[i][j][1],6.0)*pow(r,-9.0));
}

double e6d4f(int i, int j, double r)
{
	// e/(a-6)*a*6(a*a*a/rm/rm/rm/rm*exp(a*(1-r/rm)-rm^6*504/r^10)

	return external.p[i][j][0]/(external.p[i][j][2]-6.0)*external.p[i][j][2]*6.0*(external.p[i][j][2]*external.p[i][j][2]*external.p[i][j][2]/external.p[i][j][1]/external.p[i][j][1]/external.p[i][j][1]/external.p[i][j][1]*exp(external.p[i][j][2]*(1.0-r/external.p[i][j][1]))-504.0*pow(external.p[i][j][1],6.0)*pow(r,-10.0));
}

double e6d5f(int i, int j, double r)
{
	// e/(a-6)*a*6(a*a*a*a/rm/rm/rm/rm/rm*exp(a*(1-r/rm)-rm^6*5040/r^11)

	return external.p[i][j][0]/(external.p[i][j][2]-6.0)*external.p[i][j][2]*6.0*(external.p[i][j][2]*external.p[i][j][2]*external.p[i][j][2]*external.p[i][j][2]/external.p[i][j][1]/external.p[i][j][1]/external.p[i][j][1]/external.p[i][j][1]/external.p[i][j][1]*exp(external.p[i][j][2]*(1.0-r/external.p[i][j][1]))-5040.0*pow(external.p[i][j][1],6.0)*pow(r,-11.0));
}

double e6I1(int i, int j, double r)
{
	//

	return -(external.p[i][j][0]/3.0)*external.p[i][j][1]*(-18.0*exp(external.p[i][j][2])*pow(external.p[i][j][2],2.0)*pow(r,5.0)-36.0*external.p[i][j][2]*exp(external.p[i][j][2])*external.p[i][j][1]*pow(r,4.0)-36.0*exp(external.p[i][j][2])*pow(external.p[i][j][1],2.0)*pow(r,3.0)+pow(external.p[i][j][2],4.0)*pow(external.p[i][j][1],5.0)*exp(r*(external.p[i][j][2]/external.p[i][j][1])))*exp(-r*(external.p[i][j][2]/external.p[i][j][1]))/(pow(external.p[i][j][2],3.0)*(external.p[i][j][2]-6.0)*pow(r,3.0));
}

double e6I2(int i, int j, double r)
{
	// 

	return 2.0*external.p[i][j][0]*(pow(external.p[i][j][1],6.0)*pow(external.p[i][j][2],4.0)*exp(r*(external.p[i][j][2]/external.p[i][j][1]))-3.0*exp(external.p[i][j][2])*pow(r,6.0)*pow(external.p[i][j][2],3.0)-9.0*exp(external.p[i][j][2])*pow(r,5.0)*pow(external.p[i][j][2],2.0)*external.p[i][j][1]-18.0*exp(external.p[i][j][2])*pow(r,4.0)*external.p[i][j][2]*pow(external.p[i][j][1],2.0)-18.0*exp(external.p[i][j][2])*pow(r,3.0)*pow(external.p[i][j][1],3.0))*exp(-r*(external.p[i][j][2]/external.p[i][j][1]))/(pow(external.p[i][j][2],3.0)*(external.p[i][j][2]-6.0)*pow(r,3.0));;
}

int e6(int *flag/*flag<>0 is condition of fast exit*/, int i, int j, FILE *file)
{
	*internal.rm[i][j] = external.p[i][j][1];//analitic minimum point

	_f[i][j] = e6f;
	_df[i][j] = e6df;
	_ddf[i][j] = e6ddf;
	_d3f[i][j] = e6d3f;
	_d4f[i][j] = e6d4f;
	_d5f[i][j] = e6d5f;
	_I1[i][j] = e6I1;
	_I2[i][j] = e6I2;

	_f[j][i] = e6f;
	_df[j][i] = e6df;
	_ddf[j][i] = e6ddf;
	_d3f[j][i] = e6d3f;
	_d4f[j][i] = e6d4f;
	_d5f[j][i] = e6d5f;
	_I1[j][i] = e6I1;
	_I2[j][i] = e6I2;

	newtonlogmul(i,j);

	fprintf(file,"Exp-6 potential\n pair=%i %i\n e=%e\n rm=%e\n a=%e\n c*=%e\n\n",i,j,external.p[i][j][0],external.p[i][j][1],external.p[i][j][2],*internal.Rlocut[i][j]);

	return 0;
}

double janaf_cp0(int i)
{
	double t, cp0;
	t = inter_tdc.e_scale / internal.b / KILO;
	cp0 = exter_tdc.cal_parms[i][0]+exter_tdc.cal_parms[i][1]*t+exter_tdc.cal_parms[i][2]*t*t+exter_tdc.cal_parms[i][3]*pow(t,3.0)+exter_tdc.cal_parms[i][4]/t/t;
	return cp0 / kb / Na; //Dimension reduction
}

double janaf_h0(int i)
{
	double t, h0;
	t = inter_tdc.e_scale / internal.b / KILO;
	h0 = exter_tdc.cal_parms[i][0]*t+exter_tdc.cal_parms[i][1]*t*t/2.0+exter_tdc.cal_parms[i][2]*pow(t,3.0)/3.0+exter_tdc.cal_parms[i][3]*pow(t,4.0)/4.0-exter_tdc.cal_parms[i][4]/t+exter_tdc.cal_parms[i][5]-exter_tdc.cal_parms[i][7];
	return h0 / kb / Na / t; //Dimension reduction
}

double janaf_s0(int i)
{
	double t, s0;
	t = inter_tdc.e_scale / internal.b / KILO;
	s0 = exter_tdc.cal_parms[i][0]*log(t)+exter_tdc.cal_parms[i][1]*t+exter_tdc.cal_parms[i][2]*pow(t,2.0)/2.0+exter_tdc.cal_parms[i][3]*pow(t,3.0)/3.0-exter_tdc.cal_parms[i][4]/2.0/t/t+exter_tdc.cal_parms[i][6];
	return s0 / kb / Na; //Dimension reduction
}

double ivtan_cp0(int i)
{
	double t, cp0;
	t = inter_tdc.e_scale / internal.b / TENKILO;
	cp0 = exter_tdc.cal_parms[i][3]+2.0*exter_tdc.cal_parms[i][4]/t/t+2.0*exter_tdc.cal_parms[i][6]*t+6.0*exter_tdc.cal_parms[i][7]*t*t+12.0*exter_tdc.cal_parms[i][8]*pow(t,3.0);
	return cp0 / kb / Na;
}

double ivtan_h0(int i)
{
	double t, h0;
	t = inter_tdc.e_scale / internal.b / TENKILO;
	h0 = exter_tdc.cal_parms[i][0]*KILO-exter_tdc.cal_parms[i][1]*KILO+TENKILO*(exter_tdc.cal_parms[i][3]*t-2.0*exter_tdc.cal_parms[i][4]/t-exter_tdc.cal_parms[i][5]+exter_tdc.cal_parms[i][6]*t*t+2.0*exter_tdc.cal_parms[i][7]*pow(t,3.0)+3.0*exter_tdc.cal_parms[i][8]*pow(t,4.0));
	return h0 / kb / Na / inter_tdc.e_scale * internal.b;;
}

double ivtan_s0(int i)
{
	double t, s0;
	t = inter_tdc.e_scale / internal.b / TENKILO;
	s0 = exter_tdc.cal_parms[i][2]+exter_tdc.cal_parms[i][3]+exter_tdc.cal_parms[i][3]*log(t)-exter_tdc.cal_parms[i][4]/t/t+2.0*exter_tdc.cal_parms[i][6]*t+3.0*exter_tdc.cal_parms[i][7]*t*t+4.0*exter_tdc.cal_parms[i][8]*pow(t,3.0);
	return s0 / kb / Na;
}

double mole_w(double *mole_fr)
{
	int i,j;
	double m_w, m_w_f;
	m_w = 0.0;
	for (i = 0; i < external.n_molec_type; i++)
	{
		m_w_f = 0.0;
		for (j = 0; j < exter_tdc.n_atom_type; j++)
		{
			m_w_f += exter_tdc.atom_weights[j] * (double)exter_tdc.mole_compos[i][j];
		}
		m_w += mole_fr[i] *  m_w_f;
	}
	return m_w;
}

/* ===== Main Code ===== */

int getpotential(int *flag, int i, int j, FILE *file)
	/*get input parameters for potential*/
{
	if (*flag == 0)
	{
		switch (*external.poten_type[i][j])
		{
			case 1:
				softspheres(flag,i,j,file);
				break;

			case 2:
				lj(flag,i,j,file);
				break;

			case 3:
				e6(flag,i,j,file);
				break;

			case 4:
				morse(flag,i,j,file);
				break;

			default:
				*flag = 1;
				break;
		}
		switch (*external.cl_type[i][j])
		{
			case 1:
				_closure[i][j] = hmsa;
				_bridge[i][j] = Bhmsa;
				break;

			case 2:
				_closure[i][j] = ms;
				_bridge[i][j] = Bms;
				break;

			default:
				*flag = 1;
				break;
		}
		switch (*external.sw_type[i][j])
		{
			case 1:
				_swfu[i][j] = sw_exp;
				break;
			default:
				*flag = 1;
				break;
		}
	}
	return 0;
}

int calcKUZ(double *ans, double ***T, double ***C, double ***fsw, double *x, double *dbpdrhoj, double *bp, int jjj, int j, FILE *file)
{
	int i,k,kk;
	double  gr2mul, r_cutoff, r2_cutoff, dou, sum, tmp, Z, Uast, G, Ztmp, Ztmp2, Utmp, Utmp2, P, U, mutmp;

	dou = sqrt(external.delta*external.delta/internal.dr)/(double)external.nh;

	if (jjj == 0) // dbpdrhoj for j
	{
		sum = 0.0;
		for (kk = 0; kk < external.n_molec_type; kk++)
		{
			tmp = 0.0;
			for (i = 0; i < external.nh; i++)
			{
				tmp += internal.rmul[i] * C[j][kk][i];
			}
			sum += tmp * x[kk];
		}
		*dbpdrhoj = 1.0 - 4.0 * pi * external.rho * sum * internal.dr / 3.0;
	}
	else
	{
		if (jjj == external.Nrho + 1) // U, Z, dbpdrhoj for all
		{
			for (k = 0; k < external.n_molec_type; k++)
			{
				sum = 0.0;
				for (kk = 0; kk < external.n_molec_type; kk++)
				{
					tmp = 0.0;
					for (i = 0; i < external.nh; i++)
					{
						tmp += internal.rmul[i] * C[k][kk][i];
					}
					sum += tmp * x[kk];
				}
				dbpdrhoj[k] = 1.0 - 4.0 * pi * external.rho * sum * internal.dr / 3.0;
			}
		#ifdef CALC_RDF
			flockfile(file);
		#endif
			Uast = 0.0;
			Z = 0.0;
			tmp = 0.0;
			r_cutoff = internal.r[external.nh - 1] + internal.dr;
			r2_cutoff = r_cutoff * r_cutoff;
			for (k = 0; k < external.n_molec_type; k++)
			{
				Utmp2 = 0.0;
				Ztmp2 = 0.0;
				inter_tdc.mu_thermal_ex[k] = 0.0;
				for (kk = 0; kk < external.n_molec_type; kk++)
				{
					Utmp = 0.0;
					Ztmp = 0.0;
					mutmp = 0.0;
					for (i = 0; i < external.nh; i++)
					{
						G = internal.r[i] + C[k][kk][i] + T[k][kk][i];
					//#ifdef CALC_RDF
					//	fprintf(file,"\nk:%i kk:%i i:%i r:%e g:%e",k,kk,i,internal.r[i],G/internal.r[i]);
					//#endif
						if (fabs(G)<(dou)) // Noise Gate!
						{
							G = 0.0;
						}
					#ifdef CALC_RDF
						fprintf(file,"\nk:%i kk:%i i:%i r:%e g:%e",k,kk,i,internal.r[i],G/internal.r[i]);
					#endif
						gr2mul = G * internal.rmul[i];
						Utmp += f(k,kk,internal.r[i]) * gr2mul;
						Ztmp += df(k,kk,internal.r[i]) * internal.r[i] * gr2mul;
						// Bridge function integration
						{
							double B, Bint, lambda;
							int ii;
							B = bridge(T[k][kk],C[k][kk],fsw[k][kk],k,kk,i,1.0,file);
							Bint = 0.0;
							for (ii = 0; ii < external.nh + 1; ii++)//+1 is correct! No zero shifting!
							{
								lambda = (double)ii * internal.dlambda;
								Bint += internal.Simp0[ii] * bridge(T[k][kk],C[k][kk],fsw[k][kk],k,kk,i,lambda,file);
								//Bint += internal.Simp0[ii] * 1.0;
							}
							Bint = Bint * internal.dlambda / 3.0;
							if (fabs(B)<(dou)) // Noise Gate!
							{
								B = 0.0;
								Bint = 0.0;
							}
							fprintf(file," B:%e Bint:%e",B,Bint);
						
						//r*r*(B+(t+c)*(B-Bint)+(t*t+t*c)/2-c)
							mutmp += internal.rmul[i] * (internal.r[i] * B + (T[k][kk][i] + C[k][kk][i]) * (B - Bint) + T[k][kk][i] * (T[k][kk][i] + C[k][kk][i]) / 2.0 / internal.r[i] - C[k][kk][i]);
						}
					}
					Utmp2 += x[kk] * ((Utmp + f(k,kk,r_cutoff) * r2_cutoff) * internal.dr / 3.0 + I1(k,kk,r_cutoff));
					Ztmp2 += x[kk] * ((Ztmp + df(k,kk,r_cutoff) * r_cutoff * r2_cutoff) * internal.dr / 3.0 + I2(k,kk,r_cutoff));
					inter_tdc.mu_thermal_ex[k] += x[kk] * mutmp * internal.dr / 3.0;
				}
				Uast += x[k] * Utmp2;
				tmp += x[k] * Ztmp2;
				Z += x[k];
				inter_tdc.mu_thermal_ex[k] = inter_tdc.mu_thermal_ex[k] * 4.0 * pi; // * external.rho / internal.b;
				//inter_tdc.mu_thermal_ex[k] = inter_tdc.mu_thermal_ex[k] * kb * Na * internal.b / inter_tdc.e_scale;

				//===WARNING!===
				//mu_ex is WRONG!!! TODO: Fix Choundhury-Ghosh integration or change to classic Kirkwood charging formula
				//==============

			}
			Uast = Uast * 2.0 * pi * external.rho * internal.b;
			Z -= (2.0 / 3.0) * pi * external.rho * internal.b * tmp;
			P = Z * kb * inter_tdc.e_scale / internal.b * external.rho / pow(inter_tdc.r_scale / pow(10.0, 10.0), 3.0);
			U = Uast * kb * Na / internal.b * inter_tdc.e_scale;

			for (k = 0; k < external.n_molec_type; k++)
			{
				inter_tdc.mu_thermal_id[k] = log(external.mol_fr[k] * P / IUPAC1982_P);
				//inter_tdc.mu_thermal_ex[k] = 0.0;
				inter_tdc.mu_caloric[k] = h0(k) - s0(k);
				inter_tdc.mu[k] = inter_tdc.mu_thermal_id[k] + inter_tdc.mu_thermal_ex[k] + inter_tdc.mu_caloric[k];
				ans[k] = inter_tdc.mu[k];
			}

			ans[external.n_molec_type] = Z;
			ans[external.n_molec_type + 1] = Uast;
			//TODO H,S

		#ifndef CALC_RDF
			flockfile(file);
		#endif
			//fprintf(file,"\n T = %e [K] R = %e V = %e [cc/mol] mole_w %e [g/mol]", inter_tdc.e_scale / internal.b, kb * Na, 1.0 / external.rho * pow(inter_tdc.e_scale * pow (10.0, - 8.0), 3.0) * Na, mole_w(external.mol_fr));
			fprintf(file,"\n U* = %e\n Z = %e\n P = %e [Pa]\n U = %e [J/mol]",Uast,Z,P,U);
			for (k = 0; k < external.n_molec_type; k++)
			{
				fprintf(file,"\n fr %i\n mu_thermal_id[%i] = %e\n mu_thermal_ex[%i] = %e\n mu_caloric[%i] = %e\n mu[%i] = %e\nWARNING! mu_ex IS WRONG!",k,k,inter_tdc.mu_thermal_id[k],k,inter_tdc.mu_thermal_ex[k],k,inter_tdc.mu_caloric[k],k,inter_tdc.mu[k]);
			}
			funlockfile(file);
		}
		else // bp
		{
			*bp = 0.0;
			Z = 0.0;
			tmp = 0.0;
			r_cutoff = internal.r[external.nh - 1] + internal.dr;
			r2_cutoff = r_cutoff * r_cutoff;
			for (k = 0; k < external.n_molec_type; k++)
			{
				Ztmp2 = 0.0;
				for (kk = 0; kk < external.n_molec_type; kk++)
				{
					Ztmp = 0.0;
					for (i = 0; i < external.nh; i++)
					{
						G = internal.r[i] + C[k][kk][i] + T[k][kk][i];
						if (fabs(G)<(dou)) // Noise Gate!
						{
							G = 0.0;
						}
						gr2mul = G * internal.rmul[i];
						Ztmp += df(k,kk,internal.r[i]) * internal.r[i] * gr2mul;
					}
					Ztmp2 += x[kk] * ((Ztmp + df(k,kk,r_cutoff) * r_cutoff * r2_cutoff) * internal.dr / 3.0 + I2(k,kk,r_cutoff));
				}
				tmp += x[k] * Ztmp2;
				*bp += x[k];
			}
			*bp -= (2.0 / 3.0) * pi * external.rho * internal.b * tmp;
			*bp = *bp * external.rho;
		}
	}
	return 0;
}

int solveIUR(double ***T, double ***C, double ***fsw, double *x, int *status, int l, FILE *file)
{
	double ***TT, ***Ttmp, ***CC, **LU, **L, **U, **V, *v, *y, d, dd, mul_forward, mul_backward;
	double  yy;  // SB
	int i,ii,j,jj,jjj,k;

	createv(external.n_molec_type,&v);
	createv(external.n_molec_type,&y);
	createm(external.n_molec_type,external.n_molec_type,&LU);
	createm(external.n_molec_type,external.n_molec_type,&L);
	createm(external.n_molec_type,external.n_molec_type,&U);
	createm(external.n_molec_type,external.n_molec_type,&V);
	createm_tri(external.n_molec_type,external.nh,&TT);
	createm_tri(external.n_molec_type,external.nh,&Ttmp);
	createm_tri(external.n_molec_type,external.nh,&CC);

	// TT means T^
	d = external.delta + 1.0;
	ii = 0;
	for (j = 0; j < external.n_molec_type; j++)
	{
		for (jj = j; jj < external.n_molec_type; jj++)
		{
			closure(T[j][jj],C[j][jj],fsw[j][jj],j,jj,file);
			//fourier(external.nh,C[j][jj],CC[j][jj],(4.0 * pi * internal.dr),internal.si);
		}
	}

	mul_forward = 2.0 * pi * internal.dr;
	mul_backward = internal.dq / 4.0 / pi / pi;

	fftw_execute_r2r(internal.plan,C[0][0],CC[0][0]);
	cblas_dscal(external.nh * internal.pair_count,mul_forward,CC[0][0],1);//TODO:Fix for all precisions

#ifndef PARALLEL
#ifdef DEBUG3
	for (j = 0; j < external.n_molec_type; j++)
	{
		for (jj = j; jj < external.n_molec_type; jj++)
		{
			fprintf(file,"T[%i,%i]",j,jj);
			outv(external.nh,T[j][jj],file);
			fprintf(file,"C[%i,%i]",j,jj);
			outv(external.nh,C[j][jj],file);
			fprintf(file,"CC[%i,%i]",j,jj);
			outv(external.nh,CC[j][jj],file);
		}
	}
#endif
#endif

	while ((d > external.delta)&(ii < external.itermaxbig))
	{
		for (k = 0; k < BLIND_ITER; k++)
		{
			// TODO: Totally rewrite this piece of code to increase performance
			// BEGIN SLOWMO
			for (i = 0; i < external.nh; i++)
			{
				for (j = 0; j < external.n_molec_type; j++)
				{
					for (jj = 0; jj < external.n_molec_type; jj++)
					{
						V[j][jj] = external.rho * x[jj] * CC[jj][j][i];
						if (j == jj)
						{
							LU[j][jj] = - V[j][jj] + internal.q[i];
						#ifndef PARALLEL
						#ifdef DEBUG3
							fprintf(file,"\nLU[%i,%i] = %e",j,jj,LU[j][jj]);
						#endif
						#endif

						}
						else
						{
							LU[j][jj] = - V[j][jj];
						#ifndef PARALLEL
						#ifdef DEBUG3
							fprintf(file,"\nLU[%i,%i] = %e",j,jj,LU[j][jj]);
						#endif
						#endif

						}
					}
				}

				lu(external.n_molec_type,LU,L,U,file);

				for (jjj = 0; jjj < external.n_molec_type; jjj++)
				{
					for (j = 0; j < external.n_molec_type; j++)
					{
						v[j] = 0;
						for (jj = 0; jj < external.n_molec_type; jj++)
						{			
							v[j] += V[j][jj] * CC[jjj][jj][i];
						#ifndef PARALLEL
						#ifdef DEBUG3
							fprintf(file,"\nv[%i] = %e",j,v[j]);
						#endif
						#endif
						}
					}
					solvel(external.n_molec_type,L,y,v,file);
					solveu(external.n_molec_type,U,v,y,file);
					for (j = 0; j < external.n_molec_type; j++)
					{
						TT[jjj][j][i] = v[j];
					}
				}
			}
			// END SLOWMO
			d = 0;
			fftw_execute_r2r(internal.plan,TT[0][0],Ttmp[0][0]);
			cblas_dscal(external.nh * internal.pair_count,mul_backward,Ttmp[0][0],1);//TODO:Fix for all precisions

			for (j = 0; j < external.n_molec_type; j++)
			{
				for (jj = j; jj < external.n_molec_type; jj++)
				{
					//fourier(external.nh,TT[j][jj],Ttmp[j][jj],internal.dq / (2.0 * pi * pi),internal.si);
					closure(Ttmp[j][jj],C[j][jj],fsw[j][jj],j,jj,file);
					//fourier(external.nh,C[j][jj],CC[j][jj],(4.0 * pi * internal.dr),internal.si);
				}
			}
			fftw_execute_r2r(internal.plan,C[0][0],CC[0][0]);
			cblas_dscal(external.nh * internal.pair_count,mul_forward,CC[0][0],1);//TODO:Fix for all precisions

		#ifndef PARALLEL
		#ifdef DEBUG3
			for (j = 0; j < external.n_molec_type; j++)
			{
				for (jj = j; jj < external.n_molec_type; jj++)
				{
					fprintf(file,"TT[%i,%i]",j,jj);
					outv(external.nh,TT[j][jj],file);
					fprintf(file,"Ttmp[%i,%i]",j,jj);
					outv(external.nh,Ttmp[j][jj],file);
					fprintf(file,"C[%i,%i]",j,jj);
					outv(external.nh,C[j][jj],file);
					fprintf(file,"CC[%i,%i]",j,jj);
					outv(external.nh,CC[j][jj],file);
				}
			}
		#endif
		#endif
		}
		for (j = 0; j < external.n_molec_type; j++)
		{
			for (jj = j; jj < external.n_molec_type; jj++)
			{
				dd = 0;
				for (i = 0; i < external.nh; i++)
				{
					yy = Ttmp[j][jj][i] - T[j][jj][i];
					dd += yy * yy;
					T[j][jj][i]=Ttmp[j][jj][i];
					if (T[j][jj][i] != T[j][jj][i])
					{
					#ifndef PARALLEL
					#ifdef DEBUG
						printf("\nNaN!");
						*status = 1;
					#endif
					#endif
					}
				}
				dd = sqrt(internal.dr * dd);
				if (dd > d)
				{
					d = dd;
				}
			}
		}
	#ifndef PARALLEL
	#ifdef DEBUG
		printf("\nsolveIUR: iter:%i d:%e",ii,d);
	#endif
	#endif
		ii++;
	}

	if (ii > external.itermaxbig-1)
	{
	#ifdef PARELLEL
		flockfile(file);
	#endif
		fprintf(file,"\nERROR! Bad initial closure!\n");
	#ifdef PARELLEL
		funlockfile(file);
	#endif
		*status=1;
	}

	freem_tri(external.n_molec_type,CC);
	freem_tri(external.n_molec_type,Ttmp);
	freem_tri(external.n_molec_type,TT);
	freem(external.n_molec_type,V);
	freem(external.n_molec_type,U);
	freem(external.n_molec_type,L);
	freem(external.n_molec_type,LU);
	free(y);
	free(v);

	return 0;
}

int solver(double *ans, FILE *file, FILE *file1, FILE *file2)
{
	double tmp,tmp1,mul0,mul1,mul2,mul3,mul4,mul5,**x,***xx,*a,***Ttmp,***Tlast,**D,***DD,dKK,***fsw, ***T, ***C;
		/*if j>i+1 si[i][j] is not correct. use si[j][i]*/
	int i,ii,j,flagg,l,status;

	//mul temporary double
	//Te means temperature
	//b means reversed temperature
	//rho means density
	//a means swiching parameter
	//delta means accuracy parameter
	//fsw means swiching function
	//si means matrix of sin for fourier transformation
	//Nrho means number of items in density series

	status = 0;

	//internal.b = 1.0 / (external.Te * kb); //dimension rduction: kb vanished //moved to general
	//internal.b = 1.0 / external.Te;

	//fprintf(file,"\nb=%le\n",external.Te);
	fprintf(file,"\nb=%le\n",internal.b);

	fprintf(file,"rho0=%le\n",external.rho);
	fprintf(file,"drho=%le\n",external.drho);

	internal.dq = pi / ((double)external.nh + 1.0) / internal.dr;//nh+1 is correct. take a look in the theory. they use N-1 for nh nodes

	createv(external.nh,&internal.r);
	createv(external.nh,&internal.q);
	createm(external.n_molec_type,external.Nrho + 1,&x);
	createm_tri(external.n_molec_type,external.nh,&internal.fi0);
	createm_tri(external.n_molec_type,external.nh,&internal.fi1);
	createm_tri(external.n_molec_type,external.nh,&Tlast);
	createm_tri(external.n_molec_type,external.nh,&Ttmp);
	fftw_r2r_kind kind;
	kind = FFTW_RODFT00; 
	internal.plan = fftw_plan_many_r2r(1,&external.nh,internal.pair_count,Tlast[0][0],NULL,1,external.nh,Ttmp[0][0],NULL,1,external.nh,&kind,FFTW_PATIENT);//FFTW PATIENT of FFTW_ESTIMATE
#ifndef PARALLEL
	createm_tri(external.n_molec_type,external.nh,&T);
	createm_tri(external.n_molec_type,external.nh,&C);
	createm_tri(external.n_molec_type,external.nh,&fsw);
#endif
	createm3(external.n_molec_type,external.Nrho + 1,external.n_molec_type,&xx);
	createm(external.n_molec_type,external.Na + 1,&D);
	createm3(external.n_molec_type,external.Na + 1,external.Nrho + 1,&DD);
	//createmsin(external.nh,&internal.si);

	//Biggest Iteration Start//
	flagg = 1;

	for (i = 0; i < external.nh; i++)
	{
		internal.r[i] = ((double)i + 1.0) * internal.dr;
		internal.q[i] = ((double)i + 1.0) * internal.dq;
	}

#ifdef DEBUG3
	outv(external.nh,internal.r,file);
#endif

	createv(external.nh + 1,&internal.Simp0);//No zero point shidting
	getsimp0(external.nh + 1,internal.Simp0);
	createv(external.nh,&internal.Simp);//Zero point shifting
	getsimp(external.nh,internal.Simp);
	createv(external.nh,&internal.rmul);
	vmul(external.nh,internal.Simp,internal.r,internal.rmul);

	if (external.Nrho == 4)
	{
		for (i = 0; i < external.n_molec_type; i++)
		{
			x[i][0]=external.mol_fr[i];
			x[i][1]=external.mol_fr[i]*(1.0-3.0/2.0*external.drho);
			x[i][2]=external.mol_fr[i]*(1.0-external.drho/2.0);
			x[i][3]=external.mol_fr[i]*(1.0+external.drho/2.0);
			x[i][4]=external.mol_fr[i]*(1.0+3.0/2.0*external.drho);
		}

	}
	else
	{
		//Nrho=2;
		for (i = 0; i < external.n_molec_type; i++)
		{
			x[i][0]=external.mol_fr[i];
			x[i][1]=external.mol_fr[i]*(1.0-external.drho);
			x[i][2]=external.mol_fr[i]*(1.0+external.drho);
		}
	}

	for (i = 0; i < external.n_molec_type; i++)
	{
		for (j = 0; j <= external.Nrho; j++)
		{
			for (ii = 0; ii < external.n_molec_type; ii++)
			{
				if (ii == i)
				{
					xx[i][j][ii] = x[ii][j];
				}
				else
				{
					xx[i][j][ii] = x[ii][0];
				}
			}
		}
	}

	if (external.inp__type == 3) //TODO: FIX DUMP READING INP _TYPES
	{
		for (j = 0; j < external.n_molec_type; j++)
		{
			for (ii = j; ii < external.n_molec_type; ii++)
			{
				for (i = 0; i < external.nh; i++)
				{
					Tlast[j][ii][i] = 0.0;//place for normal initial estimation =0
				}
			}
		}
	}


	for (j = 0; j < external.n_molec_type; j++)
	{
		for (ii = j; ii < external.n_molec_type; ii++)
		{
			mul0 = f(j,ii,*internal.rm[j][ii]);
			mul1 = df(j,ii,*internal.rm[j][ii]);
			mul2 = ddf(j,ii,*internal.rm[j][ii]) / 2.0;
			mul3 = d3f(j,ii,*internal.rm[j][ii]) / 6.0;
			mul4 = d4f(j,ii,*internal.rm[j][ii]) / 24.0;
			mul5 = d5f(j,ii,*internal.rm[j][ii]) / 120.0;

			switch (*external.Nphi[j][ii])
			{
				case 0:
					for (i = *internal.nl[j][ii]; i < *internal.nrm[j][ii] - 1; i++)
					{
						tmp = internal.r[i] - *internal.rm[j][ii];
						tmp1 =  mul0;
						internal.fi0[j][ii][i] = f(j,ii,internal.r[i]) - tmp1;
						internal.fi1[j][ii][i] = tmp1;
					}
					break;

				case 1:
					for (i = *internal.nl[j][ii]; i < *internal.nrm[j][ii] - 1; i++)
					{
						tmp = internal.r[i] - *internal.rm[j][ii];
						tmp1 =  mul0 + mul1 * tmp;
						internal.fi0[j][ii][i] = f(j,ii,internal.r[i]) - tmp1;
						internal.fi1[j][ii][i] = tmp1;
					}
					break;

				case 2:
					for (i = *internal.nl[j][ii]; i < *internal.nrm[j][ii] - 1; i++)
					{
						tmp = internal.r[i] - *internal.rm[j][ii];
						tmp1 =  mul0 + mul1 * tmp * tmp + mul2 * tmp * tmp;
						internal.fi0[j][ii][i] = f(j,ii,internal.r[i]) - tmp1;
						internal.fi1[j][ii][i] = tmp1;
					}
					break;

				case 3:
					for (i = *internal.nl[j][ii]; i < *internal.nrm[j][ii] - 1; i++)
					{
						tmp = internal.r[i] - *internal.rm[j][ii];
						tmp1 =  mul0 + mul1 * tmp * tmp + mul2 * tmp * tmp + mul3 * tmp * tmp * tmp;
						internal.fi0[j][ii][i] = f(j,ii,internal.r[i]) - tmp1;
						internal.fi1[j][ii][i] = tmp1;
					}
					break;

				case 4:
					for (i = *internal.nl[j][ii]; i < *internal.nrm[j][ii] - 1; i++)
					{
						tmp = internal.r[i] - *internal.rm[j][ii];
						tmp1 =  mul0 + mul1 * tmp * tmp + mul2 * tmp * tmp + mul3 * tmp * tmp * tmp + mul4 * tmp * tmp * tmp * tmp;
						internal.fi0[j][ii][i] = f(j,ii,internal.r[i]) - tmp1;
						internal.fi1[j][ii][i] = tmp1;
					}
					break;

				case 5:
					for (i = *internal.nl[j][ii]; i < *internal.nrm[j][ii] - 1; i++)
					{
						tmp = internal.r[i] - *internal.rm[j][ii];
						tmp1 =  mul0 + mul1 * tmp * tmp + mul2 * tmp * tmp + mul3 * tmp * tmp * tmp + mul4 * tmp * tmp * tmp * tmp + 
						mul5 * tmp * tmp * tmp * tmp * tmp;
						internal.fi0[j][ii][i] = f(j,ii,internal.r[i]) - tmp1;
						internal.fi1[j][ii][i] = tmp1;
					}
					break;

			}

			for (i = *internal.nrm[j][ii] - 1; i < external.nh; i++)
			{
				internal.fi0[j][ii][i] = 0.0;
				internal.fi1[j][ii][i] = f(j,ii,internal.r[i]);
			}
		#ifdef DEBUG
			fprintf(file,"\nfi0[%i,%i]\n",j,ii);
			for (i = *internal.nl[j][ii]; i < external.nh; i++)
			{
				fprintf(file,"\nfi0[%i][%i][%i]%e",j,ii,i,internal.fi0[j][ii][i]);
			}
			fprintf(file,"\nfi1[%i,%i]\n",j,ii);
			for (i = *internal.nl[j][ii]; i < external.nh; i++)
			{
				fprintf(file,"\nfi1[%i][%i][%i]%e",j,ii,i,internal.fi1[j][ii][i]);
			}
		#endif
		}
	}

	//Bigger Iteration Start//
	l = 0;

	createv(external.n_molec_type,&a);//Warning! Array creation!//
	for (i = 0 ; i < external.n_molec_type; i++)
	{
		a[i] = external.a0[i];
	}

	dKK = external.dK + 1.0;

#ifdef PARALLEL
	omp_set_nested(1);
#endif

	while ((dKK > external.dK)&&(flagg > 0)&&(l < external.itermaxbiggest)&&(status == 0))
	{
	#ifdef DEBUG
		fprintf(file,"\n_________________________________________________________biggest step %i",l);
		printf("\nbiggest step %i",l);
	#endif
	#ifdef PARALLEL
		//omp_set_num_threads(external.n_molec_type * (external.Na + 1) * (external.Nrho + 1) + 1);
	#pragma omp parallel sections private (fsw,T,C)
	#endif
		{
		#ifdef PARALLEL
		#pragma omp section
		#endif
			{
				int k,kk,i;
				double *dbpdrhoj, bp;
				createv(external.n_molec_type,&dbpdrhoj);
			#ifdef PARALLEL
				createm_tri(external.n_molec_type,external.nh,&fsw);
				createm_tri(external.n_molec_type,external.nh,&T);
				createm_tri(external.n_molec_type,external.nh,&C);
			#endif
				for (k = 0; k < external.n_molec_type; k++)
				{
					for (kk = k; kk < external.n_molec_type; kk++)
					{
						if (k == kk)
						{
							for (i = 0 ; i < external.nh ; i++)
							{
								fsw[k][kk][i] = swfu(k,kk,internal.r[i],a[k]);
							}
						}
						else
						{
							//double _tmp = (a[k] + a[kk]) / 2.0;
							double _tmp = (a[k] * external.p[k][k][1] + a[kk] * external.p[kk][kk][1]) / 2.0 / external.p[k][kk][1];
							for (i = 0 ; i < external.nh ; i++)
							{
								fsw[k][kk][i] = swfu(k,kk,internal.r[i],_tmp);
							}
						}
					}
				}
				bp = 0;
				copym_tri(external.n_molec_type,external.nh,Tlast,T);
				solveIUR(T,C,fsw,xx[0][0],&status,l,file);
				calcKUZ(ans,T,C,fsw,xx[0][0],dbpdrhoj,&bp,external.Nrho + 1,-1,file);

				for (i = 0; i < external.n_molec_type; i++)
				{
					DD[i][0][0] = dbpdrhoj[i];
				}
			#ifdef STRICT
				copym_tri(external.n_molec_type,external.nh,T,Ttmp);
			#endif
			#ifndef STRICT
			#ifndef PARALLEL
				copym_tri(external.n_molec_type,external.nh,T,Tlast);
			#endif
			#ifdef PARALLEL
				copym_tri(external.n_molec_type,external.nh,T,Ttmp);
			#endif
			#endif
			#ifdef PARALLEL
				freem_tri(external.n_molec_type,C);
				freem_tri(external.n_molec_type,T);
				freem_tri(external.n_molec_type,fsw);
			#endif
				free(dbpdrhoj);
			}

		#ifdef PARALLEL
		#pragma omp section
		#endif
			{
			#ifdef PARALLEL
			#pragma omp parallel for private (fsw,T,C)
			#endif
				for (j = 0; j < external.n_molec_type; j++)
				{
					int jj;
					double **aa;

					if (external.Na == 4)
					{
						//Na=4;
						createm(external.n_molec_type,5,&aa);//Warning! Array creation!//
						for (jj = 0 ; jj < external.n_molec_type; jj++)
						{
							aa[jj][0] = a[jj];
						}
					}
					else
					{
						//Na=2;
						createm(external.n_molec_type,3,&aa);
						for (jj = 0 ; jj < external.n_molec_type; jj++)
						{
							aa[jj][0] = a[jj];
						}
					}

					for (jj = 0; jj < external.n_molec_type; jj++)
					{
						if (external.Na == 4)
						{
							//Na=4;
							if (jj == j)
							{
								aa[jj][1]=aa[jj][0]*(1.0-3.0/2.0*external.da);
								aa[jj][2]=aa[jj][0]*(1.0-external.da/2.0);
								aa[jj][3]=aa[jj][0]*(1.0+external.da/2.0);
								aa[jj][4]=aa[jj][0]*(1.0+3.0/2.0*external.da);
							}
							else
							{
								aa[jj][1]=aa[jj][0];
								aa[jj][2]=aa[jj][0];
								aa[jj][3]=aa[jj][0];
								aa[jj][4]=aa[jj][0];
							}
						}
						else
						{
							//Na=2;
							if (jj == j)
							{
								aa[jj][1]=aa[jj][0]*(1.0-external.da);
								aa[jj][2]=aa[jj][0]*(1.0+external.da);
							}
							else
							{
								aa[jj][1]=aa[jj][0];
								aa[jj][2]=aa[jj][0];
							}
						}
					}

				#ifdef PARALLEL
				#pragma omp parallel for private (fsw,T,C)
				#endif
					for (jj = 0; jj <= external.Na; jj++)
					{
						int jjj,k,kk,i;
					#ifdef PARALLEL
						createm_tri(external.n_molec_type,external.nh,&fsw);
					#endif
						for (k = 0; k < external.n_molec_type; k++)
						{
							for (kk = k; kk < external.n_molec_type; kk++)
							{
								if (k == kk)
								{
									for (i = 0 ; i < external.nh ; i++)
									{
										fsw[k][kk][i] = swfu(k,kk,internal.r[i],aa[k][jj]);
									}
								}
								else
								{
									double ttmp;
									//ttmp = (aa[k][jj] + aa[kk][jj]) / 2.0;
									ttmp = (aa[k][jj] * external.p[k][k][1] + aa[kk][jj] * external.p[kk][kk][1]) / (external.p[k][k][1] + external.p[kk][kk][1]);
									for (i = 0 ; i < external.nh ; i++)
									{
										fsw[k][kk][i] = swfu(k,kk,internal.r[i],ttmp);
									}
								}
							}
						}

					#ifdef PARALLEL
					#pragma omp parallel for private (T,C)
					#endif
						for (jjj = 0; jjj <= external.Nrho; jjj++)
						{
							double dbpdrhoj, bp;
						#ifdef PARALLEL
							createm_tri(external.n_molec_type,external.nh,&T);
							createm_tri(external.n_molec_type,external.nh,&C);
						#endif
						//#ifndef PARALLEL
							if ((jjj == 0)&&(jj == 0))
							{
							#ifdef PARALLEL
								freem_tri(external.n_molec_type,C);
								freem_tri(external.n_molec_type,T);
							#endif
								continue;
							}
						//#endif
							bp = 0;
						#ifndef PARALLEL //TODO: use tmpfiles to perthread logging
						#ifdef DEBUG
							fprintf(file,"\n_________________________________________________________big step j:%i jj:%i jjj:%i",j,jj,jjj);
							printf("\nbig step j:%i jj:%i jjj:%i",j,jj,jjj);

						#endif
						#endif
							copym_tri(external.n_molec_type,external.nh,Tlast,T);
							solveIUR(T,C,fsw,xx[j][jjj],&status,l,file);
							calcKUZ(ans,T,C,fsw,xx[j][jjj],&dbpdrhoj,&bp,jjj,j,file);

							if (jjj == 0) // We are interested in c
							{
								DD[j][jj][0] = dbpdrhoj;
							}
							else // We are interested in g
							{
								DD[j][jj][jjj] = bp;
							#ifndef PARALLEL
							#ifdef DEBUG2
								fprintf(file,"LOOOOOOCK!!! bp:%e",DD[j][jj][jjj]);
							#endif
							#endif
							}
						#ifdef PARALLEL
							freem_tri(external.n_molec_type,C);
							freem_tri(external.n_molec_type,T);
						#endif
						}
					#ifdef PARALLEL
						freem_tri(external.n_molec_type,fsw);
					#endif
					}
					freem(external.n_molec_type,aa);
				}
			}
		}
	#ifdef PARALLEL
	#pragma omp barrier
	#endif
		dKK = 0;
		for (j = 0; j < external.n_molec_type; j++)
		{
			int jj;
			double dlg, mul, _dKK;
			for (jj = 0; jj <= external.Na; jj++)
			{
				//calc_dF
				if (external.Nrho == 4)
				{
					D[j][jj] = DD[j][jj][0] - (DD[j][jj][1] - 27.0 * DD[j][jj][2] + 27.0 * DD[j][jj][3] - DD[j][jj][4]) / 24.0 / external.drho / x[j][0] / external.rho;
				#ifdef DEBUG2
					fprintf(file,"\ni:%i j:%i D:%e DD0:%e DD1:%e DD2:%e DD3:%e DD4:%e",j,jj,D[j][jj],DD[j][jj][0],DD[j][jj][1],DD[j][jj][2],DD[j][jj][3],DD[j][jj][4]);
					printf("\n_________solver: D:%e",D[j][jj]);
				#endif
				}
				else
				{
					D[j][jj] = DD[j][jj][0] - (DD[j][jj][2] - DD[j][jj][1]) / 2.0 / external.drho / x[j][0] / external.rho;
				#ifdef DEBUG2
					fprintf(file,"\ni:%i j:%i D:%e DD0:%e DD1:%e DD2:%e",j,jj,D[j][jj],DD[j][jj][0],DD[j][jj][1],DD[j][jj][2]);
					printf("\n_________solver: D:%e",D[j][jj]);
				#endif
				}
			}
			//calc_dD
			if (external.Nrho == 4)
			{
				dlg = - D[j][0] * 24.0 * external.da / (D[j][1] - 27.0 * D[j][2] + 27.0 * D[j][3] - D[j][4]);
			}
			else
			{
				dlg = - D[j][0] * 2.0 * external.da / (D[j][2] - D[j][1]);
			}
			mul = fabs( external.maxabsdlg / dlg);
		#ifdef DEBUG
			fprintf(file,"\ndlg:%e mul:%e",dlg,mul);
		#endif
			if (mul > 0.3)
			{
				mul = 0.3;
			}
			if ((dlg <= DBL_MAX) && (dlg >= - DBL_MAX))
			{
				a[j] = a[j] * exp(mul * dlg);
			}
			fprintf(file,"\na[%i]:%e",j,a[j]);
			printf("\n");
			_dKK = fabs(D[j][0]);
			if (_dKK > dKK)
			{
				dKK = _dKK;
			}
		}
	#ifdef PARALLEL
		copym_tri(external.n_molec_type,external.nh,Ttmp,Tlast);
	#endif
	#ifndef PARALLEL
	#ifdef STRICT
		copym_tri(external.n_molec_type,external.nh,Ttmp,Tlast);
	#endif
	#endif
		l++;
	}
//
	if (status > 0)
	{
		for (i = 0; i < external.n_molec_type + 4; i++)
		{
			ans[i] = INFINITY;
		}
	}
//
	free(internal.rmul);
	free(internal.Simp);
	free(internal.Simp0);

	free(a);

	//freem(external.nh,internal.si);
	freem3(external.n_molec_type,external.Na + 1,DD);
	freem(external.n_molec_type,D);
	freem3(external.n_molec_type,external.Nrho + 1,xx);
#ifndef PARALLEL
	freem_tri(external.n_molec_type,fsw);
	freem_tri(external.n_molec_type,C);
	freem_tri(external.n_molec_type,T);
#endif
	fftw_destroy_plan(internal.plan);
	freem_tri(external.n_molec_type,Ttmp);
	freem_tri(external.n_molec_type,Tlast);
	freem_tri(external.n_molec_type,internal.fi1);
	freem_tri(external.n_molec_type,internal.fi0);
	freem(external.n_molec_type,x);
	free(internal.q);
	free(internal.r);

	return 0;
}

int general(double *in, double *ans, FILE *file, FILE *file1, FILE *file2)
{
	int flag,i,j,l;
	double mul,nn,tmp;

	tmp = 0.0;
	for (i = 0; i < external.n_molec_type; i++)
	{
		tmp += in[i];
	}
	for (i = 0; i < external.n_molec_type; i++)
	{
		external.mol_fr[i] = in[i] / tmp;//mole fraction
		//
		fprintf(file,"\nmole_fraction[%i] = %lf",i,external.mol_fr[i]);
	}
	fprintf(file,"\n");

	internal.b = 1.0 / in[external.n_molec_type + 0];
	external.rho = in[external.n_molec_type + 1];

	createa_tri(external.n_molec_type,&internal.nrm);
	createa_tri(external.n_molec_type,&internal.nl);
	createv_tri(external.n_molec_type,&internal.Rlocut);
	createv_tri(external.n_molec_type,&internal.rm);

	internal.dr = external.Rhicut / (double)external.nh;
	internal.dlambda = 1.0 / (double)external.nh;

	flag = 0;
	
	for (i = 0; i < external.n_molec_type; i++)
	{
		for (j = i; j < external.n_molec_type; j++)
		{
			//k = (2 * (external.n_molec_type + 1) - i + 1) / 2 * i + j;
			getpotential(&flag,i,j,file);

			*internal.nl[i][j] = (int)floor(*internal.Rlocut[i][j] / internal.dr + 1.0);
			if (fmod((double)*internal.nl[i][j],2.0) > 0.0)
			{
				if (fmod((double)external.nh,2.0) > 0.0)
				{
					(*internal.nl[i][j])++;
				}
			}
			else
			{
				if (fmod((double)external.nh,2.0) == 0.0)
				{
					(*internal.nl[i][j])++;
				}
			}
			*internal.nrm[i][j] = (int)floor(*internal.rm[i][j] / internal.dr + 1.0);
			fprintf(file,"i = %i j = %i nrm = %d nl = %d\n", i,j,*internal.nrm[i][j],*internal.nl[i][j]);
		}
	}

	if (flag == 0)
	{
	        fprintf(file,"nh = %d \n", external.nh);

		nn = 120.0;

		for (i = 0; i < external.n_molec_type; i++)
		{
			for (j = i; j < external.n_molec_type; j++)
			{
				//k = (2 * (external.n_molec_type + 1) - i + 1) / 2 * i + j;

				if (*external.r_type[i][j]==1)
				{

					mul = pow( 1.0 + pow((external.rho / sqrt(2.0)), nn / 3.0), - 1.0 / nn);

					*internal.nrm[i][j] = (int)floor(*internal.rm[i][j] * mul / internal.dr + 1.0);

					fprintf(file,"nrm[%i,%i] -> nlmb = %d\n",i,j,*internal.nrm[i][j]);

					*internal.nrm[i][j] = (int)floor(*internal.rm[i][j] * mul * *external.lmb[i][j] / internal.dr + 1.0);

				}

				fprintf(file,"nrm[%i,%i] -> nlmb_corr[%i,%i] = %d\n",i,j,i,j,*internal.nrm[i][j]);

				fprintf(file,"      r       |      f(r)     |    df(r)/dr    |       I1       |      I2      \n");
				fprintf(file,"_______________________________________________________________________________\n\n");

				for (l = *internal.nl[i][j] - 1; l < external.nh; l++)
				{
					fprintf(file,"%9e	 %9e	 %9e	 %9e	 %9e\n",((double)l + 1.0) * internal.dr,f(i,j,((double)l + 1.0) * internal.dr),df(i,j,((double)l + 1.0) * internal.dr),I1(i,j,((double)l + 1.0) * internal.dr),I2(i,j,((double)l + 1.0) * internal.dr));
				}
			}
		}
		solver(ans,file,file1,file2);
	}

	freea_tri(external.n_molec_type,internal.nrm);
	freea_tri(external.n_molec_type,internal.nl);
	freev_tri(external.n_molec_type,internal.Rlocut);
	freev_tri(external.n_molec_type,internal.rm);

	return 0;
}

int getjacob(int n, double *x, double **LU, FILE *file, FILE *file1, FILE *file2)
{
	int i,j,k,l;
	double *in, *ans, *tmp;

	createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&in);
	createv(external.n_molec_type + 4,&ans);//P,U,H,S
	createv(exter_tdc.Nnewt,&tmp);

	for (i = 0; i < external.n_molec_type + 2; i++)
	{
		for (j = 0; j < external.n_molec_type + 2; j++)
		{
			//
			fprintf(file,"\ntdc debug froz %i %i",exter_tdc.frozen[i],exter_tdc.frozen[j]);
			//
			if ((exter_tdc.frozen[i] == 0)||(exter_tdc.frozen[j] == 0))
			{
				if (i == j)
				{
					LU[i][j] = 1.0;
				}
				else
				{
					LU[i][j] = 0.0;
				}
			}
			else
			{
				if (exter_tdc.Nnewt == 4)
				{
					tmp[0]=x[j]*(1.0-3.0/2.0*exter_tdc.dnewt);
					tmp[1]=x[j]*(1.0-exter_tdc.dnewt/2.0);
					tmp[2]=x[j]*(1.0+exter_tdc.dnewt/2.0);
					tmp[3]=x[j]*(1.0+3.0/2.0*exter_tdc.dnewt);
				}
				else
				{
				//Nrho=2;
					tmp[0]=x[j]*(1.0-exter_tdc.dnewt);
					tmp[1]=x[j]*(1.0+exter_tdc.dnewt);
				}

				for (k = 0; k < exter_tdc.Nnewt; k++)
				{
					for (l = 0; l < n; l++)
					{
						in[l] = x[l];
					}

					in[j] = tmp[k];
					//
					fprintf(file,"\ntdc debug i=%i, j=%i, k=%i",i,j,k);
					//
					general(in,ans,file,file1,file2);
					if (i < external.n_molec_type)
					{
						tmp[k] = ans[i];
					}
					else
					{
						switch (exter_tdc.problem_type)
						{
							case 1:
								//TV
								//Nothing to do
								break;
							case 2:
								//TP	
								tmp[k] = ans[external.n_molec_type];
								break;
							case 3:
								//UV
								tmp[k] = ans[external.n_molec_type + 1] / mole_w(in); //1 kg of system
								break;
							case 4:
								//HP
								if (i == external.n_molec_type)
								{
									tmp[k] = ans[external.n_molec_type + 2] / mole_w(in);
								}
								else
								{
									tmp[k] = ans[external.n_molec_type];
								}
								break;
							case 5:
								//SV
								tmp[k] = ans[external.n_molec_type + 3] / mole_w(in);
								break;
							case 6:
								//SP
								if (i == external.n_molec_type)
								{
									tmp[k] = ans[external.n_molec_type + 3] / mole_w(in);
								}
								else
								{
									tmp[k] = ans[external.n_molec_type];
								}
								break;
							case 7:
								//Sh
								//TODO
								break;
							case 8:
								//D
								//TODO
								break;
						}
					}
				}

				//calc_dF
				if (exter_tdc.Nnewt == 4)
				{
					LU[i][j] = (tmp[0] - 27.0 * tmp[1] + 27.0 * tmp[2] - tmp[3]) / 24.0 / exter_tdc.dnewt / x[j];
				}
				else
				{
					LU[i][j] = (tmp[1] - tmp[0]) / 2.0 / exter_tdc.dnewt / x[j];
				}
			}
		}
	}

	general(x,ans,file,file1,file2);//To print values at center point

	free(tmp);
	free(ans);
	free(in);

	for (i = 0; i < external.n_molec_type; i++)
	{
		for (j = 0; j < exter_tdc.n_atom_type; j++)
		{
			LU[i][external.n_molec_type + 2 + j] = - inter_tdc.a[j][i];
			LU[external.n_molec_type + 2 + j][i] = inter_tdc.a[j][i];
		}
	}

	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < exter_tdc.n_atom_type; j++)
		{
			LU[external.n_molec_type + i][external.n_molec_type + 2 + j] = 0.0;
			LU[external.n_molec_type + 2 + j][external.n_molec_type + i] = 0.0;
		}
	}

	for (i = 0; i < exter_tdc.n_atom_type; i++)
	{
		for (j = 0; j < exter_tdc.n_atom_type; j++)
		{
			LU[external.n_molec_type + 2 + i][external.n_molec_type + 2 + j] = 0.0;
		}
	}

	//
	outm(n,LU,file);
	//

	return 0;
}

int solver_tdc_0(FILE *file, FILE *file1, FILE *file2)
{
	switch(exter_tdc.problem_type)
	{
		case 0:
		//	external.Te = exter_tdc.initials[0];
		//	external.rho = exter_tdc.initials[1];
		//	solver(file,file1,file2);
		//	TODO
			break;
		case 7:
			break;
		case 8:
			break;
		default:
		{
			int i,j;			
			double *x,*y;
			createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&x);
			createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&y);
			for (i = 0; i < external.n_molec_type; i++)
			{
				x[i] = external.mol_fr[i] / mole_w(external.mol_fr);// kmol/kg
				y[i] = 0.0;// mu[j] / RT - sumi a[i][j] pi[i] = 0
			}
			x[external.n_molec_type + 0] = exter_tdc.initials[0];
			x[external.n_molec_type + 1] = exter_tdc.initials[1];
			y[external.n_molec_type + 0] = exter_tdc.initials[2];
			y[external.n_molec_type + 1] = exter_tdc.initials[3];
			for (i = 0; i < exter_tdc.n_atom_type ; i++)
			{
				x[external.n_molec_type + 2 + i] = 0.0;//pi initial
				//TODO Improve pi initial
				y[external.n_molec_type + 2 + i] = 0.0;//sumj a[i][j] n[j] = b[i]
				for (j = 0; j < external.n_molec_type; j++)
				{
					y[external.n_molec_type + 2 + i] += inter_tdc.a[i][j] * x[j];
				}
			}
			//general(x,file,file1,file2);
			newtonv(external.n_molec_type + 2 + exter_tdc.n_atom_type,x,y,file,file1,file2);
			free(y);
			free(x);
			break;
		}

	}
	return 0;
}

double residual(const gsl_vector *x, void *p)
{
	int i,j;
	double result, *in, *ans;
	struct parameters *params;
	params = p;
	gsl_vector *eq = gsl_vector_alloc(external.n_molec_type + 2 + exter_tdc.n_atom_type);

	createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&in);
	createv(external.n_molec_type + 4,&ans);//P,U,H,S

	for (i = 0; i < external.n_molec_type + 2 + exter_tdc.n_atom_type; i++)
	{
		in[i] = gsl_vector_get(x,i);
	}

	general(in,ans,params->file,params->file1,params->file2);

	for (j = 0; j < external.n_molec_type; j++)
	{
		double tmp = ans[j];
		for (i = 0; i < exter_tdc.n_atom_type; i++)
		{
			tmp -= inter_tdc.a[i][j] * in[external.n_molec_type + 2 + i];
		}
		gsl_vector_set(eq,j,tmp);
	}

	switch (exter_tdc.problem_type)
	{
		case 1:
			//TV
			gsl_vector_set(eq,external.n_molec_type + 0,in[external.n_molec_type + 0]);
			gsl_vector_set(eq,external.n_molec_type + 1,in[external.n_molec_type + 1]);
			break;
		case 2:
			//TP
			gsl_vector_set(eq,external.n_molec_type + 0,in[external.n_molec_type + 0]);
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 0]);
			break;
		case 3:
			//UV
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 1] / mole_w(in)); // 1 kg of system
			gsl_vector_set(eq,external.n_molec_type + 1,in[external.n_molec_type + 1]);
			break;
		case 4:
			//HP
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 2] / mole_w(in));
			gsl_vector_set(eq,external.n_molec_type + 1,ans[external.n_molec_type + 0]);
			break;
		case 5:
			//SV
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 3] / mole_w(in));
			gsl_vector_set(eq,external.n_molec_type + 1,in[external.n_molec_type + 1]);
			break;
		case 6:
			//SP
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 3] / mole_w(in));
			gsl_vector_set(eq,external.n_molec_type + 1,ans[external.n_molec_type + 0]);
			break;
		case 7:
			//Sh
			//TODO
			break;
		case 8:
			//D
			//TODO
			break;
	}
	
	for (i = 0; i < exter_tdc.n_atom_type; i++)
	{
		double tmp = 0.0;
		for (j = 0; j < external.n_molec_type; j++)
		{
			tmp += inter_tdc.a[i][j] * in[j];
		}
		gsl_vector_set(eq,external.n_molec_type + 2 + i,tmp);
	}
//
	fprintf(params->file,"\ntdc");
	for (i = 0; i < external.n_molec_type + 2 + exter_tdc.n_atom_type; i++)
	{
		fprintf(params->file," %e %e",gsl_vector_get(eq,i),gsl_vector_get(params->y,i));
	}
	fprintf(params->file,"\n");
//
	gsl_vector_sub(eq,params->y);
	/*
	for (i = 0; i < external.n_molec_type + 2 + exter_tdc.n_atom_type; i++)
	{
		if (gsl_vector_get(params->y,i) != 0.0)
		{
			gsl_vector_set(eq,i,gsl_vector_get(eq,i) / gsl_vector_get(params->y,i));
		}
	}
	*/
	gsl_blas_ddot(eq,eq,&result);

	free(ans);
	free(in);
	gsl_vector_free(eq);

	return result;
}

int solver_tdc_1(FILE *file, FILE *file1, FILE *file2)
{
	switch(exter_tdc.problem_type)
	{
		case 0:
		//	external.Te = exter_tdc.initials[0];
		//	external.rho = exter_tdc.initials[1];
		//	solver(file,file1,file2);
		//	TODO
			break;
		case 7:
			break;
		case 8:
			break;
		default:
		{
			int i,j,v_len;
			struct parameters params;

			params.file = file;
			params.file1 = file1;
			params.file2 = file2;

			v_len = external.n_molec_type + 2 + exter_tdc.n_atom_type;

			gsl_vector *x = gsl_vector_alloc(v_len);
			params.y = gsl_vector_alloc(v_len);
			gsl_vector *delta = gsl_vector_alloc(v_len);
			const gsl_rng_type *rng_type;
			gsl_rng *rng;

			gsl_rng_env_setup();
			rng_type = gsl_rng_default;
			rng = gsl_rng_alloc(rng_type);

			for (i = 0; i < external.n_molec_type; i++)
			{
				gsl_vector_set(x,i,external.mol_fr[i] / mole_w(external.mol_fr));// kmol/kg
				gsl_vector_set(params.y,i,0.0);// mu[j] / RT - sumi a[i][j] pi[i] = 0
			}
			gsl_vector_set(x,external.n_molec_type + 0,exter_tdc.initials[0]);
			gsl_vector_set(x,external.n_molec_type + 1,exter_tdc.initials[1]);
			gsl_vector_set(params.y,external.n_molec_type + 0,exter_tdc.initials[2]);
			gsl_vector_set(params.y,external.n_molec_type + 1,exter_tdc.initials[3]);
			for (i = 0; i < exter_tdc.n_atom_type ; i++)
			{
				gsl_vector_set(x,external.n_molec_type + 2 + i,1.0);//pi initial
				//TODO Improve pi initial
				double tmp = 0.0;//sumj a[i][j] n[j] = b[i]
				for (j = 0; j < external.n_molec_type; j++)
				{
					tmp += inter_tdc.a[i][j] * gsl_vector_get(x,j);
				}
				gsl_vector_set(params.y,external.n_molec_type + 2 + i,tmp);
			}

			for (i = 0; i < v_len; i++)
			{
				//gsl_vector_set(delta,i,gsl_rng_uniform(rng));
				gsl_vector_set(delta,i,1.0);
			}
			gsl_vector_mul(delta,x);
			gsl_vector_scale(delta,exter_tdc.dnewt);

			const gsl_multimin_fminimizer_type *T = 
			gsl_multimin_fminimizer_nmsimplex2;
			gsl_multimin_fminimizer *s = NULL;
			gsl_multimin_function minex_func;

			int status;
			double size;

			minex_func.n = v_len;
			minex_func.f = residual;
			minex_func.params = &params;

			s = gsl_multimin_fminimizer_alloc (T, v_len);
			gsl_multimin_fminimizer_set (s, &minex_func, x, delta);

			i = 0;

			do
			{
				i++;
				status = gsl_multimin_fminimizer_iterate(s);

				if (status) 
					break;

				size = gsl_multimin_fminimizer_size (s);
				status = gsl_multimin_test_size (size,exter_tdc.deltanewt);

				if (status == GSL_SUCCESS)
				{
					fprintf (file,"converged to minimum at\n");
				}

				fprintf(file,"\ntdc f() = %e size = %e",s[0].fval,size);//s[0]. = s->
				for (j = 0; j < v_len; j++)
				{
					fprintf(file," %e",gsl_vector_get(s->x,j));
				}
				fprintf(file,"\n");
			}

			while (status == GSL_CONTINUE && i < exter_tdc.itermaxnewt);

			gsl_multimin_fminimizer_free(s);
			gsl_rng_free(rng);
			gsl_vector_free(delta);
			gsl_vector_free(params.y);
			gsl_vector_free(x);

			break;
		}
	}
	return 0;
}

int tdc_eqs2(const gsl_vector *x, void *p, gsl_vector *eq)
{
	int i,j;
	double result, *in, *ans;
	struct parameters *params;
	params = p;

	createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&in);
	createv(external.n_molec_type + 4,&ans);//P,U,H,S

	for (i = 0; i < external.n_molec_type + 2 + exter_tdc.n_atom_type; i++)
	{
		in[i] = gsl_vector_get(x,i);
	}

	general(in,ans,params->file,params->file1,params->file2);

	for (j = 0; j < external.n_molec_type; j++)
	{
		double tmp = ans[j];
		for (i = 0; i < exter_tdc.n_atom_type; i++)
		{
			tmp -= inter_tdc.a[i][j] * in[external.n_molec_type + 2 + i];
		}
		gsl_vector_set(eq,j,tmp);
	}

	switch (exter_tdc.problem_type)
	{
		case 1:
			//TV
			gsl_vector_set(eq,external.n_molec_type + 0,in[external.n_molec_type + 0]);
			gsl_vector_set(eq,external.n_molec_type + 1,in[external.n_molec_type + 1]);
			break;
		case 2:
			//TP
			gsl_vector_set(eq,external.n_molec_type + 0,in[external.n_molec_type + 0]);
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 0]);
			break;
		case 3:
			//UV
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 1] / mole_w(in)); // 1 kg of system
			gsl_vector_set(eq,external.n_molec_type + 1,in[external.n_molec_type + 1]);
			break;
		case 4:
			//HP
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 2] / mole_w(in));
			gsl_vector_set(eq,external.n_molec_type + 1,ans[external.n_molec_type + 0]);
			break;
		case 5:
			//SV
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 3] / mole_w(in));
			gsl_vector_set(eq,external.n_molec_type + 1,in[external.n_molec_type + 1]);
			break;
		case 6:
			//SP
			gsl_vector_set(eq,external.n_molec_type + 0,ans[external.n_molec_type + 3] / mole_w(in));
			gsl_vector_set(eq,external.n_molec_type + 1,ans[external.n_molec_type + 0]);
			break;
		case 7:
			//Sh
			//TODO
			break;
		case 8:
			//D
			//TODO
			break;
	}

	for (i = 0; i < exter_tdc.n_atom_type; i++)
	{
		double tmp = 0.0;
		for (j = 0; j < external.n_molec_type; j++)
		{
			tmp += inter_tdc.a[i][j] * in[j];
		}
		gsl_vector_set(eq,external.n_molec_type + 2 + i,tmp);
	}
//
	fprintf(params->file,"\ntdc");
	for (i = 0; i < external.n_molec_type + 2 + exter_tdc.n_atom_type; i++)
	{
		fprintf(params->file," %e %e %e",gsl_vector_get(x,i),gsl_vector_get(eq,i),gsl_vector_get(params->y,i));
	}
	fprintf(params->file,"\n");
//

	gsl_vector_sub(eq,params->y);
	gsl_blas_ddot(eq,eq,&result);
	fprintf(params->file,"\ntdc residual %e",result);

	free(ans);
	free(in);

	return GSL_SUCCESS;
}

int solver_tdc2(FILE *file, FILE *file1, FILE *file2)
{
	switch(exter_tdc.problem_type)
	{
		case 0:
		//      external.Te = exter_tdc.initials[0];
		//      external.rho = exter_tdc.initials[1];
		//      solver(file,file1,file2);
		//      TODO
			break;
		case 7:
			break;
		case 8:
			break;
		default:
		{
			int i,j,v_len;
			struct parameters params;

			params.file = file;
			params.file1 = file1;
			params.file2 = file2;

			v_len = external.n_molec_type + 2 + exter_tdc.n_atom_type;

			gsl_vector *x = gsl_vector_alloc(v_len);
			params.y = gsl_vector_alloc(v_len);
			const gsl_rng_type *rng_type;
			gsl_rng *rng;

			gsl_rng_env_setup();
			rng_type = gsl_rng_default;
			rng = gsl_rng_alloc(rng_type);

			for (i = 0; i < external.n_molec_type; i++)
			{
				gsl_vector_set(x,i,external.mol_fr[i] / mole_w(external.mol_fr));// kmol/kg
				gsl_vector_set(params.y,i,0.0);// mu[j] / RT - sumi a[i][j] pi[i] = 0
			}
			gsl_vector_set(x,external.n_molec_type + 0,exter_tdc.initials[0]);
			gsl_vector_set(x,external.n_molec_type + 1,exter_tdc.initials[1]);
			gsl_vector_set(params.y,external.n_molec_type + 0,exter_tdc.initials[2]);
			gsl_vector_set(params.y,external.n_molec_type + 1,exter_tdc.initials[3]);
			for (i = 0; i < exter_tdc.n_atom_type ; i++)
			{
				gsl_vector_set(x,external.n_molec_type + 2 + i,0.0);//pi initial
				//TODO Improve pi initial
				double tmp = 0.0;//sumj a[i][j] n[j] = b[i]
				for (j = 0; j < external.n_molec_type; j++)
				{
					tmp += inter_tdc.a[i][j] * gsl_vector_get(x,j);
				}
				gsl_vector_set(params.y,external.n_molec_type + 2 + i,tmp);
			}

			const gsl_multiroot_fsolver_type *T;
			gsl_multiroot_fsolver *s;

			int status;
			i = 0;

			gsl_multiroot_function f = {&tdc_eqs2,v_len,&params};

			T = gsl_multiroot_fsolver_hybrids;
			//T = gsl_multiroot_fsolver_dnewton;

			s = gsl_multiroot_fsolver_alloc (T, v_len);
			gsl_multiroot_fsolver_set (s, &f, x);

			do
			{
				i++;
				status = gsl_multiroot_fsolver_iterate (s);

			if (status)   /* check if solver is stuck */
				break;

			status = gsl_multiroot_test_residual (s->f,exter_tdc.deltanewt);
			}

			while (status == GSL_CONTINUE && i < exter_tdc.itermaxnewt);

			fprintf (file,"\nstatus = %s\n", gsl_strerror (status));

			gsl_multiroot_fsolver_free (s);
			gsl_rng_free(rng);
			gsl_vector_free(params.y);
			gsl_vector_free(x);

			break;
		}
	}
	return 0;
}

int tdc_eqs(const gsl_vector *x, void *p, gsl_vector *eq)
{
	//TV
	int i,j;
	double result, *in, *ans;
	struct parameters *params;
	params = p;

	createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&in);
	createv(external.n_molec_type + 4,&ans);//

	for (i = 0; i < external.n_molec_type; i++)
	{
		in[i] = gsl_vector_get(x,i);
	}
	
	in[external.n_molec_type + 0] = exter_tdc.initials[0];
	in[external.n_molec_type + 1] = exter_tdc.initials[1];

	for (i = 0; i < exter_tdc.n_atom_type; i++)
	{
		in[external.n_molec_type + 2 + i] = gsl_vector_get(x,external.n_molec_type + i);
	}

	general(in,ans,params->file,params->file1,params->file2);

	for (j = 0; j < external.n_molec_type; j++)
	{
		double tmp = ans[j];
		for (i = 0; i < exter_tdc.n_atom_type; i++)
		{
			tmp -= inter_tdc.a[i][j] * in[external.n_molec_type + 2 + i];
		}
		gsl_vector_set(eq,j,tmp);
	}

	for (i = 0; i < exter_tdc.n_atom_type; i++)
	{
		double tmp = 0.0;
		for (j = 0; j < external.n_molec_type; j++)
		{
			tmp += inter_tdc.a[i][j] * in[j];
		}
		gsl_vector_set(eq,external.n_molec_type + i,tmp);
	}
//
	fprintf(params->file,"\ntdc");
	for (i = 0; i < external.n_molec_type + exter_tdc.n_atom_type; i++)
	{
		fprintf(params->file," %e %e %e",gsl_vector_get(x,i),gsl_vector_get(eq,i),gsl_vector_get(params->y,i));
	}
	fprintf(params->file,"\n");
//

	gsl_vector_sub(eq,params->y);
	gsl_blas_ddot(eq,eq,&result);
	fprintf(params->file,"\ntdc residual %e",result);

	free(ans);
	free(in);

	return GSL_SUCCESS;
}

int solver_tdc(FILE *file, FILE *file1, FILE *file2)
{
	switch(exter_tdc.problem_type)
	{
		case 0:
		{
			//EOS
			int i;
			double *in, *ans;

			createv(external.n_molec_type + 2 + exter_tdc.n_atom_type,&in);
			createv(external.n_molec_type + 4,&ans);//

			for (i = 0; i < external.n_molec_type; i++)
			{
				in[i] = external.mol_fr[i] / mole_w(external.mol_fr);
			}

			in[external.n_molec_type + 0] = exter_tdc.initials[0];
			in[external.n_molec_type + 1] = exter_tdc.initials[1];

			general(in,ans,file,file1,file2);

			free(ans);
			free(in);
			
			break;
		}
		case 7:
			break;
		case 8:
			break;
		case 1:
		{
			int i,j,v_len;
			struct parameters params;

			params.file = file;
			params.file1 = file1;
			params.file2 = file2;

			v_len = external.n_molec_type + exter_tdc.n_atom_type;

			gsl_vector *x = gsl_vector_alloc(v_len);
			params.y = gsl_vector_alloc(v_len);
			const gsl_rng_type *rng_type;
			gsl_rng *rng;

			gsl_rng_env_setup();
			rng_type = gsl_rng_default;
			rng = gsl_rng_alloc(rng_type);

			for (i = 0; i < external.n_molec_type; i++)
			{
				gsl_vector_set(x,i,external.mol_fr[i] / mole_w(external.mol_fr));// kmol/kg
				gsl_vector_set(params.y,i,0.0);// mu[j] / RT - sumi a[i][j] pi[i] = 0
			}
			for (i = 0; i < exter_tdc.n_atom_type ; i++)
			{
				gsl_vector_set(x,external.n_molec_type + i,0.0);//pi initial
				//TODO Improve pi initial
				double tmp = 0.0;//sumj a[i][j] n[j] = b[i]
				for (j = 0; j < external.n_molec_type; j++)
				{
					tmp += inter_tdc.a[i][j] * gsl_vector_get(x,j);
				}
				gsl_vector_set(params.y,external.n_molec_type + i,tmp);
			}

			const gsl_multiroot_fsolver_type *T;
			gsl_multiroot_fsolver *s;

			int status;
			i = 0;

			gsl_multiroot_function f = {&tdc_eqs,v_len,&params};

			T = gsl_multiroot_fsolver_hybrids;
			//T = gsl_multiroot_fsolver_dnewton;

			s = gsl_multiroot_fsolver_alloc (T, v_len);
			gsl_multiroot_fsolver_set (s, &f, x);

			do
			{
				i++;
				status = gsl_multiroot_fsolver_iterate (s);

			if (status)   /* check if solver is stuck */
				break;

			status = gsl_multiroot_test_residual (s->f,exter_tdc.deltanewt);
			}

			while (status == GSL_CONTINUE && i < exter_tdc.itermaxnewt);

			fprintf (file,"\nstatus = %s\n", gsl_strerror (status));

			gsl_multiroot_fsolver_free (s);
			gsl_rng_free(rng);
			gsl_vector_free(params.y);
			gsl_vector_free(x);

			break;
		}
	}
	return 0;
}

int inter(FILE *logfile, FILE *infile)
{
	FILE *file1 = NULL,*file2 = NULL;
	char path1[PATH_MAX] = {0}, path2[PATH_MAX] = {0}, word[BUFSIZ];
	int flag,i,j,k,l;

	flag = 0;
	i = 0;
	while (flag == 0)
	{
		i++;
		fscanf_safe(infile,"%s\n",word);
		if (strcasecmp(word,"start") == 0)
		{
			fprintf(logfile,"start\n");
			fscanf_safe(infile,"inpresfile %i ",&external.inp__type);
			if (external.inp__type == 1)
			{
				fscanf_safe(infile,"%s",path1);
			}
			if (external.inp__type == 2)
			{
				if (i == 1)
				{
					external.inp__type = 3;
				}
				else
				{
					strcpy(path1,path2);
				}
			}
			fscanf_safe(infile,"\noutfile %s",path2);
			fscanf_safe(infile,"\nproblem_type %i",&exter_tdc.problem_type);
			createv(MAX_INITIALS,&exter_tdc.initials);
			switch (exter_tdc.problem_type)
			{
				case 0:
					//EOS
					fscanf_safe(infile,"\nT[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nrho[g/cc] %lf",&exter_tdc.initials[1]);
					break;
				case 1:
					//TV
					fscanf_safe(infile,"\nT[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nrho[g/cc] %lf",&exter_tdc.initials[1]);
					break;
				case 2:
					//TP
					fscanf_safe(infile,"\nT[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nP[Pa] %lf",&exter_tdc.initials[3]);
					fscanf_safe(infile,"\nrho_in[g/cc] %lf",&exter_tdc.initials[1]);
					break;
				case 3:
					//UV
					fscanf_safe(infile,"\nU[J/mol] %lf",&exter_tdc.initials[2]);
					fscanf_safe(infile,"\nrho[g/cc] %lf",&exter_tdc.initials[1]);
					fscanf_safe(infile,"\nT_in[K] %lf",&exter_tdc.initials[0]);
					break;
				case 4:
					//HP
					fscanf_safe(infile,"\nH[J/mol] %lf",&exter_tdc.initials[2]);
					fscanf_safe(infile,"\nP[Pa] %lf",&exter_tdc.initials[3]);
					fscanf_safe(infile,"\nT_in[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nrho_in[g/cc] %lf",&exter_tdc.initials[1]);
					break;
				case 5:
					//SV
					fscanf_safe(infile,"\nS[J/K*mol] %lf",&exter_tdc.initials[2]);
					fscanf_safe(infile,"\nrho[g/cc] %lf",&exter_tdc.initials[1]);
					fscanf_safe(infile,"\nT_in[K] %lf", &exter_tdc.initials[0]);
					exter_tdc.initials[3] = exter_tdc.initials[1];
					break;
				case 6:
					//SP
					fscanf_safe(infile,"\nS[J/K*mol] %lf",&exter_tdc.initials[2]);
					fscanf_safe(infile,"\nP[Pa] %lf",&exter_tdc.initials[3]);
					fscanf_safe(infile,"\nT_in[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nrho_in[g/cc] %lf",&exter_tdc.initials[1]);
					break;
				case 7:
					//Sh
					fscanf_safe(infile,"\nT0[K] %lf",&exter_tdc.initials[2]);
					fscanf_safe(infile,"\nrho0[g/cc] %lf",&exter_tdc.initials[3]);
					fscanf_safe(infile,"\nrho/rho0[1] %lf",&exter_tdc.initials[4]);
					fscanf_safe(infile,"\nT_in[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nrho_in[g/cc] %lf",&exter_tdc.initials[1]);
					break;
				case 8:
					//D
					fscanf_safe(infile,"\nT_in[K] %lf",&exter_tdc.initials[0]);
					fscanf_safe(infile,"\nrho_in[g/cc] %lf",&exter_tdc.initials[1]);
					break;
			}
#ifndef AUTO_SCALE
			fscanf_safe(infile,"\ne_scale[K] %lf",&inter_tdc.e_scale);
			fscanf_safe(infile,"\nr_scale[A] %lf",&inter_tdc.r_scale);
#endif
			fscanf_safe(infile,"\ncreate_mesh %lf %i",&external.Rhicut,&external.nh);
			fscanf_safe(infile,"\nn_molec_type %i",&external.n_molec_type);
			fscanf_safe(infile,"\nn_atom_type %i",&exter_tdc.n_atom_type);
		
			internal.pair_count = 0;

			if ( external.n_molec_type > MAX_N_MOLEC_TYPE )
			{
				fprintf(logfile,"Too big n_molec_type! You should change MAX_NMOLEC_TYPE in the source code!");
				return 1;
			}

			createv(external.n_molec_type,&external.mol_fr);
			createm_tri(external.n_molec_type,MAX_PAR_COUNT,&external.p);
			createv(external.n_molec_type,&external.a0);
			createv_tri(external.n_molec_type,&external.rfe);
			createv_tri(external.n_molec_type,&external.d);
			createv_tri(external.n_molec_type,&external.lmb);
			createa_tri(external.n_molec_type,&external.itermax);
			createa_tri(external.n_molec_type,&external.Nphi);
			createa_tri(external.n_molec_type,&external.r_type);
			createa_tri(external.n_molec_type,&external.cl_type);
			createa_tri(external.n_molec_type,&external.sw_type);
			createa_tri(external.n_molec_type,&external.poten_type);
			createma(external.n_molec_type,exter_tdc.n_atom_type,&exter_tdc.mole_compos);
			createa(external.n_molec_type,&exter_tdc.calor_style);
			createv(exter_tdc.n_atom_type,&exter_tdc.atom_weights);
			createm(external.n_molec_type,MAX_CAL_PARMS,&exter_tdc.cal_parms);
			createv(external.n_molec_type,&inter_tdc.mu_thermal_id);
			createv(external.n_molec_type,&inter_tdc.mu_thermal_ex);
			createv(external.n_molec_type,&inter_tdc.mu_caloric);
			createv(external.n_molec_type,&inter_tdc.mu);
			createa(external.n_molec_type + 2 + exter_tdc.n_atom_type,&exter_tdc.frozen);
			createv(external.n_molec_type,&inter_tdc.mol_wv);
			createm(exter_tdc.n_atom_type,external.n_molec_type,&inter_tdc.a);

			fscanf_safe(infile,"\natom_weights %lf",&exter_tdc.atom_weights[0]);//To prevent fscanf_safe r=0 error
			for (l = 1; l < exter_tdc.n_atom_type; l++)
			{
				fscanf_safe(infile," %lf",&exter_tdc.atom_weights[l]);
			}

			for (j = 0; j < external.n_molec_type; j++)
			{
				fscanf_safe(infile,"\n==========\nmole_fraction %lf %i",&external.mol_fr[j],&exter_tdc.frozen[j]);
				fscanf_safe(infile,"\nmole_composition %i",&exter_tdc.mole_compos[j][0]);//To prevent fscanf_safe r=0 error
				for (l = 1; l < exter_tdc.n_atom_type; l++)
				{
					fscanf_safe(infile," %i",&exter_tdc.mole_compos[j][l]);
				}
				fscanf_safe(infile,"\ncalor_style %i",&exter_tdc.calor_style[j]);
				switch (exter_tdc.calor_style[j])
				{
					case 1:
						_cp0[j] = janaf_cp0;
						_h0[j] = janaf_h0;
						_s0[j] = janaf_s0;
						fscanf_safe(infile,"\nA %lf",&exter_tdc.cal_parms[j][0]);
						fscanf_safe(infile,"\nB %lf",&exter_tdc.cal_parms[j][1]);
						fscanf_safe(infile,"\nC %lf",&exter_tdc.cal_parms[j][2]);
						fscanf_safe(infile,"\nD %lf",&exter_tdc.cal_parms[j][3]);
						fscanf_safe(infile,"\nE %lf",&exter_tdc.cal_parms[j][4]);
						fscanf_safe(infile,"\nF %lf",&exter_tdc.cal_parms[j][5]);
						fscanf_safe(infile,"\nG %lf",&exter_tdc.cal_parms[j][6]);
						fscanf_safe(infile,"\nH %lf",&exter_tdc.cal_parms[j][7]);
						break;
					case 2:
						//exter_tdc.thermo_path = ;
						_cp0[j] = janaf_cp0;
						_h0[j] = janaf_h0;
						_s0[j] = janaf_s0;
						fscanf_safe(infile,"janaf_inputfile %s",&exter_tdc.thermo_path);
						//TODO:JANAF parser
						break;
					case 3:
						//exter_tdc.thermo_path = ;
						_cp0[j] = ivtan_cp0;
						_h0[j] = ivtan_h0;
						_s0[j] = ivtan_s0;
						fscanf_safe(infile,"\ndH0f[kJ/mol] %lf",&exter_tdc.cal_parms[j][0]);
						fscanf_safe(infile,"\ndH00[kJ/mol] %lf",&exter_tdc.cal_parms[j][1]);
						fscanf_safe(infile,"\nf1 %lf",&exter_tdc.cal_parms[j][2]);
						fscanf_safe(infile,"\nf2 %lf",&exter_tdc.cal_parms[j][3]);
						fscanf_safe(infile,"\nf3 %lf",&exter_tdc.cal_parms[j][4]);
						fscanf_safe(infile,"\nf4 %lf",&exter_tdc.cal_parms[j][5]);
						fscanf_safe(infile,"\nf5 %lf",&exter_tdc.cal_parms[j][6]);
						fscanf_safe(infile,"\nf6 %lf",&exter_tdc.cal_parms[j][7]);
						fscanf_safe(infile,"\nf7 %lf",&exter_tdc.cal_parms[j][8]);
						break;
					case 4:
						//exter_tdc.thermo_path = ;
						_cp0[j] = ivtan_cp0;
						_h0[j] = ivtan_h0;
						_s0[j] = ivtan_s0;
						fscanf_safe(infile,"ivtan_inputfile %s",&exter_tdc.thermo_path);
						//TODO:IVTAN parser
						break;

				}
				fscanf_safe(infile,"\ninitial_self-const %lf",&external.a0[j]);
				for (k = j; k < external.n_molec_type; k++)
				{
					internal.pair_count++;
					fscanf_safe(infile,"\nget_potential %i",external.poten_type[j][k]);
					for (l = 0; l < MAX_PAR_COUNT; l++)
					{
						fscanf_safe(infile," %lf", &external.p[j][k][l]);
					}
					fscanf_safe(infile,"\nget_closure %i %i",external.cl_type[j][k],external.sw_type[j][k]);
					fscanf_safe(infile,"\nfind_max %lf %lf %i",external.rfe[j][k],external.d[j][k],external.itermax[j][k]);
					fscanf_safe(infile,"\nanti-aliasing_potential %i %i %lf",external.Nphi[j][k],external.r_type[j][k],external.lmb[j][k]);
				}
			}

			//Reduction of dimensions
#ifdef AUTO_SCALE
			for (j = 0; j < external.n_molec_type; j++)
			{
				for (k = j; k < external.n_molec_type; k++)
				{
					if (external.p[j][k][0] > inter_tdc.e_scale)
					{
						inter_tdc.e_scale = external.p[j][k][0];// Use max e as e_scale
						inter_tdc.r_scale = external.p[j][k][1];// Use respecpive rm as r_scale
					}
				}
			}
#endif

			for (j = 0; j < external.n_molec_type; j++)
			{
				for (k = j; k < external.n_molec_type; k++)
				{
					external.p[j][k][0] = external.p[j][k][0] / inter_tdc.e_scale; // [e] K
					external.p[j][k][1] = external.p[j][k][1] / inter_tdc.r_scale; // [rm] A
				}
			}

			exter_tdc.initials[0] = exter_tdc.initials[0] / inter_tdc.e_scale; // [T_in] K
			exter_tdc.initials[1] = exter_tdc.initials[1] * pow(inter_tdc.r_scale / pow(10.0, 8.0), 3.0) / mole_w(external.mol_fr) * Na; // [rho_in] g/cc

			switch (exter_tdc.problem_type)
			{
				case 0:
					//EOS
					break;
				case 1:
					//TV
					exter_tdc.initials[2] = exter_tdc.initials[0]; // [T] 1 already
					exter_tdc.initials[3] = exter_tdc.initials[1]; // rho 1 already
					exter_tdc.frozen[external.n_molec_type + 0] = 0;
					exter_tdc.frozen[external.n_molec_type + 1] = 0;
					break;
				case 2:
					//TP
					exter_tdc.initials[2] = exter_tdc.initials[0]; // [T] 1 already
					exter_tdc.initials[3] = exter_tdc.initials[3]; // TODO [P] Pa
					exter_tdc.frozen[external.n_molec_type + 0] = 0;
					exter_tdc.frozen[external.n_molec_type + 1] = 1;
					break;
				case 3:
					//UV
					exter_tdc.initials[2] = exter_tdc.initials[2]; // TODO [U] J/mol
					exter_tdc.initials[3] = exter_tdc.initials[1]; // [rho] 1 already
					exter_tdc.frozen[external.n_molec_type + 0] = 1;
					exter_tdc.frozen[external.n_molec_type + 1] = 0;
					break;
				case 4:
					//HP
					exter_tdc.initials[2] = exter_tdc.initials[2]; // TODO [H] J/mol
					exter_tdc.initials[3] = exter_tdc.initials[3]; // TODO [P] Pa
					exter_tdc.frozen[external.n_molec_type + 0] = 1;
					exter_tdc.frozen[external.n_molec_type + 1] = 1;
					break;
				case 5:
					//SV
					exter_tdc.initials[2] = exter_tdc.initials[2]; // TODO [S] J/(mol*K)
					exter_tdc.initials[3] = exter_tdc.initials[1]; // [rho] 1 already
					exter_tdc.frozen[external.n_molec_type + 0] = 1;
					exter_tdc.frozen[external.n_molec_type + 1] = 0;
					break;
				case 6:
					//SP
					exter_tdc.initials[2] = exter_tdc.initials[2]; // TODO [S] J/(mol*K)
					exter_tdc.initials[3] = exter_tdc.initials[3]; // TODO [P] Pa
					exter_tdc.frozen[external.n_molec_type + 0] = 1;
					exter_tdc.frozen[external.n_molec_type + 1] = 1;
					break;
				case 7:
					//Sh
					exter_tdc.initials[2] = exter_tdc.initials[2]; // TODO [T0] K
					exter_tdc.initials[3] = exter_tdc.initials[3]; // TODO [rho0] g/cc
					//exter_tdc.initials[4]; // [rho/rho0] 1
					exter_tdc.frozen[external.n_molec_type + 0] = 0;
					exter_tdc.frozen[external.n_molec_type + 1] = 0;
					break;
				case 8:
					//D
					break;
			}

			// Set no freeze for additional Lagrange variables
			for (i = 0; i < exter_tdc.n_atom_type; i++)
			{
				exter_tdc.frozen[external.n_molec_type + 2 + i] = 1;
			}

			//Calculate molecular weights and a[i][j]
			for (i = 0; i < external.n_molec_type; i++)
			{
				inter_tdc.mol_wv[i] = 0.0;
				for (j = 0; j < exter_tdc.n_atom_type; j++)
				{
					inter_tdc.a[j][i] = exter_tdc.atom_weights[j] * (double)exter_tdc.mole_compos[i][j];
					inter_tdc.mol_wv[i] += inter_tdc.a[j][i];
				}
			}

			fscanf_safe(infile,"\n==========\nget_sheme %i %i %i %lf %i %lf %i %lf %i %lf %lf %lf",&exter_tdc.Nnewt,&external.Na,&external.Nrho,&exter_tdc.deltanewt,&exter_tdc.itermaxnewt,&external.dK,&external.itermaxbiggest,&external.delta,&external.itermaxbig,&exter_tdc.dnewt,&external.drho,&external.da);
			external.maxabsdlg = 2.0;

			if ((external.Nrho != 4)&&(external.Nrho != 2))
			{
				fprintf(logfile,"Nrho should be 2 or 4, but now it's %i\n", external.Nrho);
				continue;
			}

			if ((external.Na != 4)&&(external.Na != 2))
			{
				fprintf(logfile,"Na should be 2 or 4, but now it's %i\n", external.Na);
				continue;
			}

			if (*path1)
				file1 = fopen(path1, "r"/*read only*/);

			if (file1 == NULL)
			{
				fprintf(logfile,"InpRes file is not already exist!\n");
				external.inp__type = 3;
			#ifndef OUTRESFILES
				solver_tdc(logfile,file2,file1);
			#endif
			#ifdef OUTRESFILES
				int flag1;
				flag1 = 0;
				while (flag1 == 0)
				{
					file2 = fopen(path2,"r"/*read only*/);
					if (file2 == NULL)
					{
						file2 = fopen(path2,"w");
						solver_tdc(logfile,file2,file1);
						flag1 = 1;
						fclose(file2);
					}
					else
					{
						fclose(file2);
						fprintf(logfile,"OutRes file is already exist!Creation of new directory\n");
						strcat(path2,"_1");
					}
				}
			#endif
			}
			else
			{
			#ifndef OUTRESFILES
				solver_tdc(logfile,file2,file1);
			#endif
			#ifdef OUTRESFILES
				int flag1;
				flag1 = 0;
				while (flag1 == 0)
				{
					file2 = fopen(path2,"r"/*read only*/);
					if (file2 == NULL)
					{
						file2 = fopen(path2,"w");
						solver_tdc(logfile,file2,file1);
						flag1 = 1;
						fclose(file2);
					}
					else
					{
						fprintf(logfile,"OutRes file is already exist!Creation of new directory\n");
						strcat(path2,"_1");
					}
				}
			#endif
				fclose(file1);
			}

			freem(exter_tdc.n_atom_type,inter_tdc.a);
			free(inter_tdc.mol_wv);
			free(exter_tdc.frozen);
			free(inter_tdc.mu);
			free(inter_tdc.mu_caloric);
			free(inter_tdc.mu_thermal_ex);
			free(inter_tdc.mu_thermal_id);
			freem(external.n_molec_type,exter_tdc.cal_parms);
			free(exter_tdc.atom_weights);
			free(exter_tdc.calor_style);
			freema(external.n_molec_type,exter_tdc.mole_compos);
			free(external.mol_fr);
			freem_tri(external.n_molec_type,external.p);
			free(external.a0);
			freev_tri(external.n_molec_type,external.rfe);
			freev_tri(external.n_molec_type,external.d);
			freev_tri(external.n_molec_type,external.lmb);
			freea_tri(external.n_molec_type,external.itermax);
			freea_tri(external.n_molec_type,external.Nphi);
			freea_tri(external.n_molec_type,external.r_type);
			freea_tri(external.n_molec_type,external.cl_type);
			freea_tri(external.n_molec_type,external.sw_type);
			freea_tri(external.n_molec_type,external.poten_type);
			free(exter_tdc.initials);
		}
		else
		{
			fprintf(logfile,"\nend");
			flag = 1;
		}
	}
	return 0;
}

int main(int argc, char *argv[])
{
	FILE *logfile, *infile;

	if (argc <= 2)
	{
		printf("Not enought arguments! I need logfile and infile paths\n");
	}
	else
	{
		logfile = fopen(argv[1],"r"/*read only*/);

		if (logfile == 0)
		{
			logfile = fopen(argv[1],"w"/*create and write only*/);
			fprintf(logfile,"SCOZA TS v0.1\n");
			infile = fopen(argv[2],"r"/*read only*/);

			if (infile == 0)
			{
				printf("Infile is not already exist!\n");
			}
			else
			{
				inter(logfile,infile);
				fclose(infile);
				fclose(logfile);
			}
		}
		else
		{
			fclose(logfile);
			printf("Logfile is already exists!\n");
		}
	}
	return 0;
}
