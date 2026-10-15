#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

////////// COMPILATION DIRECTIVES //////////
#define _SUBTRACT_MEAN_POISSON_ 1				/// Activate subtract mean from phi in Poisson solver
#define _OUTPUT_HOOK_FUNCTION_ 0				/// Activate output hook function diagnostics

/// AUXILIAR PARAMETERS ///
const double pi = 2.0*asin(1.0);	                        /// pi number (fixed)
const int fstream_precision = 15;               		/// Fstream precision (fixed)

/// PROBLEM PARAMETERS ///
const double R_specific = 287.058;				/// Specific gas constant
const double gamma_0    = 1.4;					/// Heat capacity ratio
//const double c_p        = gamma_0*R_specific/( gamma_0 - 1.0 );	/// Isobaric heat capacity
const double Re_lambda	= 38.0;					/// Reynolds number based on the Taylor microscale
const double Re_L	= Re_lambda*Re_lambda/15.0;		/// Reynolds number based on the integral microscale
const double Ma_t	= 0.049;				/// Turbulent Mach number
//const double Pr         = 0.71;					/// Prandtl number
const double T_0	= 273.15;				/// Reference temperature
const double P_0	= 101325.0;				/// Reference pressure
const double rho_0	= P_0/(R_specific*T_0);			/// Reference density
const double mu_0	= 1.716e-05;				/// Dynamic viscosity at T_0 
const double u_0	= Ma_t*sqrt(gamma_0*R_specific*T_0/3.0);/// Root-mean-square velocity fluctuations
const double L_0	= (mu_0/P_0)*(Re_L/Ma_t)*sqrt(3.0*R_specific*T_0/gamma_0);	/// Integral lengthscale 
//const double t_0	= L_0/u_0;				/// Large-eddy turn-over time
const double L_box 	= 5.5*L_0;				/// Domain size
const double epsilon_0  = u_0*u_0*u_0/L_0;			/// Energy dissipation rate[m2/s3]
const double alpha_P    = 0.00001;                              /// Magnitude of pressure perturbations
const double tau_L 	= L_0/u_0;				/// Timescale based on integral lengthscale
const double num_modes  = 200.0; 				/// Number of modes for the initial condition

/// FORCING PARAMETERS ///
const double p = 2.0;						/// Parameter to adjust the forcing strength ... it has to be p >= 1
const double norm_tol = 1.0e-4;					/// Normalized tolerance of the iterative Poisson solver
const int ite_max = 1e8;					/// Maximum number of iterations for the Poisson solver
const double weight_SOR = 1.6;					/// Relaxation weight of the Poisson solver

////////// myRHEA CLASS //////////

double myRHEA::passotPouquetTurbulentFlowSpectrum(const double &k) {

	/// Passot-Pouquet turbulent flow spectrum
  	/// http://www.wseas.us/e-library/conferences/2011/Corfu/ASM/ASM-19.pdf
	
	/// Parameters pq spectrum
	double ke = 2.0*pi/(L/50.0);		/// Wavelength based on box size divided by 50
	double uavg = u_0;
	
	/// Energy in the corresponding band
	double kke = k/ke;
	double espec = ( 16.0*uavg*uavg/ke )*sqrt( 2.0/pi )*( kke*kke*kke*kke )*exp( -2.0*kke*kke );

	return espec;

};

void myRHEA::setInitialConditions() {
            
    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    /// srand( my_rank );
    srand( 0 );			/// Impose same seed to obtain same random angles in each partition

    /// All (inner, halo, boundary): u, v, w, P and T
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
		u_field[I1D(i,j,k)] = 0.0;
                v_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = 0.0;
                P_field[I1D(i,j,k)] = P_0;	
		T_field[I1D(i,j,k)] = thermodynamics->calculateTemperatureFromPressureDensity( P_field[I1D(i,j,k)], rho_0 );
		/// Fields created in myRHEA
		phi_field[I1D(i,j,k)] = 0.0;
            }
        }
    }
    
    double wn1 = 2.0*pi/L;			/// Smallest wavenumber that can be represented on the grid
    double wnn = pi/Delta; 			/// Highest wavenumber that can be represented on the grid (nyquist limit)
    double dk  = ( wnn - wn1 )/num_modes;	/// Wave number step

    for(int m = 0; m < num_modes; m++) {
	/// Compute random angles
	double phi_angle   = 2.0*pi*( (double) rand()/( RAND_MAX ) );
	double nu_angle    = ( (double) rand()/( RAND_MAX ) );
	double theta_angle = acos( 2.0*nu_angle - 1.0 );
	double psi_angle   = pi*( (double) rand()/( RAND_MAX ) ) - pi/2.0;
	
	/// Wavenumber at cell center
	double wn = wn1 + 0.5*dk + m*dk;
	
	/// Wavenumber vector from random angles
	double kx = sin( theta_angle )*cos( phi_angle )*wn;
	double ky = sin( theta_angle )*sin( phi_angle )*wn;
	double kz = cos( theta_angle )*wn;

	/// Divergence vector
	double ktx = sin( kx*Delta/2.0 )/Delta;
	double kty = sin( ky*Delta/2.0 )/Delta;
	double ktz = sin( kz*Delta/2.0 )/Delta;

	/// Enforce mass conservation
	double phi_1   = 2.0*pi*( (double) rand()/( RAND_MAX ) );
	double nu_1    = ( (double) rand()/( RAND_MAX ) );
	double theta_1 = acos( 2.0*nu_1 - 1.0 );
	
	double zetax = sin( theta_1 )*cos( phi_1 );
	double zetay = sin( theta_1 )*sin( phi_1 );
	double zetaz = cos( theta_1 );

	double sxm = zetay*ktz - zetaz*kty;
	double sym = -(zetax*ktz - zetaz*ktx);
	double szm = zetax*kty - zetay*ktx;
	double smag = sqrt( sxm*sxm + sym*sym + szm*szm );
	sxm = sxm/smag;
	sym = sym/smag;
	szm = szm/smag;
	
    	double espec_m = this->passotPouquetTurbulentFlowSpectrum( wn );	/// Energy at the wavenumber
	espec_m = max( espec_m, 0.0 ); 						/// Clip espec
	double um = sqrt( espec_m*dk );						/// Energy contained in a band
    		
	for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            	for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
		    double arg = kx*( x_field[I1D(i,j,k)] ) + ky*( y_field[I1D(i,j,k)] ) + kz*( z_field[I1D(i,j,k)] ) - psi_angle;
		    double bmx = 2.0*um*cos( arg - kx*Delta/2.0 );
		    double bmy = 2.0*um*cos( arg - ky*Delta/2.0 );
		    double bmz = 2.0*um*cos( arg - kz*Delta/2.0 );
		    u_field[I1D(i,j,k)] += bmx*sxm;
		    v_field[I1D(i,j,k)] += bmy*sym;
		    w_field[I1D(i,j,k)] += bmz*szm;
		}
	    }
	}
    }
    
    #pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_], phi_field.vector[0:_ls_])
    /// Update halo values
    u_field.update();
    v_field.update();
    w_field.update();
    P_field.update();
    T_field.update();
    phi_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_], phi_field.vector[0:_ls_])
#endif

};

void myRHEA::calculateSourceTerms() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration
    
    /// Evaluate w_i = sqrt(rho)*u_i, its time-dependent volume average and the volume average of u_i, u_i_u_i and u_i*( var_P/var_i )
    
    /// Calculate local values
    double local_sum_u_volume = 0.0;
    double local_sum_v_volume = 0.0;
    double local_sum_w_volume = 0.0;
    double local_sum_u_i_u_i_volume = 0.0;
    double local_sum_u_i_var_P_var_i_volume = 0.0;
    double local_sum_w_x_volume = 0.0;
    double local_sum_w_y_volume = 0.0;
    double local_sum_w_z_volume = 0.0;

    /// Inner points: w_i, u_i, u_i_u_i, u_i_var_P_var_i & volume
    #pragma acc parallel loop collapse(3) present(this, rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], w_x_field.vector[0:_ls_], w_y_field.vector[0:_ls_], w_z_field.vector[0:_ls_]) reduction(+:local_sum_u_volume) reduction(+:local_sum_v_volume) reduction(+:local_sum_w_volume) reduction(+:local_sum_u_i_u_i_volume) reduction(+:local_sum_u_i_var_P_var_i_volume) reduction(+:local_sum_w_x_volume) reduction(+:local_sum_w_y_volume) reduction(+:local_sum_w_z_volume)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Evaluate w_i fields
		w_x_field[I1D(i,j,k)] = sqrt( rho_field[I1D(i,j,k)] )*u_field[I1D(i,j,k)];
		w_y_field[I1D(i,j,k)] = sqrt( rho_field[I1D(i,j,k)] )*v_field[I1D(i,j,k)];
		w_z_field[I1D(i,j,k)] = sqrt( rho_field[I1D(i,j,k)] )*w_field[I1D(i,j,k)];
		/// Sum values
		local_sum_u_volume += ( u_field[I1D(i,j,k)] )*cell_volume; 
		local_sum_v_volume += ( v_field[I1D(i,j,k)] )*cell_volume; 
		local_sum_w_volume += ( w_field[I1D(i,j,k)] )*cell_volume; 
		local_sum_u_i_u_i_volume += ( pow( u_field[I1D(i,j,k)], 2.0 ) + pow( v_field[I1D(i,j,k)], 2.0 ) + pow( w_field[I1D(i,j,k)], 2.0 ) )*cell_volume;
		local_sum_u_i_var_P_var_i_volume += ( u_field[I1D(i,j,k)]*( ( P_field[I1D(i+1,j,k)] - P_field[I1D(i-1,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ) )
                                                    + v_field[I1D(i,j,k)]*( ( P_field[I1D(i,j+1,k)] - P_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ) ) 
                                                    + w_field[I1D(i,j,k)]*( ( P_field[I1D(i,j,k+1)] - P_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] ) ) )*cell_volume;
                local_sum_w_x_volume += w_x_field[I1D(i,j,k)]*cell_volume;
                local_sum_w_y_volume += w_y_field[I1D(i,j,k)]*cell_volume;
                local_sum_w_z_volume += w_z_field[I1D(i,j,k)]*cell_volume;
            }
        }
    }

    /// Update halo values
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(w_x_field.vector[0:_ls_], w_y_field.vector[0:_ls_],w_z_field.vector[0:_ls_])
#endif
    w_x_field.update();
    w_y_field.update();
    w_z_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(w_x_field.vector[0:_ls_], w_y_field.vector[0:_ls_],w_z_field.vector[0:_ls_])
#endif

    /// Communicate local values to obtain global & volume average values
    double global_sum_u_volume;
    MPI_Allreduce(&local_sum_u_volume, &global_sum_u_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_v_volume;
    MPI_Allreduce(&local_sum_v_volume, &global_sum_v_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_w_volume;
    MPI_Allreduce(&local_sum_w_volume, &global_sum_w_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_u_i_u_i_volume;
    MPI_Allreduce(&local_sum_u_i_u_i_volume, &global_sum_u_i_u_i_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_u_i_var_P_var_i_volume;
    MPI_Allreduce(&local_sum_u_i_var_P_var_i_volume, &global_sum_u_i_var_P_var_i_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_w_x_volume;
    MPI_Allreduce(&local_sum_w_x_volume, &global_sum_w_x_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_w_y_volume;
    MPI_Allreduce(&local_sum_w_y_volume, &global_sum_w_y_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_w_z_volume;
    MPI_Allreduce(&local_sum_w_z_volume, &global_sum_w_z_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double vol_avg_u = global_sum_u_volume/domain_volume;
    double vol_avg_v = global_sum_v_volume/domain_volume;
    double vol_avg_w = global_sum_w_volume/domain_volume;
    double vol_avg_u_i_u_i = global_sum_u_i_u_i_volume/domain_volume;
    double vol_avg_u_i_var_P_var_i = global_sum_u_i_var_P_var_i_volume/domain_volume;
    double vol_avg_w_x = global_sum_w_x_volume/domain_volume;
    double vol_avg_w_y = global_sum_w_y_volume/domain_volume;
    double vol_avg_w_z = global_sum_w_z_volume/domain_volume;

    double sum_norm_vel_var = ( ( vol_avg_u_i_u_i - ( vol_avg_u*vol_avg_u + vol_avg_v*vol_avg_v + vol_avg_w*vol_avg_w ) )/(3.0*u_0*u_0) ); 	/// Normalized velocity variance (u_rms_normalized^2) 
    double alpha_k = pow( sum_norm_vel_var, ( -1.0 )*p ) ;
    
    /// Solve Poisson's equation: laplacian(phi) = div(w), to obtain the irrotational component of the modified velocity. The solenoidal component is obtained as w_S = w - w_D - <w>
    this->solveScalarPoisson();

    /// Evaluate w_S_i and its time-dependent volume average square product
    double local_sum_w_S_i_w_S_i_volume = 0.0;
    double d_phi_dx, d_phi_dy, d_phi_dz;
    #pragma acc parallel loop collapse(3) present(this, phi_field.vector[0:_ls_], w_S_x_field.vector[0:_ls_], w_S_y_field.vector[0:_ls_], w_S_z_field.vector[0:_ls_]) reduction(+:local_sum_w_S_i_w_S_i_volume)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
		/// Gradients of the scalar potential
		d_phi_dx = ( phi_field[I1D(i+1,j,k)] - phi_field[I1D(i-1,j,k)] ) / (2.0*Delta);
		d_phi_dy = ( phi_field[I1D(i,j+1,k)] - phi_field[I1D(i,j-1,k)] ) / (2.0*Delta);
		d_phi_dz = ( phi_field[I1D(i,j,k+1)] - phi_field[I1D(i,j,k-1)] ) / (2.0*Delta);
                /// Calculate w^S_i = curl ( velocity potential ) 
		w_S_x_field[I1D(i,j,k)] = w_x_field[I1D(i,j,k)] - d_phi_dx - vol_avg_w_x;
		w_S_y_field[I1D(i,j,k)] = w_y_field[I1D(i,j,k)] - d_phi_dy - vol_avg_w_y;
		w_S_z_field[I1D(i,j,k)] = w_z_field[I1D(i,j,k)] - d_phi_dz - vol_avg_w_z;
		/// Sum values
                local_sum_w_S_i_w_S_i_volume += ( pow( w_S_x_field[I1D(i,j,k)], 2.0 ) + pow( w_S_y_field[I1D(i,j,k)], 2.0 ) + pow( w_S_z_field[I1D(i,j,k)], 2.0 ) )*cell_volume;
            }
        }
    }

    /// Update halo values
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(w_S_x_field.vector[0:_ls_], w_S_y_field.vector[0:_ls_],w_S_z_field.vector[0:_ls_])
#endif
    w_S_x_field.update();
    w_S_y_field.update();
    w_S_z_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(w_S_x_field.vector[0:_ls_], w_S_y_field.vector[0:_ls_],w_S_z_field.vector[0:_ls_])
#endif

    /// Communicate local values to obtain global & volume average values
    double global_sum_w_S_i_w_S_i_volume;
    MPI_Allreduce(&local_sum_w_S_i_w_S_i_volume, &global_sum_w_S_i_w_S_i_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double vol_avg_w_S_i_w_S_i = global_sum_w_S_i_w_S_i_volume/domain_volume;
    
    /// Calculate C_S
    double C_S = ( vol_avg_u_i_var_P_var_i + rho_0*epsilon_0 )/( vol_avg_w_S_i_w_S_i + 1.0e-10 );
   
    /// Inner points: f_rhou, f_rhov, f_rhow and ensemble of the work of the momentum forces
    double f_rhouvw;
    double local_sum_work_momentum = 0.0;
    #pragma acc parallel loop collapse(3) present(this, rho_field.vector[0:_ls_], f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], w_S_x_field.vector[0:_ls_], w_S_y_field.vector[0:_ls_], w_S_z_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_]) reduction(+:local_sum_work_momentum)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
		/// Evaluate momentum forces
		f_rhou_field[I1D(i,j,k)] = C_S*alpha_k*sqrt( rho_field[I1D(i,j,k)] )*w_S_x_field[I1D(i,j,k)];
                f_rhov_field[I1D(i,j,k)] = C_S*alpha_k*sqrt( rho_field[I1D(i,j,k)] )*w_S_y_field[I1D(i,j,k)];
                f_rhow_field[I1D(i,j,k)] = C_S*alpha_k*sqrt( rho_field[I1D(i,j,k)] )*w_S_z_field[I1D(i,j,k)];
                /// Work of momentum sources
                f_rhouvw = f_rhou_field[I1D(i,j,k)]*u_field[I1D(i,j,k)] + f_rhov_field[I1D(i,j,k)]*v_field[I1D(i,j,k)] + f_rhow_field[I1D(i,j,k)]*w_field[I1D(i,j,k)];
		local_sum_work_momentum += f_rhouvw*cell_volume;
            }
        }
    }
    
    double global_sum_work_momentum;
    MPI_Allreduce(&local_sum_work_momentum, &global_sum_work_momentum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double vol_avg_work_momentum = global_sum_work_momentum/domain_volume;

    /// Inner points: f_rhoE
    #pragma acc parallel loop collapse(3) present(this, f_rhoE_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Evaluate energy forces
                f_rhoE_field[I1D(i,j,k)] = ( -1.0 )*vol_avg_work_momentum;
            }
        }
    }

    /// Update halo values
    //f_rhou_field.update();
    //f_rhov_field.update();
    //f_rhow_field.update();
    //f_rhoE_field.update();
 
};

void myRHEA::solveScalarPoisson() {

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /// Calculate RHS of the Poisson equation:
    /// lap(phi) = ( phi_w + phi_e + phi_s + phi_n + phi_b + phi_f - 6*phi_p )/(Delta^2) = div(w)
    /// ( phi_w + phi_e + phi_s + phi_n + phi_b + phi_f - 6*phi_p ) = Delta^2*div(w) = (Delta/2)*(w_x_e - w_x_w + w_y_n - w_y_s + w_z_f - w_z_b)
    /// double dw_x_dx, dw_y_dy, dw_z_dz;
    #pragma acc parallel loop collapse(3) present( phi_source_field.vector[0:_ls_],  w_x_field.vector[0:_ls_], w_y_field.vector[0:_ls_], w_z_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
		///dw_x_dx = ( w_x_field[I1D(i+1,j,k)] - w_x_field[I1D(i-1,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] );
		///dw_y_dy = ( w_y_field[I1D(i,j+1,k)] - w_y_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] );
		///dw_z_dz = ( w_z_field[I1D(i,j,k+1)] - w_z_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
		phi_source_field[I1D(i,j,k)] = (Delta/2.0)*( w_x_field[I1D(i+1,j,k)] - w_x_field[I1D(i-1,j,k)] + w_y_field[I1D(i,j+1,k)] - w_y_field[I1D(i,j-1,k)] + w_z_field[I1D(i,j,k+1)] - w_z_field[I1D(i,j,k-1)] );
   	    }
	}
    }

    /// SOR Method: x^{n+1} = x^{n} + weight_SOR*(x_hat^{n+1}-x^{n})
    /// Convergence criteria:
    // residual = || r ||*(L_ref/(rho_ref*u_ref))  < epsilon

    double global_rel_error = norm_tol + 1.0;
    double local_abs_error;
    int ite = 0;
    while( global_rel_error > norm_tol && ite < ite_max ) {

	double phi_hat;
	/// Red/Even cells
        #pragma acc parallel loop collapse(3) present(this, phi_source_field.vector[0:_ls_], phi_field.vector[0:_ls_])
	for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
            	    if ((i + j + k) % 2 == 0) {  // Red points
			/// Assuming uniform grid spacing
    		    	phi_hat = ( phi_field[I1D(i+1,j,k)] + phi_field[I1D(i-1,j,k)]
          	    	          + phi_field[I1D(i,j+1,k)] + phi_field[I1D(i,j-1,k)]
    		    	          + phi_field[I1D(i,j,k+1)] + phi_field[I1D(i,j,k-1)] - phi_source_field[I1D(i,j,k)] ) / ( 6.0 );
    		    	/// Update values
    		    	phi_field[I1D(i,j,k)] = ( 1.0 - weight_SOR )*phi_field[I1D(i,j,k)] + ( weight_SOR * phi_hat );

    		    	/// if( my_rank==0 && i==1 && j==1 && k==1 ) { phi_field[I1D(i,j,k)] = 0.0; } /// Make laplacian non-singular
		    }
		}
   	    }
        }
        
        /// Update halo values
#if _GPU_AWARE_MPI_DEACTIVATED_ 
        #pragma acc update host(phi_field.vector[0:_ls_])
#endif
        phi_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
        #pragma acc update device(phi_field.vector[0:_ls_])
#endif

	/// Black/Odd cells
        #pragma acc parallel loop collapse(3) present(this, phi_source_field.vector[0:_ls_], phi_field.vector[0:_ls_])
	for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
            	    if ((i + j + k) % 2 == 1) {  // Black points
			/// Assuming uniform grid spacing
    		    	phi_hat = ( phi_field[I1D(i+1,j,k)] + phi_field[I1D(i-1,j,k)]
          	    	          + phi_field[I1D(i,j+1,k)] + phi_field[I1D(i,j-1,k)]
    		    	          + phi_field[I1D(i,j,k+1)] + phi_field[I1D(i,j,k-1)] - phi_source_field[I1D(i,j,k)] ) / ( 6.0 );
    		    	/// Update values
    		    	phi_field[I1D(i,j,k)] = ( 1.0 - weight_SOR )*phi_field[I1D(i,j,k)] + ( weight_SOR * phi_hat );

    		    	/// if( my_rank==0 && i==1 && j==1 && k==1 ) { phi_field[I1D(i,j,k)] = 0.0; } /// Make laplacian non-singular
		    }
		}
   	    }
        }

        /// Update halo values
#if _GPU_AWARE_MPI_DEACTIVATED_ 
        #pragma acc update host(phi_field.vector[0:_ls_])
#endif
        phi_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
        #pragma acc update device(phi_field.vector[0:_ls_])
#endif

        /// Residual calculation
        /// Set errors to 0
        double local_sum_phi = 0.0;
        double local_sum_res_phi = 0.0;
        double local_res;
        #pragma acc parallel loop collapse(3) present(this, phi_source_field.vector[0:_ls_], phi_field.vector[0:_ls_]) reduction(+: local_sum_phi) reduction(+:local_sum_res_phi)
        for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
            	local_res = ( phi_field[I1D(i+1,j,k)] + phi_field[I1D(i-1,j,k)]
                                + phi_field[I1D(i,j+1,k)] + phi_field[I1D(i,j-1,k)]
                                + phi_field[I1D(i,j,k+1)] + phi_field[I1D(i,j,k-1)] - 6.0*phi_field[I1D(i,j,k)] - phi_source_field[I1D(i,j,k)] );
                    local_sum_phi += phi_field[I1D(i,j,k)]*phi_field[I1D(i,j,k)];
                    local_sum_res_phi += local_res*local_res;
                }
            }
        }

        double global_sum_res_phi;
        MPI_Allreduce(&local_sum_res_phi, &global_sum_res_phi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_phi;
        MPI_Allreduce(&local_sum_phi, &global_sum_phi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double residual_normalized = sqrt(global_sum_res_phi)*(L_0/(sqrt(rho_0)*u_0)); /// ||r|| = ||r_dimensional||*(L_0/(rho_0*u_0))
        double residual_using_phi  = sqrt(global_sum_res_phi/global_sum_phi);
        global_rel_error = max( residual_using_phi, residual_normalized );
        ite++;
        ///if ( ite%500 == 0 ) { if (my_rank == 0) { cout << "SOR Iteration number: " << ite <<". Relative error: " << global_rel_error << ". Relative tolerance: " << norm_tol << endl; } }

    }

#if _SUBTRACT_MEAN_POISSON_
    double local_sum_mean_phi = 0.0;
    #pragma acc parallel loop collapse(3) present(this, phi_field.vector[0:_ls_]) reduction(+: local_sum_mean_phi) 
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
		local_sum_mean_phi += phi_field[I1D(i,j,k)]*cell_volume;
   	    }
	}
    }
   
    double global_sum_mean_phi;
    MPI_Allreduce(&local_sum_mean_phi, &global_sum_mean_phi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mean_phi = global_sum_mean_phi/domain_volume;
   
    #pragma acc parallel loop collapse(3) present(this, phi_field.vector[0:_ls_]) 
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
		phi_field[I1D(i,j,k)] -= mean_phi;
   	    }
	}
    }
#endif
   
};

void myRHEA::temporalHookFunction() {
    
    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

#if _OUTPUT_HOOK_FUNCTION_
    /// Output variables post-process: <u_i_u_i>, <u_i>, <tauijSij>/<rho>, <mu>, <u_rms>/u_0, tauijSij/(rho_0epsilon_0), <T>
    const string hook_output_file_name = "hook_function_data.csv";		/// Name of hook output file

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if( current_time_iter%25 == 0 ) {

        /// Calculate local values
        double div_uvw, tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
        double d_u_x, d_u_y, d_u_z, d_v_x, d_v_y, d_v_z, d_w_x, d_w_y, d_w_z;
        double div_uvw_tau_rhoe;
        double local_sum_hook_rhoe = 0.0;
        double local_sum_hook_u_i_u_i = 0.0;
        double local_sum_hook_u = 0.0;
        double local_sum_hook_v = 0.0;
        double local_sum_hook_w = 0.0;
        double local_sum_hook_mu  = 0.0;
        double local_sum_hook_T   = 0.0;
        #pragma acc parallel loop collapse(3) present(this, mu_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], T_field.vector[0:_ls_]) reduction(+:local_sum_work_momentum) reduction(+:local_sum_hook_rhoe) reduction(+:local_sum_hook_u) reduction(+:local_sum_hook_v)reduction(+:local_sum_hook_w) reduction(+:local_sum_hook_mu) reduction(+:local_sum_hook_T)
        for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                    /// Velocity derivatives
                    d_u_x = ( u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)] )/( 2.0*Delta );
                    d_u_y = ( u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)] )/( 2.0*Delta );
                    d_u_z = ( u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)] )/( 2.0*Delta );
                    d_v_x = ( v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)] )/( 2.0*Delta );
                    d_v_y = ( v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)] )/( 2.0*Delta );
                    d_v_z = ( v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)] )/( 2.0*Delta );
                    d_w_x = ( w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)] )/( 2.0*Delta );
                    d_w_y = ( w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)] )/( 2.0*Delta );
                    d_w_z = ( w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)] )/( 2.0*Delta );
                    /// Divergence of velocity
                    div_uvw = d_u_x + d_v_y + d_w_z;
                    /// Viscous stresses ( symmetric tensor )
                    tau_xx = 2.0*mu_field[I1D(i,j,k)]*( d_u_x - ( div_uvw/3.0 ) );
                    tau_xy = mu_field[I1D(i,j,k)]*( d_u_y + d_v_x );
                    tau_xz = mu_field[I1D(i,j,k)]*( d_u_z + d_w_x );
                    tau_yy = 2.0*mu_field[I1D(i,j,k)]*( d_v_y - ( div_uvw/3.0 ) );
                    tau_yz = mu_field[I1D(i,j,k)]*( d_v_z + d_w_y );
                    tau_zz = 2.0*mu_field[I1D(i,j,k)]*( d_w_z - ( div_uvw/3.0 ) );
                    /// Work of viscous stresses for internal energy
                    div_uvw_tau_rhoe = tau_xx*d_u_x + tau_xy*d_u_y + tau_xz*d_u_z
                                     + tau_xy*d_v_x + tau_yy*d_v_y + tau_yz*d_v_z
                                     + tau_xz*d_w_x + tau_yz*d_w_y + tau_zz*d_w_z;
                    /// Update volume averaged quantities
                    local_sum_hook_rhoe += div_uvw_tau_rhoe*cell_volume;
                    local_sum_hook_u_i_u_i += ( u_field[I1D(i,j,k)]*u_field[I1D(i,j,k)] + v_field[I1D(i,j,k)]*v_field[I1D(i,j,k)] + w_field[I1D(i,j,k)]*w_field[I1D(i,j,k)] )*cell_volume;
                    local_sum_hook_u += ( u_field[I1D(i,j,k)] )*cell_volume;
                    local_sum_hook_v += ( v_field[I1D(i,j,k)] )*cell_volume;
                    local_sum_hook_w += ( w_field[I1D(i,j,k)] )*cell_volume;
                    local_sum_hook_mu += ( mu_field[I1D(i,j,k)] )*cell_volume;
                    local_sum_hook_T += ( T_field[I1D(i,j,k)] )*cell_volume;
                }
            }
        }

        /// Communicate local values to obtain global & volume average values
        double global_sum_hook_rhoe;
        MPI_Allreduce(&local_sum_hook_rhoe, &global_sum_hook_rhoe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_hook_u_i_u_i;
        MPI_Allreduce(&local_sum_hook_u_i_u_i, &global_sum_hook_u_i_u_i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_hook_u;
        MPI_Allreduce(&local_sum_hook_u, &global_sum_hook_u, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_hook_v;
        MPI_Allreduce(&local_sum_hook_v, &global_sum_hook_v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_hook_w;
        MPI_Allreduce(&local_sum_hook_w, &global_sum_hook_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_hook_mu;
        MPI_Allreduce(&local_sum_hook_mu, &global_sum_hook_mu, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_sum_hook_T;
        MPI_Allreduce(&local_sum_hook_T, &global_sum_hook_T, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double vol_avg_rhoe = global_sum_hook_rhoe /domain_volume;
        double vol_avg_u_i_u_i = global_sum_hook_u_i_u_i /domain_volume;
        double vol_avg_u = global_sum_hook_u /domain_volume;
        double vol_avg_v = global_sum_hook_v /domain_volume;
        double vol_avg_w = global_sum_hook_w /domain_volume;
        double vol_avg_mu = global_sum_hook_mu /domain_volume;
        double vol_avg_T = global_sum_hook_T /domain_volume;

        double u_rms = sqrt ( ( vol_avg_u_i_u_i - ( vol_avg_u*vol_avg_u + vol_avg_v*vol_avg_v +  vol_avg_w*vol_avg_w ))/(3.0));

        if( my_rank == 0 ) {
            /// Check if file exists
            ifstream check_output_file( hook_output_file_name );

            int line_counter = 0;
            while( check_output_file.peek() != EOF ) {
                line_counter++;
                break;
            }
            check_output_file.close();

            /// Open file to write (append)
            fstream output_file( hook_output_file_name, ios::out | ios::app );

            /// Write header if file does not exist
            if( line_counter < 1 ) {
                const string output_header_string = "# time, t_norm, <u_i>2, <u_i_u_i>, rho_0*epsilon_0, tauijSij, <mu>, <T>, u_rms/u_0, tauijSij/(rho_0*epsilon_0)";
                output_file << output_header_string << endl;
            }

            /// Generate data string
            ostringstream sstr; sstr.precision( fstream_precision ); sstr << fixed;
            sstr << current_time;
            sstr << "," << current_time/tau_L;
            sstr << "," << ( vol_avg_u*vol_avg_u + vol_avg_v*vol_avg_v + vol_avg_w*vol_avg_w );
            sstr << "," << vol_avg_u_i_u_i;
            sstr << "," << rho_0*epsilon_0;
            sstr << "," << vol_avg_rhoe;
            sstr << "," << vol_avg_mu;
            sstr << "," << vol_avg_T;
            sstr << "," << u_rms/u_0;
            sstr << "," << vol_avg_rhoe/(rho_0*epsilon_0);
            string output_data_string = sstr.str();

            /// Write data to file
            output_file << output_data_string << endl;
            output_file.close();
        }
    }
#endif

};

void myRHEA::tagImmersedBoundaryMethod() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Set velocity, temperature & interpolation power parameter of IBM
    immersed_boundary_method->setVelocityIBM( 0.0, 0.0, 0.0 );			// [m/s]
    immersed_boundary_method->setTemperatureIBM( 300.0 );			// [K]
    
    /// Set tags of IBM (tagging)

    /// All (inner, halo, boundary): tag_IBM
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                tag_IBM_field[I1D(i,j,k)] = 0.0;
		//tag_IBM_field[I1D(i,j,k)] = 0.5;
		//tag_IBM_field[I1D(i,j,k)] = 1.0;
            }
        }
    }

    /// Update halo values
    tag_IBM_field.update();

};

void myRHEA::setInitialParticlesPositionsVelocities() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration	

    /// If empty (default), particles have already been initialized in random positions uniformly and with zero velocity
   
    /// Skip if there are no particles
    if( point_particles->get_num_prts_total() < 1 ) return;       	

    /// Set positions and velocities of particles
    double x_position_particle, y_position_particle, z_position_particle;
    double u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid;    
    for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

        /// Get Lagrangian positions ... by default, are randonmly distributed
        x_position_particle = point_particles->get_position_0_x_prt( p );
        y_position_particle = point_particles->get_position_0_y_prt( p );
        z_position_particle = point_particles->get_position_0_z_prt( p );	    

        /// Get Lagrangian-Euler values
	this->obtainLagrangianEulerianVelocityDynamicViscosityValues( u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid, p );	

        /// Set Lagrangian positions ... IMPORTANT: particles positions have to be set within the subdomain limits of each task
        point_particles->set_position_0_x_prt( p, x_position_particle );
        point_particles->set_position_0_y_prt( p, y_position_particle );
        point_particles->set_position_0_z_prt( p, z_position_particle );
        point_particles->set_position_x_prt( p, x_position_particle );
        point_particles->set_position_y_prt( p, y_position_particle );
        point_particles->set_position_z_prt( p, z_position_particle );

        /// Set Lagrangian velocities ... initialized with same velocity as fluid
        point_particles->set_velocity_0_x_prt( p, u_velocity_fluid_particle );
        point_particles->set_velocity_0_y_prt( p, v_velocity_fluid_particle );
        point_particles->set_velocity_0_z_prt( p, w_velocity_fluid_particle );
        point_particles->set_velocity_x_prt( p, u_velocity_fluid_particle );
        point_particles->set_velocity_y_prt( p, v_velocity_fluid_particle );
        point_particles->set_velocity_z_prt( p, w_velocity_fluid_particle );

    }

};

void myRHEA::timeAdvanceVelocityPointParticles() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Explicit Euler time-integration of particles velocity
    int i_local_index, j_local_index, k_local_index;
    double x_position_particle, y_position_particle, z_position_particle;
    double u_velocity_particle, v_velocity_particle, w_velocity_particle;
    double u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid, relaxation_time_particle;
    delta_t = this->delta_t;
    int capacity = point_particles->get_prt_capacity();
    #pragma acc parallel loop collapse (1) private(  i_local_index, j_local_index, k_local_index, x_position_particle, y_position_particle, z_position_particle, u_velocity_particle, v_velocity_particle, w_velocity_particle,  u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid, relaxation_time_particle ) present( this, mesh, point_particles, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], mu_field.vector[0:_ls_], point_particles->local_prts_densities[0:capacity], point_particles->local_prts_diameters[0:capacity], point_particles->local_prts_positions_0_x[0:capacity], point_particles->local_prts_positions_0_y[0:capacity], point_particles->local_prts_positions_0_z[0:capacity], point_particles->local_prts_velocities_x[0:capacity], point_particles->local_prts_velocities_y[0:capacity], point_particles->local_prts_velocities_z[0:capacity], point_particles->local_prts_velocities_0_x[0:capacity], point_particles->local_prts_velocities_0_y[0:capacity], point_particles->local_prts_velocities_0_z[0:capacity], point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity] ) copyin( delta_t )
    for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

        /// Obtain Lagrangian-Eulerian indexes 0
        i_local_index = point_particles->local_prts_indexes_0_i[p];
        j_local_index = point_particles->local_prts_indexes_0_j[p];
        k_local_index = point_particles->local_prts_indexes_0_k[p];

        /// Obtain Lagrangian position 0
        x_position_particle = point_particles->local_prts_positions_0_x[p];
        y_position_particle = point_particles->local_prts_positions_0_y[p];
        z_position_particle = point_particles->local_prts_positions_0_z[p];

        /// Obtain Lagrangian velocities 0
	u_velocity_particle = point_particles->local_prts_velocities_0_x[p];
        v_velocity_particle = point_particles->local_prts_velocities_0_y[p];
        w_velocity_particle = point_particles->local_prts_velocities_0_z[p];

        /// Interpolate (trilinear) values: u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid 
        u_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index)], u_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index)], u_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index,k_local_index-1)], u_field[I1D(i_local_index,j_local_index,k_local_index)], u_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index)], u_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index)], u_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
        v_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index)], v_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index)], v_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index,k_local_index-1)], v_field[I1D(i_local_index,j_local_index,k_local_index)], v_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index)], v_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index)], v_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
        w_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index)], w_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index)], w_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index,k_local_index-1)], w_field[I1D(i_local_index,j_local_index,k_local_index)], w_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index)], w_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index)], w_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
        dynamic_viscosity_fluid = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index,k_local_index)], mu_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );

        /// Calculate relaxation time particle
        //relaxation_time_particle = point_particles->calculate_relaxation_time_prt( p, dynamic_viscosity_fluid );
	relaxation_time_particle = ( point_particles->local_prts_densities[p]*pow( point_particles->local_prts_diameters[p], 2.0 ) )/( 18.0*dynamic_viscosity_fluid );

        /// Update particle velocity
        u_velocity_particle = u_velocity_particle + delta_t*( u_velocity_fluid_particle - u_velocity_particle )/relaxation_time_particle;
        v_velocity_particle = v_velocity_particle + delta_t*( v_velocity_fluid_particle - v_velocity_particle )/relaxation_time_particle;
        w_velocity_particle = w_velocity_particle + delta_t*( w_velocity_fluid_particle - w_velocity_particle )/relaxation_time_particle;
	point_particles->local_prts_velocities_x[p] = u_velocity_particle;
        point_particles->local_prts_velocities_y[p] = v_velocity_particle;
        point_particles->local_prts_velocities_z[p] = w_velocity_particle;

    }

};


////////// MAIN //////////
int main(int argc, char** argv) {

    /// Initialize MPI
    MPI_Init(&argc, &argv);

#ifdef _OPENACC
    /// OpenACC distribution on multiple accelerators (GPU)
    acc_device_t my_device_type;
    int num_devices, gpuId, local_rank;
    MPI_Comm shmcomm;    

    MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm );
    MPI_Comm_rank( shmcomm, &local_rank );           
    my_device_type = acc_get_device_type();                      
    num_devices = acc_get_num_devices( my_device_type );
    gpuId = local_rank % num_devices;
    acc_set_device_num( gpuId, my_device_type );
//    /// OpenACC distribution on multiple accelerators (GPU)
//    acc_device_t device_type = acc_get_device_type();
//    if ( acc_device_nvidia == device_type ) {
//       int ngpus = acc_get_num_devices( acc_device_nvidia );
//       int devicenum = atoi( getenv( "OMPI_COMM_WORLD_LOCAL_RANK" ) );
//       acc_set_device_num( devicenum, acc_device_nvidia );
//    }
//    acc_init(device_type);
#endif

    /// Process command line arguments
    string configuration_file;
    if( argc >= 2 ) {
        configuration_file = argv[1];
    } else {
        cout << "Proper usage: RHEA.exe configuration_file.yaml" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }

    /// Construct my RHEA
    myRHEA my_RHEA( configuration_file );
	
    /// Execute my RHEA
    my_RHEA.execute();

    /// Destruct my RHEA ... destructor is called automatically

    /// Finalize MPI
    MPI_Finalize();

    /// Return exit code of program
    return 0;

}
