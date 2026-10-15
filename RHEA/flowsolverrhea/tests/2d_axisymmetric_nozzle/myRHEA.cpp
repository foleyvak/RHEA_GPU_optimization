#include "myRHEA.hpp"
#include <cmath>
#include <cstdio>

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

////////// FIXED PARAMETERS //////////
const double epsilon    = 1.0e-10;			/// Small epsilon number (fixed)
const double pi         = 3.14159265358979323846;	/// Pi number (fixed)

/// PROBLEM PARAMETERS ///
const double gamma_0    = 1.4;				/// Heat capacity ratio [-]
const double R_specific = 287.058;			/// Specific gas constant [J/(kg K)]
const double U_inlet    = 20.0;				/// Inlet velocity [m/s]
const double P_ref      = 101325.0;			/// Reference pressure [Pa]
const double T_ref      = 400.0;			/// Reference temperature [K]

const double A_x        = 0.0;               		/// Grid stretching factor in x-direction
const double A_y        = 0.0;               		/// Grid stretching factor in y-direction
const double A_z        = 0.0;               		/// Grid stretching factor in z-direction

const double x_0        = 0.0;               		/// Domain origin in x-direction [m]
const double y_0        = 0.0;               		/// Domain origin in y-direction [m]
const double z_0        = 0.0;               		/// Domain origin in z-direction [m]

const double r_t        = 1.0e-3;         		/// Nozzle throat radius (for nozzle geometries)
const double r_c        = 7.0e-3;         		/// Nozzle chamber radius (for nozzle geometries)
const double R1_rt      = 10.0;         	    	/// Convergent-throat arc ratio as (R1/rt)
const double R2_R1      = 3.0;               		/// Chamber-convergent arc ratio as (R2/R1)
const double Rexp_rt    = 30.0;              		/// Expansion arc ratio as (Rexp/rt)
const double theta      = 15.0;              		/// Convergent segment inclination angle [deg]
const double alpha      = 3.5;               		/// Conical nozzle half-angle [deg]
const double L_N        = 10.0e-3;           		/// Conical section length [m]
const double L_c        = 5.0e-3;            		/// Chamber section length [m]

const double R1         = r_t*R1_rt;         		/// Convergent-Throat arc radius
const double R2         = R1*R2_R1;           		/// Chamber-Convergent arc radius
const double Rexp       = r_t*Rexp_rt;           	/// Expansion arc radius
const double theta_rad  = theta * pi / 180.0;         	/// Convergent segment inclination angle [rad]
const double alpha_rad  = alpha * pi / 180.0;         	/// Conical nozzle half-angle [rad]
const double x_c        = L_c;                       	/// Chamber section end location [m]
const double r2         = r_c - R2 * (1.0 - cos(theta_rad));
const double r1         = r_t + R1 * (1.0 - cos(theta_rad));
const double x2         = x_c + R2 * sin(theta_rad);
const double x1         = x2 + (r2 - r1) / tan(theta_rad);
const double x_t        = x1 + R1 * sin(theta_rad);		/// Throat location [m]
const double x_exp      = x_t + Rexp * sin(alpha_rad);  	/// Expansion section start location [m]
const double r_exp      = r_t + Rexp * (1.0 - cos(alpha_rad));	/// Expansion section radius [m]

const double L          = 1.0;					
const double local_Lx   = x_t + L_N + L_c;  		/// Total domain length in x-direction [m]
const double local_Ly   = 1.0*L;			/// Total domain length in y-direction [m]
const double local_Lz   = 0.001*L;			/// Total domain length in z-direction [m]

////////// myRHEA CLASS //////////

myRHEA::myRHEA(const std::string name_configuration_file) : FlowSolverRHEA(name_configuration_file, [this](FlowSolverRHEA* solver ) {

    /// MPI rank and size
    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    /// MPI coordinates
    const int p_x = my_rank % solver->np_x;
    const int p_y = ( my_rank / solver->np_x ) % solver->np_y;
    const int p_z = my_rank / ( solver->np_x * solver->np_y );

    /// Number of local points
    int _lNx_ = solver->_lNx_;
    int _lNy_ = solver->_lNy_;
    int _lNz_ = solver->_lNz_;

    /// Physical stride between processors is only the interior cells
    const int stride_x = _lNx_ - 2;
    const int stride_y = _lNy_ - 2;
    const int stride_z = _lNz_ - 2;

    /// All (inner, halo, boundary) points: x, y and z
    int I = -1, J = -1, K = -1;
    for(int i = solver->topo->iter_common[_ALL_][_INIX_]; i <= solver->topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = solver->topo->iter_common[_ALL_][_INIY_]; j <= solver->topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = solver->topo->iter_common[_ALL_][_INIZ_]; k <= solver->topo->iter_common[_ALL_][_ENDZ_]; k++) {
                
                // Global indices
                I = i + ( p_x*stride_x );
                J = j + ( p_y*stride_y );
                K = k + ( p_z*stride_z );
                
                // 1. Calculate X coordinate (remains uniform)
                double eta_x = ( I - 0.5 )/( double )solver->num_grid_x; // Normalized X from 0 to 1
                double x_phys = x_0 + local_Lx*eta_x + A_x*( 0.5*local_Lx - local_Lx*eta_x )*( 1.0 - eta_x )*eta_x;  // Map to [x_0, x_0 + local_Lx]
                solver->x_field[I1D(i,j,k)] = x_phys;

                // 2. Calculate local nozzle half-height H(x)
                double local_Hx = 0.0;

                if (x_phys <= x_c) {
                    local_Hx = r_c;
                } else if (x_c < x_phys && x_phys <= x2) {
                    local_Hx = r_c - R2 * (1.0 - sqrt( 1.0 - ( ( x_phys - x_c ) / R2)*( ( x_phys - x_c ) / R2) ) );
                } else if (x2 < x_phys && x_phys <= x1) {
                    local_Hx = r1 - ( x_phys - x1 ) * tan(theta_rad);
                } else if (x1 < x_phys && x_phys <= x_t) {
                    local_Hx = r_t + R1 * (1.0 - sqrt( 1.0 - ( ( x_phys - x_t ) / R1)*( ( x_phys - x_t ) / R1) ) );
                } else if (x_t < x_phys && x_phys <= x_exp) {
                    local_Hx = r_t + Rexp * (1.0 - sqrt( 1.0 - ( ( x_phys - x_t ) / Rexp)*( ( x_phys - x_t ) / Rexp) ) );
                } else {
                    local_Hx = r_exp + ( x_phys - x_exp ) * tan(alpha_rad);
                }

                // local_Hx = 0.1 + (x_phys - 0.5*local_Lx)*(x_phys - 0.5*local_Lx);
                // local_Hx = 1.0;
                
                // 3. Map Y coordinate (scale based on local height)
                double eta_y = ( J - 0.5 )/( double )solver->num_grid_y; // Normalized Y from 0 to 1
                double y_phys = y_0 + local_Hx*eta_y + A_y*( 0.5*local_Hx - local_Hx*eta_y )*( 1.0 - eta_y )*eta_y;  // Map to [-H, +H]
                solver->y_field[I1D(i,j,k)] = y_phys;
                
                // 4. Calculate Z coordinate (assuming 2D planar extrusion)
                double eta_z = ( K - 0.5 )/( double )solver->num_grid_z; // Normalized Z from 0 to 1
                double z_phys = z_0 + local_Lz*eta_z + A_z*( 0.5*local_Lz - local_Lz*eta_z )*( 1.0 - eta_z )*eta_z;  // Map to [z_0, z_0 + local_Lz]
                solver->z_field[I1D(i,j,k)] = z_phys;
            }
        }
    }

    /// Update halo values (do not activate!)
    //solver->x_field.update();
    //solver->y_field.update();
    //solver->z_field.update();

    det_Jacobian_field.setTopology(topo,"det_Jacobian_field");

    double x_xi, x_eta, x_zeta, y_xi, y_eta, y_zeta, z_xi, z_eta, z_zeta, det_J;
    for(int i = solver->topo->iter_common[_INNER_][_INIX_]; i <= solver->topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = solver->topo->iter_common[_INNER_][_INIY_]; j <= solver->topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = solver->topo->iter_common[_INNER_][_INIZ_]; k <= solver->topo->iter_common[_INNER_][_ENDZ_]; k++) {
                x_xi    = (solver->x_field[I1D(i+1,j,k)] - solver->x_field[I1D(i-1,j,k)])/2.0;
                x_eta   = (solver->x_field[I1D(i,j+1,k)] - solver->x_field[I1D(i,j-1,k)])/2.0;
                x_zeta  = (solver->x_field[I1D(i,j,k+1)] - solver->x_field[I1D(i,j,k-1)])/2.0;

                y_xi    = (solver->y_field[I1D(i+1,j,k)] - solver->y_field[I1D(i-1,j,k)])/2.0;
                y_eta   = (solver->y_field[I1D(i,j+1,k)] - solver->y_field[I1D(i,j-1,k)])/2.0;
                y_zeta  = (solver->y_field[I1D(i,j,k+1)] - solver->y_field[I1D(i,j,k-1)])/2.0;

                z_xi    = (solver->z_field[I1D(i+1,j,k)] - solver->z_field[I1D(i-1,j,k)])/2.0;
                z_eta   = (solver->z_field[I1D(i,j+1,k)] - solver->z_field[I1D(i,j-1,k)])/2.0;
                z_zeta  = (solver->z_field[I1D(i,j,k+1)] - solver->z_field[I1D(i,j,k-1)])/2.0;

                det_J = (x_xi*y_eta*z_zeta + x_eta*y_zeta*z_xi + x_zeta*y_xi*z_eta - x_zeta*y_eta*z_xi - x_eta*y_xi*z_zeta - x_xi*y_zeta*z_eta);
                det_Jacobian_field[I1D(i,j,k)] = 1.0/det_J;
            }
        }
    }

    /// Update halo values
    det_Jacobian_field.update();

    #pragma acc update device(solver->x_field.vector[0:solver->_ls_], solver->y_field.vector[0:solver->_ls_], solver->z_field.vector[0:solver->_ls_], det_Jacobian_field.vector[0:solver->_ls_])

})  // Closes the lambda '}' and the initializer list ')'
{
    // Empty constructor body for myRHEA
}   // Closes the constructor definition. No semicolon here!

void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// All (inner, halo, boundary): u, v, w, P and T    
    #pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_]) 
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] = U_inlet;
                v_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = 0.0;
                P_field[I1D(i,j,k)] = P_ref;
                T_field[I1D(i,j,k)] = T_ref;
            }
        }
    }

    #pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_])
    /// Update halo values
    u_field.update();
    v_field.update();
    w_field.update();
    P_field.update();
    T_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_])
#endif

};

void myRHEA::calculateSourceTerms() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Inner points: f_rhou, f_rhov, f_rhow and f_rhoE
    #pragma acc parallel loop collapse(3) present(f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                f_rhou_field[I1D(i,j,k)] = 0.0;
                f_rhov_field[I1D(i,j,k)] = 0.0;
                f_rhow_field[I1D(i,j,k)] = 0.0;
                f_rhoE_field[I1D(i,j,k)] = 0.0;
            }
        }
    }

    /// Update halo values
    //f_rhou_field.update();
    //f_rhov_field.update();
    //f_rhow_field.update();
    //f_rhoE_field.update();

};

void myRHEA::temporalHookFunction() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration	

};

void myRHEA::calculateInviscidFluxes() {

    /// Unsplit method for Euler equations:
    /// E. F. Toro.
    /// Riemann solvers and numerical methods for fluid dynamics.
    /// Springer, 2009.

    /// Inner points: rho, rhou, rhov, rhow and rhoE
    int index_LLL, index_LL, index_L, index_R, index_RR, index_RRR, var_type;
    double delta_x, delta_y, delta_z;
    double rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, P_rhouvw_LLL, T_LLL, a_LLL;
    double rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, P_rhouvw_LL, T_LL, a_LL;
    double rho_L, u_L, v_L, w_L, E_L, s_L, P_L, P_rhouvw_L, T_L, a_L;
    double rho_R, u_R, v_R, w_R, E_R, s_R, P_R, P_rhouvw_R, T_R, a_R;
    double rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, P_rhouvw_RR, T_RR, a_RR;
    double rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, P_rhouvw_RRR, T_RRR, a_RRR;
    double rho_F_p, rho_F_m, rhou_F_p, rhou_F_m, rhov_F_p;
    double rhov_F_m, rhow_F_p, rhow_F_m, rhoE_F_p, rhoE_F_m;
    double rhoV_n_F_p, rhoV_t1_F_p, rhoV_t2_F_p, rhoV_n_F_m, rhoV_t1_F_m, rhoV_t2_F_m;
    double x_xi_avg, x_eta_avg, x_zeta_avg, y_xi_avg, y_eta_avg, y_zeta_avg, z_xi_avg, z_eta_avg, z_zeta_avg;
    double xi_x_avg, xi_y_avg, xi_z_avg, eta_x_avg, eta_y_avg, eta_z_avg, zeta_x_avg, zeta_y_avg, zeta_z_avg;
    double grad_xi_avg, grad_eta_avg, grad_zeta_avg, grad_t1, grad_t2;
    double n_x_avg, n_y_avg, n_z_avg, t1_x_avg, t1_y_avg, t1_z_avg, t2_x_avg, t2_y_avg, t2_z_avg;
    double V_n_LLL, V_n_LL, V_n_L, V_n_R, V_n_RR, V_n_RRR;
    double V_t1_LLL, V_t1_LL, V_t1_L, V_t1_R, V_t1_RR, V_t1_RRR;
    double V_t2_LLL, V_t2_LL, V_t2_L, V_t2_R, V_t2_RR, V_t2_RRR; 

    #pragma acc parallel loop collapse(3) private(index_LLL, index_LL, index_L, index_R, index_RR, index_RRR, var_type, delta_x, delta_y, delta_z, rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, P_rhouvw_RRR, T_RRR, a_RRR, rho_F_p, rho_F_m, rhou_F_p, rhou_F_m, rhov_F_p, rhov_F_m, rhow_F_p, rhow_F_m, rhoE_F_p, rhoE_F_m, rhoV_n_F_p, rhoV_t1_F_p, rhoV_t2_F_p, rhoV_n_F_m, rhoV_t1_F_m, rhoV_t2_F_m, x_xi_avg, x_eta_avg, x_zeta_avg, y_xi_avg, y_eta_avg, y_zeta_avg, z_xi_avg, z_eta_avg, z_zeta_avg, xi_x_avg, xi_y_avg, xi_z_avg, eta_x_avg, eta_y_avg, eta_z_avg, zeta_x_avg, zeta_y_avg, zeta_z_avg, grad_xi_avg, grad_eta_avg, grad_zeta_avg, grad_t1, grad_t2, n_x_avg, n_y_avg, n_z_avg, t1_x_avg, t1_y_avg, t1_z_avg, t2_x_avg, t2_y_avg, t2_z_avg, V_n_LLL, V_n_LL, V_n_L, V_n_R, V_n_RR, V_n_RRR, V_t1_LLL, V_t1_LL, V_t1_L, V_t1_R, V_t1_RR, V_t1_RRR, V_t2_LLL, V_t2_LL, V_t2_L, V_t2_R, V_t2_RR, V_t2_RRR) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], det_Jacobian_field.vector[0:_ls_], rho_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], sos_field.vector[0:_ls_], rho_inv_flux.vector[0:_ls_], rhou_inv_flux.vector[0:_ls_], rhov_inv_flux.vector[0:_ls_], rhow_inv_flux.vector[0:_ls_], rhoE_inv_flux.vector[0:_ls_] )  
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 1.0; 
                delta_y = 1.0; 
                delta_z = 1.0;
                /// x-direction i+1/2
                index_LLL = i - 2; index_LL = i - 1; index_L = i;
		if( i <= ( topo->iter_common[_INNER_][_INIX_] ) ) { index_LLL = i; index_LL = i; index_L = i; }
                index_R = i + 1; index_RR = i + 2; index_RRR = i + 3;
        	if( i >= ( topo->iter_common[_INNER_][_ENDX_] - 1 ) ) { index_R = i + 1; index_RR = i + 1; index_RRR = i + 1; }
                rho_LLL = rho_field[I1D(index_LLL,j,k)]; rho_RRR = rho_field[I1D(index_RRR,j,k)]; 
                rho_LL  = rho_field[I1D(index_LL,j,k)];  rho_RR  = rho_field[I1D(index_RR,j,k)]; 
                rho_L   = rho_field[I1D(index_L,j,k)];   rho_R   = rho_field[I1D(index_R,j,k)];
                u_LLL   = u_field[I1D(index_LLL,j,k)];   u_RRR   = u_field[I1D(index_RRR,j,k)]; 
                u_LL    = u_field[I1D(index_LL,j,k)];    u_RR    = u_field[I1D(index_RR,j,k)]; 
                u_L     = u_field[I1D(index_L,j,k)];     u_R     = u_field[I1D(index_R,j,k)];
                v_LLL   = v_field[I1D(index_LLL,j,k)];   v_RRR   = v_field[I1D(index_RRR,j,k)]; 
                v_LL    = v_field[I1D(index_LL,j,k)];    v_RR    = v_field[I1D(index_RR,j,k)]; 
                v_L     = v_field[I1D(index_L,j,k)];     v_R     = v_field[I1D(index_R,j,k)];
                w_LLL   = w_field[I1D(index_LLL,j,k)];   w_RRR   = w_field[I1D(index_RRR,j,k)]; 
                w_LL    = w_field[I1D(index_LL,j,k)];    w_RR    = w_field[I1D(index_RR,j,k)]; 
                w_L     = w_field[I1D(index_L,j,k)];     w_R     = w_field[I1D(index_R,j,k)];
                E_LLL   = E_field[I1D(index_LLL,j,k)];   E_RRR   = E_field[I1D(index_RRR,j,k)]; 
                E_LL    = E_field[I1D(index_LL,j,k)];    E_RR    = E_field[I1D(index_RR,j,k)]; 
                E_L     = E_field[I1D(index_L,j,k)];     E_R     = E_field[I1D(index_R,j,k)];
                s_LLL   = s_field[I1D(index_LLL,j,k)];   s_RRR   = s_field[I1D(index_RRR,j,k)]; 
                s_LL    = s_field[I1D(index_LL,j,k)];    s_RR    = s_field[I1D(index_RR,j,k)]; 
                s_L     = s_field[I1D(index_L,j,k)];     s_R     = s_field[I1D(index_R,j,k)];		
                P_LLL   = P_field[I1D(index_LLL,j,k)];   P_RRR   = P_field[I1D(index_RRR,j,k)]; 
                P_LL    = P_field[I1D(index_LL,j,k)];    P_RR    = P_field[I1D(index_RR,j,k)]; 
                P_L     = P_field[I1D(index_L,j,k)];     P_R     = P_field[I1D(index_R,j,k)];
                T_LLL   = T_field[I1D(index_LLL,j,k)];   T_RRR   = T_field[I1D(index_RRR,j,k)]; 
                T_LL    = T_field[I1D(index_LL,j,k)];    T_RR    = T_field[I1D(index_RR,j,k)]; 
                T_L     = T_field[I1D(index_L,j,k)];     T_R     = T_field[I1D(index_R,j,k)];
                a_LLL   = sos_field[I1D(index_LLL,j,k)]; a_RRR   = sos_field[I1D(index_RRR,j,k)]; 
                a_LL    = sos_field[I1D(index_LL,j,k)];  a_RR    = sos_field[I1D(index_RR,j,k)]; 
                a_L     = sos_field[I1D(index_L,j,k)];   a_R     = sos_field[I1D(index_R,j,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                
                /// Geometric components
                x_xi_avg    = 0.0; x_eta_avg   = 0.0; x_zeta_avg  = 0.0; 
                y_xi_avg    = 0.0; y_eta_avg   = 0.0; y_zeta_avg  = 0.0; 
                z_xi_avg    = 0.0; z_eta_avg   = 0.0; z_zeta_avg  = 0.0; 

                x_xi_avg    = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];                                                                           
                x_eta_avg   = 0.5 * ((x_field[I1D(i+1,j+1,k)] - x_field[I1D(i+1,j-1,k)])/2.0 + (x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0);
                x_zeta_avg  = 0.5 * ((x_field[I1D(i+1,j,k+1)] - x_field[I1D(i+1,j,k-1)])/2.0 + (x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0); 
                y_xi_avg    = y_field[I1D(i+1,j,k)] - y_field[I1D(i,j,k)];                                                                           
                y_eta_avg   = 0.5 * ((y_field[I1D(i+1,j+1,k)] - y_field[I1D(i+1,j-1,k)])/2.0 + (y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0);
                y_zeta_avg  = 0.5 * ((y_field[I1D(i+1,j,k+1)] - y_field[I1D(i+1,j,k-1)])/2.0 + (y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0); 
                z_xi_avg    = z_field[I1D(i+1,j,k)] - z_field[I1D(i,j,k)];                                                                           
                z_eta_avg   = 0.5 * ((z_field[I1D(i+1,j+1,k)] - z_field[I1D(i+1,j-1,k)])/2.0 + (z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0);
                z_zeta_avg  = 0.5 * ((z_field[I1D(i+1,j,k+1)] - z_field[I1D(i+1,j,k-1)])/2.0 + (z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0); 

                // det_J = x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg); 
                // det_J = 1.0/det_J;

                xi_x_avg = y_eta_avg*z_zeta_avg - z_eta_avg*y_zeta_avg;    xi_y_avg = x_zeta_avg*z_eta_avg - x_eta_avg*z_zeta_avg;    xi_z_avg = x_eta_avg*y_zeta_avg - x_zeta_avg*y_eta_avg; /// In the form of metric/det_Jacobian evaluated at the cell face
                grad_xi_avg = sqrt( xi_x_avg*xi_x_avg + xi_y_avg*xi_y_avg + xi_z_avg*xi_z_avg);

                n_x_avg = xi_x_avg/grad_xi_avg;     n_y_avg = xi_y_avg/grad_xi_avg;     n_z_avg = xi_z_avg/grad_xi_avg;
                t1_x_avg = -1.0*n_y_avg;            t1_y_avg =  1.0*n_x_avg;            t1_z_avg =  0.0;     
                t2_x_avg = -1.0*n_x_avg*n_z_avg;    t2_y_avg = -1.0*n_y_avg*n_z_avg;    t2_z_avg = n_x_avg*n_x_avg + n_y_avg*n_y_avg; 

                grad_t1 = sqrt(t1_x_avg*t1_x_avg + t1_y_avg*t1_y_avg + t1_z_avg*t1_z_avg);
                grad_t2 = sqrt(t2_x_avg*t2_x_avg + t2_y_avg*t2_y_avg + t2_z_avg*t2_z_avg);

                t1_x_avg = t1_x_avg/grad_t1;
                t1_y_avg = t1_y_avg/grad_t1;
                t1_z_avg = t1_z_avg/grad_t1;
                t2_x_avg = t2_x_avg/grad_t2;
                t2_y_avg = t2_y_avg/grad_t2;
                t2_z_avg = t2_z_avg/grad_t2;

                // printf("normal vector = [%.8f, %.8f, %.8f]\n",n_x_avg,n_y_avg,n_z_avg);
                // printf("tangential vector 1 = [%.8f, %.8f, %.8f]\n",t1_x_avg,t1_y_avg,t1_z_avg);
                // printf("tangential vector 2 = [%.8f, %.8f, %.8f]\n",t2_x_avg,t2_y_avg,t2_z_avg);

                V_n_L      = u_L   * n_x_avg  + v_L   * n_y_avg + w_L    * n_z_avg;   
                V_n_LL     = u_LL  * n_x_avg  + v_LL  * n_y_avg + w_LL   * n_z_avg;
                V_n_LLL    = u_LLL * n_x_avg  + v_LLL * n_y_avg + w_LLL  * n_z_avg;
                V_t1_L     = u_L   * t1_x_avg + v_L   * t1_y_avg + w_L   * t1_z_avg;
                V_t1_LL    = u_LL  * t1_x_avg + v_LL  * t1_y_avg + w_LL  * t1_z_avg;
                V_t1_LLL   = u_LLL * t1_x_avg + v_LLL * t1_y_avg + w_LLL * t1_z_avg;
                V_t2_L     = u_L   * t2_x_avg + v_L   * t2_y_avg + w_L   * t2_z_avg;
                V_t2_LL    = u_LL  * t2_x_avg + v_LL  * t2_y_avg + w_LL  * t2_z_avg;
                V_t2_LLL   = u_LLL * t2_x_avg + v_LLL * t2_y_avg + w_LLL * t2_z_avg;

                V_n_R      = u_R   * n_x_avg  + v_R   * n_y_avg  + w_R   * n_z_avg;   
                V_n_RR     = u_RR  * n_x_avg  + v_RR  * n_y_avg  + w_RR  * n_z_avg;
                V_n_RRR    = u_RRR * n_x_avg  + v_RRR * n_y_avg  + w_RRR * n_z_avg;
                V_t1_R     = u_R   * t1_x_avg + v_R   * t1_y_avg + w_R   * t1_z_avg;
                V_t1_RR    = u_RR  * t1_x_avg + v_RR  * t1_y_avg + w_RR  * t1_z_avg;
                V_t1_RRR   = u_RRR * t1_x_avg + v_RRR * t1_y_avg + w_RRR * t1_z_avg;
                V_t2_R     = u_R   * t2_x_avg + v_R   * t2_y_avg + w_R   * t2_z_avg;
                V_t2_RR    = u_RR  * t2_x_avg + v_RR  * t2_y_avg + w_RR  * t2_z_avg;
                V_t2_RRR   = u_RRR * t2_x_avg + v_RRR * t2_y_avg + w_RRR * t2_z_avg;

                /// rho
                var_type = 0;
                rho_F_p  = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhou
                var_type = 1;
                rhoV_n_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhov
                var_type = 2;
                rhoV_t1_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhow
                var_type = 3;
                rhoV_t2_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                
                /// Reconstruct flux terms in global physical axes
                rho_F_p     = rho_F_p * grad_xi_avg;
                rhou_F_p    = (rhoV_n_F_p * n_x_avg + rhoV_t1_F_p * t1_x_avg + rhoV_t2_F_p * t2_x_avg) * grad_xi_avg;
                rhov_F_p    = (rhoV_n_F_p * n_y_avg + rhoV_t1_F_p * t1_y_avg + rhoV_t2_F_p * t2_y_avg) * grad_xi_avg;
                rhow_F_p    = (rhoV_n_F_p * n_z_avg + rhoV_t1_F_p * t1_z_avg + rhoV_t2_F_p * t2_z_avg) * grad_xi_avg;
                rhoE_F_p    = rhoE_F_p * grad_xi_avg;

                /// x-direction i-1/2
                index_LLL = i - 3; index_LL = i - 2; index_L = i - 1;
		if( i <= ( topo->iter_common[_INNER_][_INIX_] + 1 ) ) { index_LLL = i - 1; index_LL = i - 1; index_L = i - 1; }
                index_R = i; index_RR = i + 1; index_RRR = i + 2;
                if( i >= ( topo->iter_common[_INNER_][_ENDX_] ) ) { index_R = i; index_RR = i; index_RRR = i; }
                rho_LLL = rho_field[I1D(index_LLL,j,k)]; rho_RRR = rho_field[I1D(index_RRR,j,k)]; 
                rho_LL  = rho_field[I1D(index_LL,j,k)];  rho_RR  = rho_field[I1D(index_RR,j,k)]; 
                rho_L   = rho_field[I1D(index_L,j,k)];   rho_R   = rho_field[I1D(index_R,j,k)];
                u_LLL   = u_field[I1D(index_LLL,j,k)];   u_RRR   = u_field[I1D(index_RRR,j,k)]; 
                u_LL    = u_field[I1D(index_LL,j,k)];    u_RR    = u_field[I1D(index_RR,j,k)]; 
                u_L     = u_field[I1D(index_L,j,k)];     u_R     = u_field[I1D(index_R,j,k)];
                v_LLL   = v_field[I1D(index_LLL,j,k)];   v_RRR   = v_field[I1D(index_RRR,j,k)]; 
                v_LL    = v_field[I1D(index_LL,j,k)];    v_RR    = v_field[I1D(index_RR,j,k)]; 
                v_L     = v_field[I1D(index_L,j,k)];     v_R     = v_field[I1D(index_R,j,k)];
                w_LLL   = w_field[I1D(index_LLL,j,k)];   w_RRR   = w_field[I1D(index_RRR,j,k)]; 
                w_LL    = w_field[I1D(index_LL,j,k)];    w_RR    = w_field[I1D(index_RR,j,k)]; 
                w_L     = w_field[I1D(index_L,j,k)];     w_R     = w_field[I1D(index_R,j,k)];
                E_LLL   = E_field[I1D(index_LLL,j,k)];   E_RRR   = E_field[I1D(index_RRR,j,k)]; 
                E_LL    = E_field[I1D(index_LL,j,k)];    E_RR    = E_field[I1D(index_RR,j,k)]; 
                E_L     = E_field[I1D(index_L,j,k)];     E_R     = E_field[I1D(index_R,j,k)];
                s_LLL   = s_field[I1D(index_LLL,j,k)];   s_RRR   = s_field[I1D(index_RRR,j,k)]; 
                s_LL    = s_field[I1D(index_LL,j,k)];    s_RR    = s_field[I1D(index_RR,j,k)]; 
                s_L     = s_field[I1D(index_L,j,k)];     s_R     = s_field[I1D(index_R,j,k)];		
                P_LLL   = P_field[I1D(index_LLL,j,k)];   P_RRR   = P_field[I1D(index_RRR,j,k)]; 
                P_LL    = P_field[I1D(index_LL,j,k)];    P_RR    = P_field[I1D(index_RR,j,k)]; 
                P_L     = P_field[I1D(index_L,j,k)];     P_R     = P_field[I1D(index_R,j,k)];
                T_LLL   = T_field[I1D(index_LLL,j,k)];   T_RRR   = T_field[I1D(index_RRR,j,k)]; 
                T_LL    = T_field[I1D(index_LL,j,k)];    T_RR    = T_field[I1D(index_RR,j,k)]; 
                T_L     = T_field[I1D(index_L,j,k)];     T_R     = T_field[I1D(index_R,j,k)];
                a_LLL   = sos_field[I1D(index_LLL,j,k)]; a_RRR   = sos_field[I1D(index_RRR,j,k)]; 
                a_LL    = sos_field[I1D(index_LL,j,k)];  a_RR    = sos_field[I1D(index_RR,j,k)]; 
                a_L     = sos_field[I1D(index_L,j,k)];   a_R     = sos_field[I1D(index_R,j,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                
                /// Geometric components
                x_xi_avg    = 0.0; x_eta_avg   = 0.0; x_zeta_avg  = 0.0; 
                y_xi_avg    = 0.0; y_eta_avg   = 0.0; y_zeta_avg  = 0.0; 
                z_xi_avg    = 0.0; z_eta_avg   = 0.0; z_zeta_avg  = 0.0;

                x_xi_avg    = x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)];                                                                           
                x_eta_avg   = 0.5 * ((x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0 + (x_field[I1D(i-1,j+1,k)] - x_field[I1D(i-1,j-1,k)])/2.0);
                x_zeta_avg  = 0.5 * ((x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0 + (x_field[I1D(i-1,j,k+1)] - x_field[I1D(i-1,j,k-1)])/2.0); 
                y_xi_avg    = y_field[I1D(i,j,k)] - y_field[I1D(i-1,j,k)];                                                                           
                y_eta_avg   = 0.5 * ((y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0 + (y_field[I1D(i-1,j+1,k)] - y_field[I1D(i-1,j-1,k)])/2.0);
                y_zeta_avg  = 0.5 * ((y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0 + (y_field[I1D(i-1,j,k+1)] - y_field[I1D(i-1,j,k-1)])/2.0); 
                z_xi_avg    = z_field[I1D(i,j,k)] - z_field[I1D(i-1,j,k)];                                                                           
                z_eta_avg   = 0.5 * ((z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0 + (z_field[I1D(i-1,j+1,k)] - z_field[I1D(i-1,j-1,k)])/2.0);
                z_zeta_avg  = 0.5 * ((z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0 + (z_field[I1D(i-1,j,k+1)] - z_field[I1D(i-1,j,k-1)])/2.0); 

                // det_J = x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg); 
                // det_J = 1.0/det_J;

                xi_x_avg = y_eta_avg*z_zeta_avg - z_eta_avg*y_zeta_avg;    xi_y_avg = x_zeta_avg*z_eta_avg - x_eta_avg*z_zeta_avg;    xi_z_avg = x_eta_avg*y_zeta_avg - x_zeta_avg*y_eta_avg; /// In the form of metric/det_Jacobian evaluated at the cell face
                grad_xi_avg = sqrt( xi_x_avg*xi_x_avg + xi_y_avg*xi_y_avg + xi_z_avg*xi_z_avg);

                n_x_avg = xi_x_avg/grad_xi_avg;     n_y_avg = xi_y_avg/grad_xi_avg;     n_z_avg = xi_z_avg/grad_xi_avg;
                t1_x_avg = -1.0*n_y_avg;            t1_y_avg =  1.0*n_x_avg;            t1_z_avg =  0.0;     
                t2_x_avg = -1.0*n_x_avg*n_z_avg;    t2_y_avg = -1.0*n_y_avg*n_z_avg;    t2_z_avg = n_x_avg*n_x_avg + n_y_avg*n_y_avg; 

                grad_t1 = sqrt(t1_x_avg*t1_x_avg + t1_y_avg*t1_y_avg + t1_z_avg*t1_z_avg);
                grad_t2 = sqrt(t2_x_avg*t2_x_avg + t2_y_avg*t2_y_avg + t2_z_avg*t2_z_avg);

                t1_x_avg = t1_x_avg/grad_t1;
                t1_y_avg = t1_y_avg/grad_t1;
                t1_z_avg = t1_z_avg/grad_t1;
                t2_x_avg = t2_x_avg/grad_t2;
                t2_y_avg = t2_y_avg/grad_t2;
                t2_z_avg = t2_z_avg/grad_t2;

                // printf("normal vector = [%.8f, %.8f, %.8f]\n",n_x_avg,n_y_avg,n_z_avg);
                // printf("tangential vector 1 = [%.8f, %.8f, %.8f]\n",t1_x_avg,t1_y_avg,t1_z_avg);
                // printf("tangential vector 2 = [%.8f, %.8f, %.8f]\n",t2_x_avg,t2_y_avg,t2_z_avg);

                V_n_L      = u_L   * n_x_avg  + v_L   * n_y_avg  + w_L   * n_z_avg;   
                V_n_LL     = u_LL  * n_x_avg  + v_LL  * n_y_avg  + w_LL  * n_z_avg;
                V_n_LLL    = u_LLL * n_x_avg  + v_LLL * n_y_avg  + w_LLL * n_z_avg;
                V_t1_L     = u_L   * t1_x_avg + v_L   * t1_y_avg + w_L   * t1_z_avg;
                V_t1_LL    = u_LL  * t1_x_avg + v_LL  * t1_y_avg + w_LL  * t1_z_avg;
                V_t1_LLL   = u_LLL * t1_x_avg + v_LLL * t1_y_avg + w_LLL * t1_z_avg;
                V_t2_L     = u_L   * t2_x_avg + v_L   * t2_y_avg + w_L   * t2_z_avg;
                V_t2_LL    = u_LL  * t2_x_avg + v_LL  * t2_y_avg + w_LL  * t2_z_avg;
                V_t2_LLL   = u_LLL * t2_x_avg + v_LLL * t2_y_avg + w_LLL * t2_z_avg;

                V_n_R      = u_R   * n_x_avg  + v_R   * n_y_avg  + w_R   * n_z_avg;   
                V_n_RR     = u_RR  * n_x_avg  + v_RR  * n_y_avg  + w_RR  * n_z_avg;
                V_n_RRR    = u_RRR * n_x_avg  + v_RRR * n_y_avg  + w_RRR * n_z_avg;
                V_t1_R     = u_R   * t1_x_avg + v_R   * t1_y_avg + w_R   * t1_z_avg;
                V_t1_RR    = u_RR  * t1_x_avg + v_RR  * t1_y_avg + w_RR  * t1_z_avg;
                V_t1_RRR   = u_RRR * t1_x_avg + v_RRR * t1_y_avg + w_RRR * t1_z_avg;
                V_t2_R     = u_R   * t2_x_avg + v_R   * t2_y_avg + w_R   * t2_z_avg;
                V_t2_RR    = u_RR  * t2_x_avg + v_RR  * t2_y_avg + w_RR  * t2_z_avg;
                V_t2_RRR   = u_RRR * t2_x_avg + v_RRR * t2_y_avg + w_RRR * t2_z_avg;

                /// rho
                var_type = 0;
                rho_F_m  = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhou
                var_type = 1;
                rhoV_n_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhov
                var_type = 2;
                rhoV_t1_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhow
                var_type = 3;
                rhoV_t2_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );

                /// Reconstruct flux terms in global physical axes
                rho_F_m     = rho_F_m * grad_xi_avg;
                rhou_F_m    = (rhoV_n_F_m * n_x_avg + rhoV_t1_F_m * t1_x_avg + rhoV_t2_F_m * t2_x_avg) * grad_xi_avg;
                rhov_F_m    = (rhoV_n_F_m * n_y_avg + rhoV_t1_F_m * t1_y_avg + rhoV_t2_F_m * t2_y_avg) * grad_xi_avg;
                rhow_F_m    = (rhoV_n_F_m * n_z_avg + rhoV_t1_F_m * t1_z_avg + rhoV_t2_F_m * t2_z_avg) * grad_xi_avg;
                rhoE_F_m    = rhoE_F_m * grad_xi_avg;
                
                /// Fluxes x-direction
                rho_inv_flux[I1D(i,j,k)]  = det_Jacobian_field[I1D(i,j,k)] * ( rho_F_p - rho_F_m );
                rhou_inv_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * ( rhou_F_p - rhou_F_m );
                rhov_inv_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * ( rhov_F_p - rhov_F_m );
                rhow_inv_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * ( rhow_F_p - rhow_F_m );
                rhoE_inv_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * ( rhoE_F_p - rhoE_F_m );
                
            	/// y-direction j+1/2
                index_LLL = j - 2; index_LL = j - 1; index_L = j;
		if( j <= ( topo->iter_common[_INNER_][_INIY_] ) ) { index_LLL = j; index_LL = j; index_L = j; }
                index_R = j + 1; index_RR = j + 2; index_RRR = j + 3;
                if( j >= ( topo->iter_common[_INNER_][_ENDY_] - 1 ) ) { index_R = j + 1; index_RR = j + 1; index_RRR = j + 1; }
                rho_LLL = rho_field[I1D(i,index_LLL,k)]; rho_RRR = rho_field[I1D(i,index_RRR,k)]; 
                rho_LL  = rho_field[I1D(i,index_LL,k)];  rho_RR  = rho_field[I1D(i,index_RR,k)]; 
                rho_L   = rho_field[I1D(i,index_L,k)];   rho_R   = rho_field[I1D(i,index_R,k)];
                u_LLL   = u_field[I1D(i,index_LLL,k)];   u_RRR   = u_field[I1D(i,index_RRR,k)]; 
                u_LL    = u_field[I1D(i,index_LL,k)];    u_RR    = u_field[I1D(i,index_RR,k)]; 
                u_L     = u_field[I1D(i,index_L,k)];     u_R     = u_field[I1D(i,index_R,k)];
                v_LLL   = v_field[I1D(i,index_LLL,k)];   v_RRR   = v_field[I1D(i,index_RRR,k)]; 
                v_LL    = v_field[I1D(i,index_LL,k)];    v_RR    = v_field[I1D(i,index_RR,k)]; 
                v_L     = v_field[I1D(i,index_L,k)];     v_R     = v_field[I1D(i,index_R,k)];
                w_LLL   = w_field[I1D(i,index_LLL,k)];   w_RRR   = w_field[I1D(i,index_RRR,k)]; 
                w_LL    = w_field[I1D(i,index_LL,k)];    w_RR    = w_field[I1D(i,index_RR,k)]; 
                w_L     = w_field[I1D(i,index_L,k)];     w_R     = w_field[I1D(i,index_R,k)];
                E_LLL   = E_field[I1D(i,index_LLL,k)];   E_RRR   = E_field[I1D(i,index_RRR,k)]; 
                E_LL    = E_field[I1D(i,index_LL,k)];    E_RR    = E_field[I1D(i,index_RR,k)]; 
                E_L     = E_field[I1D(i,index_L,k)];     E_R     = E_field[I1D(i,index_R,k)];
                s_LLL   = s_field[I1D(i,index_LLL,k)];   s_RRR   = s_field[I1D(i,index_RRR,k)]; 
                s_LL    = s_field[I1D(i,index_LL,k)];    s_RR    = s_field[I1D(i,index_RR,k)]; 
                s_L     = s_field[I1D(i,index_L,k)];     s_R     = s_field[I1D(i,index_R,k)];		
                P_LLL   = P_field[I1D(i,index_LLL,k)];   P_RRR   = P_field[I1D(i,index_RRR,k)]; 
                P_LL    = P_field[I1D(i,index_LL,k)];    P_RR    = P_field[I1D(i,index_RR,k)]; 
                P_L     = P_field[I1D(i,index_L,k)];     P_R     = P_field[I1D(i,index_R,k)];
                T_LLL   = T_field[I1D(i,index_LLL,k)];   T_RRR   = T_field[I1D(i,index_RRR,k)]; 
                T_LL    = T_field[I1D(i,index_LL,k)];    T_RR    = T_field[I1D(i,index_RR,k)]; 
                T_L     = T_field[I1D(i,index_L,k)];     T_R     = T_field[I1D(i,index_R,k)];
                a_LLL   = sos_field[I1D(i,index_LLL,k)]; a_RRR   = sos_field[I1D(i,index_RRR,k)]; 
                a_LL    = sos_field[I1D(i,index_LL,k)];  a_RR    = sos_field[I1D(i,index_RR,k)]; 
                a_L     = sos_field[I1D(i,index_L,k)];   a_R     = sos_field[I1D(i,index_R,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                
                /// Geometric components
                x_xi_avg    = 0.0; x_eta_avg   = 0.0; x_zeta_avg  = 0.0; 
                y_xi_avg    = 0.0; y_eta_avg   = 0.0; y_zeta_avg  = 0.0; 
                z_xi_avg    = 0.0; z_eta_avg   = 0.0; z_zeta_avg  = 0.0;

                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j+1,k)] - x_field[I1D(i-1,j+1,k)])/2.0 + (x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0);                                                                           
                x_eta_avg   = x_field[I1D(i,j+1,k)] - x_field[I1D(i,j,k)];
                x_zeta_avg  = 0.5 * ((x_field[I1D(i,j+1,k+1)] - x_field[I1D(i,j+1,k-1)])/2.0 + (x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0);
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j+1,k)] - y_field[I1D(i-1,j+1,k)])/2.0 + (y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0); 
                y_eta_avg   = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
                y_zeta_avg  = 0.5 * ((y_field[I1D(i,j+1,k+1)] - y_field[I1D(i,j+1,k-1)])/2.0 + (y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0); 
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j+1,k)] - z_field[I1D(i-1,j+1,k)])/2.0 + (z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0); 
                z_eta_avg   = z_field[I1D(i,j+1,k)] - z_field[I1D(i,j,k)];
                z_zeta_avg  = 0.5 * ((z_field[I1D(i,j+1,k+1)] - z_field[I1D(i,j+1,k-1)])/2.0 + (z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0); 

                eta_x_avg = z_xi_avg*y_zeta_avg - y_xi_avg*z_zeta_avg;    eta_y_avg = x_xi_avg*z_zeta_avg - x_zeta_avg*z_xi_avg;    eta_z_avg = x_zeta_avg*y_xi_avg - x_xi_avg*y_zeta_avg; /// In the form of metric/det_Jacobian evaluated at the cell face
                grad_eta_avg = sqrt( eta_x_avg*eta_x_avg + eta_y_avg*eta_y_avg + eta_z_avg*eta_z_avg);

                n_x_avg = eta_x_avg/grad_eta_avg;   n_y_avg = eta_y_avg/grad_eta_avg;   n_z_avg = eta_z_avg/grad_eta_avg;
                t1_x_avg = -1.0*n_y_avg;            t1_y_avg =  1.0*n_x_avg;            t1_z_avg =  0.0;     
                t2_x_avg = -1.0*n_x_avg*n_z_avg;    t2_y_avg = -1.0*n_y_avg*n_z_avg;    t2_z_avg = n_x_avg*n_x_avg + n_y_avg*n_y_avg;
                
                grad_t1 = sqrt(t1_x_avg*t1_x_avg + t1_y_avg*t1_y_avg + t1_z_avg*t1_z_avg);
                grad_t2 = sqrt(t2_x_avg*t2_x_avg + t2_y_avg*t2_y_avg + t2_z_avg*t2_z_avg);

                t1_x_avg = t1_x_avg/grad_t1;
                t1_y_avg = t1_y_avg/grad_t1;
                t1_z_avg = t1_z_avg/grad_t1;
                t2_x_avg = t2_x_avg/grad_t2;
                t2_y_avg = t2_y_avg/grad_t2;
                t2_z_avg = t2_z_avg/grad_t2;

                // printf("normal vector = [%.8f, %.8f, %.8f]\n",n_x_avg,n_y_avg,n_z_avg);
                // printf("tangential vector 1 = [%.8f, %.8f, %.8f]\n",t1_x_avg,t1_y_avg,t1_z_avg);
                // printf("tangential vector 2 = [%.8f, %.8f, %.8f]\n",t2_x_avg,t2_y_avg,t2_z_avg);

                V_n_L      = u_L   * n_x_avg + v_L    * n_y_avg  + w_L   * n_z_avg;   
                V_n_LL     = u_LL  * n_x_avg + v_LL   * n_y_avg  + w_LL  * n_z_avg;
                V_n_LLL    = u_LLL * n_x_avg + v_LLL  * n_y_avg  + w_LLL * n_z_avg;
                V_t1_L     = u_L   * t1_x_avg + v_L   * t1_y_avg + w_L   * t1_z_avg;
                V_t1_LL    = u_LL  * t1_x_avg + v_LL  * t1_y_avg + w_LL  * t1_z_avg;
                V_t1_LLL   = u_LLL * t1_x_avg + v_LLL * t1_y_avg + w_LLL * t1_z_avg;
                V_t2_L     = u_L   * t2_x_avg + v_L   * t2_y_avg + w_L   * t2_z_avg;
                V_t2_LL    = u_LL  * t2_x_avg + v_LL  * t2_y_avg + w_LL  * t2_z_avg;
                V_t2_LLL   = u_LLL * t2_x_avg + v_LLL * t2_y_avg + w_LLL * t2_z_avg;

                V_n_R      = u_R   * n_x_avg  + v_R   * n_y_avg  + w_R   * n_z_avg;   
                V_n_RR     = u_RR  * n_x_avg  + v_RR  * n_y_avg  + w_RR  * n_z_avg;
                V_n_RRR    = u_RRR * n_x_avg  + v_RRR * n_y_avg  + w_RRR * n_z_avg;
                V_t1_R     = u_R   * t1_x_avg + v_R   * t1_y_avg + w_R   * t1_z_avg;
                V_t1_RR    = u_RR  * t1_x_avg + v_RR  * t1_y_avg + w_RR  * t1_z_avg;
                V_t1_RRR   = u_RRR * t1_x_avg + v_RRR * t1_y_avg + w_RRR * t1_z_avg;
                V_t2_R     = u_R   * t2_x_avg + v_R   * t2_y_avg + w_R   * t2_z_avg;
                V_t2_RR    = u_RR  * t2_x_avg + v_RR  * t2_y_avg + w_RR  * t2_z_avg;
                V_t2_RRR   = u_RRR * t2_x_avg + v_RRR * t2_y_avg + w_RRR * t2_z_avg;
                
                /// rho
                var_type = 0;
                rho_F_p  = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhou
                var_type = 1;
                rhoV_n_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhov
                var_type = 2;
                rhoV_t1_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhow
                var_type = 3;
                rhoV_t2_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );

                /// Reconstruct flux terms in global physical axes
                rho_F_p     = rho_F_p     * grad_eta_avg;
                rhou_F_p    = (rhoV_n_F_p * n_x_avg + rhoV_t1_F_p * t1_x_avg + rhoV_t2_F_p * t2_x_avg) * grad_eta_avg;
                rhov_F_p    = (rhoV_n_F_p * n_y_avg + rhoV_t1_F_p * t1_y_avg + rhoV_t2_F_p * t2_y_avg) * grad_eta_avg;
                rhow_F_p    = (rhoV_n_F_p * n_z_avg + rhoV_t1_F_p * t1_z_avg + rhoV_t2_F_p * t2_z_avg) * grad_eta_avg;
                rhoE_F_p    = rhoE_F_p    * grad_eta_avg;
                
                /// y-direction j-1/2
                index_LLL = j - 3; index_LL = j - 2; index_L = j - 1;
		if( j <= ( topo->iter_common[_INNER_][_INIY_] + 1 ) ) { index_LLL = j - 1; index_LL = j - 1; index_L = j - 1; }
                index_R = j; index_RR = j + 1; index_RRR = j + 2;
                if( j >= ( topo->iter_common[_INNER_][_ENDY_] ) ) { index_R = j; index_RR = j; index_RRR = j; }
                rho_LLL = rho_field[I1D(i,index_LLL,k)]; rho_RRR = rho_field[I1D(i,index_RRR,k)]; 
                rho_LL  = rho_field[I1D(i,index_LL,k)];  rho_RR  = rho_field[I1D(i,index_RR,k)]; 
                rho_L   = rho_field[I1D(i,index_L,k)];   rho_R   = rho_field[I1D(i,index_R,k)];
                u_LLL   = u_field[I1D(i,index_LLL,k)];   u_RRR   = u_field[I1D(i,index_RRR,k)]; 
                u_LL    = u_field[I1D(i,index_LL,k)];    u_RR    = u_field[I1D(i,index_RR,k)]; 
                u_L     = u_field[I1D(i,index_L,k)];     u_R     = u_field[I1D(i,index_R,k)];
                v_LLL   = v_field[I1D(i,index_LLL,k)];   v_RRR   = v_field[I1D(i,index_RRR,k)]; 
                v_LL    = v_field[I1D(i,index_LL,k)];    v_RR    = v_field[I1D(i,index_RR,k)]; 
                v_L     = v_field[I1D(i,index_L,k)];     v_R     = v_field[I1D(i,index_R,k)];
                w_LLL   = w_field[I1D(i,index_LLL,k)];   w_RRR   = w_field[I1D(i,index_RRR,k)]; 
                w_LL    = w_field[I1D(i,index_LL,k)];    w_RR    = w_field[I1D(i,index_RR,k)]; 
                w_L     = w_field[I1D(i,index_L,k)];     w_R     = w_field[I1D(i,index_R,k)];
                E_LLL   = E_field[I1D(i,index_LLL,k)];   E_RRR   = E_field[I1D(i,index_RRR,k)]; 
                E_LL    = E_field[I1D(i,index_LL,k)];    E_RR    = E_field[I1D(i,index_RR,k)]; 
                E_L     = E_field[I1D(i,index_L,k)];     E_R     = E_field[I1D(i,index_R,k)];
                s_LLL   = s_field[I1D(i,index_LLL,k)];   s_RRR   = s_field[I1D(i,index_RRR,k)]; 
                s_LL    = s_field[I1D(i,index_LL,k)];    s_RR    = s_field[I1D(i,index_RR,k)]; 
                s_L     = s_field[I1D(i,index_L,k)];     s_R     = s_field[I1D(i,index_R,k)];		
                P_LLL   = P_field[I1D(i,index_LLL,k)];   P_RRR   = P_field[I1D(i,index_RRR,k)]; 
                P_LL    = P_field[I1D(i,index_LL,k)];    P_RR    = P_field[I1D(i,index_RR,k)]; 
                P_L     = P_field[I1D(i,index_L,k)];     P_R     = P_field[I1D(i,index_R,k)];
                T_LLL   = T_field[I1D(i,index_LLL,k)];   T_RRR   = T_field[I1D(i,index_RRR,k)]; 
                T_LL    = T_field[I1D(i,index_LL,k)];    T_RR    = T_field[I1D(i,index_RR,k)]; 
                T_L     = T_field[I1D(i,index_L,k)];     T_R     = T_field[I1D(i,index_R,k)];
                a_LLL   = sos_field[I1D(i,index_LLL,k)]; a_RRR   = sos_field[I1D(i,index_RRR,k)]; 
                a_LL    = sos_field[I1D(i,index_LL,k)];  a_RR    = sos_field[I1D(i,index_RR,k)]; 
                a_L     = sos_field[I1D(i,index_L,k)];   a_R     = sos_field[I1D(i,index_R,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                
                /// Geometric components
                x_xi_avg    = 0.0; x_eta_avg   = 0.0; x_zeta_avg  = 0.0; 
                y_xi_avg    = 0.0; y_eta_avg   = 0.0; y_zeta_avg  = 0.0; 
                z_xi_avg    = 0.0; z_eta_avg   = 0.0; z_zeta_avg  = 0.0;

                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0 + (x_field[I1D(i+1,j-1,k)] - x_field[I1D(i-1,j-1,k)])/2.0);                                                                           
                x_eta_avg   = x_field[I1D(i,j,k)] - x_field[I1D(i,j-1,k)];
                x_zeta_avg  = 0.5 * ((x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0 + (x_field[I1D(i,j-1,k+1)] - x_field[I1D(i,j-1,k-1)])/2.0);
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0 + (y_field[I1D(i+1,j-1,k)] - y_field[I1D(i-1,j-1,k)])/2.0); 
                y_eta_avg   = y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)];
                y_zeta_avg  = 0.5 * ((y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0 + (y_field[I1D(i,j-1,k+1)] - y_field[I1D(i,j-1,k-1)])/2.0); 
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0 + (z_field[I1D(i+1,j-1,k)] - z_field[I1D(i-1,j-1,k)])/2.0); 
                z_eta_avg   = z_field[I1D(i,j,k)] - z_field[I1D(i,j-1,k)];
                z_zeta_avg  = 0.5 * ((z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0 + (z_field[I1D(i,j-1,k+1)] - z_field[I1D(i,j-1,k-1)])/2.0); 

                eta_x_avg = z_xi_avg*y_zeta_avg - y_xi_avg*z_zeta_avg;    eta_y_avg = x_xi_avg*z_zeta_avg - x_zeta_avg*z_xi_avg;    eta_z_avg = x_zeta_avg*y_xi_avg - x_xi_avg*y_zeta_avg; /// In the form of metric/det_Jacobian evaluated at the cell face
                grad_eta_avg = sqrt( eta_x_avg*eta_x_avg + eta_y_avg*eta_y_avg + eta_z_avg*eta_z_avg);

                n_x_avg = eta_x_avg/grad_eta_avg;   n_y_avg = eta_y_avg/grad_eta_avg;   n_z_avg = eta_z_avg/grad_eta_avg;
                t1_x_avg = -1.0*n_y_avg;            t1_y_avg =  1.0*n_x_avg;            t1_z_avg =  0.0;     
                t2_x_avg = -1.0*n_x_avg*n_z_avg;    t2_y_avg = -1.0*n_y_avg*n_z_avg;    t2_z_avg = n_x_avg*n_x_avg + n_y_avg*n_y_avg; 

                grad_t1 = sqrt(t1_x_avg*t1_x_avg + t1_y_avg*t1_y_avg + t1_z_avg*t1_z_avg);
                grad_t2 = sqrt(t2_x_avg*t2_x_avg + t2_y_avg*t2_y_avg + t2_z_avg*t2_z_avg);

                t1_x_avg = t1_x_avg/grad_t1;
                t1_y_avg = t1_y_avg/grad_t1;
                t1_z_avg = t1_z_avg/grad_t1;
                t2_x_avg = t2_x_avg/grad_t2;
                t2_y_avg = t2_y_avg/grad_t2;
                t2_z_avg = t2_z_avg/grad_t2;

                // printf("normal vector = [%.8f, %.8f, %.8f]\n",n_x_avg,n_y_avg,n_z_avg);
                // printf("tangential vector 1 = [%.8f, %.8f, %.8f]\n",t1_x_avg,t1_y_avg,t1_z_avg);
                // printf("tangential vector 2 = [%.8f, %.8f, %.8f]\n",t2_x_avg,t2_y_avg,t2_z_avg);

                V_n_L      = u_L   * n_x_avg  + v_L   * n_y_avg  + w_L   * n_z_avg;   
                V_n_LL     = u_LL  * n_x_avg  + v_LL  * n_y_avg  + w_LL  * n_z_avg;
                V_n_LLL    = u_LLL * n_x_avg  + v_LLL * n_y_avg  + w_LLL * n_z_avg;
                V_t1_L     = u_L   * t1_x_avg + v_L   * t1_y_avg + w_L   * t1_z_avg;
                V_t1_LL    = u_LL  * t1_x_avg + v_LL  * t1_y_avg + w_LL  * t1_z_avg;
                V_t1_LLL   = u_LLL * t1_x_avg + v_LLL * t1_y_avg + w_LLL * t1_z_avg;
                V_t2_L     = u_L   * t2_x_avg + v_L   * t2_y_avg + w_L   * t2_z_avg;
                V_t2_LL    = u_LL  * t2_x_avg + v_LL  * t2_y_avg + w_LL  * t2_z_avg;
                V_t2_LLL   = u_LLL * t2_x_avg + v_LLL * t2_y_avg + w_LLL * t2_z_avg;

                V_n_R      = u_R   * n_x_avg  + v_R   * n_y_avg  + w_R   * n_z_avg;   
                V_n_RR     = u_RR  * n_x_avg  + v_RR  * n_y_avg  + w_RR  * n_z_avg;
                V_n_RRR    = u_RRR * n_x_avg  + v_RRR * n_y_avg  + w_RRR * n_z_avg;
                V_t1_R     = u_R   * t1_x_avg + v_R   * t1_y_avg + w_R   * t1_z_avg;
                V_t1_RR    = u_RR  * t1_x_avg + v_RR  * t1_y_avg + w_RR  * t1_z_avg;
                V_t1_RRR   = u_RRR * t1_x_avg + v_RRR * t1_y_avg + w_RRR * t1_z_avg;
                V_t2_R     = u_R   * t2_x_avg + v_R   * t2_y_avg + w_R   * t2_z_avg;
                V_t2_RR    = u_RR  * t2_x_avg + v_RR  * t2_y_avg + w_RR  * t2_z_avg;
                V_t2_RRR   = u_RRR * t2_x_avg + v_RRR * t2_y_avg + w_RRR * t2_z_avg;
                
                /// rho
                var_type = 0;
                rho_F_m  = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhou
                var_type = 1;
                rhoV_n_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhov
                var_type = 2;
                rhoV_t1_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhow
                var_type = 3;
                rhoV_t2_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );

                /// Reconstruct flux terms in global physical axes
                rho_F_m     = rho_F_m * grad_eta_avg;
                rhou_F_m    = (rhoV_n_F_m * n_x_avg + rhoV_t1_F_m * t1_x_avg + rhoV_t2_F_m * t2_x_avg) * grad_eta_avg;
                rhov_F_m    = (rhoV_n_F_m * n_y_avg + rhoV_t1_F_m * t1_y_avg + rhoV_t2_F_m * t2_y_avg) * grad_eta_avg;
                rhow_F_m    = (rhoV_n_F_m * n_z_avg + rhoV_t1_F_m * t1_z_avg + rhoV_t2_F_m * t2_z_avg) * grad_eta_avg;
                rhoE_F_m    = rhoE_F_m * grad_eta_avg;
            
                /// Fluxes y-direction
                rho_inv_flux[I1D(i,j,k)]  += det_Jacobian_field[I1D(i,j,k)] * ( rho_F_p - rho_F_m );
                rhou_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhou_F_p - rhou_F_m );
                rhov_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhov_F_p - rhov_F_m );
                rhow_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhow_F_p - rhow_F_m );
                rhoE_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhoE_F_p - rhoE_F_m );
		
                /// z-direction k+1/2
                index_LLL = k - 2; index_LL = k - 1; index_L = k;
		if( k <= ( topo->iter_common[_INNER_][_INIZ_] ) ) { index_LLL = k; index_LL = k; index_L = k; }
                index_R = k + 1; index_RR = k + 2; index_RRR = k + 3;
                if( k >= ( topo->iter_common[_INNER_][_ENDZ_] - 1 ) ) { index_R = k + 1; index_RR = k + 1; index_RRR = k + 1; }
                rho_LLL = rho_field[I1D(i,j,index_LLL)]; rho_RRR = rho_field[I1D(i,j,index_RRR)]; 
                rho_LL  = rho_field[I1D(i,j,index_LL)];  rho_RR  = rho_field[I1D(i,j,index_RR)]; 
                rho_L   = rho_field[I1D(i,j,index_L)];   rho_R   = rho_field[I1D(i,j,index_R)];
                u_LLL   = u_field[I1D(i,j,index_LLL)];   u_RRR   = u_field[I1D(i,j,index_RRR)]; 
                u_LL    = u_field[I1D(i,j,index_LL)];    u_RR    = u_field[I1D(i,j,index_RR)]; 
                u_L     = u_field[I1D(i,j,index_L)];     u_R     = u_field[I1D(i,j,index_R)];
                v_LLL   = v_field[I1D(i,j,index_LLL)];   v_RRR   = v_field[I1D(i,j,index_RRR)]; 
                v_LL    = v_field[I1D(i,j,index_LL)];    v_RR    = v_field[I1D(i,j,index_RR)]; 
                v_L     = v_field[I1D(i,j,index_L)];     v_R     = v_field[I1D(i,j,index_R)];
                w_LLL   = w_field[I1D(i,j,index_LLL)];   w_RRR   = w_field[I1D(i,j,index_RRR)]; 
                w_LL    = w_field[I1D(i,j,index_LL)];    w_RR    = w_field[I1D(i,j,index_RR)]; 
                w_L     = w_field[I1D(i,j,index_L)];     w_R     = w_field[I1D(i,j,index_R)];
                E_LLL   = E_field[I1D(i,j,index_LLL)];   E_RRR   = E_field[I1D(i,j,index_RRR)]; 
                E_LL    = E_field[I1D(i,j,index_LL)];    E_RR    = E_field[I1D(i,j,index_RR)]; 
                E_L     = E_field[I1D(i,j,index_L)];     E_R     = E_field[I1D(i,j,index_R)];
                s_LLL   = s_field[I1D(i,j,index_LLL)];   s_RRR   = s_field[I1D(i,j,index_RRR)]; 
                s_LL    = s_field[I1D(i,j,index_LL)];    s_RR    = s_field[I1D(i,j,index_RR)]; 
                s_L     = s_field[I1D(i,j,index_L)];     s_R     = s_field[I1D(i,j,index_R)];		
                P_LLL   = P_field[I1D(i,j,index_LLL)];   P_RRR   = P_field[I1D(i,j,index_RRR)]; 
                P_LL    = P_field[I1D(i,j,index_LL)];    P_RR    = P_field[I1D(i,j,index_RR)]; 
                P_L     = P_field[I1D(i,j,index_L)];     P_R     = P_field[I1D(i,j,index_R)];
                T_LLL   = T_field[I1D(i,j,index_LLL)];   T_RRR   = T_field[I1D(i,j,index_RRR)]; 
                T_LL    = T_field[I1D(i,j,index_LL)];    T_RR    = T_field[I1D(i,j,index_RR)]; 
                T_L     = T_field[I1D(i,j,index_L)];     T_R     = T_field[I1D(i,j,index_R)];
                a_LLL   = sos_field[I1D(i,j,index_LLL)]; a_RRR   = sos_field[I1D(i,j,index_RRR)]; 
                a_LL    = sos_field[I1D(i,j,index_LL)];  a_RR    = sos_field[I1D(i,j,index_RR)]; 
                a_L     = sos_field[I1D(i,j,index_L)];   a_R     = sos_field[I1D(i,j,index_R)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated

                /// Geometric components
                x_xi_avg    = 0.0; x_eta_avg   = 0.0; x_zeta_avg  = 0.0; 
                y_xi_avg    = 0.0; y_eta_avg   = 0.0; y_zeta_avg  = 0.0; 
                z_xi_avg    = 0.0; z_eta_avg   = 0.0; z_zeta_avg  = 0.0;

                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j,k+1)] - x_field[I1D(i-1,j,k+1)])/2.0 + (x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0);
                x_eta_avg   = 0.5 * ((x_field[I1D(i,j+1,k+1)] - x_field[I1D(i,j-1,k+1)])/2.0 + (x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0);
                x_zeta_avg  = x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k)];
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j,k+1)] - y_field[I1D(i-1,j,k+1)])/2.0 + (y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0);
                y_eta_avg   = 0.5 * ((y_field[I1D(i,j+1,k+1)] - y_field[I1D(i,j-1,k+1)])/2.0 + (y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0);
                y_zeta_avg  = y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k)];
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j,k+1)] - z_field[I1D(i-1,j,k+1)])/2.0 + (z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0);
                z_eta_avg   = 0.5 * ((z_field[I1D(i,j+1,k+1)] - z_field[I1D(i,j-1,k+1)])/2.0 + (z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0);
                z_zeta_avg  = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];

                zeta_x_avg = y_xi_avg*z_eta_avg - y_eta_avg*z_xi_avg;    zeta_y_avg = z_xi_avg*x_eta_avg - z_eta_avg*x_xi_avg;    zeta_z_avg = x_xi_avg*y_eta_avg - x_eta_avg*y_xi_avg; /// In the form of metric/det_Jacobian evaluated at the cell face
                grad_zeta_avg = sqrt( zeta_x_avg*zeta_x_avg + zeta_y_avg*zeta_y_avg + zeta_z_avg*zeta_z_avg);

                n_x_avg = zeta_x_avg/grad_zeta_avg;     n_y_avg = zeta_y_avg/grad_zeta_avg;                 n_z_avg = zeta_z_avg/grad_zeta_avg;
                t1_x_avg =  1.0*n_z_avg;                t1_y_avg =  0.0;                                    t1_z_avg = -1.0*n_x_avg;
                t2_x_avg = -1.0*n_x_avg*n_y_avg;        t2_y_avg = n_x_avg*n_x_avg + n_z_avg*n_z_avg;       t2_z_avg = -1.0*n_y_avg*n_z_avg;

                grad_t1 = sqrt(t1_x_avg*t1_x_avg + t1_y_avg*t1_y_avg + t1_z_avg*t1_z_avg);
                grad_t2 = sqrt(t2_x_avg*t2_x_avg + t2_y_avg*t2_y_avg + t2_z_avg*t2_z_avg);

                t1_x_avg = t1_x_avg/grad_t1;
                t1_y_avg = t1_y_avg/grad_t1;
                t1_z_avg = t1_z_avg/grad_t1;
                t2_x_avg = t2_x_avg/grad_t2;
                t2_y_avg = t2_y_avg/grad_t2;
                t2_z_avg = t2_z_avg/grad_t2;

                // printf("normal vector = [%.8f, %.8f, %.8f]\n",n_x_avg,n_y_avg,n_z_avg);
                // printf("tangential vector 1 = [%.8f, %.8f, %.8f]\n",t1_x_avg,t1_y_avg,t1_z_avg);
                // printf("tangential vector 2 = [%.8f, %.8f, %.8f]\n",t2_x_avg,t2_y_avg,t2_z_avg);

                V_n_L      = u_L   * n_x_avg  + v_L   * n_y_avg  + w_L   * n_z_avg;   
                V_n_LL     = u_LL  * n_x_avg  + v_LL  * n_y_avg  + w_LL  * n_z_avg;
                V_n_LLL    = u_LLL * n_x_avg  + v_LLL * n_y_avg  + w_LLL * n_z_avg;
                V_t1_L     = u_L   * t1_x_avg + v_L   * t1_y_avg + w_L   * t1_z_avg;
                V_t1_LL    = u_LL  * t1_x_avg + v_LL  * t1_y_avg + w_LL  * t1_z_avg;
                V_t1_LLL   = u_LLL * t1_x_avg + v_LLL * t1_y_avg + w_LLL * t1_z_avg;
                V_t2_L     = u_L   * t2_x_avg + v_L   * t2_y_avg + w_L   * t2_z_avg;
                V_t2_LL    = u_LL  * t2_x_avg + v_LL  * t2_y_avg + w_LL  * t2_z_avg;
                V_t2_LLL   = u_LLL * t2_x_avg + v_LLL * t2_y_avg + w_LLL * t2_z_avg;

                V_n_R      = u_R   * n_x_avg  + v_R   * n_y_avg  + w_R   * n_z_avg;   
                V_n_RR     = u_RR  * n_x_avg  + v_RR  * n_y_avg  + w_RR  * n_z_avg;
                V_n_RRR    = u_RRR * n_x_avg  + v_RRR * n_y_avg  + w_RRR * n_z_avg;
                V_t1_R     = u_R   * t1_x_avg + v_R   * t1_y_avg + w_R   * t1_z_avg;
                V_t1_RR    = u_RR  * t1_x_avg + v_RR  * t1_y_avg + w_RR  * t1_z_avg;
                V_t1_RRR   = u_RRR * t1_x_avg + v_RRR * t1_y_avg + w_RRR * t1_z_avg;
                V_t2_R     = u_R   * t2_x_avg + v_R   * t2_y_avg + w_R   * t2_z_avg;
                V_t2_RR    = u_RR  * t2_x_avg + v_RR  * t2_y_avg + w_RR  * t2_z_avg;
                V_t2_RRR   = u_RRR * t2_x_avg + v_RRR * t2_y_avg + w_RRR * t2_z_avg;

                /// rho
                var_type = 0;
                rho_F_p  = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhou
                var_type = 1;
                rhoV_n_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhov
                var_type = 2;
                rhoV_t1_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhow
                var_type = 3;
                rhoV_t2_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                
                // printf("i = %d, j = %d, k = %d // %.8f, %.8f, %.8f\n",i,j,k,rhoV_n_F_p, rhoV_t1_F_p, rhoV_t2_F_p);

                rho_F_p     = rho_F_p * grad_zeta_avg;
                rhou_F_p    = (rhoV_n_F_p * n_x_avg + rhoV_t1_F_p * t1_x_avg + rhoV_t2_F_p * t2_x_avg) * grad_zeta_avg;
                rhov_F_p    = (rhoV_n_F_p * n_y_avg + rhoV_t1_F_p * t1_y_avg + rhoV_t2_F_p * t2_y_avg) * grad_zeta_avg;
                rhow_F_p    = (rhoV_n_F_p * n_z_avg + rhoV_t1_F_p * t1_z_avg + rhoV_t2_F_p * t2_z_avg) * grad_zeta_avg;
                rhoE_F_p    = rhoE_F_p * grad_zeta_avg;

                /// z-direction k-1/2
                index_LLL = k - 3; index_LL = k - 2; index_L = k - 1;
		if( k <= ( topo->iter_common[_INNER_][_INIZ_] + 1 ) ) { index_LLL = k - 1; index_LL = k - 1; index_L = k - 1; }
                index_R = k; index_RR = k + 1; index_RRR = k + 2;
                if( k >= ( topo->iter_common[_INNER_][_ENDZ_] ) ) { index_R = k; index_RR = k; index_RRR = k; }
                rho_LLL = rho_field[I1D(i,j,index_LLL)]; rho_RRR = rho_field[I1D(i,j,index_RRR)]; 
                rho_LL  = rho_field[I1D(i,j,index_LL)];  rho_RR  = rho_field[I1D(i,j,index_RR)]; 
                rho_L   = rho_field[I1D(i,j,index_L)];   rho_R   = rho_field[I1D(i,j,index_R)];
                u_LLL   = u_field[I1D(i,j,index_LLL)];   u_RRR   = u_field[I1D(i,j,index_RRR)]; 
                u_LL    = u_field[I1D(i,j,index_LL)];    u_RR    = u_field[I1D(i,j,index_RR)]; 
                u_L     = u_field[I1D(i,j,index_L)];     u_R     = u_field[I1D(i,j,index_R)];
                v_LLL   = v_field[I1D(i,j,index_LLL)];   v_RRR   = v_field[I1D(i,j,index_RRR)]; 
                v_LL    = v_field[I1D(i,j,index_LL)];    v_RR    = v_field[I1D(i,j,index_RR)]; 
                v_L     = v_field[I1D(i,j,index_L)];     v_R     = v_field[I1D(i,j,index_R)];
                w_LLL   = w_field[I1D(i,j,index_LLL)];   w_RRR   = w_field[I1D(i,j,index_RRR)]; 
                w_LL    = w_field[I1D(i,j,index_LL)];    w_RR    = w_field[I1D(i,j,index_RR)]; 
                w_L     = w_field[I1D(i,j,index_L)];     w_R     = w_field[I1D(i,j,index_R)];
                E_LLL   = E_field[I1D(i,j,index_LLL)];   E_RRR   = E_field[I1D(i,j,index_RRR)]; 
                E_LL    = E_field[I1D(i,j,index_LL)];    E_RR    = E_field[I1D(i,j,index_RR)]; 
                E_L     = E_field[I1D(i,j,index_L)];     E_R     = E_field[I1D(i,j,index_R)];
                s_LLL   = s_field[I1D(i,j,index_LLL)];   s_RRR   = s_field[I1D(i,j,index_RRR)]; 
                s_LL    = s_field[I1D(i,j,index_LL)];    s_RR    = s_field[I1D(i,j,index_RR)]; 
                s_L     = s_field[I1D(i,j,index_L)];     s_R     = s_field[I1D(i,j,index_R)];		
                P_LLL   = P_field[I1D(i,j,index_LLL)];   P_RRR   = P_field[I1D(i,j,index_RRR)]; 
                P_LL    = P_field[I1D(i,j,index_LL)];    P_RR    = P_field[I1D(i,j,index_RR)]; 
                P_L     = P_field[I1D(i,j,index_L)];     P_R     = P_field[I1D(i,j,index_R)];
                T_LLL   = T_field[I1D(i,j,index_LLL)];   T_RRR   = T_field[I1D(i,j,index_RRR)]; 
                T_LL    = T_field[I1D(i,j,index_LL)];    T_RR    = T_field[I1D(i,j,index_RR)]; 
                T_L     = T_field[I1D(i,j,index_L)];     T_R     = T_field[I1D(i,j,index_R)];
                a_LLL   = sos_field[I1D(i,j,index_LLL)]; a_RRR   = sos_field[I1D(i,j,index_RRR)]; 
                a_LL    = sos_field[I1D(i,j,index_LL)];  a_RR    = sos_field[I1D(i,j,index_RR)]; 
                a_L     = sos_field[I1D(i,j,index_L)];   a_R     = sos_field[I1D(i,j,index_R)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                
                /// Geometric components
                x_xi_avg    = 0.0; x_eta_avg   = 0.0; x_zeta_avg  = 0.0; 
                y_xi_avg    = 0.0; y_eta_avg   = 0.0; y_zeta_avg  = 0.0; 
                z_xi_avg    = 0.0; z_eta_avg   = 0.0; z_zeta_avg  = 0.0;

                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0 + (x_field[I1D(i+1,j,k-1)] - x_field[I1D(i-1,j,k-1)])/2.0);
                x_eta_avg   = 0.5 * ((x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0 + (x_field[I1D(i,j+1,k-1)] - x_field[I1D(i,j-1,k-1)])/2.0);
                x_zeta_avg  = x_field[I1D(i,j,k)] - x_field[I1D(i,j,k-1)];
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0 + (y_field[I1D(i+1,j,k-1)] - y_field[I1D(i-1,j,k-1)])/2.0);
                y_eta_avg   = 0.5 * ((y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0 + (y_field[I1D(i,j+1,k-1)] - y_field[I1D(i,j-1,k-1)])/2.0);
                y_zeta_avg  = y_field[I1D(i,j,k)] - y_field[I1D(i,j,k-1)];
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0 + (z_field[I1D(i+1,j,k-1)] - z_field[I1D(i-1,j,k-1)])/2.0);
                z_eta_avg   = 0.5 * ((z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0 + (z_field[I1D(i,j+1,k-1)] - z_field[I1D(i,j-1,k-1)])/2.0);
                z_zeta_avg  = z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)];

                zeta_x_avg = y_xi_avg*z_eta_avg - y_eta_avg*z_xi_avg;    zeta_y_avg = z_xi_avg*x_eta_avg - z_eta_avg*x_xi_avg;    zeta_z_avg = x_xi_avg*y_eta_avg - x_eta_avg*y_xi_avg; /// In the form of metric/det_Jacobian evaluated at the cell face
                grad_zeta_avg = sqrt( zeta_x_avg*zeta_x_avg + zeta_y_avg*zeta_y_avg + zeta_z_avg*zeta_z_avg);

                n_x_avg = zeta_x_avg/grad_zeta_avg;     n_y_avg = zeta_y_avg/grad_zeta_avg;                 n_z_avg = zeta_z_avg/grad_zeta_avg;
                t1_x_avg =  1.0*n_z_avg;                t1_y_avg =  0.0;                                    t1_z_avg = -1.0*n_x_avg;
                t2_x_avg = -1.0*n_x_avg*n_y_avg;        t2_y_avg = n_x_avg*n_x_avg + n_z_avg*n_z_avg;       t2_z_avg = -1.0*n_y_avg*n_z_avg;

                grad_t1 = sqrt(t1_x_avg*t1_x_avg + t1_y_avg*t1_y_avg + t1_z_avg*t1_z_avg);
                grad_t2 = sqrt(t2_x_avg*t2_x_avg + t2_y_avg*t2_y_avg + t2_z_avg*t2_z_avg);

                t1_x_avg = t1_x_avg/grad_t1;
                t1_y_avg = t1_y_avg/grad_t1;
                t1_z_avg = t1_z_avg/grad_t1;
                t2_x_avg = t2_x_avg/grad_t2;
                t2_y_avg = t2_y_avg/grad_t2;
                t2_z_avg = t2_z_avg/grad_t2;

                // printf("normal vector = [%.8f, %.8f, %.8f]\n",n_x_avg,n_y_avg,n_z_avg);
                // printf("tangential vector 1 = [%.8f, %.8f, %.8f]\n",t1_x_avg,t1_y_avg,t1_z_avg);
                // printf("tangential vector 2 = [%.8f, %.8f, %.8f]\n",t2_x_avg,t2_y_avg,t2_z_avg);

                V_n_L      = u_L   * n_x_avg + v_L    * n_y_avg + w_L    * n_z_avg;   
                V_n_LL     = u_LL  * n_x_avg + v_LL   * n_y_avg + w_LL   * n_z_avg;
                V_n_LLL    = u_LLL * n_x_avg + v_LLL  * n_y_avg + w_LLL  * n_z_avg;
                V_t1_L     = u_L   * t1_x_avg + v_L   * t1_y_avg + w_L   * t1_z_avg;
                V_t1_LL    = u_LL  * t1_x_avg + v_LL  * t1_y_avg + w_LL  * t1_z_avg;
                V_t1_LLL   = u_LLL * t1_x_avg + v_LLL * t1_y_avg + w_LLL * t1_z_avg;
                V_t2_L     = u_L   * t2_x_avg + v_L   * t2_y_avg + w_L   * t2_z_avg;
                V_t2_LL    = u_LL  * t2_x_avg + v_LL  * t2_y_avg + w_LL  * t2_z_avg;
                V_t2_LLL   = u_LLL * t2_x_avg + v_LLL * t2_y_avg + w_LLL * t2_z_avg;

                V_n_R      = u_R   * n_x_avg  + v_R   * n_y_avg  + w_R   * n_z_avg;   
                V_n_RR     = u_RR  * n_x_avg  + v_RR  * n_y_avg  + w_RR  * n_z_avg;
                V_n_RRR    = u_RRR * n_x_avg  + v_RRR * n_y_avg  + w_RRR * n_z_avg;
                V_t1_R     = u_R   * t1_x_avg + v_R   * t1_y_avg + w_R   * t1_z_avg;
                V_t1_RR    = u_RR  * t1_x_avg + v_RR  * t1_y_avg + w_RR  * t1_z_avg;
                V_t1_RRR   = u_RRR * t1_x_avg + v_RRR * t1_y_avg + w_RRR * t1_z_avg;
                V_t2_R     = u_R   * t2_x_avg + v_R   * t2_y_avg + w_R   * t2_z_avg;
                V_t2_RR    = u_RR  * t2_x_avg + v_RR  * t2_y_avg + w_RR  * t2_z_avg;
                V_t2_RRR   = u_RRR * t2_x_avg + v_RRR * t2_y_avg + w_RRR * t2_z_avg;

                // printf("i = %d, j = %d, k = %d // %.8f, %.8f, %.8f\n",i,j,k, V_n_L, V_t1_L, V_t2_L);

                /// rho
                var_type = 0;
                rho_F_m  = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhou
                var_type = 1;
                rhoV_n_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhov
                var_type = 2;
                rhoV_t1_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhow
                var_type = 3;
                rhoV_t2_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, V_n_LLL, V_t1_LLL, V_t2_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, V_n_LL, V_t1_LL, V_t2_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, V_n_L, V_t1_L, V_t2_L, E_L, s_L, P_L, T_L, a_L, rho_R, V_n_R, V_t1_R, V_t2_R, E_R, s_R, P_R, T_R, a_R, rho_RR, V_n_RR, V_t1_RR, V_t2_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, V_n_RRR, V_t1_RRR, V_t2_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                
                rho_F_m     = rho_F_m * grad_zeta_avg;
                rhou_F_m    = (rhoV_n_F_m * n_x_avg + rhoV_t1_F_m * t1_x_avg + rhoV_t2_F_m * t2_x_avg) * grad_zeta_avg;
                rhov_F_m    = (rhoV_n_F_m * n_y_avg + rhoV_t1_F_m * t1_y_avg + rhoV_t2_F_m * t2_y_avg) * grad_zeta_avg;
                rhow_F_m    = (rhoV_n_F_m * n_z_avg + rhoV_t1_F_m * t1_z_avg + rhoV_t2_F_m * t2_z_avg) * grad_zeta_avg;
                rhoE_F_m    = rhoE_F_m * grad_zeta_avg;

                /// Fluxes z-direction
                rho_inv_flux[I1D(i,j,k)]  += det_Jacobian_field[I1D(i,j,k)] * ( rho_F_p - rho_F_m );
                rhou_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhou_F_p - rhou_F_m );
                rhov_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhov_F_p - rhov_F_m );
                rhow_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhow_F_p - rhow_F_m );
                rhoE_inv_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * ( rhoE_F_p - rhoE_F_m );

                // printf("i = %d, j = %d, k = %d // %.8f, %.8f, %.8f, %.8f, %.8f\n",i,j,k,( rho_F_p - rho_F_m ), ( rhou_F_p - rhou_F_m ), ( rhov_F_p - rhov_F_m ), ( rhow_F_p - rhow_F_m ), ( rhoE_F_p - rhoE_F_m ));
            }
        }
    }

    /// Update halo values
    //rho_inv_flux.update();
    //rhou_inv_flux.update();
    //rhov_inv_flux.update();
    //rhow_inv_flux.update();
    //rhoE_inv_flux.update();

};

void myRHEA::calculateViscousFluxes() {

    /// Second-order central finite differences for derivatives:
    /// P. Moin.
    /// Fundamentals of engineering numerical analysis.
    /// Cambridge University Press, 2010.

    /// Inner points: rho_vis_flux, rhou_vis_flux, rhov_vis_flux, rhow_vis_flux, rhoE_vis_flux and work_vis_rhoe_flux 
    double d_u_x, d_u_y, d_u_z, d_v_x, d_v_y, d_v_z, d_w_x, d_w_y, d_w_z, d_T_x, d_T_y, d_T_z;
    double x_xi_avg, x_eta_avg, x_zeta_avg, y_xi_avg, y_eta_avg, y_zeta_avg, z_xi_avg, z_eta_avg, z_zeta_avg;
    double xi_x_avg, xi_y_avg, xi_z_avg, eta_x_avg, eta_y_avg, eta_z_avg, zeta_x_avg, zeta_y_avg, zeta_z_avg;
    double xi_x_J_avg, xi_y_J_avg, xi_z_J_avg, eta_x_J_avg, eta_y_J_avg, eta_z_J_avg, zeta_x_J_avg, zeta_y_J_avg, zeta_z_J_avg;
    double mu_avg, kappa_avg, u_avg, v_avg, w_avg, det_J;
    double tau_xx, tau_yy, tau_zz, tau_xy, tau_xz, tau_yz;
    double q_x, q_y, q_z, rhoE_x, rhoE_y, rhoE_z;
    double rhou_F_p, rhou_F_m, rhov_F_p, rhov_F_m, rhow_F_p, rhow_F_m, rhoE_F_p, rhoE_F_m;
    double rhou_G_p, rhou_G_m, rhov_G_p, rhov_G_m, rhow_G_p, rhow_G_m, rhoE_G_p, rhoE_G_m;
    double rhou_H_p, rhou_H_m, rhov_H_p, rhov_H_m, rhow_H_p, rhow_H_m, rhoE_H_p, rhoE_H_m;
    double div_uvw, div_tau_x, div_tau_y, div_tau_z;
    double div_q, div_uvw_tau_rhoe, div_uvw_tau_rhoke, div_uvw_tau_rhoE; 

    #pragma acc parallel loop collapse(3) private(d_u_x, d_u_y, d_u_z, d_v_x, d_v_y, d_v_z, d_w_x, d_w_y, d_w_z, d_T_x, d_T_y, d_T_z, div_uvw, div_tau_x, div_tau_y, div_tau_z, div_q, div_uvw_tau_rhoe, div_uvw_tau_rhoke, div_uvw_tau_rhoE, x_xi_avg, x_eta_avg, x_zeta_avg, y_xi_avg, y_eta_avg, y_zeta_avg, z_xi_avg, z_eta_avg, z_zeta_avg, xi_x_avg, xi_y_avg, xi_z_avg, eta_x_avg, eta_y_avg, eta_z_avg, zeta_x_avg, zeta_y_avg, zeta_z_avg, xi_x_J_avg, xi_y_J_avg, xi_z_J_avg, eta_x_J_avg, eta_y_J_avg, eta_z_J_avg, zeta_x_J_avg, zeta_y_J_avg, zeta_z_J_avg, mu_avg, kappa_avg, u_avg, v_avg, w_avg, det_J, tau_xx, tau_yy, tau_zz, tau_xy, tau_xz, tau_yz, q_x, q_y, q_z, rhoE_x, rhoE_y, rhoE_z, rhou_F_p, rhou_F_m, rhov_F_p, rhov_F_m, rhow_F_p, rhow_F_m, rhoE_F_p, rhoE_F_m, rhou_G_p, rhou_G_m, rhov_G_p, rhov_G_m, rhow_G_p, rhow_G_m, rhoE_G_p, rhoE_G_m, rhou_H_p, rhou_H_m, rhov_H_p, rhov_H_m, rhow_H_p, rhow_H_m, rhoE_H_p, rhoE_H_m) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], T_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], rhou_vis_flux.vector[0:_ls_], rhov_vis_flux.vector[0:_ls_], rhow_vis_flux.vector[0:_ls_], rhoE_vis_flux.vector[0:_ls_], work_vis_rhoe_flux.vector[0:_ls_], det_Jacobian_field.vector[0:_ls_])  
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
            	/// x-direction i+1/2
                /// Metrics
                x_xi_avg    = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];                                                                           
                x_eta_avg   = 0.5 * ((x_field[I1D(i+1,j+1,k)] - x_field[I1D(i+1,j-1,k)])/2.0 + (x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0);
                x_zeta_avg  = 0.5 * ((x_field[I1D(i+1,j,k+1)] - x_field[I1D(i+1,j,k-1)])/2.0 + (x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0); 
                y_xi_avg    = y_field[I1D(i+1,j,k)] - y_field[I1D(i,j,k)];                                                                           
                y_eta_avg   = 0.5 * ((y_field[I1D(i+1,j+1,k)] - y_field[I1D(i+1,j-1,k)])/2.0 + (y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0);
                y_zeta_avg  = 0.5 * ((y_field[I1D(i+1,j,k+1)] - y_field[I1D(i+1,j,k-1)])/2.0 + (y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0); 
                z_xi_avg    = z_field[I1D(i+1,j,k)] - z_field[I1D(i,j,k)];                                                                           
                z_eta_avg   = 0.5 * ((z_field[I1D(i+1,j+1,k)] - z_field[I1D(i+1,j-1,k)])/2.0 + (z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0);
                z_zeta_avg  = 0.5 * ((z_field[I1D(i+1,j,k+1)] - z_field[I1D(i+1,j,k-1)])/2.0 + (z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0); 

                det_J = (x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta_avg*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg)); 
                det_J = 1.0/det_J;

                xi_x_avg    = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg) * det_J; 
                xi_y_avg    = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg) * det_J;
                xi_z_avg    = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg) * det_J;
                eta_x_avg   = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg) * det_J;
                eta_y_avg   = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg) * det_J;
                eta_z_avg   = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg) * det_J;
                zeta_x_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg ) * det_J;
                zeta_y_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg ) * det_J;
                zeta_z_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg ) * det_J;

                xi_x_J_avg  = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg);
                xi_y_J_avg  = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg);
                xi_z_J_avg  = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg);

                /// Average derivatives
                d_u_x = xi_x_avg * ( u_field[I1D(i+1,j,k)] - u_field[I1D(i,j,k)]) + 0.5 * eta_x_avg * ((u_field[I1D(i+1,j+1,k)] - u_field[I1D(i+1,j-1,k)])/2.0 + (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((u_field[I1D(i+1,j,k+1)] - u_field[I1D(i+1,j,k-1)])/2.0 + (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0);
                d_u_y = xi_y_avg * ( u_field[I1D(i+1,j,k)] - u_field[I1D(i,j,k)]) + 0.5 * eta_y_avg * ((u_field[I1D(i+1,j+1,k)] - u_field[I1D(i+1,j-1,k)])/2.0 + (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((u_field[I1D(i+1,j,k+1)] - u_field[I1D(i+1,j,k-1)])/2.0 + (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0);
                d_u_z = xi_z_avg * ( u_field[I1D(i+1,j,k)] - u_field[I1D(i,j,k)]) + 0.5 * eta_z_avg * ((u_field[I1D(i+1,j+1,k)] - u_field[I1D(i+1,j-1,k)])/2.0 + (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((u_field[I1D(i+1,j,k+1)] - u_field[I1D(i+1,j,k-1)])/2.0 + (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0);
                d_v_x = xi_x_avg * ( v_field[I1D(i+1,j,k)] - v_field[I1D(i,j,k)]) + 0.5 * eta_x_avg * ((v_field[I1D(i+1,j+1,k)] - v_field[I1D(i+1,j-1,k)])/2.0 + (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((v_field[I1D(i+1,j,k+1)] - v_field[I1D(i+1,j,k-1)])/2.0 + (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0);
                d_v_y = xi_y_avg * ( v_field[I1D(i+1,j,k)] - v_field[I1D(i,j,k)]) + 0.5 * eta_y_avg * ((v_field[I1D(i+1,j+1,k)] - v_field[I1D(i+1,j-1,k)])/2.0 + (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((v_field[I1D(i+1,j,k+1)] - v_field[I1D(i+1,j,k-1)])/2.0 + (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0);
                d_v_z = xi_z_avg * ( v_field[I1D(i+1,j,k)] - v_field[I1D(i,j,k)]) + 0.5 * eta_z_avg * ((v_field[I1D(i+1,j+1,k)] - v_field[I1D(i+1,j-1,k)])/2.0 + (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((v_field[I1D(i+1,j,k+1)] - v_field[I1D(i+1,j,k-1)])/2.0 + (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0);
                d_w_x = xi_x_avg * ( w_field[I1D(i+1,j,k)] - w_field[I1D(i,j,k)]) + 0.5 * eta_x_avg * ((w_field[I1D(i+1,j+1,k)] - w_field[I1D(i+1,j-1,k)])/2.0 + (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((w_field[I1D(i+1,j,k+1)] - w_field[I1D(i+1,j,k-1)])/2.0 + (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0);
                d_w_y = xi_y_avg * ( w_field[I1D(i+1,j,k)] - w_field[I1D(i,j,k)]) + 0.5 * eta_y_avg * ((w_field[I1D(i+1,j+1,k)] - w_field[I1D(i+1,j-1,k)])/2.0 + (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((w_field[I1D(i+1,j,k+1)] - w_field[I1D(i+1,j,k-1)])/2.0 + (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0);
                d_w_z = xi_z_avg * ( w_field[I1D(i+1,j,k)] - w_field[I1D(i,j,k)]) + 0.5 * eta_z_avg * ((w_field[I1D(i+1,j+1,k)] - w_field[I1D(i+1,j-1,k)])/2.0 + (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((w_field[I1D(i+1,j,k+1)] - w_field[I1D(i+1,j,k-1)])/2.0 + (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0);
                d_T_x = xi_x_avg * ( T_field[I1D(i+1,j,k)] - T_field[I1D(i,j,k)]) + 0.5 * eta_x_avg * ((T_field[I1D(i+1,j+1,k)] - T_field[I1D(i+1,j-1,k)])/2.0 + (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((T_field[I1D(i+1,j,k+1)] - T_field[I1D(i+1,j,k-1)])/2.0 + (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0);
                d_T_y = xi_y_avg * ( T_field[I1D(i+1,j,k)] - T_field[I1D(i,j,k)]) + 0.5 * eta_y_avg * ((T_field[I1D(i+1,j+1,k)] - T_field[I1D(i+1,j-1,k)])/2.0 + (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((T_field[I1D(i+1,j,k+1)] - T_field[I1D(i+1,j,k-1)])/2.0 + (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0);
                d_T_z = xi_z_avg * ( T_field[I1D(i+1,j,k)] - T_field[I1D(i,j,k)]) + 0.5 * eta_z_avg * ((T_field[I1D(i+1,j+1,k)] - T_field[I1D(i+1,j-1,k)])/2.0 + (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((T_field[I1D(i+1,j,k+1)] - T_field[I1D(i+1,j,k-1)])/2.0 + (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0);

                /// Stress tensor
                mu_avg      = 0.5 * (mu_field[I1D(i+1,j,k)] + mu_field[I1D(i,j,k)]);
                kappa_avg   = 0.5 * (kappa_field[I1D(i+1,j,k)] + kappa_field[I1D(i,j,k)]); 
                u_avg       = 0.5 * (u_field[I1D(i+1,j,k)] + u_field[I1D(i,j,k)]);
                v_avg       = 0.5 * (v_field[I1D(i+1,j,k)] + v_field[I1D(i,j,k)]);
                w_avg       = 0.5 * (w_field[I1D(i+1,j,k)] + w_field[I1D(i,j,k)]);

                tau_xx      = 2.0 * mu_avg * (d_u_x - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_yy      = 2.0 * mu_avg * (d_v_y - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_zz      = 2.0 * mu_avg * (d_w_z - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_xy      = mu_avg * (d_u_y + d_v_x);
                tau_xz      = mu_avg * (d_u_z + d_w_x);
                tau_yz      = mu_avg * (d_v_z + d_w_y);

                q_x = -1.0 * kappa_avg * d_T_x;
                q_y = -1.0 * kappa_avg * d_T_y;
                q_z = -1.0 * kappa_avg * d_T_z;

                rhoE_x = u_avg * tau_xx + v_avg * tau_xy + w_avg * tau_xz - q_x;
                rhoE_y = u_avg * tau_xy + v_avg * tau_yy + w_avg * tau_yz - q_y; 
                rhoE_z = u_avg * tau_xz + v_avg * tau_yz + w_avg * tau_zz - q_z;

                /// Flux terms
                rhou_F_p = xi_x_J_avg * tau_xx + xi_y_J_avg * tau_xy + xi_z_J_avg * tau_xz;
                rhov_F_p = xi_x_J_avg * tau_xy + xi_y_J_avg * tau_yy + xi_z_J_avg * tau_yz;
                rhow_F_p = xi_x_J_avg * tau_xz + xi_y_J_avg * tau_yz + xi_z_J_avg * tau_zz;
                rhoE_F_p = xi_x_J_avg * rhoE_x + xi_y_J_avg * rhoE_y + xi_z_J_avg * rhoE_z;

            	/// x-direction i-1/2
                /// Metrics
                x_xi_avg    = x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)];                                                                           
                x_eta_avg   = 0.5 * ((x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0 + (x_field[I1D(i-1,j+1,k)] - x_field[I1D(i-1,j-1,k)])/2.0);
                x_zeta_avg  = 0.5 * ((x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0 + (x_field[I1D(i-1,j,k+1)] - x_field[I1D(i-1,j,k-1)])/2.0); 
                y_xi_avg    = y_field[I1D(i,j,k)] - y_field[I1D(i-1,j,k)];                                                                           
                y_eta_avg   = 0.5 * ((y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0 + (y_field[I1D(i-1,j+1,k)] - y_field[I1D(i-1,j-1,k)])/2.0);
                y_zeta_avg  = 0.5 * ((y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0 + (y_field[I1D(i-1,j,k+1)] - y_field[I1D(i-1,j,k-1)])/2.0); 
                z_xi_avg    = z_field[I1D(i,j,k)] - z_field[I1D(i-1,j,k)];                                                                           
                z_eta_avg   = 0.5 * ((z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0 + (z_field[I1D(i-1,j+1,k)] - z_field[I1D(i-1,j-1,k)])/2.0);
                z_zeta_avg  = 0.5 * ((z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0 + (z_field[I1D(i-1,j,k+1)] - z_field[I1D(i-1,j,k-1)])/2.0); 

                det_J = (x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta_avg*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg)); 
                det_J = 1.0/det_J;

                xi_x_avg    = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg) * det_J; 
                xi_y_avg    = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg) * det_J;
                xi_z_avg    = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg) * det_J;
                eta_x_avg   = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg) * det_J;
                eta_y_avg   = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg) * det_J;
                eta_z_avg   = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg) * det_J;
                zeta_x_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg ) * det_J;
                zeta_y_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg ) * det_J;
                zeta_z_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg ) * det_J;

                xi_x_J_avg  = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg);
                xi_y_J_avg  = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg);
                xi_z_J_avg  = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg);

                /// Average derivatives
                d_u_x = xi_x_avg * ( u_field[I1D(i,j,k)] - u_field[I1D(i-1,j,k)]) + 0.5 * eta_x_avg * ((u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0 + (u_field[I1D(i-1,j+1,k)] - u_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0 + (u_field[I1D(i-1,j,k+1)] - u_field[I1D(i-1,j,k-1)])/2.0);
                d_u_y = xi_y_avg * ( u_field[I1D(i,j,k)] - u_field[I1D(i-1,j,k)]) + 0.5 * eta_y_avg * ((u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0 + (u_field[I1D(i-1,j+1,k)] - u_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0 + (u_field[I1D(i-1,j,k+1)] - u_field[I1D(i-1,j,k-1)])/2.0);
                d_u_z = xi_z_avg * ( u_field[I1D(i,j,k)] - u_field[I1D(i-1,j,k)]) + 0.5 * eta_z_avg * ((u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0 + (u_field[I1D(i-1,j+1,k)] - u_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0 + (u_field[I1D(i-1,j,k+1)] - u_field[I1D(i-1,j,k-1)])/2.0);
                d_v_x = xi_x_avg * ( v_field[I1D(i,j,k)] - v_field[I1D(i-1,j,k)]) + 0.5 * eta_x_avg * ((v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0 + (v_field[I1D(i-1,j+1,k)] - v_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0 + (v_field[I1D(i-1,j,k+1)] - v_field[I1D(i-1,j,k-1)])/2.0);
                d_v_y = xi_y_avg * ( v_field[I1D(i,j,k)] - v_field[I1D(i-1,j,k)]) + 0.5 * eta_y_avg * ((v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0 + (v_field[I1D(i-1,j+1,k)] - v_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0 + (v_field[I1D(i-1,j,k+1)] - v_field[I1D(i-1,j,k-1)])/2.0);
                d_v_z = xi_z_avg * ( v_field[I1D(i,j,k)] - v_field[I1D(i-1,j,k)]) + 0.5 * eta_z_avg * ((v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0 + (v_field[I1D(i-1,j+1,k)] - v_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0 + (v_field[I1D(i-1,j,k+1)] - v_field[I1D(i-1,j,k-1)])/2.0);
                d_w_x = xi_x_avg * ( w_field[I1D(i,j,k)] - w_field[I1D(i-1,j,k)]) + 0.5 * eta_x_avg * ((w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0 + (w_field[I1D(i-1,j+1,k)] - w_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0 + (w_field[I1D(i-1,j,k+1)] - w_field[I1D(i-1,j,k-1)])/2.0);
                d_w_y = xi_y_avg * ( w_field[I1D(i,j,k)] - w_field[I1D(i-1,j,k)]) + 0.5 * eta_y_avg * ((w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0 + (w_field[I1D(i-1,j+1,k)] - w_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0 + (w_field[I1D(i-1,j,k+1)] - w_field[I1D(i-1,j,k-1)])/2.0);
                d_w_z = xi_z_avg * ( w_field[I1D(i,j,k)] - w_field[I1D(i-1,j,k)]) + 0.5 * eta_z_avg * ((w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0 + (w_field[I1D(i-1,j+1,k)] - w_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0 + (w_field[I1D(i-1,j,k+1)] - w_field[I1D(i-1,j,k-1)])/2.0);
                d_T_x = xi_x_avg * ( T_field[I1D(i,j,k)] - T_field[I1D(i-1,j,k)]) + 0.5 * eta_x_avg * ((T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0 + (T_field[I1D(i-1,j+1,k)] - T_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_x_avg * ((T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0 + (T_field[I1D(i-1,j,k+1)] - T_field[I1D(i-1,j,k-1)])/2.0);
                d_T_y = xi_y_avg * ( T_field[I1D(i,j,k)] - T_field[I1D(i-1,j,k)]) + 0.5 * eta_y_avg * ((T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0 + (T_field[I1D(i-1,j+1,k)] - T_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_y_avg * ((T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0 + (T_field[I1D(i-1,j,k+1)] - T_field[I1D(i-1,j,k-1)])/2.0);
                d_T_z = xi_z_avg * ( T_field[I1D(i,j,k)] - T_field[I1D(i-1,j,k)]) + 0.5 * eta_z_avg * ((T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0 + (T_field[I1D(i-1,j+1,k)] - T_field[I1D(i-1,j-1,k)])/2.0) + 0.5 * zeta_z_avg * ((T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0 + (T_field[I1D(i-1,j,k+1)] - T_field[I1D(i-1,j,k-1)])/2.0);

                /// Stress tensor
                mu_avg      = 0.5 * (mu_field[I1D(i,j,k)] + mu_field[I1D(i-1,j,k)]);
                kappa_avg   = 0.5 * (kappa_field[I1D(i,j,k)] + kappa_field[I1D(i-1,j,k)]); 
                u_avg       = 0.5 * (u_field[I1D(i,j,k)] + u_field[I1D(i-1,j,k)]);
                v_avg       = 0.5 * (v_field[I1D(i,j,k)] + v_field[I1D(i-1,j,k)]);
                w_avg       = 0.5 * (w_field[I1D(i,j,k)] + w_field[I1D(i-1,j,k)]);

                tau_xx      = 2.0 * mu_avg * (d_u_x - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_yy      = 2.0 * mu_avg * (d_v_y - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_zz      = 2.0 * mu_avg * (d_w_z - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_xy      = mu_avg * (d_u_y + d_v_x);
                tau_xz      = mu_avg * (d_u_z + d_w_x);
                tau_yz      = mu_avg * (d_v_z + d_w_y);

                q_x = -1.0 * kappa_avg * d_T_x;
                q_y = -1.0 * kappa_avg * d_T_y;
                q_z = -1.0 * kappa_avg * d_T_z;

                rhoE_x = u_avg * tau_xx + v_avg * tau_xy + w_avg * tau_xz - q_x;
                rhoE_y = u_avg * tau_xy + v_avg * tau_yy + w_avg * tau_yz - q_y; 
                rhoE_z = u_avg * tau_xz + v_avg * tau_yz + w_avg * tau_zz - q_z;

                /// Flux terms
                rhou_F_m = xi_x_J_avg * tau_xx + xi_y_J_avg * tau_xy + xi_z_J_avg * tau_xz;
                rhov_F_m = xi_x_J_avg * tau_xy + xi_y_J_avg * tau_yy + xi_z_J_avg * tau_yz;
                rhow_F_m = xi_x_J_avg * tau_xz + xi_y_J_avg * tau_yz + xi_z_J_avg * tau_zz;
                rhoE_F_m = xi_x_J_avg * rhoE_x + xi_y_J_avg * rhoE_y + xi_z_J_avg * rhoE_z;

            	/// y-direction j+1/2
                /// Metrics
                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j+1,k)] - x_field[I1D(i-1,j+1,k)])/2.0 + (x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0);                                                                           
                x_eta_avg   = x_field[I1D(i,j+1,k)] - x_field[I1D(i,j,k)];
                x_zeta_avg  = 0.5 * ((x_field[I1D(i,j+1,k+1)] - x_field[I1D(i,j+1,k-1)])/2.0 + (x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0);
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j+1,k)] - y_field[I1D(i-1,j+1,k)])/2.0 + (y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0); 
                y_eta_avg   = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
                y_zeta_avg  = 0.5 * ((y_field[I1D(i,j+1,k+1)] - y_field[I1D(i,j+1,k-1)])/2.0 + (y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0); 
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j+1,k)] - z_field[I1D(i-1,j+1,k)])/2.0 + (z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0); 
                z_eta_avg   = z_field[I1D(i,j+1,k)] - z_field[I1D(i,j,k)];
                z_zeta_avg  = 0.5 * ((z_field[I1D(i,j+1,k+1)] - z_field[I1D(i,j+1,k-1)])/2.0 + (z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0); 

                det_J = (x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta_avg*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg)); 
                det_J = 1.0/det_J;

                xi_x_avg    = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg) * det_J; 
                xi_y_avg    = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg) * det_J;
                xi_z_avg    = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg) * det_J;
                eta_x_avg   = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg) * det_J;
                eta_y_avg   = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg) * det_J;
                eta_z_avg   = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg) * det_J;
                zeta_x_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg ) * det_J;
                zeta_y_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg ) * det_J;
                zeta_z_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg ) * det_J;

                eta_x_J_avg  = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg);
                eta_y_J_avg  = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg);
                eta_z_J_avg  = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg);

                /// Average derivatives
                d_u_x = 0.5 * xi_x_avg * ((u_field[I1D(i+1,j+1,k)] - u_field[I1D(i-1,j+1,k)])/2.0 + (u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0) + eta_x_avg * (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j,k)]) + 0.5 * zeta_x_avg * ((u_field[I1D(i,j+1,k+1)] - u_field[I1D(i,j+1,k-1)])/2.0 + (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0);
                d_u_y = 0.5 * xi_y_avg * ((u_field[I1D(i+1,j+1,k)] - u_field[I1D(i-1,j+1,k)])/2.0 + (u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0) + eta_y_avg * (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j,k)]) + 0.5 * zeta_y_avg * ((u_field[I1D(i,j+1,k+1)] - u_field[I1D(i,j+1,k-1)])/2.0 + (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0);
                d_u_z = 0.5 * xi_z_avg * ((u_field[I1D(i+1,j+1,k)] - u_field[I1D(i-1,j+1,k)])/2.0 + (u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0) + eta_z_avg * (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j,k)]) + 0.5 * zeta_z_avg * ((u_field[I1D(i,j+1,k+1)] - u_field[I1D(i,j+1,k-1)])/2.0 + (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0);
                d_v_x = 0.5 * xi_x_avg * ((v_field[I1D(i+1,j+1,k)] - v_field[I1D(i-1,j+1,k)])/2.0 + (v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0) + eta_x_avg * (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j,k)]) + 0.5 * zeta_x_avg * ((v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j+1,k-1)])/2.0 + (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0);
                d_v_y = 0.5 * xi_y_avg * ((v_field[I1D(i+1,j+1,k)] - v_field[I1D(i-1,j+1,k)])/2.0 + (v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0) + eta_y_avg * (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j,k)]) + 0.5 * zeta_y_avg * ((v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j+1,k-1)])/2.0 + (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0);
                d_v_z = 0.5 * xi_z_avg * ((v_field[I1D(i+1,j+1,k)] - v_field[I1D(i-1,j+1,k)])/2.0 + (v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0) + eta_z_avg * (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j,k)]) + 0.5 * zeta_z_avg * ((v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j+1,k-1)])/2.0 + (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0);
                d_w_x = 0.5 * xi_x_avg * ((w_field[I1D(i+1,j+1,k)] - w_field[I1D(i-1,j+1,k)])/2.0 + (w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0) + eta_x_avg * (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j,k)]) + 0.5 * zeta_x_avg * ((w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j+1,k-1)])/2.0 + (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0);
                d_w_y = 0.5 * xi_y_avg * ((w_field[I1D(i+1,j+1,k)] - w_field[I1D(i-1,j+1,k)])/2.0 + (w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0) + eta_y_avg * (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j,k)]) + 0.5 * zeta_y_avg * ((w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j+1,k-1)])/2.0 + (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0);
                d_w_z = 0.5 * xi_z_avg * ((w_field[I1D(i+1,j+1,k)] - w_field[I1D(i-1,j+1,k)])/2.0 + (w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0) + eta_z_avg * (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j,k)]) + 0.5 * zeta_z_avg * ((w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j+1,k-1)])/2.0 + (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0);
                d_T_x = 0.5 * xi_x_avg * ((T_field[I1D(i+1,j+1,k)] - T_field[I1D(i-1,j+1,k)])/2.0 + (T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0) + eta_x_avg * (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j,k)]) + 0.5 * zeta_x_avg * ((T_field[I1D(i,j+1,k+1)] - T_field[I1D(i,j+1,k-1)])/2.0 + (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0);
                d_T_y = 0.5 * xi_y_avg * ((T_field[I1D(i+1,j+1,k)] - T_field[I1D(i-1,j+1,k)])/2.0 + (T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0) + eta_y_avg * (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j,k)]) + 0.5 * zeta_y_avg * ((T_field[I1D(i,j+1,k+1)] - T_field[I1D(i,j+1,k-1)])/2.0 + (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0);
                d_T_z = 0.5 * xi_z_avg * ((T_field[I1D(i+1,j+1,k)] - T_field[I1D(i-1,j+1,k)])/2.0 + (T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0) + eta_z_avg * (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j,k)]) + 0.5 * zeta_z_avg * ((T_field[I1D(i,j+1,k+1)] - T_field[I1D(i,j+1,k-1)])/2.0 + (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0);

                /// Stress tensor
                mu_avg      = 0.5 * (mu_field[I1D(i,j+1,k)] + mu_field[I1D(i,j,k)]);
                kappa_avg   = 0.5 * (kappa_field[I1D(i,j+1,k)] + kappa_field[I1D(i,j,k)]); 
                u_avg       = 0.5 * (u_field[I1D(i,j+1,k)] + u_field[I1D(i,j,k)]);
                v_avg       = 0.5 * (v_field[I1D(i,j+1,k)] + v_field[I1D(i,j,k)]);
                w_avg       = 0.5 * (w_field[I1D(i,j+1,k)] + w_field[I1D(i,j,k)]);

                tau_xx      = 2.0 * mu_avg * (d_u_x - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_yy      = 2.0 * mu_avg * (d_v_y - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_zz      = 2.0 * mu_avg * (d_w_z - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_xy      = mu_avg * (d_u_y + d_v_x);
                tau_xz      = mu_avg * (d_u_z + d_w_x);
                tau_yz      = mu_avg * (d_v_z + d_w_y);

                q_x = -1.0 * kappa_avg * d_T_x;
                q_y = -1.0 * kappa_avg * d_T_y;
                q_z = -1.0 * kappa_avg * d_T_z;

                rhoE_x = u_avg * tau_xx + v_avg * tau_xy + w_avg * tau_xz - q_x;
                rhoE_y = u_avg * tau_xy + v_avg * tau_yy + w_avg * tau_yz - q_y; 
                rhoE_z = u_avg * tau_xz + v_avg * tau_yz + w_avg * tau_zz - q_z;

                /// Flux terms
                rhou_G_p = eta_x_J_avg * tau_xx + eta_y_J_avg * tau_xy + eta_z_J_avg * tau_xz;
                rhov_G_p = eta_x_J_avg * tau_xy + eta_y_J_avg * tau_yy + eta_z_J_avg * tau_yz;
                rhow_G_p = eta_x_J_avg * tau_xz + eta_y_J_avg * tau_yz + eta_z_J_avg * tau_zz;
                rhoE_G_p = eta_x_J_avg * rhoE_x + eta_y_J_avg * rhoE_y + eta_z_J_avg * rhoE_z;

            	/// y-direction j-1/2
                /// Metrics
                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0 + (x_field[I1D(i+1,j-1,k)] - x_field[I1D(i-1,j-1,k)])/2.0);                                                                           
                x_eta_avg   = x_field[I1D(i,j,k)] - x_field[I1D(i,j-1,k)];
                x_zeta_avg  = 0.5 * ((x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0 + (x_field[I1D(i,j-1,k+1)] - x_field[I1D(i,j-1,k-1)])/2.0);
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0 + (y_field[I1D(i+1,j-1,k)] - y_field[I1D(i-1,j-1,k)])/2.0); 
                y_eta_avg   = y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)];
                y_zeta_avg  = 0.5 * ((y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0 + (y_field[I1D(i,j-1,k+1)] - y_field[I1D(i,j-1,k-1)])/2.0); 
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0 + (z_field[I1D(i+1,j-1,k)] - z_field[I1D(i-1,j-1,k)])/2.0); 
                z_eta_avg   = z_field[I1D(i,j,k)] - z_field[I1D(i,j-1,k)];
                z_zeta_avg  = 0.5 * ((z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0 + (z_field[I1D(i,j-1,k+1)] - z_field[I1D(i,j-1,k-1)])/2.0); 

                det_J = (x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta_avg*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg)); 
                det_J = 1.0/det_J;

                xi_x_avg    = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg) * det_J; 
                xi_y_avg    = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg) * det_J;
                xi_z_avg    = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg) * det_J;
                eta_x_avg   = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg) * det_J;
                eta_y_avg   = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg) * det_J;
                eta_z_avg   = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg) * det_J;
                zeta_x_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg ) * det_J;
                zeta_y_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg ) * det_J;
                zeta_z_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg ) * det_J;

                eta_x_J_avg  = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg);
                eta_y_J_avg  = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg);
                eta_z_J_avg  = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg);

                /// Average derivatives
                d_u_x = 0.5 * xi_x_avg * ((u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0 + (u_field[I1D(i+1,j-1,k)] - u_field[I1D(i-1,j-1,k)])/2.0) + eta_x_avg * (u_field[I1D(i,j,k)] - u_field[I1D(i,j-1,k)]) + 0.5 * zeta_x_avg * ((u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0 + (u_field[I1D(i,j-1,k+1)] - u_field[I1D(i,j-1,k-1)])/2.0);
                d_u_y = 0.5 * xi_y_avg * ((u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0 + (u_field[I1D(i+1,j-1,k)] - u_field[I1D(i-1,j-1,k)])/2.0) + eta_y_avg * (u_field[I1D(i,j,k)] - u_field[I1D(i,j-1,k)]) + 0.5 * zeta_y_avg * ((u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0 + (u_field[I1D(i,j-1,k+1)] - u_field[I1D(i,j-1,k-1)])/2.0);
                d_u_z = 0.5 * xi_z_avg * ((u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0 + (u_field[I1D(i+1,j-1,k)] - u_field[I1D(i-1,j-1,k)])/2.0) + eta_z_avg * (u_field[I1D(i,j,k)] - u_field[I1D(i,j-1,k)]) + 0.5 * zeta_z_avg * ((u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)])/2.0 + (u_field[I1D(i,j-1,k+1)] - u_field[I1D(i,j-1,k-1)])/2.0);
                d_v_x = 0.5 * xi_x_avg * ((v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0 + (v_field[I1D(i+1,j-1,k)] - v_field[I1D(i-1,j-1,k)])/2.0) + eta_x_avg * (v_field[I1D(i,j,k)] - v_field[I1D(i,j-1,k)]) + 0.5 * zeta_x_avg * ((v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0 + (v_field[I1D(i,j-1,k+1)] - v_field[I1D(i,j-1,k-1)])/2.0);
                d_v_y = 0.5 * xi_y_avg * ((v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0 + (v_field[I1D(i+1,j-1,k)] - v_field[I1D(i-1,j-1,k)])/2.0) + eta_y_avg * (v_field[I1D(i,j,k)] - v_field[I1D(i,j-1,k)]) + 0.5 * zeta_y_avg * ((v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0 + (v_field[I1D(i,j-1,k+1)] - v_field[I1D(i,j-1,k-1)])/2.0);
                d_v_z = 0.5 * xi_z_avg * ((v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0 + (v_field[I1D(i+1,j-1,k)] - v_field[I1D(i-1,j-1,k)])/2.0) + eta_z_avg * (v_field[I1D(i,j,k)] - v_field[I1D(i,j-1,k)]) + 0.5 * zeta_z_avg * ((v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)])/2.0 + (v_field[I1D(i,j-1,k+1)] - v_field[I1D(i,j-1,k-1)])/2.0);
                d_w_x = 0.5 * xi_x_avg * ((w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0 + (w_field[I1D(i+1,j-1,k)] - w_field[I1D(i-1,j-1,k)])/2.0) + eta_x_avg * (w_field[I1D(i,j,k)] - w_field[I1D(i,j-1,k)]) + 0.5 * zeta_x_avg * ((w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0 + (w_field[I1D(i,j-1,k+1)] - w_field[I1D(i,j-1,k-1)])/2.0);
                d_w_y = 0.5 * xi_y_avg * ((w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0 + (w_field[I1D(i+1,j-1,k)] - w_field[I1D(i-1,j-1,k)])/2.0) + eta_y_avg * (w_field[I1D(i,j,k)] - w_field[I1D(i,j-1,k)]) + 0.5 * zeta_y_avg * ((w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0 + (w_field[I1D(i,j-1,k+1)] - w_field[I1D(i,j-1,k-1)])/2.0);
                d_w_z = 0.5 * xi_z_avg * ((w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0 + (w_field[I1D(i+1,j-1,k)] - w_field[I1D(i-1,j-1,k)])/2.0) + eta_z_avg * (w_field[I1D(i,j,k)] - w_field[I1D(i,j-1,k)]) + 0.5 * zeta_z_avg * ((w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)])/2.0 + (w_field[I1D(i,j-1,k+1)] - w_field[I1D(i,j-1,k-1)])/2.0);
                d_T_x = 0.5 * xi_x_avg * ((T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0 + (T_field[I1D(i+1,j-1,k)] - T_field[I1D(i-1,j-1,k)])/2.0) + eta_x_avg * (T_field[I1D(i,j,k)] - T_field[I1D(i,j-1,k)]) + 0.5 * zeta_x_avg * ((T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0 + (T_field[I1D(i,j-1,k+1)] - T_field[I1D(i,j-1,k-1)])/2.0);
                d_T_y = 0.5 * xi_y_avg * ((T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0 + (T_field[I1D(i+1,j-1,k)] - T_field[I1D(i-1,j-1,k)])/2.0) + eta_y_avg * (T_field[I1D(i,j,k)] - T_field[I1D(i,j-1,k)]) + 0.5 * zeta_y_avg * ((T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0 + (T_field[I1D(i,j-1,k+1)] - T_field[I1D(i,j-1,k-1)])/2.0);
                d_T_z = 0.5 * xi_z_avg * ((T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0 + (T_field[I1D(i+1,j-1,k)] - T_field[I1D(i-1,j-1,k)])/2.0) + eta_z_avg * (T_field[I1D(i,j,k)] - T_field[I1D(i,j-1,k)]) + 0.5 * zeta_z_avg * ((T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)])/2.0 + (T_field[I1D(i,j-1,k+1)] - T_field[I1D(i,j-1,k-1)])/2.0);

                /// Stress tensor
                mu_avg      = 0.5 * (mu_field[I1D(i,j,k)] + mu_field[I1D(i,j-1,k)]);
                kappa_avg   = 0.5 * (kappa_field[I1D(i,j,k)] + kappa_field[I1D(i,j-1,k)]); 
                u_avg       = 0.5 * (u_field[I1D(i,j,k)] + u_field[I1D(i,j-1,k)]);
                v_avg       = 0.5 * (v_field[I1D(i,j,k)] + v_field[I1D(i,j-1,k)]);
                w_avg       = 0.5 * (w_field[I1D(i,j,k)] + w_field[I1D(i,j-1,k)]);

                tau_xx      = 2.0 * mu_avg * (d_u_x - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_yy      = 2.0 * mu_avg * (d_v_y - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_zz      = 2.0 * mu_avg * (d_w_z - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_xy      = mu_avg * (d_u_y + d_v_x);
                tau_xz      = mu_avg * (d_u_z + d_w_x);
                tau_yz      = mu_avg * (d_v_z + d_w_y);

                q_x = -1.0 * kappa_avg * d_T_x;
                q_y = -1.0 * kappa_avg * d_T_y;
                q_z = -1.0 * kappa_avg * d_T_z;

                rhoE_x = u_avg * tau_xx + v_avg * tau_xy + w_avg * tau_xz - q_x;
                rhoE_y = u_avg * tau_xy + v_avg * tau_yy + w_avg * tau_yz - q_y; 
                rhoE_z = u_avg * tau_xz + v_avg * tau_yz + w_avg * tau_zz - q_z;

                /// Flux terms
                rhou_G_m = eta_x_J_avg * tau_xx + eta_y_J_avg * tau_xy + eta_z_J_avg * tau_xz;
                rhov_G_m = eta_x_J_avg * tau_xy + eta_y_J_avg * tau_yy + eta_z_J_avg * tau_yz;
                rhow_G_m = eta_x_J_avg * tau_xz + eta_y_J_avg * tau_yz + eta_z_J_avg * tau_zz;
                rhoE_G_m = eta_x_J_avg * rhoE_x + eta_y_J_avg * rhoE_y + eta_z_J_avg * rhoE_z;

            	/// z-direction k+1/2
                /// Metrics
                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j,k+1)] - x_field[I1D(i-1,j,k+1)])/2.0 + (x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0);
                x_eta_avg   = 0.5 * ((x_field[I1D(i,j+1,k+1)] - x_field[I1D(i,j-1,k+1)])/2.0 + (x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0);
                x_zeta_avg  = x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k)];
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j,k+1)] - y_field[I1D(i-1,j,k+1)])/2.0 + (y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0);
                y_eta_avg   = 0.5 * ((y_field[I1D(i,j+1,k+1)] - y_field[I1D(i,j-1,k+1)])/2.0 + (y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0);
                y_zeta_avg  = y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k)];
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j,k+1)] - z_field[I1D(i-1,j,k+1)])/2.0 + (z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0);
                z_eta_avg   = 0.5 * ((z_field[I1D(i,j+1,k+1)] - z_field[I1D(i,j-1,k+1)])/2.0 + (z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0);
                z_zeta_avg  = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];

                det_J = (x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta_avg*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg)); 
                det_J = 1.0/det_J;

                xi_x_avg    = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg) * det_J; 
                xi_y_avg    = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg) * det_J;
                xi_z_avg    = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg) * det_J;
                eta_x_avg   = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg) * det_J;
                eta_y_avg   = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg) * det_J;
                eta_z_avg   = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg) * det_J;
                zeta_x_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg ) * det_J;
                zeta_y_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg ) * det_J;
                zeta_z_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg ) * det_J;

                zeta_x_J_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg );
                zeta_y_J_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg );
                zeta_z_J_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg );

                /// Average derivatives
                d_u_x = 0.5 * xi_x_avg * ((u_field[I1D(i+1,j,k+1)] - u_field[I1D(i-1,j,k+1)])/2.0 + (u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_x_avg * ((u_field[I1D(i,j+1,k+1)] - u_field[I1D(i,j-1,k+1)])/2.0 + (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0) + zeta_x_avg * (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k)]);
                d_u_y = 0.5 * xi_y_avg * ((u_field[I1D(i+1,j,k+1)] - u_field[I1D(i-1,j,k+1)])/2.0 + (u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_y_avg * ((u_field[I1D(i,j+1,k+1)] - u_field[I1D(i,j-1,k+1)])/2.0 + (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0) + zeta_y_avg * (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k)]);
                d_u_z = 0.5 * xi_z_avg * ((u_field[I1D(i+1,j,k+1)] - u_field[I1D(i-1,j,k+1)])/2.0 + (u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_z_avg * ((u_field[I1D(i,j+1,k+1)] - u_field[I1D(i,j-1,k+1)])/2.0 + (u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0) + zeta_z_avg * (u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k)]);
                d_v_x = 0.5 * xi_x_avg * ((v_field[I1D(i+1,j,k+1)] - v_field[I1D(i-1,j,k+1)])/2.0 + (v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_x_avg * ((v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j-1,k+1)])/2.0 + (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0) + zeta_x_avg * (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k)]);
                d_v_y = 0.5 * xi_y_avg * ((v_field[I1D(i+1,j,k+1)] - v_field[I1D(i-1,j,k+1)])/2.0 + (v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_y_avg * ((v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j-1,k+1)])/2.0 + (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0) + zeta_y_avg * (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k)]);
                d_v_z = 0.5 * xi_z_avg * ((v_field[I1D(i+1,j,k+1)] - v_field[I1D(i-1,j,k+1)])/2.0 + (v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_z_avg * ((v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j-1,k+1)])/2.0 + (v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0) + zeta_z_avg * (v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k)]);
                d_w_x = 0.5 * xi_x_avg * ((w_field[I1D(i+1,j,k+1)] - w_field[I1D(i-1,j,k+1)])/2.0 + (w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_x_avg * ((w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j-1,k+1)])/2.0 + (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0) + zeta_x_avg * (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k)]);
                d_w_y = 0.5 * xi_y_avg * ((w_field[I1D(i+1,j,k+1)] - w_field[I1D(i-1,j,k+1)])/2.0 + (w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_y_avg * ((w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j-1,k+1)])/2.0 + (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0) + zeta_y_avg * (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k)]);
                d_w_z = 0.5 * xi_z_avg * ((w_field[I1D(i+1,j,k+1)] - w_field[I1D(i-1,j,k+1)])/2.0 + (w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_z_avg * ((w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j-1,k+1)])/2.0 + (w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0) + zeta_z_avg * (w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k)]);
                d_T_x = 0.5 * xi_x_avg * ((T_field[I1D(i+1,j,k+1)] - T_field[I1D(i-1,j,k+1)])/2.0 + (T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_x_avg * ((T_field[I1D(i,j+1,k+1)] - T_field[I1D(i,j-1,k+1)])/2.0 + (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0) + zeta_x_avg * (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k)]);
                d_T_y = 0.5 * xi_y_avg * ((T_field[I1D(i+1,j,k+1)] - T_field[I1D(i-1,j,k+1)])/2.0 + (T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_y_avg * ((T_field[I1D(i,j+1,k+1)] - T_field[I1D(i,j-1,k+1)])/2.0 + (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0) + zeta_y_avg * (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k)]);
                d_T_z = 0.5 * xi_z_avg * ((T_field[I1D(i+1,j,k+1)] - T_field[I1D(i-1,j,k+1)])/2.0 + (T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0) + 0.5 * eta_z_avg * ((T_field[I1D(i,j+1,k+1)] - T_field[I1D(i,j-1,k+1)])/2.0 + (T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0) + zeta_z_avg * (T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k)]);
                
                /// Stress tensor
                mu_avg      = 0.5 * (mu_field[I1D(i,j,k+1)] + mu_field[I1D(i,j,k)]);
                kappa_avg   = 0.5 * (kappa_field[I1D(i,j,k+1)] + kappa_field[I1D(i,j,k)]); 
                u_avg       = 0.5 * (u_field[I1D(i,j,k+1)] + u_field[I1D(i,j,k)]);
                v_avg       = 0.5 * (v_field[I1D(i,j,k+1)] + v_field[I1D(i,j,k)]);
                w_avg       = 0.5 * (w_field[I1D(i,j,k+1)] + w_field[I1D(i,j,k)]);

                tau_xx      = 2.0 * mu_avg * (d_u_x - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_yy      = 2.0 * mu_avg * (d_v_y - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_zz      = 2.0 * mu_avg * (d_w_z - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_xy      = mu_avg * (d_u_y + d_v_x);
                tau_xz      = mu_avg * (d_u_z + d_w_x);
                tau_yz      = mu_avg * (d_v_z + d_w_y);

                q_x = -1.0 * kappa_avg * d_T_x;
                q_y = -1.0 * kappa_avg * d_T_y;
                q_z = -1.0 * kappa_avg * d_T_z;

                rhoE_x = u_avg * tau_xx + v_avg * tau_xy + w_avg * tau_xz - q_x;
                rhoE_y = u_avg * tau_xy + v_avg * tau_yy + w_avg * tau_yz - q_y; 
                rhoE_z = u_avg * tau_xz + v_avg * tau_yz + w_avg * tau_zz - q_z;

                /// Flux terms
                rhou_H_p = zeta_x_J_avg * tau_xx + zeta_y_J_avg * tau_xy + zeta_z_J_avg * tau_xz;
                rhov_H_p = zeta_x_J_avg * tau_xy + zeta_y_J_avg * tau_yy + zeta_z_J_avg * tau_yz;
                rhow_H_p = zeta_x_J_avg * tau_xz + zeta_y_J_avg * tau_yz + zeta_z_J_avg * tau_zz;
                rhoE_H_p = zeta_x_J_avg * rhoE_x + zeta_y_J_avg * rhoE_y + zeta_z_J_avg * rhoE_z;

            	/// z-direction k-1/2
                /// Metrics
                x_xi_avg    = 0.5 * ((x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0 + (x_field[I1D(i+1,j,k-1)] - x_field[I1D(i-1,j,k-1)])/2.0);
                x_eta_avg   = 0.5 * ((x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0 + (x_field[I1D(i,j+1,k-1)] - x_field[I1D(i,j-1,k-1)])/2.0);
                x_zeta_avg  = x_field[I1D(i,j,k)] - x_field[I1D(i,j,k-1)];
                y_xi_avg    = 0.5 * ((y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0 + (y_field[I1D(i+1,j,k-1)] - y_field[I1D(i-1,j,k-1)])/2.0);
                y_eta_avg   = 0.5 * ((y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0 + (y_field[I1D(i,j+1,k-1)] - y_field[I1D(i,j-1,k-1)])/2.0);
                y_zeta_avg  = y_field[I1D(i,j,k)] - y_field[I1D(i,j,k-1)];
                z_xi_avg    = 0.5 * ((z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0 + (z_field[I1D(i+1,j,k-1)] - z_field[I1D(i-1,j,k-1)])/2.0);
                z_eta_avg   = 0.5 * ((z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0 + (z_field[I1D(i,j+1,k-1)] - z_field[I1D(i,j-1,k-1)])/2.0);
                z_zeta_avg  = z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)];

                det_J = (x_xi_avg*y_eta_avg*z_zeta_avg + x_eta_avg*y_zeta_avg*z_xi_avg + x_zeta_avg*y_xi_avg*z_eta_avg - (x_zeta_avg*y_eta_avg*z_xi_avg + x_eta_avg*y_xi_avg*z_zeta_avg + x_xi_avg*y_zeta_avg*z_eta_avg)); 
                det_J = 1.0/det_J;

                xi_x_avg    = (y_eta_avg* z_zeta_avg - z_eta_avg * y_zeta_avg) * det_J; 
                xi_y_avg    = (z_eta_avg* x_zeta_avg - x_eta_avg * z_zeta_avg) * det_J;
                xi_z_avg    = (x_eta_avg* y_zeta_avg - y_eta_avg * x_zeta_avg) * det_J;
                eta_x_avg   = (z_xi_avg * y_zeta_avg - y_xi_avg  * z_zeta_avg) * det_J;
                eta_y_avg   = (x_xi_avg * z_zeta_avg - z_xi_avg  * x_zeta_avg) * det_J;
                eta_z_avg   = (y_xi_avg * x_zeta_avg - x_xi_avg  * y_zeta_avg) * det_J;
                zeta_x_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg ) * det_J;
                zeta_y_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg ) * det_J;
                zeta_z_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg ) * det_J;

                zeta_x_J_avg  = (y_xi_avg * z_eta_avg  - z_xi_avg  * y_eta_avg);
                zeta_y_J_avg  = (z_xi_avg * x_eta_avg  - x_xi_avg  * z_eta_avg);
                zeta_z_J_avg  = (x_xi_avg * y_eta_avg  - y_xi_avg  * x_eta_avg);

                /// Average derivatives
                d_u_x = 0.5 * xi_x_avg * ((u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0 + (u_field[I1D(i+1,j,k-1)] - u_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_x_avg * ((u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0 + (u_field[I1D(i,j+1,k-1)] - u_field[I1D(i,j-1,k-1)])/2.0) + zeta_x_avg * (u_field[I1D(i,j,k)] - u_field[I1D(i,j,k-1)]);
                d_u_y = 0.5 * xi_y_avg * ((u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0 + (u_field[I1D(i+1,j,k-1)] - u_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_y_avg * ((u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0 + (u_field[I1D(i,j+1,k-1)] - u_field[I1D(i,j-1,k-1)])/2.0) + zeta_y_avg * (u_field[I1D(i,j,k)] - u_field[I1D(i,j,k-1)]);
                d_u_z = 0.5 * xi_z_avg * ((u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)])/2.0 + (u_field[I1D(i+1,j,k-1)] - u_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_z_avg * ((u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)])/2.0 + (u_field[I1D(i,j+1,k-1)] - u_field[I1D(i,j-1,k-1)])/2.0) + zeta_z_avg * (u_field[I1D(i,j,k)] - u_field[I1D(i,j,k-1)]);
                d_v_x = 0.5 * xi_x_avg * ((v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0 + (v_field[I1D(i+1,j,k-1)] - v_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_x_avg * ((v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0 + (v_field[I1D(i,j+1,k-1)] - v_field[I1D(i,j-1,k-1)])/2.0) + zeta_x_avg * (v_field[I1D(i,j,k)] - v_field[I1D(i,j,k-1)]);
                d_v_y = 0.5 * xi_y_avg * ((v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0 + (v_field[I1D(i+1,j,k-1)] - v_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_y_avg * ((v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0 + (v_field[I1D(i,j+1,k-1)] - v_field[I1D(i,j-1,k-1)])/2.0) + zeta_y_avg * (v_field[I1D(i,j,k)] - v_field[I1D(i,j,k-1)]);
                d_v_z = 0.5 * xi_z_avg * ((v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)])/2.0 + (v_field[I1D(i+1,j,k-1)] - v_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_z_avg * ((v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)])/2.0 + (v_field[I1D(i,j+1,k-1)] - v_field[I1D(i,j-1,k-1)])/2.0) + zeta_z_avg * (v_field[I1D(i,j,k)] - v_field[I1D(i,j,k-1)]);
                d_w_x = 0.5 * xi_x_avg * ((w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0 + (w_field[I1D(i+1,j,k-1)] - w_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_x_avg * ((w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0 + (w_field[I1D(i,j+1,k-1)] - w_field[I1D(i,j-1,k-1)])/2.0) + zeta_x_avg * (w_field[I1D(i,j,k)] - w_field[I1D(i,j,k-1)]);
                d_w_y = 0.5 * xi_y_avg * ((w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0 + (w_field[I1D(i+1,j,k-1)] - w_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_y_avg * ((w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0 + (w_field[I1D(i,j+1,k-1)] - w_field[I1D(i,j-1,k-1)])/2.0) + zeta_y_avg * (w_field[I1D(i,j,k)] - w_field[I1D(i,j,k-1)]);
                d_w_z = 0.5 * xi_z_avg * ((w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)])/2.0 + (w_field[I1D(i+1,j,k-1)] - w_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_z_avg * ((w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)])/2.0 + (w_field[I1D(i,j+1,k-1)] - w_field[I1D(i,j-1,k-1)])/2.0) + zeta_z_avg * (w_field[I1D(i,j,k)] - w_field[I1D(i,j,k-1)]);
                d_T_x = 0.5 * xi_x_avg * ((T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0 + (T_field[I1D(i+1,j,k-1)] - T_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_x_avg * ((T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0 + (T_field[I1D(i,j+1,k-1)] - T_field[I1D(i,j-1,k-1)])/2.0) + zeta_x_avg * (T_field[I1D(i,j,k)] - T_field[I1D(i,j,k-1)]);
                d_T_y = 0.5 * xi_y_avg * ((T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0 + (T_field[I1D(i+1,j,k-1)] - T_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_y_avg * ((T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0 + (T_field[I1D(i,j+1,k-1)] - T_field[I1D(i,j-1,k-1)])/2.0) + zeta_y_avg * (T_field[I1D(i,j,k)] - T_field[I1D(i,j,k-1)]);
                d_T_z = 0.5 * xi_z_avg * ((T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)])/2.0 + (T_field[I1D(i+1,j,k-1)] - T_field[I1D(i-1,j,k-1)])/2.0) + 0.5 * eta_z_avg * ((T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)])/2.0 + (T_field[I1D(i,j+1,k-1)] - T_field[I1D(i,j-1,k-1)])/2.0) + zeta_z_avg * (T_field[I1D(i,j,k)] - T_field[I1D(i,j,k-1)]);
 
                /// Stress tensor
                mu_avg      = 0.5 * (mu_field[I1D(i,j,k)] + mu_field[I1D(i,j,k-1)]);
                kappa_avg   = 0.5 * (kappa_field[I1D(i,j,k)] + kappa_field[I1D(i,j,k-1)]); 
                u_avg       = 0.5 * (u_field[I1D(i,j,k)] + u_field[I1D(i,j,k-1)]);
                v_avg       = 0.5 * (v_field[I1D(i,j,k)] + v_field[I1D(i,j,k-1)]);
                w_avg       = 0.5 * (w_field[I1D(i,j,k)] + w_field[I1D(i,j,k-1)]);

                tau_xx      = 2.0 * mu_avg * (d_u_x - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_yy      = 2.0 * mu_avg * (d_v_y - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_zz      = 2.0 * mu_avg * (d_w_z - 1.0/3.0 * (d_u_x + d_v_y + d_w_z));
                tau_xy      = mu_avg * (d_u_y + d_v_x);
                tau_xz      = mu_avg * (d_u_z + d_w_x);
                tau_yz      = mu_avg * (d_v_z + d_w_y);

                q_x = -1.0 * kappa_avg * d_T_x;
                q_y = -1.0 * kappa_avg * d_T_y;
                q_z = -1.0 * kappa_avg * d_T_z;

                rhoE_x = u_avg * tau_xx + v_avg * tau_xy + w_avg * tau_xz - q_x;
                rhoE_y = u_avg * tau_xy + v_avg * tau_yy + w_avg * tau_yz - q_y; 
                rhoE_z = u_avg * tau_xz + v_avg * tau_yz + w_avg * tau_zz - q_z;

                /// Flux terms
                rhou_H_m = zeta_x_J_avg * tau_xx + zeta_y_J_avg * tau_xy + zeta_z_J_avg * tau_xz;
                rhov_H_m = zeta_x_J_avg * tau_xy + zeta_y_J_avg * tau_yy + zeta_z_J_avg * tau_yz;
                rhow_H_m = zeta_x_J_avg * tau_xz + zeta_y_J_avg * tau_yz + zeta_z_J_avg * tau_zz;
                rhoE_H_m = zeta_x_J_avg * rhoE_x + zeta_y_J_avg * rhoE_y + zeta_z_J_avg * rhoE_z;

                /// Viscous fluxes
                /// x-direction
                rhou_vis_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * (rhou_F_p - rhou_F_m);
                rhov_vis_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * (rhov_F_p - rhov_F_m);
                rhow_vis_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * (rhow_F_p - rhow_F_m);
                rhoE_vis_flux[I1D(i,j,k)] = det_Jacobian_field[I1D(i,j,k)] * (rhoE_F_p - rhoE_F_m);

                /// y-direction
                rhou_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhou_G_p - rhou_G_m);
                rhov_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhov_G_p - rhov_G_m);
                rhow_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhow_G_p - rhow_G_m);
                rhoE_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhoE_G_p - rhoE_G_m);

                /// z-direction
                rhou_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhou_H_p - rhou_H_m);
                rhov_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhov_H_p - rhov_H_m);
                rhow_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhow_H_p - rhow_H_m);
                rhoE_vis_flux[I1D(i,j,k)] += det_Jacobian_field[I1D(i,j,k)] * (rhoE_H_p - rhoE_H_m);

                /// Work of viscous stresses for internal energy
                // div_uvw_tau_rhoe = tau_xx*d_u_x + tau_xy*d_u_y + tau_xz*d_u_z
                //                  + tau_xy*d_v_x + tau_yy*d_v_y + tau_yz*d_v_z
                //                  + tau_xz*d_w_x + tau_yz*d_w_y + tau_zz*d_w_z;
                /// Work of viscous stresses for kinetic energy
                // div_uvw_tau_rhoke = u_field[I1D(i,j,k)]*div_tau_x + v_field[I1D(i,j,k)]*div_tau_y + w_field[I1D(i,j,k)]*div_tau_z;
                /// Work of viscous stresses for total energy
                // div_uvw_tau_rhoE = div_uvw_tau_rhoe + div_uvw_tau_rhoke;
                /// Viscous fluxes
                // rhou_vis_flux[I1D(i,j,k)]      = div_tau_x;
                // rhov_vis_flux[I1D(i,j,k)]      = div_tau_y;
                // rhow_vis_flux[I1D(i,j,k)]      = div_tau_z;
                // rhoE_vis_flux[I1D(i,j,k)]      = ( -1.0 )*div_q + div_uvw_tau_rhoE;
                // work_vis_rhoe_flux[I1D(i,j,k)] = ( -1.0 )*div_q + div_uvw_tau_rhoe;
                work_vis_rhoe_flux[I1D(i,j,k)] = 1.0;
            }
        }
    }

    /// Update halo values
    //rhou_vis_flux.update();
    //rhov_vis_flux.update();
    //rhow_vis_flux.update();
    //rhoE_vis_flux.update();
    //work_vis_rhoe_flux.update();

};

void myRHEA::calculateTimeStep() {

    /// Inviscid time step size for explicit schemes:
    /// E. F. Toro.
    /// Riemann solvers and numerical methods for fluid dynamics.
    /// Springer, 2009.

    /// Viscous time step size for explicit schemes:
    /// E. Turkel, R.C. Swanson, V. N. Vatsa, J.A. White.
    /// Multigrid for hypersonic viscous two- and three-dimensional flows.
    /// NASA Contractor Report 187603, 1991.

    /// Initialize to largest double value
    double local_delta_t = numeric_limits<double>::max();

    /// Inner points: find minimum (local) delta_t
    double sos, c_p;
    double delta_x, delta_y, delta_z;
    double S_x, S_y, S_z;
    double x_xi, x_eta, x_zeta, y_xi, y_eta, y_zeta, z_xi, z_eta, z_zeta;
    double xi_x, xi_y, xi_z, eta_x, eta_y, eta_z, zeta_x, zeta_y, zeta_z;
    double det_J, grad_xi, grad_eta, grad_zeta;
    double U, V, W;
    #pragma acc parallel loop collapse(3) independent private(delta_x,delta_y,delta_z, S_x,S_y,S_z,c_p,sos,x_xi, x_eta, x_zeta, y_xi, y_eta, y_zeta, z_xi, z_eta, z_zeta, xi_x, xi_y, xi_z, eta_x, eta_y, eta_z, zeta_x, zeta_y, zeta_z, det_J, grad_xi, grad_eta, grad_zeta, U, V, W) reduction(min:local_delta_t) present(this, rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_p_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_])  
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Speed of sound
		        sos = sos_field[I1D(i,j,k)];
                /// Heat capacities
                //thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                c_p = c_p_field[I1D(i,j,k)];
                /// Geometric stuff
                delta_x = 1.0; 
                delta_y = 1.0; 
                delta_z = 1.0;
                /// Metrics
                x_xi    = (x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)])/2.0;
                x_eta   = (x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0;
                x_zeta  = (x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0;

                y_xi    = (y_field[I1D(i+1,j,k)] - y_field[I1D(i-1,j,k)])/2.0;
                y_eta   = (y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0;
                y_zeta  = (y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0;

                z_xi    = (z_field[I1D(i+1,j,k)] - z_field[I1D(i-1,j,k)])/2.0;
                z_eta   = (z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0;
                z_zeta  = (z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0;

                det_J = (x_xi*y_eta*z_zeta + x_eta*y_zeta*z_xi + x_zeta*y_xi*z_eta - x_zeta*y_eta*z_xi - x_eta*y_xi*z_zeta - x_xi*y_zeta*z_eta);
                det_J = 1.0/det_J;

                xi_x   = (y_eta*z_zeta - y_zeta*z_eta) * det_J;
                xi_y   = (z_eta*x_zeta - z_zeta*x_eta) * det_J;
                xi_z   = (x_eta*y_zeta - x_zeta*y_eta) * det_J;
                eta_x  = (y_zeta*z_xi  - y_xi*z_zeta ) * det_J;
                eta_y  = (z_zeta*x_xi  - z_xi*x_zeta ) * det_J;
                eta_z  = (x_zeta*y_xi  - x_xi*y_zeta ) * det_J;
                zeta_x = (y_xi*z_eta   - y_eta*z_xi  ) * det_J;
                zeta_y = (z_xi*x_eta   - z_eta*x_xi  ) * det_J;
                zeta_z = (x_xi*y_eta   - x_eta*y_xi  ) * det_J;

                grad_xi     = sqrt(xi_x*xi_x + xi_y*xi_y + xi_z*xi_z);
                grad_eta    = sqrt(eta_x*eta_x + eta_y*eta_y + eta_z*eta_z);
                grad_zeta   = sqrt(zeta_x*zeta_x + zeta_y*zeta_y + zeta_z*zeta_z);
                /// Contravariant velocities
                U = xi_x    * u_field[I1D(i,j,k)] + xi_y    * v_field[I1D(i,j,k)] + xi_z    * w_field[I1D(i,j,k)];
                V = eta_x   * u_field[I1D(i,j,k)] + eta_y   * v_field[I1D(i,j,k)] + eta_z   * w_field[I1D(i,j,k)]; 
                W = zeta_x  * u_field[I1D(i,j,k)] + zeta_y  * v_field[I1D(i,j,k)] + zeta_z  * w_field[I1D(i,j,k)];
                /// x-direction inviscid, viscous & thermal terms
                S_x           = abs(U) + sos*grad_xi;
                local_delta_t = min( local_delta_t, CFL*delta_x/S_x );											/// acoustic scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*pow( delta_x, 2.0 )/max( mu_field[I1D(i,j,k)], epsilon ) );		/// viscous scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*c_p*pow( delta_x, 2.0 )/max( kappa_field[I1D(i,j,k)], epsilon ) );	/// thermal diffusivity scale
                /// y-direction inviscid, viscous & thermal terms
                S_y           = abs(V) + sos*grad_eta;
                local_delta_t = min( local_delta_t, CFL*delta_y/S_y );											/// acoustic scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*pow( delta_y, 2.0 )/max( mu_field[I1D(i,j,k)], epsilon ) );		/// viscous scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*c_p*pow( delta_y, 2.0 )/max( kappa_field[I1D(i,j,k)], epsilon ) );	/// thermal diffusivity scale
                /// z-direction inviscid, viscous & thermal terms
                S_z           = abs(W) + sos*grad_zeta;
                local_delta_t = min( local_delta_t, CFL*delta_z/S_z );											/// acoustic scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*pow( delta_z, 2.0 )/max( mu_field[I1D(i,j,k)], epsilon ) );		/// viscous scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*c_p*pow( delta_z, 2.0 )/max( kappa_field[I1D(i,j,k)], epsilon ) );	/// thermal diffusivity scale
            }
        }
    }

    /// Iterate through local particles in use
    if( ( point_particles->get_num_prts_total() > 0 ) and !activate_pure_tracer_particles ) {

        int i_local_index, j_local_index, k_local_index;
        double u_velocity_particle, v_velocity_particle, w_velocity_particle;   
	int capacity = point_particles->get_prt_capacity();
        #pragma acc parallel loop collapse (1) private(i_local_index, j_local_index, k_local_index, u_velocity_particle, v_velocity_particle, w_velocity_particle, delta_z, delta_y, delta_x) present(this, mesh, topo, point_particles, x_field.vector[0:_ls_], v_field.vector[0:_ls_], z_field.vector[0:_ls_], point_particles->local_prts_velocities_0_x[0:capacity], point_particles->local_prts_velocities_0_y[0:capacity], point_particles->local_prts_velocities_0_z[0:capacity], point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity]) reduction(min:local_delta_t)
        for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

            /// Obtain Lagrangian-Eulerian indexes 0
            i_local_index = point_particles->local_prts_indexes_0_i[p];		/// Local index i
            j_local_index = point_particles->local_prts_indexes_0_j[p];		/// Local index j
            k_local_index = point_particles->local_prts_indexes_0_k[p];		/// Local index k

            /// Obtain velocity of point particle
            u_velocity_particle = point_particles->local_prts_velocities_0_x[p];
            v_velocity_particle = point_particles->local_prts_velocities_0_y[p];
            w_velocity_particle = point_particles->local_prts_velocities_0_z[p];

            // / Geometric stuff
            delta_x = 0.5*( x_field[I1D(i_local_index+1,j_local_index,k_local_index)] - x_field[I1D(i_local_index-1,j_local_index,k_local_index)] ); 
            delta_y = 0.5*( y_field[I1D(i_local_index,j_local_index+1,k_local_index)] - y_field[I1D(i_local_index,j_local_index-1,k_local_index)] ); 
            delta_z = 0.5*( z_field[I1D(i_local_index,j_local_index,k_local_index+1)] - z_field[I1D(i_local_index,j_local_index,k_local_index-1)] );

            /// x-direction particle term
            local_delta_t = min( local_delta_t, CFL*delta_x/max( abs( u_velocity_particle ), epsilon ) );	/// Particle scale
            /// y-direction particle term
            local_delta_t = min( local_delta_t, CFL*delta_y/max( abs( v_velocity_particle ), epsilon ) );	/// Particle scale
            /// z-direction particle term
            local_delta_t = min( local_delta_t, CFL*delta_z/max( abs( w_velocity_particle ), epsilon ) );	/// Particle scale
        
        }

    }

    /// Find minimum (global) delta_t
    double global_delta_t;
    MPI_Allreduce(&local_delta_t, &global_delta_t, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    /// Set new time step
    delta_t = global_delta_t;
   
};

void myRHEA::updateBoundaries() {

    /// General form: w_g*phi_g + w_in*phi_in = phi_b
/// phi_g is ghost cell value
    /// phi_in is inner cell value
    /// phi_b is boundary value/flux
    /// w_g is ghost cell weight
    /// w_in is inner cell weight

    /// Declare weights and ghost & inner values
    double wg_g = 0.0, wg_in = 0.0;
    double u_g, v_g, w_g, P_g, T_g, rho_g, e_g, ke_g, E_g;
    double u_in, v_in, w_in, P_in, T_in;
    double Delta_g;

    /// Parameters iterative solver for subsonic NSCBC
    int max_iter = 10;
    double rel_tol = 1.0e-5;
    
    /// West boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, x_field.vector[0:_ls_])  
    for(int i = topo->iter_bound[_WEST_][_INIX_]; i <= topo->iter_bound[_WEST_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_WEST_][_INIY_]; j <= topo->iter_bound[_WEST_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_WEST_][_INIZ_]; k <= topo->iter_bound[_WEST_][_ENDZ_]; k++) {
                if( ( bocos_type[_WEST_] == _DIRICHLET_ ) or ( bocos_type[_WEST_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_WEST_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) - x_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( x_field[I1D(i+1,j,k)] - 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
                if( bocos_type[_WEST_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
                // printf("WEST // wg_g = %.8f // wg_in = %.8f\n",wg_g, wg_in);
		/// Get/calculate inner values
                u_in = u_field[I1D(i+1,j,k)];
                v_in = v_field[I1D(i+1,j,k)];
                w_in = w_field[I1D(i+1,j,k)];
                P_in = P_field[I1D(i+1,j,k)];
                T_in = T_field[I1D(i+1,j,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_WEST_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_WEST_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_WEST_] - wg_in*w_in )/wg_g;
        	/// Calculate transformation metrics
                double x_xi_in    = (x_field[I1D(i+2,j,k)] - x_field[I1D(i+1,j,k)]);
                double x_eta_in   = 0.5 * ((x_field[I1D(i+1,j+1,k)] - x_field[I1D(i+1,j-1,k)])/2.0 + (x_field[I1D(i+2,j+1,k)] - x_field[I1D(i+2,j-1,k)])/2.0);
                double x_zeta_in  = 0.5 * ((x_field[I1D(i+1,j,k+1)] - x_field[I1D(i+1,j,k-1)])/2.0 + (x_field[I1D(i+2,j,k+1)] - x_field[I1D(i+2,j,k-1)])/2.0);
                double y_xi_in    = (y_field[I1D(i+2,j,k)] - y_field[I1D(i+1,j,k)]);
                double y_eta_in   = 0.5 * ((y_field[I1D(i+1,j+1,k)] - y_field[I1D(i+1,j-1,k)])/2.0 + (y_field[I1D(i+2,j+1,k)] - y_field[I1D(i+2,j-1,k)])/2.0);
                double y_zeta_in  = 0.5 * ((y_field[I1D(i+1,j,k+1)] - y_field[I1D(i+1,j,k-1)])/2.0 + (y_field[I1D(i+2,j,k+1)] - y_field[I1D(i+2,j,k-1)])/2.0);
                double z_xi_in    = (z_field[I1D(i+2,j,k)] - z_field[I1D(i+1,j,k)]);
                double z_eta_in   = 0.5 * ((z_field[I1D(i+1,j+1,k)] - z_field[I1D(i+1,j-1,k)])/2.0 + (z_field[I1D(i+2,j+1,k)] - z_field[I1D(i+2,j-1,k)])/2.0);
                double z_zeta_in  = 0.5 * ((z_field[I1D(i+1,j,k+1)] - z_field[I1D(i+1,j,k-1)])/2.0 + (z_field[I1D(i+2,j,k+1)] - z_field[I1D(i+2,j,k-1)])/2.0);

                double x_xi_g    = (x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)]);
                double x_eta_g   = 0.5 * ((x_field[I1D(i+1,j+1,k)] - x_field[I1D(i+1,j-1,k)])/2.0 + (x_field[I1D(i,j+1,k)] - x_field[I1D(i,j-1,k)])/2.0);
                double x_zeta_g  = 0.5 * ((x_field[I1D(i+1,j,k+1)] - x_field[I1D(i+1,j,k-1)])/2.0 + (x_field[I1D(i,j,k+1)] - x_field[I1D(i,j,k-1)])/2.0);
                double y_xi_g    = (y_field[I1D(i+1,j,k)] - y_field[I1D(i,j,k)]);
                double y_eta_g   = 0.5 * ((y_field[I1D(i+1,j+1,k)] - y_field[I1D(i+1,j-1,k)])/2.0 + (y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)])/2.0);
                double y_zeta_g  = 0.5 * ((y_field[I1D(i+1,j,k+1)] - y_field[I1D(i+1,j,k-1)])/2.0 + (y_field[I1D(i,j,k+1)] - y_field[I1D(i,j,k-1)])/2.0);
                double z_xi_g    = (z_field[I1D(i+1,j,k)] - z_field[I1D(i,j,k)]);
                double z_eta_g   = 0.5 * ((z_field[I1D(i+1,j+1,k)] - z_field[I1D(i+1,j-1,k)])/2.0 + (z_field[I1D(i,j+1,k)] - z_field[I1D(i,j-1,k)])/2.0);
                double z_zeta_g  = 0.5 * ((z_field[I1D(i+1,j,k+1)] - z_field[I1D(i+1,j,k-1)])/2.0 + (z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)])/2.0);

                double det_J_in = (x_xi_in*y_eta_in*z_zeta_in + x_eta_in*y_zeta_in*z_xi_in + x_zeta_in*y_xi_in*z_eta_in - x_zeta_in*y_eta_in*z_xi_in - x_eta_in*y_xi_in*z_zeta_in - x_xi_in*y_zeta_in*z_eta_in);
                det_J_in = 1.0/det_J_in;

                double det_J_g = (x_xi_g*y_eta_g*z_zeta_g + x_eta_g*y_zeta_g*z_xi_g + x_zeta_g*y_xi_g*z_eta_g - x_zeta_g*y_eta_g*z_xi_g - x_eta_g*y_xi_g*z_zeta_g - x_xi_g*y_zeta_g*z_eta_g);
                det_J_g = 1.0/det_J_g;

                double xi_x_in   = (y_eta_in*z_zeta_in - y_zeta_in*z_eta_in) * det_J_in;
                double xi_y_in   = (z_eta_in*x_zeta_in - z_zeta_in*x_eta_in) * det_J_in;
                double xi_z_in   = (x_eta_in*y_zeta_in - x_zeta_in*y_eta_in) * det_J_in;
                double eta_x_in  = (y_zeta_in*z_xi_in  - y_xi_in*z_zeta_in ) * det_J_in;
                double eta_y_in  = (z_zeta_in*x_xi_in  - z_xi_in*x_zeta_in ) * det_J_in;
                double eta_z_in  = (x_zeta_in*y_xi_in  - x_xi_in*y_zeta_in ) * det_J_in;
                double zeta_x_in = (y_xi_in*z_eta_in   - y_eta_in*z_xi_in  ) * det_J_in;
                double zeta_y_in = (z_xi_in*x_eta_in   - z_eta_in*x_xi_in  ) * det_J_in;
                double zeta_z_in = (x_xi_in*y_eta_in   - x_eta_in*y_xi_in  ) * det_J_in;

                double xi_x_g   = (y_eta_g*z_zeta_g - y_zeta_g*z_eta_g) * det_J_g;
                double xi_y_g   = (z_eta_g*x_zeta_g - z_zeta_g*x_eta_g) * det_J_g;
                double xi_z_g   = (x_eta_g*y_zeta_g - x_zeta_g*y_eta_g) * det_J_g;
                double eta_x_g  = (y_zeta_g*z_xi_g  - y_xi_g*z_zeta_g ) * det_J_g;
                double eta_y_g  = (z_zeta_g*x_xi_g  - z_xi_g*x_zeta_g ) * det_J_g;
                double eta_z_g  = (x_zeta_g*y_xi_g  - x_xi_g*y_zeta_g ) * det_J_g;
                double zeta_x_g = (y_xi_g*z_eta_g   - y_eta_g*z_xi_g  ) * det_J_g;
                double zeta_y_g = (z_xi_g*x_eta_g   - z_eta_g*x_xi_g  ) * det_J_g;
                double zeta_z_g = (x_xi_g*y_eta_g   - x_eta_g*y_xi_g  ) * det_J_g;

                if( ( bocos_type[_WEST_] == _DIRICHLET_ ) and ( bocos_P[_WEST_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_WEST_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_WEST_] == _DIRICHLET_ ) and ( bocos_T[_WEST_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_WEST_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_WEST_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in_in = x_field[I1D(i+2,j,k)] - x_field[I1D(i+1,j,k)];
		    double rho_in = rho_field[I1D(i+1,j,k)]; 
		    double sos_in = sos_field[I1D(i+1,j,k)];
		    double drho_dx_in_in = ( rho_field[I1D(i+2,j,k)] - rho_field[I1D(i+1,j,k)] )/Delta_in_in;
		    double du_dx_in_in   = (   u_field[I1D(i+2,j,k)] -   u_field[I1D(i+1,j,k)] )/Delta_in_in;
		    double dP_dx_in_in   = (   P_field[I1D(i+2,j,k)] -   P_field[I1D(i+1,j,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dx_in_in - rho_in*sos_in*du_dx_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_WEST_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter; ite++ ) {
                        /// Aitken's x_1
                        double drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        double du_dx_g   = (   u_in -   u_g )/Delta_g;
                        double dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        du_dx_g   = (   u_in -   u_g )/Delta_g;
                        dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            break;		/// If the result is within tolerance, leave the loop!
			}
			x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
		    }
		} else if( bocos_type[_WEST_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in = x_field[I1D(i+2,j,k)] - x_field[I1D(i+1,j,k)];
                    double rho_in = rho_field[I1D(i+1,j,k)]; 
                    double sos_in = sos_field[I1D(i+1,j,k)];
                    double Ma_in  = u_in/sos_in;
                    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/local_Lx;
                    double drho_dx_in = ( rho_field[I1D(i+2,j,k)] - rho_field[I1D(i+1,j,k)] )/Delta_in;
                    double du_dx_in   = (   u_field[I1D(i+2,j,k)] -   u_field[I1D(i+1,j,k)] )/Delta_in;
                    double dv_dx_in   = (   v_field[I1D(i+2,j,k)] -   v_field[I1D(i+1,j,k)] )/Delta_in;
                    double dw_dx_in   = (   w_field[I1D(i+2,j,k)] -   w_field[I1D(i+1,j,k)] )/Delta_in;
                    double dP_dx_in   = (   P_field[I1D(i+2,j,k)] -   P_field[I1D(i+1,j,k)] )/Delta_in;
                    double lambda_5_in = u_in + sos_in;
                    double L_1_lambda_1_in = dP_dx_in - rho_in*sos_in*du_dx_in;
                    double L_2_lambda_2_in = sos_in*sos_in*drho_dx_in - dP_dx_in;
                    double L_3_lambda_3_in = dv_dx_in;
                    double L_4_lambda_4_in = dw_dx_in;
		            double L_5_lambda_5_in = K_in*( P_in - bocos_P[_WEST_] )/lambda_5_in;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_3_dx_in = L_3_lambda_3_in;
                    double dQ_4_dx_in = L_4_lambda_4_in;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_WEST_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i+1,j,k)]; 
		    double sos_in = sos_field[I1D(i+1,j,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_WEST_], bocos_T[_WEST_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_WEST_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_WEST_] - wg_in*T_in )/wg_g;
		    double drho_dx_g = ( rho_field[I1D(i+1,j,k)] - rho_g )/Delta_g;
		    double du_dx_g   = (   u_field[I1D(i+1,j,k)] - u_g   )/Delta_g;
		    double dv_dx_g   = (   v_field[I1D(i+1,j,k)] - v_g   )/Delta_g;
		    double dw_dx_g   = (   w_field[I1D(i+1,j,k)] - w_g   )/Delta_g;
		    double dP_dx_g   = (   P_field[I1D(i+1,j,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dx_g - rho_in*sos_in*du_dx_g;
                    double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                    double L_3_lambda_3_g = dv_dx_g;
                    double L_4_lambda_4_g = dw_dx_g;
                    double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_3_dx_in = L_3_lambda_3_g;
                    double dQ_4_dx_in = L_4_lambda_4_g;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );			
		} else if( bocos_type[_WEST_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// East boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, x_field.vector[0:_ls_]) 
    for(int i = topo->iter_bound[_EAST_][_INIX_]; i <= topo->iter_bound[_EAST_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_EAST_][_INIY_]; j <= topo->iter_bound[_EAST_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_EAST_][_INIZ_]; k <= topo->iter_bound[_EAST_][_ENDZ_]; k++) {
                if( ( bocos_type[_EAST_] == _DIRICHLET_ ) or ( bocos_type[_EAST_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_EAST_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( x_field[I1D(i+1,j,k)] - 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) - x_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
                if( bocos_type[_EAST_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
                //printf("EAST // wg_g = %.8f // wg_in = %.8f\n",wg_g, wg_in);
		/// Get/calculate inner values
                u_in = u_field[I1D(i-1,j,k)];
                v_in = v_field[I1D(i-1,j,k)];
                w_in = w_field[I1D(i-1,j,k)];
                P_in = P_field[I1D(i-1,j,k)];
                T_in = T_field[I1D(i-1,j,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_EAST_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_EAST_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_EAST_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_EAST_] == _DIRICHLET_ ) and ( bocos_P[_EAST_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_EAST_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_EAST_] == _DIRICHLET_ ) and ( bocos_T[_EAST_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_EAST_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_EAST_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = x_field[I1D(i-1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in_in = x_field[I1D(i-2,j,k)] - x_field[I1D(i-1,j,k)];
		    double rho_in = rho_field[I1D(i-1,j,k)]; 
		    double sos_in = sos_field[I1D(i-1,j,k)];
		    double drho_dx_in_in = ( rho_field[I1D(i-2,j,k)] - rho_field[I1D(i-1,j,k)] )/Delta_in_in;
		    double du_dx_in_in   = (   u_field[I1D(i-2,j,k)] -   u_field[I1D(i-1,j,k)] )/Delta_in_in;
		    double dP_dx_in_in   = (   P_field[I1D(i-2,j,k)] -   P_field[I1D(i-1,j,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dx_in_in - rho_in*sos_in*du_dx_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_EAST_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter; ite++ ) {
                        /// Aitken's x_1
                        double drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        double du_dx_g   = (   u_in -   u_g )/Delta_g;
                        double dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        du_dx_g   = (   u_in -   u_g )/Delta_g;
                        dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            break;		/// If the result is within tolerance, leave the loop!
			}
			x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
		    }
		} else if( bocos_type[_EAST_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = x_field[I1D(i-1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in = x_field[I1D(i-2,j,k)] - x_field[I1D(i-1,j,k)];
		    double rho_in = rho_field[I1D(i-1,j,k)]; 
		    double sos_in = sos_field[I1D(i-1,j,k)];
		    double Ma_in  = u_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/local_Lx;
		    double drho_dx_in = ( rho_field[I1D(i-2,j,k)] - rho_field[I1D(i-1,j,k)] )/Delta_in;
		    double du_dx_in   = (   u_field[I1D(i-2,j,k)] -   u_field[I1D(i-1,j,k)] )/Delta_in;
		    double dv_dx_in   = (   v_field[I1D(i-2,j,k)] -   v_field[I1D(i-1,j,k)] )/Delta_in;
		    double dw_dx_in   = (   w_field[I1D(i-2,j,k)] -   w_field[I1D(i-1,j,k)] )/Delta_in;
		    double dP_dx_in   = (   P_field[I1D(i-2,j,k)] -   P_field[I1D(i-1,j,k)] )/Delta_in;
		    double lambda_1_in = u_in - sos_in;
		    double L_1_lambda_1_in = K_in*( P_in - bocos_P[_EAST_] )/lambda_1_in;
                    double L_2_lambda_2_in = sos_in*sos_in*drho_dx_in - dP_dx_in;
                    double L_3_lambda_3_in = dv_dx_in;
                    double L_4_lambda_4_in = dw_dx_in;
                    double L_5_lambda_5_in = dP_dx_in + rho_in*sos_in*du_dx_in;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_3_dx_in = L_3_lambda_3_in;
                    double dQ_4_dx_in = L_4_lambda_4_in;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_EAST_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = x_field[I1D(i-1,j,k)] - x_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i-1,j,k)]; 
		    double sos_in = sos_field[I1D(i-1,j,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_EAST_], bocos_T[_EAST_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_EAST_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_EAST_] - wg_in*T_in )/wg_g;
		    double drho_dx_g = ( rho_field[I1D(i-1,j,k)] - rho_g )/Delta_g;
		    double du_dx_g   = (   u_field[I1D(i-1,j,k)] - u_g   )/Delta_g;
		    double dv_dx_g   = (   v_field[I1D(i-1,j,k)] - v_g   )/Delta_g;
		    double dw_dx_g   = (   w_field[I1D(i-1,j,k)] - w_g   )/Delta_g;
		    double dP_dx_g   = (   P_field[I1D(i-1,j,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dx_g - rho_in*sos_in*du_dx_g;
                    double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                    double L_3_lambda_3_g = dv_dx_g;
                    double L_4_lambda_4_g = dw_dx_g;
                    double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_3_dx_in = L_3_lambda_3_g;
                    double dQ_4_dx_in = L_4_lambda_4_g;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_EAST_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;		
            }
        }
    }

    /// South boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, y_field.vector[0:_ls_]) 
    for(int i = topo->iter_bound[_SOUTH_][_INIX_]; i <= topo->iter_bound[_SOUTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_SOUTH_][_INIY_]; j <= topo->iter_bound[_SOUTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_SOUTH_][_INIZ_]; k <= topo->iter_bound[_SOUTH_][_ENDZ_]; k++) {
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) or ( bocos_type[_SOUTH_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_SOUTH_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) - y_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( y_field[I1D(i,j+1,k)] - 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
                if( bocos_type[_SOUTH_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
                //printf("SOUTH // wg_g = %.8f // wg_in = %.8f\n",wg_g, wg_in);
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j+1,k)];
                v_in = v_field[I1D(i,j+1,k)];
                w_in = w_field[I1D(i,j+1,k)];
                P_in = P_field[I1D(i,j+1,k)];
                T_in = T_field[I1D(i,j+1,k)];	
		/// Calculate ghost primitive variables
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_u[_SOUTH_] < 0.0 ) ) {
                    u_g = u_in;
                } else {
                    u_g = ( bocos_u[_SOUTH_] - wg_in*u_in )/wg_g;
                }
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_v[_SOUTH_] < 0.0 ) ) {
                    v_g = v_in;
                } else {
                    v_g = ( bocos_v[_SOUTH_] - wg_in*v_in )/wg_g;
                }
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_w[_SOUTH_] < 0.0 ) ) {
                    w_g = w_in;
                } else {
                    w_g = ( bocos_w[_SOUTH_] - wg_in*w_in )/wg_g;
                }
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_P[_SOUTH_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_SOUTH_] - wg_in*P_in )/wg_g;
                }

                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_T[_SOUTH_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_SOUTH_] - wg_in*T_in )/wg_g;
                }

                if( bocos_type[_SOUTH_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in_in = y_field[I1D(i,j+2,k)] - y_field[I1D(i,j+1,k)];
		    double rho_in = rho_field[I1D(i,j+1,k)]; 
		    double sos_in = sos_field[I1D(i,j+1,k)];
		    double drho_dy_in_in = ( rho_field[I1D(i,j+2,k)] - rho_field[I1D(i,j+1,k)] )/Delta_in_in;
		    double dv_dy_in_in   = (   v_field[I1D(i,j+2,k)] -   v_field[I1D(i,j+1,k)] )/Delta_in_in;
		    double dP_dy_in_in   = (   P_field[I1D(i,j+2,k)] -   P_field[I1D(i,j+1,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dy_in_in - rho_in*sos_in*dv_dy_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_SOUTH_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter; ite++ ) {
                        /// Aitken's x_1
                        double drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        double dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        double dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            break;		/// If the result is within tolerance, leave the loop!
			}
			x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
		    }
		} else if( bocos_type[_SOUTH_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in = y_field[I1D(i,j+2,k)] - y_field[I1D(i,j+1,k)];
		    double rho_in = rho_field[I1D(i,j+1,k)]; 
		    double sos_in = sos_field[I1D(i,j+1,k)];
		    double Ma_in  = v_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/local_Ly;
		    double drho_dy_in = ( rho_field[I1D(i,j+2,k)] - rho_field[I1D(i,j+1,k)] )/Delta_in;
		    double du_dy_in   = (   u_field[I1D(i,j+2,k)] -   u_field[I1D(i,j+1,k)] )/Delta_in;
		    double dv_dy_in   = (   v_field[I1D(i,j+2,k)] -   v_field[I1D(i,j+1,k)] )/Delta_in;
		    double dw_dy_in   = (   w_field[I1D(i,j+2,k)] -   w_field[I1D(i,j+1,k)] )/Delta_in;
		    double dP_dy_in   = (   P_field[I1D(i,j+2,k)] -   P_field[I1D(i,j+1,k)] )/Delta_in;
		    double lambda_5_in = v_in + sos_in;
                    double L_1_lambda_1_in = dP_dy_in - rho_in*sos_in*dv_dy_in;
                    double L_2_lambda_2_in = du_dy_in;
                    double L_3_lambda_3_in = sos_in*sos_in*drho_dy_in - dP_dy_in;
                    double L_4_lambda_4_in = dw_dy_in;
		    double L_5_lambda_5_in = K_in*( P_in - bocos_P[_SOUTH_] )/lambda_5_in;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dy_in = L_2_lambda_2_in;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_4_dy_in = L_4_lambda_4_in;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_SOUTH_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j+1,k)]; 
		    double sos_in = sos_field[I1D(i,j+1,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_SOUTH_], bocos_T[_SOUTH_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_SOUTH_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_SOUTH_] - wg_in*T_in )/wg_g;
		    double drho_dy_g = ( rho_field[I1D(i,j+1,k)] - rho_g )/Delta_g;
		    double du_dy_g   = (   u_field[I1D(i,j+1,k)] - u_g   )/Delta_g;
		    double dv_dy_g   = (   v_field[I1D(i,j+1,k)] - v_g   )/Delta_g;
		    double dw_dy_g   = (   w_field[I1D(i,j+1,k)] - w_g   )/Delta_g;
		    double dP_dy_g   = (   P_field[I1D(i,j+1,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dy_g - rho_in*sos_in*dv_dy_g;
                    double L_2_lambda_2_g = du_dy_g;
                    double L_3_lambda_3_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                    double L_4_lambda_4_g = dw_dy_g;
                    double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dy_in = L_2_lambda_2_g;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_4_dy_in = L_4_lambda_4_g;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_SOUTH_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// North boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, y_field.vector[0:_ls_]) 
    for(int i = topo->iter_bound[_NORTH_][_INIX_]; i <= topo->iter_bound[_NORTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_NORTH_][_INIY_]; j <= topo->iter_bound[_NORTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_NORTH_][_INIZ_]; k <= topo->iter_bound[_NORTH_][_ENDZ_]; k++) {
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) or ( bocos_type[_NORTH_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_NORTH_] == _SUPERSONIC_INFLOW_ ) ) { 
                    wg_g  = 1.0 - ( y_field[I1D(i,j+1,k)] - 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) - y_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
                if( bocos_type[_NORTH_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
                //printf("NORTH // wg_g = %.8f // wg_in = %.8f\n",wg_g, wg_in);
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j-1,k)];
                v_in = v_field[I1D(i,j-1,k)];
                w_in = w_field[I1D(i,j-1,k)];
                P_in = P_field[I1D(i,j-1,k)];
                T_in = T_field[I1D(i,j-1,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_NORTH_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_NORTH_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_NORTH_] - wg_in*w_in )/wg_g;
        	/// Calculate transformation metrics
                double x_xi    = (x_field[I1D(i+1,j-1,k)] - x_field[I1D(i-1,j-1,k)])/2.0;
                double x_eta   = (x_field[I1D(i,j,k)] - x_field[I1D(i,j-2,k)])/2.0;
                double x_zeta  = (x_field[I1D(i,j-1,k+1)] - x_field[I1D(i,j-1,k-1)])/2.0;
                double y_xi    = (y_field[I1D(i+1,j-1,k)] - y_field[I1D(i-1,j-1,k)])/2.0;
                double y_eta   = (y_field[I1D(i,j,k)] - y_field[I1D(i,j-2,k)])/2.0;
                double y_zeta  = (y_field[I1D(i,j-1,k+1)] - y_field[I1D(i,j-1,k-1)])/2.0;
                double z_xi    = (z_field[I1D(i+1,j-1,k)] - z_field[I1D(i-1,j-1,k)])/2.0;
                double z_eta   = (z_field[I1D(i,j,k)] - z_field[I1D(i,j-2,k)])/2.0;
                double z_zeta  = (z_field[I1D(i,j-1,k+1)] - z_field[I1D(i,j-1,k-1)])/2.0;

                double det_J = (x_xi*y_eta*z_zeta + x_eta*y_zeta*z_xi + x_zeta*y_xi*z_eta - x_zeta*y_eta*z_xi - x_eta*y_xi*z_zeta - x_xi*y_zeta*z_eta);
                det_J = 1.0/det_J;

                double xi_x   = (y_eta*z_zeta - y_zeta*z_eta) * det_J;
                double xi_y   = (z_eta*x_zeta - z_zeta*x_eta) * det_J;
                double xi_z   = (x_eta*y_zeta - x_zeta*y_eta) * det_J;
                double eta_x  = (y_zeta*z_xi  - y_xi*z_zeta ) * det_J;
                double eta_y  = (z_zeta*x_xi  - z_xi*x_zeta ) * det_J;
                double eta_z  = (x_zeta*y_xi  - x_xi*y_zeta ) * det_J;
                double zeta_x = (y_xi*z_eta   - y_eta*z_xi  ) * det_J;
                double zeta_y = (z_xi*x_eta   - z_eta*x_xi  ) * det_J;
                double zeta_z = (x_xi*y_eta   - x_eta*y_xi  ) * det_J;
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) and ( bocos_P[_NORTH_] < 0.0 ) ) {
                    // double P_xi     = (P_field[I1D(i+1,j-1,k)] - P_field[I1D(i-1,j-1,k)])/2.0;
                    // double P_zeta   = (P_field[I1D(i,j-1,k+1)] - P_field[I1D(i,j-1,k-1)])/2.0;
                    // double P_rhs    = -1.0 * ((xi_x*eta_x + xi_y*eta_y + xi_z*eta_z)*P_xi + (zeta_x*eta_x + zeta_y*eta_y + zeta_z*eta_z)*P_zeta) / (eta_x*eta_x + eta_y*eta_y + eta_z*eta_z);   /// Presure equation right-hand side
                    // P_g = P_in + P_rhs;
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_NORTH_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) and ( bocos_T[_NORTH_] < 0.0 ) ) {
                    // double T_xi     = (T_field[I1D(i+1,j-1,k)] - T_field[I1D(i-1,j-1,k)])/2.0;
                    // double T_zeta   = (T_field[I1D(i,j-1,k+1)] - T_field[I1D(i,j-1,k-1)])/2.0;
                    // double T_rhs    = -1.0 * ((xi_x*eta_x + xi_y*eta_y + xi_z*eta_z)*T_xi + (zeta_x*eta_x + zeta_y*eta_y + zeta_z*eta_z)*T_zeta) / (eta_x*eta_x + eta_y*eta_y + eta_z*eta_z);   /// Presure equation right-hand side
                    // T_g = T_in + T_rhs;
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_NORTH_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_NORTH_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = y_field[I1D(i,j-1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in_in = y_field[I1D(i,j-2,k)] - y_field[I1D(i,j-1,k)];
		    double rho_in = rho_field[I1D(i,j-1,k)]; 
		    double sos_in = sos_field[I1D(i,j-1,k)];
		    double drho_dy_in_in = ( rho_field[I1D(i,j-2,k)] - rho_field[I1D(i,j-1,k)] )/Delta_in_in;
		    double dv_dy_in_in   = (   v_field[I1D(i,j-2,k)] -   v_field[I1D(i,j-1,k)] )/Delta_in_in;
		    double dP_dy_in_in   = (   P_field[I1D(i,j-2,k)] -   P_field[I1D(i,j-1,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dy_in_in - rho_in*sos_in*dv_dy_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_NORTH_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter; ite++ ) {
                        /// Aitken's x_1
                        double drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        double dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        double dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            break;		/// If the result is within tolerance, leave the loop!
			}
			x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
		    }
		} else if( bocos_type[_NORTH_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = y_field[I1D(i,j-1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in = y_field[I1D(i,j-2,k)] - y_field[I1D(i,j-1,k)];
		    double rho_in = rho_field[I1D(i,j-1,k)]; 
		    double sos_in = sos_field[I1D(i,j-1,k)];
		    double Ma_in  = v_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/local_Ly;
		    double drho_dy_in = ( rho_field[I1D(i,j-2,k)] - rho_field[I1D(i,j-1,k)] )/Delta_in;
		    double du_dy_in   = (   u_field[I1D(i,j-2,k)] -   u_field[I1D(i,j-1,k)] )/Delta_in;
		    double dv_dy_in   = (   v_field[I1D(i,j-2,k)] -   v_field[I1D(i,j-1,k)] )/Delta_in;
		    double dw_dy_in   = (   w_field[I1D(i,j-2,k)] -   w_field[I1D(i,j-1,k)] )/Delta_in;
		    double dP_dy_in   = (   P_field[I1D(i,j-2,k)] -   P_field[I1D(i,j-1,k)] )/Delta_in;
		    double lambda_1_in = v_in - sos_in;
		    double L_1_lambda_1_in = K_in*( P_in - bocos_P[_NORTH_] )/lambda_1_in;
                    double L_2_lambda_2_in = du_dy_in;
                    double L_3_lambda_3_in = sos_in*sos_in*drho_dy_in - dP_dy_in;
                    double L_4_lambda_4_in = dw_dy_in;
                    double L_5_lambda_5_in = dP_dy_in + rho_in*sos_in*dv_dy_in;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dy_in = L_2_lambda_2_in;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_4_dy_in = L_4_lambda_4_in;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_NORTH_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = y_field[I1D(i,j-1,k)] - y_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j-1,k)]; 
		    double sos_in = sos_field[I1D(i,j-1,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_NORTH_], bocos_T[_NORTH_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_NORTH_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_NORTH_] - wg_in*T_in )/wg_g;
		    double drho_dy_g = ( rho_field[I1D(i,j-1,k)] - rho_g )/Delta_g;
		    double du_dy_g   = (   u_field[I1D(i,j-1,k)] - u_g   )/Delta_g;
		    double dv_dy_g   = (   v_field[I1D(i,j-1,k)] - v_g   )/Delta_g;
		    double dw_dy_g   = (   w_field[I1D(i,j-1,k)] - w_g   )/Delta_g;
		    double dP_dy_g   = (   P_field[I1D(i,j-1,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dy_g - rho_in*sos_in*dv_dy_g;
                    double L_2_lambda_2_g = du_dy_g;
                    double L_3_lambda_3_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                    double L_4_lambda_4_g = dw_dy_g;
                    double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dy_in = L_2_lambda_2_g;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_4_dy_in = L_4_lambda_4_g;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_NORTH_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// Back boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, z_field.vector[0:_ls_])  
    for(int i = topo->iter_bound[_BACK_][_INIX_]; i <= topo->iter_bound[_BACK_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_BACK_][_INIY_]; j <= topo->iter_bound[_BACK_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_BACK_][_INIZ_]; k <= topo->iter_bound[_BACK_][_ENDZ_]; k++) {
                if( ( bocos_type[_BACK_] == _DIRICHLET_ ) or ( bocos_type[_BACK_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_BACK_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) - z_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( z_field[I1D(i,j,k+1)] - 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
                if( bocos_type[_BACK_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j,k+1)];
                v_in = v_field[I1D(i,j,k+1)];
                w_in = w_field[I1D(i,j,k+1)];
                P_in = P_field[I1D(i,j,k+1)];
                T_in = T_field[I1D(i,j,k+1)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_BACK_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_BACK_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_BACK_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_BACK_] == _DIRICHLET_ ) and ( bocos_P[_BACK_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_BACK_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_BACK_] == _DIRICHLET_ ) and ( bocos_T[_BACK_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_BACK_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_BACK_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];
                    double Delta_in_in = z_field[I1D(i,j,k+2)] - z_field[I1D(i,j,k+1)];
		    double rho_in = rho_field[I1D(i,j,k+1)]; 
		    double sos_in = sos_field[I1D(i,j,k+1)];
		    double drho_dz_in_in = ( rho_field[I1D(i,j,k+2)] - rho_field[I1D(i,j,k+1)] )/Delta_in_in;
		    double dw_dz_in_in   = (   w_field[I1D(i,j,k+2)] -   w_field[I1D(i,j,k+1)] )/Delta_in_in;
		    double dP_dz_in_in   = (   P_field[I1D(i,j,k+2)] -   P_field[I1D(i,j,k+1)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dz_in_in - rho_in*sos_in*dw_dz_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_BACK_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter; ite++ ) {
                        /// Aitken's x_1
                        double drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        double dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        double dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            break;		/// If the result is within tolerance, leave the loop!
			}
			x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
		    }
		} else if( bocos_type[_BACK_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];
                    double Delta_in = z_field[I1D(i,j,k+2)] - z_field[I1D(i,j,k+1)];
		    double rho_in = rho_field[I1D(i,j,k+1)]; 
		    double sos_in = sos_field[I1D(i,j,k+1)];
		    double Ma_in  = w_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/local_Lz;
		    double drho_dz_in = ( rho_field[I1D(i,j,k+2)] - rho_field[I1D(i,j,k+1)] )/Delta_in;
		    double du_dz_in   = (   u_field[I1D(i,j,k+2)] -   u_field[I1D(i,j,k+1)] )/Delta_in;
		    double dv_dz_in   = (   v_field[I1D(i,j,k+2)] -   v_field[I1D(i,j,k+1)] )/Delta_in;
		    double dw_dz_in   = (   w_field[I1D(i,j,k+2)] -   w_field[I1D(i,j,k+1)] )/Delta_in;
		    double dP_dz_in   = (   P_field[I1D(i,j,k+2)] -   P_field[I1D(i,j,k+1)] )/Delta_in;
		    double lambda_5_in = w_in + sos_in;
                    double L_1_lambda_1_in = dP_dz_in - rho_in*sos_in*dw_dz_in;
                    double L_2_lambda_2_in = du_dz_in;
                    double L_3_lambda_3_in = dv_dz_in;
                    double L_4_lambda_4_in = sos_in*sos_in*drho_dz_in - dP_dz_in;
		    double L_5_lambda_5_in = K_in*( P_in - bocos_P[_BACK_] )/lambda_5_in;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dz_in = L_2_lambda_2_in;
                    double dQ_3_dz_in = L_3_lambda_3_in;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_BACK_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j,k+1)]; 
		    double sos_in = sos_field[I1D(i,j,k+1)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_BACK_], bocos_T[_BACK_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_BACK_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_BACK_] - wg_in*T_in )/wg_g;
		    double drho_dz_g = ( rho_field[I1D(i,j,k+1)] - rho_g )/Delta_g;
		    double du_dz_g   = (   u_field[I1D(i,j,k+1)] - u_g   )/Delta_g;
		    double dv_dz_g   = (   v_field[I1D(i,j,k+1)] - v_g   )/Delta_g;
		    double dw_dz_g   = (   w_field[I1D(i,j,k+1)] - w_g   )/Delta_g;
		    double dP_dz_g   = (   P_field[I1D(i,j,k+1)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dz_g - rho_in*sos_in*dw_dz_g;
                    double L_2_lambda_2_g = du_dz_g;
                    double L_3_lambda_3_g = dv_dz_g;
                    double L_4_lambda_4_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                    double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dz_in = L_2_lambda_2_g;
                    double dQ_3_dz_in = L_3_lambda_3_g;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_BACK_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// Front boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, z_field.vector[0:_ls_])  
    for(int i = topo->iter_bound[_FRONT_][_INIX_]; i <= topo->iter_bound[_FRONT_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_FRONT_][_INIY_]; j <= topo->iter_bound[_FRONT_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_FRONT_][_INIZ_]; k <= topo->iter_bound[_FRONT_][_ENDZ_]; k++) {
                if( ( bocos_type[_FRONT_] == _DIRICHLET_ ) or ( bocos_type[_FRONT_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_FRONT_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( z_field[I1D(i,j,k+1)] - 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) - z_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
                if( bocos_type[_FRONT_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j,k-1)];
                v_in = v_field[I1D(i,j,k-1)];
                w_in = w_field[I1D(i,j,k-1)];
                P_in = P_field[I1D(i,j,k-1)];
                T_in = T_field[I1D(i,j,k-1)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_FRONT_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_FRONT_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_FRONT_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_FRONT_] == _DIRICHLET_ ) and ( bocos_P[_FRONT_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_FRONT_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_FRONT_] == _DIRICHLET_ ) and ( bocos_T[_FRONT_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_FRONT_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_FRONT_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = z_field[I1D(i,j,k-1)] - z_field[I1D(i,j,k)];
                    double Delta_in_in = z_field[I1D(i,j,k-2)] - z_field[I1D(i,j,k-1)];
		    double rho_in = rho_field[I1D(i,j,k-1)]; 
		    double sos_in = sos_field[I1D(i,j,k-1)];
		    double drho_dz_in_in = ( rho_field[I1D(i,j,k-2)] - rho_field[I1D(i,j,k-1)] )/Delta_in_in;
		    double dw_dz_in_in   = (   w_field[I1D(i,j,k-2)] -   w_field[I1D(i,j,k-1)] )/Delta_in_in;
		    double dP_dz_in_in   = (   P_field[I1D(i,j,k-2)] -   P_field[I1D(i,j,k-1)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dz_in_in - rho_in*sos_in*dw_dz_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_FRONT_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter; ite++ ) {
                        /// Aitken's x_1
                        double drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        double dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        double dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            break;		/// If the result is within tolerance, leave the loop!
			}
			x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
		    }
		} else if( bocos_type[_FRONT_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = z_field[I1D(i,j,k-1)] - z_field[I1D(i,j,k)];
                    double Delta_in = z_field[I1D(i,j,k-2)] - z_field[I1D(i,j,k-1)];
		    double rho_in = rho_field[I1D(i,j,k-1)]; 
		    double sos_in = sos_field[I1D(i,j,k-1)];
		    double Ma_in  = w_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/local_Lz;
		    double drho_dz_in = ( rho_field[I1D(i,j,k-2)] - rho_field[I1D(i,j,k-1)] )/Delta_in;
		    double du_dz_in   = (   u_field[I1D(i,j,k-2)] -   u_field[I1D(i,j,k-1)] )/Delta_in;
		    double dv_dz_in   = (   v_field[I1D(i,j,k-2)] -   v_field[I1D(i,j,k-1)] )/Delta_in;
		    double dw_dz_in   = (   w_field[I1D(i,j,k-2)] -   w_field[I1D(i,j,k-1)] )/Delta_in;
		    double dP_dz_in   = (   P_field[I1D(i,j,k-2)] -   P_field[I1D(i,j,k-1)] )/Delta_in;
		    double lambda_1_in = w_in - sos_in;
		    double L_1_lambda_1_in = K_in*( P_in - bocos_P[_FRONT_] )/lambda_1_in;
                    double L_2_lambda_2_in = du_dz_in;
                    double L_3_lambda_3_in = dv_dz_in;
                    double L_4_lambda_4_in = sos_in*sos_in*drho_dz_in - dP_dz_in;
                    double L_5_lambda_5_in = dP_dz_in + rho_in*sos_in*dw_dz_in;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dz_in = L_2_lambda_2_in;
                    double dQ_3_dz_in = L_3_lambda_3_in;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_FRONT_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = z_field[I1D(i,j,k-1)] - z_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j,k-1)]; 
		    double sos_in = sos_field[I1D(i,j,k-1)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_FRONT_], bocos_T[_FRONT_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_FRONT_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_FRONT_] - wg_in*T_in )/wg_g;
		    double drho_dz_g = ( rho_field[I1D(i,j,k-1)] - rho_g )/Delta_g;
		    double du_dz_g   = (   u_field[I1D(i,j,k-1)] - u_g   )/Delta_g;
		    double dv_dz_g   = (   v_field[I1D(i,j,k-1)] - v_g   )/Delta_g;
		    double dw_dz_g   = (   w_field[I1D(i,j,k-1)] - w_g   )/Delta_g;
		    double dP_dz_g   = (   P_field[I1D(i,j,k-1)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dz_g - rho_in*sos_in*dw_dz_g;
                    double L_2_lambda_2_g = du_dz_g;
                    double L_3_lambda_3_g = dv_dz_g;
                    double L_4_lambda_4_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                    double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dz_in = L_2_lambda_2_g;
                    double dQ_3_dz_in = L_3_lambda_3_g;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_FRONT_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }
    
    /// Update halo values
    //#pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_])
    //rho_field.update();
    //rhou_field.update();
    //rhov_field.update();
    //rhow_field.update();
    //rhoE_field.update();
    //u_field.update();
    u_field.fillEdgeCornerBoundaries();
    //v_field.update();
    v_field.fillEdgeCornerBoundaries();
    //w_field.update();
    w_field.fillEdgeCornerBoundaries();
    //E_field.update();
    //s_field.update();
    //P_field.update();
    // T_field.update();
    //sos_field.update();
    //c_v_field.update();
    //c_p_field.update();
    //#pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_])

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
