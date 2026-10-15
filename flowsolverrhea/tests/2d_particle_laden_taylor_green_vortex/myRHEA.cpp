#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

/// AUXILIAR PARAMETERS ///
const double pi = 2.0*asin( 1.0 );				/// pi number (fixed)

/// PROBLEM PARAMETERS ///
const double gamma_0    = 1.4;					/// Reference ratio of heat capacities
const double R_specific = 287.058;				/// Specific gas constant
const double Re         = pi;					/// Reynolds number
const double Ma         = 1.0e-1/sqrt( gamma_0 );		/// Mach number
const double l_c        = pi;					/// Characteristic length size			
const double LxLyLz     = pow( 2.0*l_c, 2.0 )*0.1;		/// Domain volume			
const double rho_0      = 1.0;					/// Reference density	
const double U_0        = 1.0;					/// Reference velocity
const double mu_0       = rho_0*U_0*l_c/Re;			/// Dynamic viscosity	
const double nu_0       = mu_0/rho_0;				/// Kinematic viscosity	
const double P_0        = rho_0*U_0*U_0/( gamma_0*Ma*Ma );	/// Reference pressure
const double d_p        = 0.01;					/// Diameter of particles
const double rho_p      = 1413.7166941154069;			/// Density of particles
const int N_p           = 5;					/// Number of particles
const int K             = int( pow( Re, 3.0/4.0 ) );		/// Largest mode (approximate) of initial vortex
const double tau_f      = Re/( K*K*nu_0 );			/// Relaxation time scale of flow
const double tau_p      = ( rho_p*d_p*d_p )/( 18.0*mu_0 );	/// Relaxation time scale of particles
const double m_p        = ( pi/6.0 )*rho_p*pow( d_p, 3.0 );	/// Mass of particles
const double n_p        = N_p/LxLyLz;				/// Number density of particles
const double St         = tau_p/tau_f;				/// Stokes number of particles
const double phi_m      = ( N_p*m_p )/( rho_0*LxLyLz );		/// Mass (loading ratio) fraction of particles


////////// myRHEA CLASS //////////

void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// All (inner, halo, boundary): u, v, w, P and T
    #pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_]) 
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] = U_0*sin( x_field[I1D(i,j,k)] )*cos( y_field[I1D(i,j,k)] );
                v_field[I1D(i,j,k)] = ( -1.0 )*U_0*cos( x_field[I1D(i,j,k)] )*sin( y_field[I1D(i,j,k)] );
                w_field[I1D(i,j,k)] = 0.0;
                P_field[I1D(i,j,k)] = P_0 + ( rho_0*U_0*U_0/4.0 )*( cos( 2.0*x_field[I1D(i,j,k)] ) + cos( 2.0*y_field[I1D(i,j,k)] ) );
                T_field[I1D(i,j,k)] = thermodynamics->calculateTemperatureFromPressureDensity( P_field[I1D(i,j,k)], rho_0 );
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

void myRHEA::tagImmersedBoundaryMethod() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Set velocity, temperature & interpolation power parameter of IBM
    immersed_boundary_method->setVelocityIBM( 0.0, 0.0, 0.0 );			// [m/s]
    immersed_boundary_method->setTemperatureIBM( 300.0 );			// [K]
    
    /// Set tags of IBM (tagging)

    /// All (inner, halo, boundary): tag_IBM
    #pragma acc parallel loop collapse(3) present(tag_IBM_field.vector[0:_ls_])
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
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(tag_IBM_field.vector[0:_ls_])
#endif
    tag_IBM_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(tag_IBM_field.vector[0:_ls_])
#endif

};

void myRHEA::calculateInviscidFluxes() {

    /// IMPORTANT: This method has been modified/overwritten according to the problem under consideration

    #pragma acc parallel loop collapse(3) present(rho_inv_flux.vector[0:_ls_], rhou_inv_flux.vector[0:_ls_], rhov_inv_flux.vector[0:_ls_], rhow_inv_flux.vector[0:_ls_], rhoE_inv_flux.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Fluxes set to zero
                rho_inv_flux[I1D(i,j,k)]  = 0.0;
                rhou_inv_flux[I1D(i,j,k)] = 0.0;
                rhov_inv_flux[I1D(i,j,k)] = 0.0;
                rhow_inv_flux[I1D(i,j,k)] = 0.0;
                rhoE_inv_flux[I1D(i,j,k)] = 0.0;
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

/*void myRHEA::calculateViscousFluxes() {

    /// IMPORTANT: This method has been modified/overwritten according to the problem under consideration

    #pragma acc parallel loop collapse(3) present(rhou_vis_flux.vector[0:_ls_], rhov_vis_flux.vector[0:_ls_], rhow_vis_flux.vector[0:_ls_], rhoE_vis_flux.vector[0:_ls_], work_vis_rhoe_flux.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Fluxes set to zero
                rhou_vis_flux[I1D(i,j,k)] = 0.0;
                rhov_vis_flux[I1D(i,j,k)] = 0.0;
                rhow_vis_flux[I1D(i,j,k)] = 0.0;
                rhoE_vis_flux[I1D(i,j,k)] = 0.0;
                work_vis_rhoe_flux[I1D(i,j,k)] = 0.0;
            }
        }
    }

    /// Update halo values
    //rhou_vis_flux.update();
    //rhov_vis_flux.update();
    //rhow_vis_flux.update();
    //rhoE_vis_flux.update();
    //work_vis_rhoe_flux.update();

};*/

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
