#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

/// FIXED PARAMETERS ///
const double epsilon = 1.0e-15;						/// Small epsilon number (fixed)
const int cout_precision = 5;		                		/// Output precision (fixed)

/// PROBLEM PARAMETERS ///
const double u_injection = 5.4;						/// Injection melocity [m/s]
const double T_injection = 137.0; 					/// Injection temperature [K] 	
const double P_chamber   = 3.98e6;					/// Chamber pressure [Pa]
const double T_chamber   = 298.0; 					/// Chamber temperature [K] 
const double T_g_min     = 100;                                         /// Minimum temperature boundary (ghost) grid points [K]               
const double T_min       = T_injection*0.99;                            /// Minimum temperature interior grid points [K]            
const double T_max       = T_chamber*1.01;                              /// Maximum temperature interior grid points [K] 
const double D_jet       = 2.1e-3;					/// Diameter of jet [m]
//const double D_jet       = 2.2e-3;					/// Diameter of jet [m]
const double R_jet       = 0.5*D_jet;					/// Radius of jet [m]
const double x_jet	 = 0.0;						/// Center of jet in x-direction [m]
const double y_jet	 = 0.0;						/// Center of jet in y-direction [m]
const double z_jet	 = 0.0;						/// Center of jet in z-direction [m]
const double alpha_u     = 1.0e-6;                                  	/// Magnitude of initial velocity perturbations [-]
//const double alpha_u     = 0.0;                                  	/// Magnitude of initial velocity perturbations [-]
const double sigma_L     = 0.9;						/// Fraction ratio of L for sponge region [-]
const double sigma_mu    = 1000.0;					/// Increment ratio of mu in sponge region [-]
const double R_outer     = 1.1*R_jet;					/// Outer radius of smoothed jet
const double alpha_tanh  = 5.0;						/// Controls steepness of transition (8 - very steep; 2 - very gradual)
const double P_tol       = 0.05;					/// Pressure clipping tolerance with respect to P_chamber [-]

/// Control bulk pressure to maintain the fixed target value P_chamber
bool control_pressure = false;

/// Control temperature to T_injection <= T <= T_chamber
bool control_temperature = true;


////////// myRHEA CLASS //////////

void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    srand( my_rank );

    /// All (inner, halo, boundary): u, v, w, P and T
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] = 0.0;
                v_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = 0.0;
                P_field[I1D(i,j,k)] = P_chamber;
                T_field[I1D(i,j,k)] = T_chamber;
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
    #pragma acc parallel loop collapse(3) present(this, f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_])
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
    double y_pos, z_pos;
    double area_fraction;
    double r, r_normalized, T_target, u_target, noise;
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
		/// Calculate area fraction of jet
		/// Hyperbolic tangent
                y_pos = z_pos = r = r_normalized = 0.0;
                y_pos = y_field[I1D(i,j,k)];
                z_pos = z_field[I1D(i,j,k)];
                r = sqrt( pow( y_pos - y_jet, 2.0 ) + pow( z_pos - z_jet, 2.0 ) );
                r_normalized = ( r - R_jet )/( R_outer - R_jet );
                // Hyperbolic tangent interpolation: 1 inside jet, 0 outside
                area_fraction = 0.5*( 1.0 - tanh( alpha_tanh * r_normalized ) );
                area_fraction = fmin( 1.0, fmax( 0.0, area_fraction ) );
                T_target = area_fraction*T_injection + ( 1.0 - area_fraction )*T_chamber;
                u_target = area_fraction*u_injection;	/// chamber assumed stagnant (zero velocity)
		noise = pseudo_random_noise( i, j, k );
		u_target += noise;
		/// Get/calculate inner values
                u_in = u_field[I1D(i+1,j,k)];
                v_in = v_field[I1D(i+1,j,k)];
                w_in = w_field[I1D(i+1,j,k)];
                P_in = P_field[I1D(i+1,j,k)];
                T_in = T_field[I1D(i+1,j,k)];	
		/// Calculate ghost primitive variables
                //u_g = ( bocos_u[_WEST_] - wg_in*u_in )/wg_g;
		//u_g = ( bocos_u[_WEST_]*area_fraction - wg_in*u_in )/wg_g;
		u_g = ( u_target - wg_in * u_in ) / wg_g;
                v_g = ( bocos_v[_WEST_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_WEST_] - wg_in*w_in )/wg_g;
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
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_x;
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
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_x;
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
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j+1,k)];
                v_in = v_field[I1D(i,j+1,k)];
                w_in = w_field[I1D(i,j+1,k)];
                P_in = P_field[I1D(i,j+1,k)];
                T_in = T_field[I1D(i,j+1,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_SOUTH_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_SOUTH_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_SOUTH_] - wg_in*w_in )/wg_g;
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
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_y;
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
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) and ( bocos_P[_NORTH_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_NORTH_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) and ( bocos_T[_NORTH_] < 0.0 ) ) {
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
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_y;
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
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_z;
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
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_z;
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
    //T_field.update();
    //sos_field.update();
    //c_v_field.update();
    //c_p_field.update();
    //#pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_])

};

void myRHEA::calculateTransportCoefficients() {
    
    /// All (inner, halo, boundary) points: mu and kappa
    //#pragma acc kernels loop collapse(3) independent
    #pragma acc parallel loop collapse(3) present(this, mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], P_field.vector[0:_ls_], rho_field.vector[0:_ls_], T_field.vector[0:_ls_])
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                mu_field[I1D(i,j,k)]    = transport_coefficients->calculateDynamicViscosity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
		if( ( x_field[I1D(i,j,k)] > ( x_jet + sigma_L*L_x ) ) || ( fabs( y_field[I1D(i,j,k)] ) > fabs( y_jet + sigma_L*0.5*L_y ) ) || ( fabs( z_field[I1D(i,j,k)] ) > fabs( z_jet + sigma_L*0.5*L_z )  ) ) {
		    mu_field[I1D(i,j,k)] = sigma_mu*mu_field[I1D(i,j,k)]; 
		}
                kappa_field[I1D(i,j,k)] = transport_coefficients->calculateThermalConductivity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
            }
        }
    }

    /// Update halo values
    //mu_field.update();
    //kappa_field.update();

};

void myRHEA::execute() {

    /// Start wall-clock timer
    auto start_wall_clock_timer = chrono::steady_clock::now();

    /// Start timer: execute
    timers->start( "execute" );

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /// Set output (cout) precision
    cout.precision( cout_precision );

    /// Start RHEA simulation
    if( my_rank == 0 ) cout << "RHEA (v" << version_number << "): START SIMULATION" << endl;

    /// Initialize flow variables from restart file or by setting initial conditions
    if( use_restart ) {

        /// Initialize from restart file
        this->initializeFromRestart();

        if( artificial_compressibility_method ) {

            /// Calculate thermodynamic (bulk) pressure
            P_thermo = this->calculateVolumeAveragedPressure();
	    #pragma acc update device(P_thermo)

            /// Calculate alpha value of artificial compressibility method
            alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();
	    #pragma acc update device(alpha_acm)

	    /// Calculate artificially modified thermodynamics
            this->calculateArtificiallyModifiedThermodynamics();

	    /// Calculate artificially modified transport coefficients
            this->calculateArtificiallyModifiedTransportCoefficients();

	}

    } else {

        /// Set initial conditions
        this->setInitialConditions();

        /// Initialize thermodynamics
        this->initializeThermodynamics();

        if( artificial_compressibility_method ) {

            /// Calculate thermodynamic (bulk) pressure
            P_thermo = this->calculateVolumeAveragedPressure();
	    #pragma acc update device(P_thermo)

            /// Calculate alpha value of artificial compressibility method
            alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();
	    #pragma acc update device(alpha_acm)

	    /// Calculate artificially modified thermodynamics
            this->calculateArtificiallyModifiedThermodynamics();

	    /// Calculate artificially modified transport coefficients
            this->calculateArtificiallyModifiedTransportCoefficients();

	} else {

            /// Calculate transport coefficients
            this->calculateTransportCoefficients();

	}

    }

    if( activate_immersed_boundary_method ) {

        /// Start timer: immersed_boundary_method
        timers->start( "immersed_boundary_method" );

        /// Tag immersed boundary method
        this->tagImmersedBoundaryMethod();

        /// Stop timer: immersed_boundary_method
        timers->stop( "immersed_boundary_method" );

    }

    /// Calculate conserved variables from primitive variables
    this->primitiveToConservedVariables();

    /// Update previous state of conserved variables
    this->updatePreviousStateConservedVariables();

    /// Start timer: time_iteration_loop
    timers->start( "time_iteration_loop" );

    // Some copyin's to the GPU, need to figure out where to put them. UNDER DEVELOPMENT
    #pragma acc enter data copyin(x_field.vector[0:_ls_],y_field.vector[0:_ls_],z_field.vector[0:_ls_])

    /// Iterate flow solver RHEA in time
    for(int time_iter = current_time_iter; time_iter < final_time_iter; time_iter++) {

        /// Start timer: calculate_time_step
        timers->start( "calculate_time_step" );

	/// Calculate time step
        this->calculateTimeStep();
	if( ( current_time + delta_t ) > final_time ) delta_t = final_time - current_time;

        /// Stop timer: calculate_time_step
        timers->stop( "calculate_time_step" );

        /// Stop timer: execute
        timers->stop( "execute" );

        /// Start timer: output_solver_state
        timers->start( "output_solver_state" );

        /// Print time iteration information (if criterion satisfied)
        if( ( current_time_iter%print_frequency_iter == 0 ) and ( my_rank == 0 ) ) {
	    auto diff_wall_clock_timer = chrono::steady_clock::now() - start_wall_clock_timer;
            cout << "Time iteration " << current_time_iter << ": "
                 << "time = " << scientific << current_time << " [s], "
                 << "time-step = " << scientific << delta_t << " [s], "
                 << "wall-clock time = " << scientific << chrono::duration< double, std::ratio<3600> >( diff_wall_clock_timer ).count() << " [h]" << endl;
        }

	/// Output current state data to file (if criterion satisfied)
	updated_cpu = false;	/// Reset CPU data update tracker
        if( current_time_iter%output_frequency_iter == 0 ) this->outputCurrentStateData();

	/// Output current 2d slices state data to file (if criterion satisfied)
        this->output2dSlicesCurrentStateData();

	/// Output temporal point probes data to files (if criterion satisfied)
	this->outputTemporalPointProbesData();

        /// Stop timer: output_solver_state
        timers->stop( "output_solver_state" );

        /// Start timer: execute
        timers->start( "execute" );

        /// Start timer: rk_iteration_loop
        timers->start( "rk_iteration_loop" );

        /// Runge-Kutta time-integration steps
        for(rk_time_stage = 1; rk_time_stage <= rk_number_stages; rk_time_stage++) {

            /// Start timer: calculate_thermophysical_properties
            timers->start( "calculate_thermophysical_properties" );

	    if( artificial_compressibility_method ) {

	        /// Calculate artificially modified transport coefficients
                this->calculateArtificiallyModifiedTransportCoefficients();

	    } else {

                /// Calculate transport coefficients
                this->calculateTransportCoefficients();

	    }

	    /// Stop timer: calculate_thermophysical_properties
            timers->stop( "calculate_thermophysical_properties" );

            /// Start timer: calculate_inviscid_fluxes
            timers->start( "calculate_inviscid_fluxes" );

            /// Calculate inviscid fluxes
            this->calculateInviscidFluxes();

	    /// Stop timer: calculate_inviscid_fluxes
            timers->stop( "calculate_inviscid_fluxes" );

            /// Start timer: calculate_viscous_fluxes
            timers->start( "calculate_viscous_fluxes" );

            /// Calculate viscous fluxes
            this->calculateViscousFluxes();
	    #pragma acc wait

            /// Stop timer: calculate_viscous_fluxes
            timers->stop( "calculate_viscous_fluxes" );

            /// Start timer: calculate_source_terms
            timers->start( "calculate_source_terms" );

            /// Calculate source terms
            this->calculateSourceTerms();

            /// Stop timer: calculate_source_terms
            timers->stop( "calculate_source_terms" );

            /// Start timer: time_advance_conserved_variables
            timers->start( "time_advance_conserved_variables" );

            /// Advance conserved variables in time
            this->timeAdvanceConservedVariables();

	    /// Stop timer: time_advance_conserved_variables
            timers->stop( "time_advance_conserved_variables" );

            /// Start timer: conserved_to_primitive_variables
            timers->start( "conserved_to_primitive_variables" );

            /// Calculate primitive variables from conserved variables
            this->conservedToPrimitiveVariables();

            /// Stop timer: conserved_to_primitive_variables
            timers->stop( "conserved_to_primitive_variables" );

            /// Start timer: calculate_thermodynamics_from_primitive_variables
            timers->start( "calculate_thermodynamics_from_primitive_variables" );

            /// Calculate thermodynamics from primitive variables
            this->calculateThermodynamicsFromPrimitiveVariables();

            if( artificial_compressibility_method ) {

                /// Calculate thermodynamic (bulk) pressure
                P_thermo = this->calculateVolumeAveragedPressure();
	        #pragma acc update device(P_thermo)

                /// Calculate alpha value of artificial compressibility method
                alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();
	        #pragma acc update device(alpha_acm)

                /// Calculate artificially modified thermodynamics
                this->calculateArtificiallyModifiedThermodynamics();

	    }

            /// Stop timer: calculate_thermodynamics_from_primitive_variables
            timers->stop( "calculate_thermodynamics_from_primitive_variables" );

            /// Start: pressure explicitly modified to (1-P_tol)*P_chamber <= P <= (1+P_tol)*P_chamber
            if( control_pressure ) {

		/// Mantain P to (1-P_tol)*P_chamber <= P <= (1+P_tol)*P_chamber
                #pragma acc parallel loop collapse(3) present(this, P_field.vector[0:_ls_])
                for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                    for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                        for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                            P_field[I1D(i,j,k)] = max( (1.0-P_tol)*P_chamber, min( (1.0+P_tol)*P_chamber, P_field[I1D(i,j,k)] ) );
                        }
                    }
                }
		
            }
            /// Stop: pressure explicitly modified to (1-P_tol)*P_chamber <= P <= (1+P_tol)*P_chamber

	    /// Start: temperature explicitly modified to T_min <= T <= T_max
            if( control_temperature ) {

	        /// Mantain T to T_min <= T <= T_max
		#pragma acc parallel loop collapse(3) present(this, T_field.vector[0:_ls_])
                for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                    for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                        for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
			    T_field[I1D(i,j,k)] = max( T_min, min( T_max, T_field[I1D(i,j,k)] ) );
  			    if( x_field[I1D(i,j,k)] < 2.0*D_jet ) {
  			        if(  y_field[I1D(i,j,k)]*y_field[I1D(i,j,k)] + z_field[I1D(i,j,k)]*z_field[I1D(i,j,k)]  >  (R_jet*1.9)*(R_jet*1.9) ) {
			            T_field[I1D(i,j,k)] = T_chamber;
			        }
			    }
			    if( x_field[I1D(i,j,k)] < 0.3*D_jet ) {
                                if(  y_field[I1D(i,j,k)]*y_field[I1D(i,j,k)] + z_field[I1D(i,j,k)]*z_field[I1D(i,j,k)]  >  (R_jet*1.4)*(R_jet*1.4) ) {
                                    T_field[I1D(i,j,k)] = T_chamber;
                                }
                            }
			}
		    }
		}

            }
	    /// Stop: temperature explicitly modified to T_b_w <= T <= T_t_w

	    /// Start: recalculate rho, E, sos, c_v and c_p from P and T
            if( control_pressure || control_temperature ) {

                double u, v, w, P, T;
                double rho, e, ke, E;
                double c_v, c_p;
		#pragma acc parallel loop collapse(3) private(ke,e,c_v,P,T,c_p) present(this, topo, thermodynamics, u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], c_p_field.vector[0:_ls_], c_v_field.vector[0:_ls_], sos_field.vector[0:_ls_])
		for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
                    for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
                        for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
		            /// Obtain primitive variables
                            u = u_field[I1D(i,j,k)];
                            v = v_field[I1D(i,j,k)];
                            w = w_field[I1D(i,j,k)];
                            P = P_field[I1D(i,j,k)];
                            T = T_field[I1D(i,j,k)];
		            /// Update rho, e, ke and E
                            thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho, e, P, T );
                            ke = 0.5*( u*u + v*v + w*w );
                            E  = e + ke;
		            /// Update conserved variables
			    rho_field[I1D(i,j,k)]  = rho;
                            rhou_field[I1D(i,j,k)] = rho*u;
                            rhov_field[I1D(i,j,k)] = rho*v;
                            rhow_field[I1D(i,j,k)] = rho*w;
                            rhoE_field[I1D(i,j,k)] = rho*E;
		            /// Update thermodynamics
		            if( artificial_compressibility_method ) {
                                sos_field[I1D(i,j,k)] = ( 1.0/( alpha_acm + epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T, rho );
                                thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T, rho );
			    } else {
                                sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P, T, rho );
                                thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P, T, rho );
			    }
                            c_v_field[I1D(i,j,k)] = c_v;
                            c_p_field[I1D(i,j,k)] = c_p;
			}
		    }
		}

	    }
	    /// Stop: recalculate rho, E, sos, c_v and c_p from P and T

            /// Start timer: update_boundaries
            timers->start( "update_boundaries" );

            /// Update boundary values
            this->updateBoundaries();

            /// Stop timer: update_boundaries
            timers->stop( "update_boundaries" );

        }

        /// Stop timer: rk_iteration_loop
        timers->stop( "rk_iteration_loop" );

        /// Start timer: temporal_hook_function
        timers->start( "temporal_hook_function" );

        /// Temporal hook function
        this->temporalHookFunction();

        /// Stop timer: temporal_hook_function
        timers->stop( "temporal_hook_function" );

        /// Start timer: update_time_averaged_quantities
        timers->start( "update_time_averaged_quantities" );

        /// Update time-averaged quantities
        if( time_averaging_active ) this->updateTimeAveragedQuantities();

        /// Stop timer: update_time_averaged_quantities
        timers->stop( "update_time_averaged_quantities" );

        /// Start timer: update_previous_state_conserved_variables
        timers->start( "update_previous_state_conserved_variables" );

        /// Update previous state of conserved variables
        this->updatePreviousStateConservedVariables();

        /// Update time and time iteration
        current_time += delta_t;
        current_time_iter += 1;

        /// Stop timer: update_previous_state_conserved_variables
        timers->stop( "update_previous_state_conserved_variables" );

        /// Check if simulation is completed: current_time > final_time
        if( current_time >= final_time ) break;

    }

    // ... deleting methods/fields from GPU
    #pragma acc exit data delete(x_field.vector[0:_ls_],y_field.vector[0:_ls_],z_field.vector[0:_ls_])

    /// Stop timer: time_iteration_loop
    timers->stop( "time_iteration_loop" );

    /// Print timers information
    if( print_timers ) timers->printTimers( timers_information_file );

    /// Print time advancement information
    if( my_rank == 0 ) {
        cout << "Time advancement completed -> "
             << "iteration = " << current_time_iter << ", "
             << "time = " << scientific << current_time << " [s]" << endl;
    }

    /// Output current state data to file
    updated_cpu = false;	/// Reset CPU data update tracker
    this->outputCurrentStateData();

    /// Output current 2d slices state data to file (if criterion satisfied)
    this->output2dSlicesCurrentStateData();

    /// Output temporal point probes data to files (if criterion satisfied)
    this->outputTemporalPointProbesData();

    /// Output current state particles data to file
    point_particles->write_to_file( current_time, current_time_iter );

    /// End RHEA simulation
    if( my_rank == 0 ) cout << "RHEA (v" << version_number << "): END SIMULATION" << endl;

    /// Stop timer: execute
    timers->stop( "execute" );

    /// Start wall-clock timer
    //auto end_wall_clock_timer = chrono::steady_clock::now();

};

#pragma acc routine seq
double myRHEA::pseudo_random_noise(int i, int j, int k) {

    unsigned int n = (unsigned int)(i * 73856093 ^ j * 19349663 ^ k * 83492791);
    n = (n >> 13) ^ n;
    n = (n * (n * n * 15731 + 789221) + 1376312589);

    return ( alpha_u*u_injection*( 1.0 - 2.0 * ( ( n & 0x7fffffff )/(double)0x7fffffff) ) );

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
