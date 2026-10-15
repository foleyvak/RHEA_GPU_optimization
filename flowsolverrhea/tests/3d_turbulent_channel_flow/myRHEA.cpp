#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

////////// COMPILATION DIRECTIVES //////////
#define _FEEDBACK_LOOP_BODY_FORCE_ 0				/// Activate feedback loop for the body force moving the flow

/// AUXILIAR PARAMETERS ///
//const double pi = 2.0*asin( 1.0 );				/// pi number (fixed)

/// PROBLEM PARAMETERS ///
//const double R_specific = 287.058;				/// Specific gas constant
const double gamma_0    = 1.4;					/// Heat capacity ratio
//const double c_p        = gamma_0*R_specific/( gamma_0 - 1.0 );	/// Isobaric heat capacity
const double delta      = 1.0;					/// Channel half-height
const double Re_tau     = 180.0;				/// Friction Reynolds number
const double Ma         = 3.0e-1;				/// Mach number
//const double Pr         = 0.71;					/// Prandtl number
const double rho_0      = 1.0;					/// Reference density	
const double u_tau      = 1.0;					/// Friction velocity
const double tau_w      = rho_0*u_tau*u_tau;			/// Wall shear stress
const double mu         = rho_0*u_tau*delta/Re_tau;		/// Dynamic viscosity	
const double nu         = u_tau*delta/Re_tau;			/// Kinematic viscosity	
//const double kappa      = c_p*mu/Pr;				/// Thermal conductivity	
const double Re_b       = pow( Re_tau/0.09, 1.0/0.88 );		/// Bulk (approximated) Reynolds number
const double u_b        = nu*Re_b/( 2.0*delta );		/// Bulk (approximated) velocity
const double P_0        = rho_0*u_b*u_b/( gamma_0*Ma*Ma );	/// Reference pressure
//const double T_0        = P_0/( rho_0*R_specific );		/// Reference temperature
//const double L_x        = 4.0*pi*delta;			/// Streamwise length
//const double L_y        = 2.0*delta;				/// Wall-normal height
//const double L_z        = 4.0*pi*delta/3.0;			/// Spanwise width
const double kappa_vK   = 0.41;                                 /// von Kármán constant
const double y_0        = nu/( 9.0*u_tau );                     /// Smooth-wall roughness
const double u_0        = ( u_tau/kappa_vK )*( log( delta/y_0 ) + ( y_0/delta ) - 1.0 );        /// Volume average of a log-law velocity profile
const double alpha_u    = 1.0;                                  /// Magnitude of velocity perturbations
const double alpha_P    = 0.1;                                  /// Magnitude of pressure perturbations

#if _FEEDBACK_LOOP_BODY_FORCE_
/// Estimated uniform body force to drive the flow
double controller_output = tau_w/delta;			        /// Initialize controller output
double controller_error  = 0.0;			        	/// Initialize controller error
double controller_K_p    = 1.0e-1;		        	/// Controller proportional gain
#endif

////////// myRHEA CLASS //////////

void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    srand( my_rank );

    /// All (inner, halo, boundary): u, v, w, P and T
    double random_number, y_dist;
    #pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_])	
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                random_number       = 2.0*( (double) rand()/( RAND_MAX ) ) - 1.0;
                y_dist              = min( y_field[I1D(i,j,k)], 2.0*delta - y_field[I1D(i,j,k)] );
                u_field[I1D(i,j,k)] = ( 2.0*u_0*y_dist/delta ) + alpha_u*u_0*random_number;
                //v_field[I1D(i,j,k)] = 0.0;
                v_field[I1D(i,j,k)] = alpha_u*u_0*random_number;
                //w_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = alpha_u*u_0*random_number;
                //P_field[I1D(i,j,k)] = P_0;
                P_field[I1D(i,j,k)] = P_0*( 1.0 + alpha_P*random_number );
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

#if _FEEDBACK_LOOP_BODY_FORCE_
    /// Evaluate numerical shear stress at walls

    /// Calculate local values
    double local_sum_u_boundary_w = 0.0;
    double local_sum_u_inner_w    = 0.0;
    double local_number_grid_points_w = 0.0;

    /// South boundary
    #pragma acc kernels loop collapse(3) independent reduction(+:local_sum_u_boundary_w) reduction(+:local_sum_u_inner_w) reduction(+:local_number_grid_points_w) present(this, u_field.vector[0:_ls_])
    for(int i = topo->iter_bound[_SOUTH_][_INIX_]; i <= topo->iter_bound[_SOUTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_SOUTH_][_INIY_]; j <= topo->iter_bound[_SOUTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_SOUTH_][_INIZ_]; k <= topo->iter_bound[_SOUTH_][_ENDZ_]; k++) {
//		if( abs( avg_u_field[I1D(i,j,k)] ) > 0.0 ) {
//                    /// Sum boundary values
//                    local_sum_u_boundary_w += avg_u_field[I1D(i,j,k)];
//                    /// Sum inner values
//                    local_sum_u_inner_w    += avg_u_field[I1D(i,j+1,k)];
//		} else {
                    /// Sum boundary values
                    local_sum_u_boundary_w += u_field[I1D(i,j,k)];
                    /// Sum inner values
                    local_sum_u_inner_w    += u_field[I1D(i,j+1,k)];
//		}
                /// Sum number grid points
                local_number_grid_points_w += 1.0;
            }
        }
    }

    /// North boundary
    #pragma acc kernels loop collapse(3) independent reduction(+:local_sum_u_boundary_w) reduction(+:local_sum_u_inner_w) reduction(+:local_number_grid_points_w) present(this, u_field.vector[0:_ls_])
    for(int i = topo->iter_bound[_NORTH_][_INIX_]; i <= topo->iter_bound[_NORTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_NORTH_][_INIY_]; j <= topo->iter_bound[_NORTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_NORTH_][_INIZ_]; k <= topo->iter_bound[_NORTH_][_ENDZ_]; k++) {
//		if( abs( avg_u_field[I1D(i,j,k)] ) > 0.0 ) {
//                    /// Sum boundary values
//                    local_sum_u_boundary_w += avg_u_field[I1D(i,j,k)];
//                    /// Sum inner values
//                    local_sum_u_inner_w    += avg_u_field[I1D(i,j-1,k)];
//		} else {
                    /// Sum boundary values
                    local_sum_u_boundary_w += u_field[I1D(i,j,k)];
                    /// Sum inner values
                    local_sum_u_inner_w    += u_field[I1D(i,j-1,k)];
//		}
                /// Sum number grid points
                local_number_grid_points_w += 1.0;
            }
        }
    }   

    /// Communicate local values to obtain global & average values
    double global_sum_u_boundary_w;
    MPI_Allreduce(&local_sum_u_boundary_w, &global_sum_u_boundary_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_u_inner_w;
    MPI_Allreduce(&local_sum_u_inner_w, &global_sum_u_inner_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_number_grid_points_w;
    MPI_Allreduce(&local_number_grid_points_w, &global_number_grid_points_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_avg_u_boundary_w = global_sum_u_boundary_w/global_number_grid_points_w;
    double global_avg_u_inner_w    = global_sum_u_inner_w/global_number_grid_points_w;   

    /// Calculate delta_y
    double delta_y = mesh->getGloby(1) - mesh->getGloby(0);

    /// Calculate tau_wall_numerical
    double tau_w_numerical = mu*( global_avg_u_inner_w - global_avg_u_boundary_w )/delta_y;
    
    /// Update controller variables
    controller_error   = ( tau_w - tau_w_numerical )/delta;
    controller_output += controller_K_p*controller_error;

    //int my_rank, world_size;
    //MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //if( my_rank == 0 ) cout << tau_w << "  " << tau_w_numerical << "  " << controller_output << "  " << controller_error << endl;    
#endif

    /// Inner points: f_rhou, f_rhov, f_rhow and f_rhoE
    #pragma acc parallel loop collapse(3) present(this, f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
#if _FEEDBACK_LOOP_BODY_FORCE_
		f_rhou_field[I1D(i,j,k)] = controller_output;
#else
		f_rhou_field[I1D(i,j,k)] = tau_w/delta;
#endif
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
