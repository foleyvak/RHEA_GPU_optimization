#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

/// AUXILIAR PARAMETERS ///
const double epsilon = 1.0e-15;			/// Small epsilon number (fixed)
const double pi = 2.0*asin( 1.0 );		/// pi number (fixed)

/// PROBLEM PARAMETERS ///
const double R_specific = 287.058;				/// Specific gas constant
const double gamma_0    = 1.4;					/// Heat capacity ratio
//const double c_p        = gamma_0*R_specific/( gamma_0 - 1.0 );	/// Isobaric heat capacity
//const double Re_delta_0_ast = 450.0;				/// Reynolds number based on inlet displacement thickness
const double Ma         = 3.0e-1;				/// Mach number
//const double Pr         = 0.71;					/// Prandtl number
const double U_infty    = 1.0;					/// Free-stream velocity
const double rho_ref    = 1.0;					/// Reference density
const double P_ref      = rho_ref*pow( U_infty, 2.0 )/( gamma_0*Ma*Ma );	/// Reference pressure
const double T_ref      = P_ref/( rho_ref*R_specific );		/// Reference temperature
const double delta_0_ast = 1.0;				  	/// Inlet displacement thickness
const double tau 	= delta_0_ast/U_infty;			/// Characteristic time scale 
//const double mu         = rho_ref*U_infty*delta_0_ast/Re_delta_0_ast;		/// Dynamic viscosity
//const double nu         = mu/rho_ref;				/// Kinematic viscosity
//const double kappa      = c_p*mu/Pr;				/// Thermal conductivity
const double L_x        = 750*delta_0_ast;			/// Domain size in x-direction 
//const double L_y        = 40*delta_0_ast;			/// Domain size in y-direction
//const double L_z        = 34*delta_0_ast;			/// Domain size in z-direction
const double alpha_u    = 1.0;                                  /// Magnitude of velocity perturbations
const double alpha_P    = 1.0e-3;                               /// Magnitude of pressure perturbations
const double x_0_domain = 0.0;					/// Domain origin in x-direction
const double y_0_domain = 0.0;					/// Domain origin in y-direction
const double L_x_forcing = 10*delta_0_ast;			/// Location of forcing downstream of inlet
const int time_iter_forcing = 10.0;				/// Forcing iteration interval
const double delta_x    = L_x/1024.0;				/// Mesh size in x-direction
double random_number;						/// Random number


////////// myRHEA CLASS //////////

void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// All (inner, halo, boundary): u, v, w, P and T
    #pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_])
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
		u_field[I1D(i,j,k)] = U_infty;
                v_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = 0.0;
                //P_field[I1D(i,j,k)] = P_ref;
                P_field[I1D(i,j,k)] = P_ref*( 1.0 + alpha_P*random_number );
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

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    srand( my_rank );
		
    if( current_time_iter%time_iter_forcing == 0 ) random_number = 2.0*( (double) rand()/( RAND_MAX ) ) - 1.0;

    /// Inner points: f_rhou, f_rhov, f_rhow and f_rhoE
    double x_position, y_position;
    #pragma acc parallel loop collapse(3) present(x_field.vector[0:_ls_], y_field.vector[0:_ls_], f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                f_rhou_field[I1D(i,j,k)] = 0.0;
                f_rhov_field[I1D(i,j,k)] = 0.0;
		x_position = x_field[I1D(i,j,k)];
		if( ( x_position > ( x_0_domain + L_x_forcing - 2.0*delta_x ) ) && ( x_position < ( x_0_domain + L_x_forcing + 2.0*delta_x ) ) ) {
		    y_position = y_field[I1D(i,j,k)];
	            if( y_position < 1.0*delta_0_ast ) {
                        f_rhov_field[I1D(i,j,k)] = rho_ref*( alpha_u*random_number*U_infty )/tau;
		    }
		}	
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
