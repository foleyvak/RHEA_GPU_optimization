#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

/// FIXED PARAMETERS ///
const double tag_IBM_tolerance   = 1.0e-5;				/// Tag IBM tolerance (fixed)
const double geometric_tolerance = 1.0e-12;				/// Geometric tolerance (fixed)
const double pi = 2.0*asin( 1.0 );					/// pi number (fixed)

/// PROBLEM PARAMETERS ///
const double R_specific  = 287.058;					/// Specific gas constant
const double gamma_ref   = 1.4;						/// Heat capacity ratio
const double c_p         = gamma_ref*R_specific/( gamma_ref - 1.0 );	/// Isobaric heat capacity
const double Re          = 50000.0;					/// Reynolds number
const double Ma          = 0.3;						/// Mach number
const double Pr          = 0.71;					/// Prandtl number
const double rho_ref     = 1.0;						/// Reference density	
const double u_infty     = 1.0;						/// Free-stream velocity
const double C_airfoil   = 1.0;						/// Chord of airfoil
const double AoA_airfoil = 12.0;					/// Angle-of-attack of airfoil
const double mu          = rho_ref*u_infty*C_airfoil/Re;		/// Dynamic viscosity
const double kappa       = c_p*mu/Pr;					/// Thermal conductivity	
const double P_ref       = rho_ref*u_infty*u_infty/( gamma_ref*Ma*Ma );	/// Reference pressure
const double x_airfoil	 = 0.5;						/// Center of mass i x-direction of airfoil 
const double y_airfoil	 = 0.0;						/// Center of mass i y-direction of airfoil 
const double z_airfoil	 = 0.0;						/// Center of mass i z-direction of airfoil 
const double u_airfoil	 = 0.0;						/// u-velocity of airfoil 
const double v_airfoil	 = 0.0;						/// v-velocity of airfoil
const double w_airfoil	 = 0.0;						/// w-velocity of airfoil 
const double T_airfoil	 = P_ref/( rho_ref*R_specific );		/// Temperature of airfoil
const double alpha_u     = 1.0e-3;                                  	/// Magnitude of velocity perturbations

////////// myRHEA CLASS //////////

void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    srand( my_rank );

    /// All (inner, halo, boundary): u, v, w, P and T
    double random_number;
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                random_number       = 2.0*( (double) rand()/( RAND_MAX ) ) - 1.0;
                //u_field[I1D(i,j,k)] = u_infty;
                u_field[I1D(i,j,k)] = u_infty*( 1.0 + alpha_u*random_number );
                v_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = 0.0;
                P_field[I1D(i,j,k)] = P_ref;
                T_field[I1D(i,j,k)] = thermodynamics->calculateTemperatureFromPressureDensity( P_field[I1D(i,j,k)], rho_ref );
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

    /// Set velocity & temperature of IBM
    immersed_boundary_method->setVelocityIBM( u_airfoil, v_airfoil, w_airfoil );	// [m/s]
    immersed_boundary_method->setTemperatureIBM( T_airfoil );				// [K]
    
    /// Set tags of IBM (tagging)

    #pragma update host(tag_IBM_field.vector[0:_ls_], x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_])
    /// Read STL
    stl_reader::StlMesh< double, unsigned int > stl_mesh( "naca0012_aoa12deg.stl" );

    /// STL (stl_tris) defined as a vector of triangles. Each triangle is defined as a size-3 vector of D3 (corners)
    vector < vector < D3 > > stl_tris( stl_mesh.num_tris(), vector < D3 > (3) );
    for(size_t itri = 0; itri < stl_mesh.num_tris(); ++itri) {
        for(size_t icorner = 0; icorner < 3; ++icorner) {
            const double *c = stl_mesh.tri_corner_coords(itri, icorner);
            stl_tris[itri][icorner] = D3(c[0], c[1], c[2]);
        }
    }    

    /// Inner: tag_IBM ... first (coarse values) iteration
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {

            /// Index first z-direction gridpoint
	    int first_k_index = topo->iter_common[_INNER_][_INIZ_]; 

            /// Gridpoint to be tagged
            D3 gridpoint(x_field[I1D(i,j,first_k_index)], y_field[I1D(i,j,first_k_index)], z_field[I1D(i,j,first_k_index)]);

            /// Obtain tag_IBM value
	    double tag_IBM_value = 0.0;
            if( this->isInSTL(gridpoint, stl_tris) ) {
	        tag_IBM_value = 1.0;
	    }

            /// Set tag_IBM value along z-direction
	    for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
	        tag_IBM_field[I1D(i,j,k)] = tag_IBM_value;
            }

        }
    }

    /// Update halo values
    //#pragma update host(tag_IBM_field.vector[0:_ls_])
    tag_IBM_field.update();
    //#pragma update device(tag_IBM_field.vector[0:_ls_])

    /// Tag mixed/interface gridpoints ... a gridpoint is considerd mixed if its tag differs from any of its neighbouring tags 
    /// In case of mixed gridpoint, the tag value is replaced by the fluid/solid ratio
    const int mc_points_direction = 25;
    
    /// Inner: tag_IBM ... second (fine values) iteration
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {

            /// Index first z-direction gridpoint
	    int first_k_index = topo->iter_common[_INNER_][_INIZ_];

            /// Gridpoint position
            double x_position = x_field[I1D(i,j,first_k_index)];
	    double y_position = y_field[I1D(i,j,first_k_index)]; 

            /// Large bounding box containing the airfoil
            if( ( fabs( x_position - x_airfoil  ) < 0.625*C_airfoil ) && ( fabs( y_position - y_airfoil  ) < 0.625*C_airfoil ) ) {

                /// Initialize bool value    
                bool is_mixed = false;
                double gridpoint_tag_IBM = tag_IBM_field[I1D(i,j,first_k_index)];

                if     ( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i-1,j-1,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i-1,j  ,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }		
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i-1,j+1,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }		
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i  ,j-1,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }
		//else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i  ,j  ,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }	
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i  ,j+1,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i+1,j-1,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i+1,j  ,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }		
		else if( fabs( gridpoint_tag_IBM - tag_IBM_field[I1D(i+1,j+1,first_k_index)] ) > tag_IBM_tolerance ) { is_mixed = true; }

                if( is_mixed ) {

                    /// Geometric stuff
                    double delta_x = 0.5*( x_field[I1D(i+1,j,first_k_index)] - x_field[I1D(i-1,j,first_k_index)] ); 
                    double delta_y = 0.5*( y_field[I1D(i,j+1,first_k_index)] - y_field[I1D(i,j-1,first_k_index)] );

		    /// Monte Carlo approach
                    double mc_in_points = 0.0, mc_total_points = 0.0;
                    for(int ii = 0; ii < mc_points_direction; ++ii) {
                        for(int jj = 0; jj < mc_points_direction; ++jj) {
				double x = ( x_field[I1D(i,j,first_k_index)] - 0.5*delta_x ) + ii*delta_x/( mc_points_direction - 1 );
				double y = ( y_field[I1D(i,j,first_k_index)] - 0.5*delta_y ) + jj*delta_y/( mc_points_direction - 1 );
                                double z = z_field[I1D(i,j,first_k_index)];
                                D3 point(x,y,z);
                                if( this->isInSTL(point, stl_tris) ) mc_in_points += 1.0;
				mc_total_points += 1.0;
                        }
                    }
		    double tag_IBM_value = mc_in_points/mc_total_points;

                    for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                        tag_IBM_field[I1D(i,j,k)] = tag_IBM_value;
                    }

		    //cout << "Mixed: " << tag_IBM_field[I1D(i,j,first_k_index)] << endl;

                //} else {

		    //cout << "Not mixed: " << tag_IBM_field[I1D(i,j,first_k_index)] << endl;

		}

                //cout << "Inside box" << endl;

            //} else {

                //cout << "Outside box" << endl;

	    }
        }
    }
    
    /// Update halo values
    #pragma acc update device(tag_IBM_field.vector[0:_ls_])
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

void myRHEA::planeLineIntersect(const D3 &p, const D3 &ray, const vector<D3> &tri, D3 &intersect, double &lambda) {

    D3 v1 = tri[0];
    D3 v2 = tri[1];
    D3 v3 = tri[2];

    D3 tri_norm = crossProduct(v1 - v3, v2 - v3); tri_norm.normalize();
    lambda = ( (v3 - p)*tri_norm )/( ray*tri_norm );

    intersect = p + ( lambda*ray );

};

bool myRHEA::isInTri(const D3 &p, const vector<D3> &tri) {

    D3 v1 = tri[0];
    D3 v2 = tri[1];
    D3 v3 = tri[2];

    double S = 0.5*( crossProduct( v1 - v3, v2 - v3 ) ).norm2();

    double S1 = 0.5*( crossProduct( v1 - p, v2 - p ) ).norm2();
    double S2 = 0.5*( crossProduct( v1 - p, v3 - p ) ).norm2();
    double S3 = 0.5*( crossProduct( v2 - p, v3 - p ) ).norm2();

    if( ( S1 + S2 + S3 - S ) > geometric_tolerance ) {
        return false;
    } else {
        return true;
    }

};

bool myRHEA::isInSTL(const D3 &p, const vector<vector<D3> > &stl_tris) {

    /// Point at infinity ... may need to be adjusted for each problem
    //const D3 p_inf( 100.0*( x_0 + L_x ), 100.0*( y_0 + L_y ), 100.0*( z_0 + L_z ) );
    const double AoA_rad = AoA_airfoil*pi/180.0;
    const D3 p_inf( x_airfoil + 100.0*(-1.0)*cos( AoA_rad ), y_airfoil + 100.0*sin( AoA_rad ), p[2] );

    /// The entire STL is iterated to count the number of ray intersections with STL triangles
    /// If the resulting number is odd, p is in the STL. Otherwise is out
    int intersect_count = 0;
    for(std::vector < vector < D3 > >::const_iterator tri = stl_tris.begin(); tri != stl_tris.end(); ++tri) {
        const vector<D3> &triangle = *tri;
        D3 v1 = triangle[0];
	D3 v2 = triangle[1];
	D3 v3 = triangle[2];
        D3 tri_norm = crossProduct(v1 - v3, v2 - v3); tri_norm.normalize();
        D3 ray = p_inf - p;
      
        if( fabs( ( ray*(1.0/ray.norm2()) )*tri_norm ) > geometric_tolerance ) {
            D3 intersect;
            double lambda;
            this->planeLineIntersect(p, ray, triangle, intersect, lambda);
            if( this->isInTri( intersect, triangle ) && lambda > geometric_tolerance ) { ++intersect_count; }
        }
    }

    if( intersect_count%2 == 0 ) {
        return false;
    } else {
	return true;
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
