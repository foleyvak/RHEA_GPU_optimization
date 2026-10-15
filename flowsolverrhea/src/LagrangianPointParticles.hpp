#ifndef _LAGRANGIAN_POINT_PARTICLES_
#define _LAGRANGIAN_POINT_PARTICLES_

////////// INCLUDES //////////
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <functional>
#include <cassert>
#include <type_traits>
#include <random>
#include <hdf5.h>
#include <mpi.h>
#include "yaml-cpp/yaml.h"
#include "MacroParameters.hpp"


////////// CLASS DECLARATION //////////
class BaseLagrangianPointParticle;			/// Base Lagrangian particle
//class DistributedPointParticles;			/// Container of distributed point particles

////////// FUNCTION DECLARATION //////////

////////// FIXED PARAMETERS //////////
#define _DEBUG_DISTRIBUTED_PARTICLES_ 0				/// Activate for debbuging
const double epsilon_geometry = 1.0e-5;				/// Small geometric number


////////// BaseLagrangianPointParticle CLASS //////////
class BaseLagrangianPointParticle {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        BaseLagrangianPointParticle();						/// Default constructor
        BaseLagrangianPointParticle(const std::string configuration_file_);	/// Parametrized constructor
        virtual ~BaseLagrangianPointParticle();					/// Destructor

	////////// GET FUNCTIONS //////////
        inline int getIdentifierPrt() { return( identifier_prt ); };
        inline double getDiameterPrt() { return( diameter_prt ); };
        inline double getDensityPrt() { return( density_prt ); };
        inline double * getPositionPrt() { return position_prt; };
        inline double * getPosition0Prt() { return position_0_prt; };
        inline double * getVelocityPrt() { return velocity_prt; };
        inline double * getVelocity0Prt() { return velocity_0_prt; };
        inline int getByteSizePrt() { return( sizeof( identifier_prt ) + sizeof( diameter_prt ) + sizeof( density_prt ) + sizeof( position_prt ) + sizeof( position_0_prt ) + sizeof( velocity_prt ) + sizeof( velocity_0_prt ) ); };

	////////// SET FUNCTIONS //////////
	inline void setIdentifierPrt(const int &identifier_prt_) { identifier_prt = identifier_prt_; };
	inline void setDiameterPrt(const double &diameter_prt_) { diameter_prt = diameter_prt_; };
	inline void setDensityPrt(const double &density_prt_) { density_prt = density_prt_; };
        inline void setPositionPrt(const double &x_position_prt, const double &y_position_prt, const double &z_position_prt) { position_prt[0] = x_position_prt; position_prt[1] = y_position_prt; position_prt[2] = z_position_prt; };
        inline void setPosition0Prt(const double &x_position_0_prt, const double &y_position_0_prt, const double &z_position_0_prt) { position_0_prt[0] = x_position_0_prt; position_0_prt[1] = y_position_0_prt; position_0_prt[2] = z_position_0_prt; };
        inline void setVelocityPrt(const double &u_velocity_prt, const double &v_velocity_prt, const double &w_velocity_prt) { velocity_prt[0] = u_velocity_prt; velocity_prt[1] = v_velocity_prt; velocity_prt[2] = w_velocity_prt; };
        inline void setVelocity0Prt(const double &u_velocity_0_prt, const double &v_velocity_0_prt, const double &w_velocity_0_prt) { velocity_0_prt[0] = u_velocity_0_prt; velocity_0_prt[1] = v_velocity_0_prt; velocity_0_prt[2] = w_velocity_0_prt; };

	////////// METHODS //////////
       
        /// Read configuration (input) file written in YAML language
        //virtual void readConfigurationFile( const std::string configuration_file ) {};
        void readConfigurationFile( const std::string configuration_file );		// ... modified for OpenACC
        
        /// Calculate volume of particle
        //virtual double calculateVolumePrt() = 0;
        double calculateVolumePrt();						// ... modified for OpenACC

        /// Calculate mass of particle
        //virtual double calculateMassPrt() = 0;
        double calculateMassPrt();						// ... modified for OpenACC

        /// Calculate relaxation time of particle
        //virtual double calculateRelaxationTimePrt(const double &mu_fluid) = 0;
        double calculateRelaxationTimePrt(const double &mu_fluid);		// ... modified for OpenACC

        /// Calculate density ratio of particle
        //virtual double calculateDensityRatioBetaPrt(const double &rho_fluid) = 0;
        double calculateDensityRatioBetaPrt(const double &rho_fluid);		// ... modified for OpenACC

        /// Append particle to a buffer
        void appendToBufferPrt(char *data) { 

            int bytes = 0;

            bytes = sizeof( identifier_prt );
            std::memcpy( data, &identifier_prt, bytes ); 
            data += bytes;

            bytes = sizeof( diameter_prt );
            std::memcpy( data, &diameter_prt, bytes ); 
            data += bytes;

            bytes = sizeof( density_prt );
            std::memcpy( data, &density_prt, bytes ); 
            data += bytes;

            bytes = sizeof( position_prt );
            std::memcpy( data, position_prt, bytes ); 
            data += bytes;
 
            bytes = sizeof( position_0_prt );
            std::memcpy( data, position_0_prt, bytes ); 
            data += bytes;
 
            bytes = sizeof( velocity_prt );
            std::memcpy( data, velocity_prt, bytes ); 
            data += bytes;

            bytes = sizeof( velocity_0_prt );
            std::memcpy( data, velocity_0_prt, bytes ); 

        };

        /// Assign particle from a buffer
        void assignFromBufferPrt(char *data) {
 
            int bytes = 0;

            bytes = sizeof( identifier_prt );
            std::memcpy( &identifier_prt, data, bytes ); 
            data += bytes;

            bytes = sizeof( diameter_prt );
            std::memcpy( &diameter_prt, data, bytes ); 
            data += bytes;

            bytes = sizeof( density_prt );
            std::memcpy( &density_prt, data, bytes ); 
            data += bytes;

            bytes = sizeof( position_prt );
            std::memcpy( position_prt, data, bytes ); 
            data += bytes;

            bytes = sizeof( position_0_prt );
            std::memcpy( position_0_prt, data, bytes ); 
            data += bytes;

            bytes = sizeof( velocity_prt );
            std::memcpy( velocity_prt, data, bytes ); 
            data += bytes;

            bytes = sizeof( velocity_0_prt );
            std::memcpy( velocity_0_prt, data, bytes ); 

        };

    protected:

        ////////// PARAMETERS //////////

        /// Identifier, diameter and density of particle
        int identifier_prt;		/// Identifier of particle
        double diameter_prt;		/// Diameter of particle
        double density_prt;		/// Density of particle

        /// Position and velocity of particle
        double position_prt[3];		/// Position of particle at time n+1: x = [0], y = [1], z = [2]
        double position_0_prt[3];	/// Position of particle at time n:   x = [0], y = [1], z = [2]
        double velocity_prt[3];		/// Velocity of particle at time n+1: u = [0], v = [1], w = [2]
        double velocity_0_prt[3];	/// Velocity of particle at time n:   u = [0], v = [1], w = [2]

    private:

};


////////// DistributedPointParticles CLASS //////////
class DistributedPointParticles {

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        DistributedPointParticles();						/// Default constructor
        DistributedPointParticles(const std::string configuration_file_);	/// Parametrized constructor
        virtual ~DistributedPointParticles();					/// Destructor

        /// Read configuration (input) file written in YAML language
        //virtual void readConfigurationFile( const std::string configuration_file ) {};
        void readConfigurationFile( const std::string configuration_file );		// ... modified for OpenACC

        /// Set subdomains distributed particles
        /// xmin, xmax specifies the domain that the current task is responsible for
        void set_subdomains_distributed_prts( double &xmin_local, double &xmax_local, double &ymin_local, double &ymax_local, double &zmin_local, double &zmax_local );

        /// Generate particles randomly within the 3D domain spread across all tasks
        void generate_prts_random(double &buffer_ratio_prts_) {

            if (get_num_prts_total() < 1) return;

            buffer_ratio_prts = buffer_ratio_prts_;

            // Estimate number of local particles and allocate arrays
            int estimate_particles_local = std::max(static_cast<int>(std::round(buffer_ratio_prts * num_prts_total / world_size)), static_cast<int>(buffer_ratio_prts));
	    
            allocateParticleArrays(estimate_particles_local);

            /// Set position of particles
    	    std::mt19937 engine;	/// Mersenne Twister random number generator
    	    engine.seed( 42 );		/// Set a fixed seed dynamically
	    std::uniform_real_distribution<double> dis_x( x_0 + epsilon_geometry*L_x, x_0 + L_x - epsilon_geometry*L_x );	/// Distribution between [x_0, x_0 + L_x]
	    std::uniform_real_distribution<double> dis_y( y_0 + epsilon_geometry*L_y, y_0 + L_y - epsilon_geometry*L_y );	/// Distribution between [y_0, y_0 + L_y]
	    std::uniform_real_distribution<double> dis_z( z_0 + epsilon_geometry*L_z, z_0 + L_z - epsilon_geometry*L_z );	/// Distribution between [z_0, z_0 + L_z]
	    std::vector<double> x_positions( num_prts_total, 0.0 );
	    std::vector<double> y_positions( num_prts_total, 0.0 );
	    std::vector<double> z_positions( num_prts_total, 0.0 );
            double random_number_x, random_number_y, random_number_z;
            
	    
	    for( int p = 0; p < num_prts_total; p++ ) {
                random_number_x = dis_x( engine );
	        x_positions[p] = random_number_x;
                random_number_y = dis_y( engine );
	        y_positions[p] = random_number_y;
                random_number_z = dis_z( engine );
	        z_positions[p] = random_number_z;
            }

            /// Add particle and provide identifier & position of local particles
            double x_position_prt, y_position_prt, z_position_prt;
            for( int p = 0; p < num_prts_total; p++ ) {

                /// Position of particle p
                x_position_prt = x_positions[p]; 
                y_position_prt = y_positions[p]; 
                z_position_prt = z_positions[p];

	        /*/// Update num_prts_local_in_use
                for( int rank = 0; rank < world_size; rank++ ) {
                    if( ( x_position_prt >= x_min_per_task[rank] ) and ( x_position_prt < x_max_per_task[rank] ) and
                        ( y_position_prt >= y_min_per_task[rank] ) and ( y_position_prt < y_max_per_task[rank] ) and
                        ( z_position_prt >= z_min_per_task[rank] ) and ( z_position_prt < z_max_per_task[rank] ) ) {
      	                num_prts_local_in_use[rank] += 1;
                        break;
	            }
	        }*/

	        /// Allocate and set attributes of local particles
    	        if( ( x_position_prt >= x_min_per_task[my_rank] ) and ( x_position_prt < x_max_per_task[my_rank] ) and
                    ( y_position_prt >= y_min_per_task[my_rank] ) and ( y_position_prt < y_max_per_task[my_rank] ) and
                    ( z_position_prt >= z_min_per_task[my_rank] ) and ( z_position_prt < z_max_per_task[my_rank] ) ) {

                     addParticle(p, diameter_prt, density_prt, x_position_prt, y_position_prt, z_position_prt, 0.0, 0.0, 0.0);
                }

            }
	    num_prts_local_in_use[my_rank] = prt_count;

#if _DEBUG_DISTRIBUTED_PARTICLES_
	    std::cout << my_rank << " (" << num_prts_local_in_use[my_rank] << "): " ;
	    for( int p = 0; p < this->get_num_prts_local_in_use( my_rank ); ++p ){
                std::cout << local_prts_positions_x[p] << " " << local_prts_positions_y[p] << " " << local_prts_positions_z[p] << std::endl;
            }
            std::cout << std::endl;
#endif

        };

        /// Has the DistributedPointParticles been created?
        //explicit operator bool() const { return local_prts_identifiers.size() > 0; };
	explicit operator bool() const { return local_prts_identifiers != nullptr; }
	

        /// Get a reference to the i'th num_prts_local_in_use
        int & get_num_prts_local_in_use(int i) { return num_prts_local_in_use[i]; };	
	
        /// Get a number of the allocated capacity of particles in the current rank
	int get_prt_capacity() const { return prt_capacity; }
	
        /// Get identifier of particle iprt
        int get_identifier_prt(int iprt) {
            return local_prts_identifiers[iprt];
        };

        /// Get diameter of particle iprt
        double get_diameter_prt(int iprt) {
            return local_prts_diameters[iprt];
        };

        /// Get density of particle iprt
        double get_density_prt(int iprt) {
            return local_prts_densities[iprt];
        };

        /// Get position x of particle iprt
        double get_position_x_prt(int iprt) {
            return local_prts_positions_x[iprt];
        };

        /// Get position y of particle iprt
        double get_position_y_prt(int iprt) {
            return local_prts_positions_y[iprt];
        };

        /// Get position z of particle iprt
        double get_position_z_prt(int iprt) {
            return local_prts_positions_z[iprt];
        };

        /// Get position 0 x of particle iprt
        double get_position_0_x_prt(int iprt) {
            return local_prts_positions_0_x[iprt];
        };

        /// Get position 0 y of particle iprt
        double get_position_0_y_prt(int iprt) {
            return local_prts_positions_0_y[iprt];
        };

        /// Get position 0 z of particle iprt
        double get_position_0_z_prt(int iprt) {
            return local_prts_positions_0_z[iprt];
        };

        /// Get velocity x of particle iprt
        double get_velocity_x_prt(int iprt) {
            return local_prts_velocities_x[iprt];
        };

        /// Get velocity y of particle iprt
        double get_velocity_y_prt(int iprt) {
            return local_prts_velocities_y[iprt];
        };

        /// Get velocity z of particle iprt
        double get_velocity_z_prt(int iprt) {
            return local_prts_velocities_z[iprt];
        };

        /// Get velocity 0 x of particle iprt
        double get_velocity_0_x_prt(int iprt) {
            return local_prts_velocities_0_x[iprt];
        };

        /// Get velocity 0 y of particle iprt
        double get_velocity_0_y_prt(int iprt) {
            return local_prts_velocities_0_y[iprt];
        };

        /// Get velocity 0 z of particle iprt
        double get_velocity_0_z_prt(int iprt) {
            return local_prts_velocities_0_z[iprt];
        };

        /// Get index 0 i of particle iprt
        int get_index_0_i_prt(int iprt) {
            return local_prts_indexes_0_i[iprt];
        };

        /// Get index 0 j of particle iprt
        int get_index_0_j_prt(int iprt) {
            return local_prts_indexes_0_j[iprt];
        };

        /// Get index 0 k of particle iprt
        int get_index_0_k_prt(int iprt) {
            return local_prts_indexes_0_k[iprt];
        };

        /// Get the byte-size of particle iprt
        int get_byte_size_prt(int iprt) {

	    int byte_size_prt = sizeof( local_prts_identifiers[iprt] ) + sizeof( local_prts_diameters[iprt] ) + sizeof( local_prts_densities[iprt] ) + sizeof( local_prts_positions_x[iprt] ) + sizeof( local_prts_positions_y[iprt] ) + sizeof( local_prts_positions_z[iprt] ) + sizeof( local_prts_positions_0_x[iprt] ) + sizeof( local_prts_positions_0_y[iprt] ) + sizeof( local_prts_positions_0_z[iprt] ) + sizeof( local_prts_velocities_x[iprt] ) + sizeof( local_prts_velocities_y[iprt] ) + sizeof( local_prts_velocities_z[iprt] ) + sizeof( local_prts_velocities_0_x[iprt] ) + sizeof( local_prts_velocities_0_y[iprt] ) + sizeof( local_prts_velocities_0_z[iprt] ) + sizeof( local_prts_indexes_0_i[iprt] ) + sizeof( local_prts_indexes_0_j[iprt] ) + sizeof( local_prts_indexes_0_k[iprt] );

            return( byte_size_prt );

        };

        /// Set identifier of particle iprt
        void set_identifier_prt(const int &iprt, const int &identifier_prt) {
            local_prts_identifiers[iprt] = identifier_prt;
        };

        /// Set diameter of particle iprt
        void set_diameter_prt(const int &iprt, const double &diameter_prt) {
            local_prts_diameters[iprt] = diameter_prt;
        };

        /// Set density of particle iprt
        void set_density_prt(const int &iprt, const double &density_prt) {
            local_prts_densities[iprt] = density_prt;
        };

        /// Set position x of particle iprt
        void set_position_x_prt(const int &iprt, const double &x_position_prt) {
            local_prts_positions_x[iprt] = x_position_prt;
        };

        /// Set position y of particle iprt
        void set_position_y_prt(const int &iprt, const double &y_position_prt) {
            local_prts_positions_y[iprt] = y_position_prt;
        };

        /// Set position z of particle iprt
        void set_position_z_prt(const int &iprt, const double &z_position_prt) {
            local_prts_positions_z[iprt] = z_position_prt;
        };

        /// Set position 0 x of particle iprt
        void set_position_0_x_prt(const int &iprt, const double &x_position_0_prt) {
            local_prts_positions_0_x[iprt] = x_position_0_prt;
        };

        /// Set position 0 y of particle iprt
        void set_position_0_y_prt(const int &iprt, const double &y_position_0_prt) {
            local_prts_positions_0_y[iprt] = y_position_0_prt;
        };

        /// Set position 0 z of particle iprt
        void set_position_0_z_prt(const int &iprt, const double &z_position_0_prt) {
            local_prts_positions_0_z[iprt] = z_position_0_prt;
        };

        /// Set velocity x of particle iprt
        void set_velocity_x_prt(const int &iprt, const double &x_velocity_prt) {
            local_prts_velocities_x[iprt] = x_velocity_prt;
        };

        /// Set velocity y of particle iprt
        void set_velocity_y_prt(const int &iprt, const double &y_velocity_prt) {
            local_prts_velocities_y[iprt] = y_velocity_prt;
        };

        /// Set velocity z of particle iprt
        void set_velocity_z_prt(const int &iprt, const double &z_velocity_prt) {
            local_prts_velocities_z[iprt] = z_velocity_prt;
        };

        /// Set velocity 0 x of particle iprt
        void set_velocity_0_x_prt(const int &iprt, const double &x_velocity_0_prt) {
            local_prts_velocities_0_x[iprt] = x_velocity_0_prt;
        };

        /// Set velocity 0 y of particle iprt
        void set_velocity_0_y_prt(const int &iprt, const double &y_velocity_0_prt) {
            local_prts_velocities_0_y[iprt] = y_velocity_0_prt;
        };

        /// Set velocity 0 z of particle iprt
        void set_velocity_0_z_prt(const int &iprt, const double &z_velocity_0_prt) {
            local_prts_velocities_0_z[iprt] = z_velocity_0_prt;
        };

        /// Set index 0 i of particle iprt
        void set_index_0_i_prt(const int &iprt, const int &i_index_0_prt) {
            local_prts_indexes_0_i[iprt] = i_index_0_prt;
        };

        /// Set index 0 j of particle iprt
        void set_index_0_j_prt(const int &iprt, const int &j_index_0_prt) {
            local_prts_indexes_0_j[iprt] = j_index_0_prt;
        };

        /// Set index 0 k of particle iprt
        void set_index_0_k_prt(const int &iprt, const int &k_index_0_prt) {
            local_prts_indexes_0_k[iprt] = k_index_0_prt;
        };

        /// Append particle to buffer
        void append_to_buffer_prt(int iprt, char *data) {

            int bytes = 0;

            bytes = sizeof( local_prts_identifiers[iprt] );
            std::memcpy( data, &local_prts_identifiers[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_diameters[iprt] );
            std::memcpy( data, &local_prts_diameters[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_densities[iprt] );
            std::memcpy( data, &local_prts_densities[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_x[iprt] );
            std::memcpy( data, &local_prts_positions_x[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_y[iprt] );
            std::memcpy( data, &local_prts_positions_y[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_z[iprt] );
            std::memcpy( data, &local_prts_positions_z[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_0_x[iprt] );
            std::memcpy( data, &local_prts_positions_0_x[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_0_y[iprt] );
            std::memcpy( data, &local_prts_positions_0_y[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_0_z[iprt] );
            std::memcpy( data, &local_prts_positions_0_z[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_x[iprt] );
            std::memcpy( data, &local_prts_velocities_x[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_y[iprt] );
            std::memcpy( data, &local_prts_velocities_y[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_z[iprt] );
            std::memcpy( data, &local_prts_velocities_z[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_0_x[iprt] );
            std::memcpy( data, &local_prts_velocities_0_x[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_0_y[iprt] );
            std::memcpy( data, &local_prts_velocities_0_y[iprt], bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_0_z[iprt] );
            std::memcpy( data, &local_prts_velocities_0_z[iprt], bytes );
            data += bytes;

            bytes = sizeof( local_prts_indexes_0_i[iprt] );
            std::memcpy( data, &local_prts_indexes_0_i[iprt], bytes );
            data += bytes;

            bytes = sizeof( local_prts_indexes_0_j[iprt] );
            std::memcpy( data, &local_prts_indexes_0_j[iprt], bytes );
            data += bytes;

            bytes = sizeof( local_prts_indexes_0_k[iprt] );
            std::memcpy( data, &local_prts_indexes_0_k[iprt], bytes );
 
        };

        /// Assign particle from buffer
        void assign_from_buffer_prt(int iprt, char *data) {

            int bytes = 0;

            bytes = sizeof( local_prts_identifiers[iprt] );
            std::memcpy( &local_prts_identifiers[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_diameters[iprt] );
            std::memcpy( &local_prts_diameters[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_densities[iprt] );
            std::memcpy( &local_prts_densities[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_x[iprt] );
            std::memcpy( &local_prts_positions_x[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_y[iprt] );
            std::memcpy( &local_prts_positions_y[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_z[iprt] );
            std::memcpy( &local_prts_positions_z[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_0_x[iprt] );
            std::memcpy( &local_prts_positions_0_x[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_0_y[iprt] );
            std::memcpy( &local_prts_positions_0_y[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_positions_0_z[iprt] );
            std::memcpy( &local_prts_positions_0_z[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_x[iprt] );
            std::memcpy( &local_prts_velocities_x[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_y[iprt] );
            std::memcpy( &local_prts_velocities_y[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_z[iprt] );
            std::memcpy( &local_prts_velocities_z[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_0_x[iprt] );
            std::memcpy( &local_prts_velocities_0_x[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_0_y[iprt] );
            std::memcpy( &local_prts_velocities_0_y[iprt], data, bytes ); 
            data += bytes;

            bytes = sizeof( local_prts_velocities_0_z[iprt] );
            std::memcpy( &local_prts_velocities_0_z[iprt], data, bytes );
            data += bytes;

            bytes = sizeof( local_prts_indexes_0_i[iprt] );
            std::memcpy( &local_prts_indexes_0_i[iprt], data, bytes );
            data += bytes;

            bytes = sizeof( local_prts_indexes_0_j[iprt] );
            std::memcpy( &local_prts_indexes_0_j[iprt], data, bytes );
            data += bytes;

            bytes = sizeof( local_prts_indexes_0_k[iprt] );
            std::memcpy( &local_prts_indexes_0_k[iprt], data, bytes );

        };

        /// Calculate volume of particle iprt
        double calculate_volume_prt(int iprt);

        /// Calculate mass of particle iprt
        double calculate_mass_prt(int iprt);

        /// Calculate relaxation time of particle iprt
        double calculate_relaxation_time_prt(int iprt, const double &mu_fluid);

        /// Calculate density ratio beta of particle iprt
        double calculate_density_ratio_beta_prt(int iprt, const double &rho_fluid);

        /// Total number of active particles across all tasks
        int get_num_prts_total() const { return num_prts_total; };
      
        /// Total number of active particles across all tasks
        double get_buffer_ratio_prts() const { return buffer_ratio_prts; };

        /// Communicate particles across CPU boundaries
        void communicate_prts() {

            /// Skip if there are no particles
            if( this->get_num_prts_total() < 1 ) return;

#if _DEBUG_DISTRIBUTED_PARTICLES_
            std::cout << "Communicating particles task: " << my_rank << " number particles: " << num_prts_local_in_use[my_rank] << std::endl;
	    MPI_Barrier( MPI_COMM_WORLD );
            int prt_count_local = num_prts_local_in_use[my_rank];
            int prt_count_global;
            MPI_Allreduce( &prt_count_local, &prt_count_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
            if( my_rank == 0 ) std::cout << "Total number particles: " << prt_count_global << std::endl;
            MPI_Allreduce( &prt_count_local, &prt_count_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	    if( prt_count_global != this->get_num_prts_total() ) {
                MPI_Abort(MPI_COMM_WORLD, 0);
	    }
#endif
    
            /// Count number of particles to send to each task, move the particles to be sent to the back of the array,
            /// and reduce the num_prts_local_in_use if a partice is to be sent.
            /// After this is done, we have all the particles to be sent in location
            double x_position_prt, y_position_prt, z_position_prt, tmp_double;
            int aux_x_prt, aux_y_prt, aux_z_prt, rank_to_send, tmp_int;
	    std::vector<int> num_to_send( world_size, 0 );
	    std::vector<int> num_to_recv( world_size, 0 );
	    std::vector<int> num_bytes_to_send( world_size, 0 );
	    std::vector<int> num_bytes_to_recv( world_size, 0 );
            for( int p = 0; p < num_prts_local_in_use[my_rank]; p++ ) {

                /// Particle i position
                x_position_prt = get_position_x_prt( p );
                y_position_prt = get_position_y_prt( p );
                z_position_prt = get_position_z_prt( p );

                /// Locate particle
    	        if( ( x_position_prt >= x_min_per_task[my_rank] ) and ( x_position_prt < x_max_per_task[my_rank] ) and
                    ( y_position_prt >= y_min_per_task[my_rank] ) and ( y_position_prt < y_max_per_task[my_rank] ) and
                    ( z_position_prt >= z_min_per_task[my_rank] ) and ( z_position_prt < z_max_per_task[my_rank] ) ) {

                    /// Particle is within the range of the local task
                    continue;

                } else {

                    /// x-direction ... at first, it is assumed to be in the local task
                    aux_x_prt = 0;
                    if( x_position_prt < x_min_per_task[my_rank] ) {
                        /// Particle is west to the local task
                        aux_x_prt = -1;
                    } else if( x_position_prt >= x_max_per_task[my_rank] ) {
                        /// Particle is east to the local task
                        aux_x_prt = 1;
	            }

                    /// y-direction ... at first, it is assumed to be in the local task
                    aux_y_prt = 0;
                    if( y_position_prt < y_min_per_task[my_rank] ) {
                        /// Particle is south to the local task
                        aux_y_prt = -1;
                    } else if( y_position_prt >= y_max_per_task[my_rank] ) {
                        /// Particle is north to the local task
                        aux_y_prt = 1;
	            }

                    /// z-direction ... at first, it is assumed to be in the local task
                    aux_z_prt = 0;
                    if( z_position_prt < z_min_per_task[my_rank] ) {
                        /// Particle is back to the local task
                        aux_z_prt = -1;
                    } else if( z_position_prt >= z_max_per_task[my_rank] ) {
                        /// Particle is front to the local task
                        aux_z_prt = 1;
	            }

                    /// !!!FUTURE WORK: ALL TYPE OF BOUNDARY CONDITIONS NEED TO BE TREATED!!!
                    if( ( bocos_type[_WEST_] == _PERIODIC_ ) and ( aux_x_prt == -1 ) and ( x_min_per_task[my_rank] < ( x_0 + epsilon_geometry*L_x ) ) ) {
	                aux_x_prt = np_x - 1;
                    } else if( ( bocos_type[_EAST_] == _PERIODIC_ ) and ( aux_x_prt == 1 ) and ( x_max_per_task[my_rank] > ( x_0 + L_x - epsilon_geometry*L_x ) ) ) {
	                aux_x_prt = 1 - np_x;
                    }
                    if( ( bocos_type[_SOUTH_] == _PERIODIC_ ) and ( aux_y_prt == -1 ) and ( y_min_per_task[my_rank] < ( y_0 + epsilon_geometry*L_y ) ) ) {
	                aux_y_prt = np_y - 1; 
                    } else if( ( bocos_type[_NORTH_] == _PERIODIC_ ) and ( aux_y_prt == 1 ) and ( y_max_per_task[my_rank] > ( y_0 + L_y - epsilon_geometry*L_y ) ) ) {
	                aux_y_prt = 1 - np_y;
                    }
                    if( ( bocos_type[_BACK_] == _PERIODIC_ ) and ( aux_z_prt == -1 ) and ( z_min_per_task[my_rank] < ( z_0 + epsilon_geometry*L_z ) ) ) {
	                aux_z_prt = np_z - 1;
                    } else if( ( bocos_type[_FRONT_] == _PERIODIC_ ) and ( aux_z_prt == 1 ) and ( z_max_per_task[my_rank] > ( z_0 + L_z - epsilon_geometry*L_z ) ) ) {
	                aux_z_prt = 1 - np_z;
                    }

                    /// Identify rank to send
                    rank_to_send = my_rank + aux_x_prt + aux_y_prt*np_x + aux_z_prt*np_x*np_y;
#if _DEBUG_DISTRIBUTED_PARTICLES_
		    std::cout << "My rank: " << my_rank << "  ... rank to send: " << rank_to_send << "  " << aux_x_prt << "  " << aux_y_prt << "  " << aux_z_prt << std::endl;           
#endif

                    /// Fill in send vectors
		    if( rank_to_send != my_rank ) {

                        num_to_send[rank_to_send] += 1;
                        num_bytes_to_send[rank_to_send] += get_byte_size_prt( p );
                        num_prts_local_in_use[my_rank] -= 1;

                        tmp_int = local_prts_identifiers[p];
                        local_prts_identifiers[p] = local_prts_identifiers[num_prts_local_in_use[my_rank]];
			local_prts_identifiers[num_prts_local_in_use[my_rank]] = tmp_int; 

                        tmp_double = local_prts_diameters[p];
                        local_prts_diameters[p] = local_prts_diameters[num_prts_local_in_use[my_rank]];
			local_prts_diameters[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_densities[p];
                        local_prts_densities[p] = local_prts_densities[num_prts_local_in_use[my_rank]];
			local_prts_densities[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_positions_x[p];
                        local_prts_positions_x[p] = local_prts_positions_x[num_prts_local_in_use[my_rank]];
			local_prts_positions_x[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_positions_y[p];
                        local_prts_positions_y[p] = local_prts_positions_y[num_prts_local_in_use[my_rank]];
			local_prts_positions_y[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_positions_z[p];
                        local_prts_positions_z[p] = local_prts_positions_z[num_prts_local_in_use[my_rank]];
			local_prts_positions_z[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_positions_0_x[p];
                        local_prts_positions_0_x[p] = local_prts_positions_0_x[num_prts_local_in_use[my_rank]];
			local_prts_positions_0_x[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_positions_0_y[p];
                        local_prts_positions_0_y[p] = local_prts_positions_0_y[num_prts_local_in_use[my_rank]];
			local_prts_positions_0_y[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_positions_0_z[p];
                        local_prts_positions_0_z[p] = local_prts_positions_0_z[num_prts_local_in_use[my_rank]];
			local_prts_positions_0_z[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_velocities_x[p];
                        local_prts_velocities_x[p] = local_prts_velocities_x[num_prts_local_in_use[my_rank]];
			local_prts_velocities_x[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_velocities_y[p];
                        local_prts_velocities_y[p] = local_prts_velocities_y[num_prts_local_in_use[my_rank]];
			local_prts_velocities_y[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_velocities_z[p];
                        local_prts_velocities_z[p] = local_prts_velocities_z[num_prts_local_in_use[my_rank]];
			local_prts_velocities_z[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_velocities_0_x[p];
                        local_prts_velocities_0_x[p] = local_prts_velocities_0_x[num_prts_local_in_use[my_rank]];
			local_prts_velocities_0_x[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_velocities_0_y[p];
                        local_prts_velocities_0_y[p] = local_prts_velocities_0_y[num_prts_local_in_use[my_rank]];
			local_prts_velocities_0_y[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_double = local_prts_velocities_0_z[p];
                        local_prts_velocities_0_z[p] = local_prts_velocities_0_z[num_prts_local_in_use[my_rank]];
			local_prts_velocities_0_z[num_prts_local_in_use[my_rank]] = tmp_double;

                        tmp_int = local_prts_indexes_0_i[p];
                        local_prts_indexes_0_i[p] = local_prts_indexes_0_i[num_prts_local_in_use[my_rank]];
			local_prts_indexes_0_i[num_prts_local_in_use[my_rank]] = tmp_int;

                        tmp_int = local_prts_indexes_0_j[p];
                        local_prts_indexes_0_j[p] = local_prts_indexes_0_j[num_prts_local_in_use[my_rank]];
			local_prts_indexes_0_j[num_prts_local_in_use[my_rank]] = tmp_int;

                        tmp_int = local_prts_indexes_0_k[p];
                        local_prts_indexes_0_k[p] = local_prts_indexes_0_k[num_prts_local_in_use[my_rank]];
			local_prts_indexes_0_k[num_prts_local_in_use[my_rank]] = tmp_int;

			p -= 1;

		    } else {

                        /// !!!FUTURE WORK: ALL TYPE OF BOUNDARY CONDITIONS NEED TO BE TREATED!!!
                        if( ( bocos_type[_WEST_] == _PERIODIC_ ) and ( x_position_prt < x_min_per_task[my_rank] ) ) {
	    	    	    x_position_prt += L_x;
                            local_prts_positions_x[p] = x_position_prt;
                        } else if( ( bocos_type[_EAST_] == _PERIODIC_ ) and ( x_position_prt >= x_max_per_task[my_rank] ) ) {
	    	    	    x_position_prt -= L_x;
                            local_prts_positions_x[p] = x_position_prt;
                        }
                        if( ( bocos_type[_SOUTH_] == _PERIODIC_ ) and ( y_position_prt < y_min_per_task[my_rank] ) ) {
	    	    	    y_position_prt += L_y;
                            local_prts_positions_y[p] = y_position_prt;
                        } else if( ( bocos_type[_NORTH_] == _PERIODIC_ ) and ( y_position_prt >= y_max_per_task[my_rank] ) ) {
	    	    	    y_position_prt -= L_y;
                            local_prts_positions_y[p] = y_position_prt;
                        }
                        if( ( bocos_type[_BACK_] == _PERIODIC_ ) and ( z_position_prt < z_min_per_task[my_rank] ) ) {
	    	    	    z_position_prt += L_z;
                            local_prts_positions_z[p] = z_position_prt;
                        } else if( ( bocos_type[_FRONT_] == _PERIODIC_ ) and ( z_position_prt >= z_max_per_task[my_rank] ) ) {
	    	    	    z_position_prt -= L_z;
                            local_prts_positions_z[p] = z_position_prt;
                        }

                    }

                }

	    }

	    /// Communicate to get number of particles to receive from each task
            for( int i = 1; i < world_size; i++ ) {

                int send_request_to = ( my_rank + i ) % world_size;
                int get_request_from = ( my_rank - i + world_size ) % world_size;

                /// Send & receive number of particles to communicate
                MPI_Status status;
                MPI_Sendrecv( &num_to_send[send_request_to], 1, MPI_INT, send_request_to, 0, &num_to_recv[get_request_from], 1, MPI_INT, get_request_from, 0, MPI_COMM_WORLD, &status );

                /// Send & receive number of bytes to communicate
                MPI_Sendrecv( &num_bytes_to_send[send_request_to], 1, MPI_INT, send_request_to, 0, &num_bytes_to_recv[get_request_from], 1, MPI_INT, get_request_from, 0, MPI_COMM_WORLD, &status );

            }

#if _DEBUG_DISTRIBUTED_PARTICLES_
            /// Show some info
            if( my_rank == 0 ) {
                for( int i = 0; i < world_size; i++ ) {
	            std::cout << "Task " << my_rank << " send to   " << i << " : " << num_to_send[i] << std::endl;
	            std::cout << "Task " << my_rank << " recv from " << i << " : " << num_to_recv[i] << std::endl;
                }
            }
#endif

            /// Total number to send and recv
            int ntot_to_send = 0;
            int ntot_to_recv = 0;
            int ntot_bytes_to_send = 0;
            int ntot_bytes_to_recv = 0;
            for( int i = 0; i < world_size; i++ ) {

                ntot_to_send += num_to_send[i];
                ntot_to_recv += num_to_recv[i];
                ntot_bytes_to_send += num_bytes_to_send[i];
                ntot_bytes_to_recv += num_bytes_to_recv[i];

            }

            /// Allocate send buffer
	    std::vector<char> send_buffer( ntot_bytes_to_send );
	    std::vector<char> recv_buffer( ntot_bytes_to_recv );

            /// Pointers to each send-recv place in the send-recv buffer
	    std::vector<int> offset_in_send_buffer( world_size, 0 );
	    std::vector<int> offset_in_recv_buffer( world_size, 0 );
	    std::vector<char *> send_buffer_by_task( world_size, send_buffer.data() );
	    std::vector<char *> recv_buffer_by_task( world_size, recv_buffer.data() );
            for( int i = 1; i < world_size; i++ ) {

                offset_in_send_buffer[i] = offset_in_send_buffer[i - 1] + num_bytes_to_send[i - 1];
                offset_in_recv_buffer[i] = offset_in_recv_buffer[i - 1] + num_bytes_to_recv[i - 1];
                send_buffer_by_task[i] = &send_buffer.data()[offset_in_send_buffer[i]];
                recv_buffer_by_task[i] = &recv_buffer.data()[offset_in_recv_buffer[i]];

            }

            /// Gather particle data
            int id_prt;
            for( int p = 0; p < ntot_to_send; p++ ) {

                /// Particle index and position
                id_prt = num_prts_local_in_use[my_rank] + p;
                x_position_prt = get_position_x_prt( id_prt );
                y_position_prt = get_position_y_prt( id_prt );
                z_position_prt = get_position_z_prt( id_prt );

                /// Locate particle to send
    	        if( ( x_position_prt >= x_min_per_task[my_rank] ) and ( x_position_prt < x_max_per_task[my_rank] ) and
                    ( y_position_prt >= y_min_per_task[my_rank] ) and ( y_position_prt < y_max_per_task[my_rank] ) and
                    ( z_position_prt >= z_min_per_task[my_rank] ) and ( z_position_prt < z_max_per_task[my_rank] ) ) {

                    /// Particle is within the range of the local task ... we should not be here as particles have been swapped
	            continue;

                } else {

                    /// x-direction ... at first, it is assumed to be in the local task
                    aux_x_prt = 0;
                    if( x_position_prt < x_min_per_task[my_rank] ) {
                        /// Particle is west to the local task
                        aux_x_prt = -1;
                    } else if( x_position_prt >= x_max_per_task[my_rank] ) {
                        /// Particle is east to the local task
                        aux_x_prt = 1;
	            }

                    /// y-direction ... at first, it is assumed to be in the local task
                    aux_y_prt = 0;
                    if( y_position_prt < y_min_per_task[my_rank] ) {
                        /// Particle is south to the local task
                        aux_y_prt = -1;
                    } else if( y_position_prt >= y_max_per_task[my_rank] ) {
                        /// Particle is north to the local task
                        aux_y_prt = 1;
	            }

                    /// z-direction ... at first, it is assumed to be in the local task
                    aux_z_prt = 0;
                    if( z_position_prt < z_min_per_task[my_rank] ) {
                        /// Particle is back to the local task
                        aux_z_prt = -1;
                    } else if( z_position_prt >= z_max_per_task[my_rank] ) {
                        /// Particle is front to the local task
                        aux_z_prt = 1;
	            }

                    /// !!!FUTURE WORK: ALL TYPE OF BOUNDARY CONDITIONS NEED TO BE TREATED!!!
                    if( ( bocos_type[_WEST_] == _PERIODIC_ ) and ( aux_x_prt == -1 ) and ( x_min_per_task[my_rank] < ( x_0 + epsilon_geometry*L_x ) ) ) {
	                aux_x_prt = np_x - 1;
	    	    	x_position_prt += L_x;
                        local_prts_positions_x[id_prt] = x_position_prt;
                    } else if( ( bocos_type[_EAST_] == _PERIODIC_ ) and ( aux_x_prt == 1 ) and ( x_max_per_task[my_rank] > ( x_0 + L_x - epsilon_geometry*L_x ) ) ) {
	                aux_x_prt = 1 - np_x;
	    	    	x_position_prt -= L_x;
                        local_prts_positions_x[id_prt] = x_position_prt;
                    }
                    if( ( bocos_type[_SOUTH_] == _PERIODIC_ ) and ( aux_y_prt == -1 ) and ( y_min_per_task[my_rank] < ( y_0 + epsilon_geometry*L_y ) ) ) {
	                aux_y_prt = np_y - 1;
	    	    	y_position_prt += L_y;
                        local_prts_positions_y[id_prt] = y_position_prt;
                    } else if( ( bocos_type[_NORTH_] == _PERIODIC_ ) and ( aux_y_prt == 1 ) and ( y_max_per_task[my_rank] > ( y_0 + L_y - epsilon_geometry*L_y ) ) ) {
	                aux_y_prt = 1 - np_y;
	    	    	y_position_prt -= L_y;
                        local_prts_positions_y[id_prt] = y_position_prt;
                    }
                    if( ( bocos_type[_BACK_] == _PERIODIC_ ) and ( aux_z_prt == -1 ) and ( z_min_per_task[my_rank] < ( z_0 + epsilon_geometry*L_z ) ) ) {
	                aux_z_prt = np_z - 1;
	    	    	z_position_prt += L_z;
                        local_prts_positions_z[id_prt] = z_position_prt;
                    } else if( ( bocos_type[_FRONT_] == _PERIODIC_ ) and ( aux_z_prt == 1 ) and ( z_max_per_task[my_rank] > ( z_0 + L_z - epsilon_geometry*L_z ) ) ) {
	                aux_z_prt = 1 - np_z;
	    	    	z_position_prt -= L_z;
                        local_prts_positions_z[id_prt] = z_position_prt;
                    }

                    /// Identify rank to send
                    rank_to_send = my_rank + aux_x_prt + aux_y_prt*np_x + aux_z_prt*np_x*np_y;

                    /// Fill in buffer vectors
	    	    if( rank_to_send != my_rank ) {

                        append_to_buffer_prt( id_prt, send_buffer_by_task[rank_to_send] );
                        send_buffer_by_task[rank_to_send] += get_byte_size_prt( id_prt );

	    	    }

                }

            }

            /// We changed the send-recv pointers above so reset them
            for( int i = 0; i < world_size; i++ ) {

                send_buffer_by_task[i] = &send_buffer.data()[offset_in_send_buffer[i]];
                recv_buffer_by_task[i] = &recv_buffer.data()[offset_in_recv_buffer[i]];

            }

            /// Communicate particles data
            for( int i = 1; i < world_size; i++ ) {

                int send_request_to  = ( my_rank + i )%world_size;
                int get_request_from = ( my_rank - i + world_size )%world_size;

                /// Send & receive buffers to communicate
                MPI_Status status;
                MPI_Sendrecv( send_buffer_by_task[send_request_to], num_bytes_to_send[send_request_to], MPI_CHAR, send_request_to, 0, recv_buffer_by_task[get_request_from], num_bytes_to_recv[get_request_from], MPI_CHAR, get_request_from, 0, MPI_COMM_WORLD, &status );

            }

            // Copy over the particle data (this also updates the total number of particles)
            copy_over_received_data( recv_buffer, ntot_to_recv );

        };

        /// Get a vector of xmin of the domain for each task
        std::vector<double> get_x_min_per_task() { return x_min_per_task; };
        
	/// Get a vector of xmax of the domain for each task
        std::vector<double> get_x_max_per_task() { return x_max_per_task; };

        /// Get a vector of ymin of the domain for each task
        std::vector<double> get_y_min_per_task() { return y_min_per_task; };
        
	/// Get a vector of ymax of the domain for each task
        std::vector<double> get_y_max_per_task() { return y_max_per_task; };

        /// Get a vector of zmin of the domain for each task
        std::vector<double> get_z_min_per_task() { return z_min_per_task; };
        
	/// Get a vector of zmax of the domain for each task
        std::vector<double> get_z_max_per_task() { return z_max_per_task; };

        /// Free all memory of the stored particles
        void free() {
	    freeParticleArrays();  // Properly deletes all arrays
        };

        /// Write data to file (HDF5 format)
        void write_to_file(const double &current_time, const int &current_time_iter) {
	    
	    /// Update CPU particles
	    this->updateHostParticles();
	    
            /// Skip if there are no particles
            if( this->get_num_prts_total() < 1 ) return;

            /// Generate file name
            char output_file_prts_name[100];
            sprintf(output_file_prts_name, "%s_%d.h5", output_data_file_name_prts.c_str(), current_time_iter);

            /// Gather global size
            int local_size = this->get_num_prts_local_in_use( my_rank );
            int global_size = this->get_num_prts_total();

            /// Compute offsets
            int offset = 0;
            MPI_Exscan(&local_size, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            /// Create property list for parallel I/O
            hid_t plist_id_1 = H5Pcreate( H5P_FILE_ACCESS );
            H5Pset_fapl_mpio( plist_id_1, MPI_COMM_WORLD, MPI_INFO_NULL );
	    hid_t plist_id_2 = H5Pcreate( H5P_DATASET_XFER );
            H5Pset_dxpl_mpio( plist_id_2, H5FD_MPIO_INDEPENDENT );

            /// Create the HDF5 file with parallel access
            hid_t file_id = H5Fcreate( output_file_prts_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_1 );
            if( file_id < 0 ) {
		std::cout << " Error creating the file " << output_file_prts_name << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 0);
            }

            /// Initialize stuff and create dataspace
            hid_t dataset_id;
            hsize_t dims_global[1] = { static_cast<hsize_t>( global_size ) };
            hsize_t dims_local[1]  = { static_cast<hsize_t>( local_size ) };
            hid_t dataspace_id = H5Screate_simple( 1, dims_global, NULL );
            hid_t memspace_id  = H5Screate_simple( 1, dims_local, NULL );		

            /// Select hyperslab or none
            hsize_t offset_dims[1] = { static_cast<hsize_t>(offset) };
            H5Sselect_hyperslab( dataspace_id, H5S_SELECT_SET, offset_dims, NULL, dims_local, NULL );

            /// Write data

            /// Flatten local particle data for HDF5 attributes
            std::vector<int> prt_int_input( local_size, -1 );
            std::vector<double> prt_double_input( local_size, 0.0 );

	    /// Write identifiers
            for( int p = 0; p < local_size; p++ ) {
                prt_int_input[p] = local_prts_identifiers[p];           
            }
            dataset_id = H5Dcreate2( file_id, "identifier", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_INT, memspace_id, dataspace_id, plist_id_2, prt_int_input.data() );
            H5Dclose( dataset_id );

            /// Write diameters
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_diameters[p];
            }
            dataset_id = H5Dcreate2( file_id, "diameter", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write densities
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_densities[p];
            }
            dataset_id = H5Dcreate2( file_id, "density", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write x_positions
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_positions_x[p];
            }
            dataset_id = H5Dcreate2( file_id, "x_position", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write y_positions
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_positions_y[p];
            }
            dataset_id = H5Dcreate2( file_id, "y_position", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write z_positions
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_positions_z[p];
            }
            dataset_id = H5Dcreate2( file_id, "z_position", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write u_velocities
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_velocities_x[p];
            }
            dataset_id = H5Dcreate2( file_id, "u_velocity", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write v_velocities
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_velocities_y[p];
            }
            dataset_id = H5Dcreate2( file_id, "v_velocity", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write w_velocities
            for( int p = 0; p < local_size; p++ ) {
                prt_double_input[p] = local_prts_velocities_z[p];
            }
            dataset_id = H5Dcreate2( file_id, "w_velocity", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id_2, prt_double_input.data() );
            H5Dclose( dataset_id );

            /// Write current_time & current_time_iter
	    hid_t attr_id;
            hsize_t dims_scalar[1] = { 1 };
            hid_t attr_space_scalar = H5Screate_simple( 1, dims_scalar, NULL );

            attr_id = H5Acreate2( file_id, "Time", H5T_NATIVE_DOUBLE, attr_space_scalar, H5P_DEFAULT, H5P_DEFAULT );
            H5Awrite( attr_id, H5T_NATIVE_DOUBLE, &current_time );
            H5Aclose( attr_id );

            attr_id = H5Acreate2( file_id, "Iteration", H5T_NATIVE_INT, attr_space_scalar, H5P_DEFAULT, H5P_DEFAULT );
            H5Awrite( attr_id, H5T_NATIVE_INT, &current_time_iter );
            H5Aclose( attr_id );

            H5Sclose( attr_space_scalar );

            /// Synchronize before closing
            MPI_Barrier( MPI_COMM_WORLD );
            H5Sclose( dataspace_id );
            H5Sclose( memspace_id );
            H5Fclose( file_id );
            H5Pclose( plist_id_1 );
            H5Pclose( plist_id_2 );

            /// Write xdmf file
	    if( generate_xdmf_file_prts ) {
	   
                if( my_rank == 0 ) {
            
		    char output_file_prts_xdmf_name[100];
                    sprintf( output_file_prts_xdmf_name, "%s_%d.xdmf", output_data_file_name_prts.c_str(), current_time_iter );
		    char output_file_prts_xdmf_name_aux[100];
                    sprintf( output_file_prts_xdmf_name_aux, "%s_%d", output_data_file_name_prts.c_str(), current_time_iter );
		   
		    std::ofstream output_file_prts_xdmf( output_file_prts_xdmf_name );

                    /// Header stuff
		    output_file_prts_xdmf << "<?xml version='1.0' ?>" << std::endl;
		    output_file_prts_xdmf << "<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>" << std::endl;
		    output_file_prts_xdmf << "<Xdmf Version='2.0'>" << std::endl;
		    output_file_prts_xdmf << "  <Domain>" << std::endl;	
		    output_file_prts_xdmf << "    <Grid Name='" << output_file_prts_xdmf_name_aux << "' GridType='Uniform'>" << std::endl;

                    /// Topology: points ( each particle is treated as a point )
                    output_file_prts_xdmf << "      <Topology TopologyType='Polyvertex' NumberOfElements='" << num_prts_total << "'/>" << std::endl;

                    /// Geometry: particle positions ( x_position, y_position, z_position )
                    output_file_prts_xdmf << "      <Geometry GeometryType='X_Y_Z'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/x_position" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>)" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/y_position" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>)" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/z_position" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>)" << std::endl;
                    output_file_prts_xdmf << "      </Geometry>)" << std::endl;

                    /// Attribute: identifier
                    output_file_prts_xdmf << "      <Attribute Name='identifier' AttributeType='Scalar' Center='Node'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Int' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/identifier" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>" << std::endl;
                    output_file_prts_xdmf << "      </Attribute>" << std::endl;

                    /// Attribute: diameter
                    output_file_prts_xdmf << "      <Attribute Name='diameter' AttributeType='Scalar' Center='Node'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/diameter" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>" << std::endl;
                    output_file_prts_xdmf << "      </Attribute>" << std::endl;

                    /// Attribute: density
                    output_file_prts_xdmf << "      <Attribute Name='density' AttributeType='Scalar' Center='Node'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/density" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>" << std::endl;
                    output_file_prts_xdmf << "      </Attribute>" << std::endl;

                    /// Attribute: u_velocity
                    output_file_prts_xdmf << "      <Attribute Name='u_velocity' AttributeType='Scalar' Center='Node'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/u_velocity" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>" << std::endl;
                    output_file_prts_xdmf << "      </Attribute>" << std::endl;

                    /// Attribute: v_velocity
                    output_file_prts_xdmf << "      <Attribute Name='v_velocity' AttributeType='Scalar' Center='Node'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/v_velocity" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>" << std::endl;
                    output_file_prts_xdmf << "      </Attribute>" << std::endl;

                    /// Attribute: w_velocity
                    output_file_prts_xdmf << "      <Attribute Name='w_velocity' AttributeType='Scalar' Center='Node'>" << std::endl;
                    output_file_prts_xdmf << "        <DataItem Dimensions='" << num_prts_total << "' DataType='Float' Format='HDF'>" << std::endl;
                    output_file_prts_xdmf << "          " << output_file_prts_name << ":/w_velocity" << std::endl;
                    output_file_prts_xdmf << "        </DataItem>" << std::endl;
                    output_file_prts_xdmf << "      </Attribute>" << std::endl;

                    /// Close the tags
		    output_file_prts_xdmf << "    </Grid>" << std::endl;
		    output_file_prts_xdmf << "  </Domain>" << std::endl;
		    output_file_prts_xdmf << "</Xdmf>" << std::endl;

		    output_file_prts_xdmf.close();

		}

	    }

        };
        
	/// Read data from file (HDF5 format)
        void read_from_file(const std::string &file_name) {

            /// Skip if there are no particles
            if( this->get_num_prts_total() < 1 ) return;

            /// Generate file name
            char file_name_aux[100];
            sprintf( file_name_aux, "%s", file_name.c_str() );

            /// Open the HDF5 file
            hid_t plist_id = H5Pcreate( H5P_FILE_ACCESS );
            H5Pset_fapl_mpio( plist_id, MPI_COMM_WORLD, MPI_INFO_NULL );
            hid_t file_id = H5Fopen( file_name_aux, H5F_ACC_RDONLY, plist_id );
            H5Pclose( plist_id );
	    hid_t dataset_id;

            if( file_id < 0 ) {
                std::cout << "Error opening the file: " << file_name_aux << std::endl;
                MPI_Abort( MPI_COMM_WORLD, 0 );
            }

            /// Read identifier data
            int prt_identifiers[num_prts_total];
            dataset_id = H5Dopen( file_id, "identifier", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_identifiers );
            H5Dclose( dataset_id );

            /// Read diameter data
            double prt_diameters[num_prts_total];
            dataset_id = H5Dopen( file_id, "diameter", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_diameters );
            H5Dclose( dataset_id );

            /// Read density data
            double prt_densities[num_prts_total];
            dataset_id = H5Dopen( file_id, "density", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_densities );
            H5Dclose( dataset_id );

            /// Read x_position
            double prt_x_positions[num_prts_total];
            dataset_id = H5Dopen( file_id, "x_position", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_x_positions );
            H5Dclose( dataset_id );

            /// Read y_position
            double prt_y_positions[num_prts_total];
            dataset_id = H5Dopen( file_id, "y_position", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_y_positions );
            H5Dclose( dataset_id );

            /// Read z_position
            double prt_z_positions[num_prts_total];
            dataset_id = H5Dopen( file_id, "z_position", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_z_positions );
            H5Dclose( dataset_id );

            /// Read u_velocity
            double prt_u_velocities[num_prts_total];
            dataset_id = H5Dopen( file_id, "u_velocity", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_u_velocities );
            H5Dclose( dataset_id );

            /// Read v_velocity
            double prt_v_velocities[num_prts_total];
            dataset_id = H5Dopen( file_id, "v_velocity", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_v_velocities );
            H5Dclose( dataset_id );

            /// Read w_velocity
            double prt_w_velocities[num_prts_total];
            dataset_id = H5Dopen( file_id, "w_velocity", H5P_DEFAULT );
            H5Dread( dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prt_w_velocities );
            H5Dclose( dataset_id );

            /// Close the file
            H5Fclose( file_id );

            // Estimate number of local particles and allocate arrays
            int estimate_particles_local = std::max(static_cast<int>(std::round(buffer_ratio_prts * num_prts_total / world_size)), static_cast<int>(buffer_ratio_prts)); 
	    
	    allocateParticleArrays(estimate_particles_local); 
            
	    /// Calculate local number of particles & allocate and provide diameter, density, position & velocity of local particles
            double x_position_prt, y_position_prt, z_position_prt;
            for( int p = 0; p < num_prts_total; p++ ) {

                /// Position of particle p
                x_position_prt = prt_x_positions[p]; 
                y_position_prt = prt_y_positions[p]; 
                z_position_prt = prt_z_positions[p];

	        /// Update num_prts_local_in_use
                for( int rank = 0; rank < world_size; rank++ ) {
                    if( ( x_position_prt >= x_min_per_task[rank] ) and ( x_position_prt < x_max_per_task[rank] ) and
                        ( y_position_prt >= y_min_per_task[rank] ) and ( y_position_prt < y_max_per_task[rank] ) and
                        ( z_position_prt >= z_min_per_task[rank] ) and ( z_position_prt < z_max_per_task[rank] ) ) {
      	                num_prts_local_in_use[rank] += 1;
                        break;
	            }
	        }

	        /// Allocate and set attributes of local particles
    	        if( ( x_position_prt >= x_min_per_task[my_rank] ) and ( x_position_prt < x_max_per_task[my_rank] ) and
                    ( y_position_prt >= y_min_per_task[my_rank] ) and ( y_position_prt < y_max_per_task[my_rank] ) and
                    ( z_position_prt >= z_min_per_task[my_rank] ) and ( z_position_prt < z_max_per_task[my_rank] ) ) {

                    /// Add particles to allocate buffer 
		     addParticle( prt_identifiers[p], prt_diameters[p], prt_densities[p], x_position_prt, y_position_prt, z_position_prt, prt_u_velocities[p], prt_v_velocities[p], prt_w_velocities[p]);
                }
		
		/// Particle count
		num_prts_local_in_use[my_rank] = prt_count;

            }

        };


	void freeParticleArrays() {
            
	    #pragma acc exit data delete(local_prts_identifiers[0:prt_capacity], local_prts_diameters[0:prt_capacity], local_prts_densities[0:prt_capacity], local_prts_positions_x[0:prt_capacity], local_prts_positions_y[0:prt_capacity], local_prts_positions_z[0:prt_capacity], local_prts_positions_0_x[0:prt_capacity], local_prts_positions_0_y[0:prt_capacity], local_prts_positions_0_z[0:prt_capacity], local_prts_velocities_x[0:prt_capacity], local_prts_velocities_y[0:prt_capacity], local_prts_velocities_z[0:prt_capacity], local_prts_velocities_0_x[0:prt_capacity], local_prts_velocities_0_y[0:prt_capacity], local_prts_velocities_0_z[0:prt_capacity])
	    
	    delete[] local_prts_identifiers;
	    delete[] local_prts_diameters;
	    delete[] local_prts_densities;
	    delete[] local_prts_positions_x;
	    delete[] local_prts_positions_y;
	    delete[] local_prts_positions_z;
	    delete[] local_prts_positions_0_x;
	    delete[] local_prts_positions_0_y;
	    delete[] local_prts_positions_0_z;
	    delete[] local_prts_velocities_x;
	    delete[] local_prts_velocities_y;
	    delete[] local_prts_velocities_z;
	    delete[] local_prts_velocities_0_x;
	    delete[] local_prts_velocities_0_y;
	    delete[] local_prts_velocities_0_z;
	    delete[] local_prts_indexes_0_i;
	    delete[] local_prts_indexes_0_j;
	    delete[] local_prts_indexes_0_k;
	    
	};
	
	void allocateParticleArrays(int capacity) {
	    
	    prt_capacity = capacity;
	    local_prts_identifiers    = new int[capacity];
	    local_prts_diameters      = new double[capacity];
	    local_prts_densities      = new double[capacity];
	    local_prts_positions_x    = new double[capacity];
	    local_prts_positions_y    = new double[capacity];
	    local_prts_positions_z    = new double[capacity];
	    local_prts_positions_0_x  = new double[capacity];
	    local_prts_positions_0_y  = new double[capacity];
	    local_prts_positions_0_z  = new double[capacity];
	    local_prts_velocities_x   = new double[capacity];
	    local_prts_velocities_y   = new double[capacity];
	    local_prts_velocities_z   = new double[capacity];
	    local_prts_velocities_0_x = new double[capacity];
	    local_prts_velocities_0_y = new double[capacity];
	    local_prts_velocities_0_z = new double[capacity];
	    local_prts_indexes_0_i    = new int[capacity];
	    local_prts_indexes_0_j    = new int[capacity];
	    local_prts_indexes_0_k    = new int[capacity];
	    prt_count = 0;

	};

	void addParticle(int id, double diam, double rho, double x, double y, double z, double u, double v, double w) {
	    
	    if (prt_count >= prt_capacity) {
	        std::cerr << "ERROR: Particle capacity exceeded!" << std::endl;
	        std::exit(EXIT_FAILURE);
	    }
	
	    int p = prt_count;
	    local_prts_identifiers[p] = id;
	    local_prts_diameters[p]   = diam;
	    local_prts_densities[p]   = rho;
	    local_prts_positions_x[p] = x;
	    local_prts_positions_y[p] = y;
	    local_prts_positions_z[p] = z;
	    local_prts_positions_0_x[p] = x;
	    local_prts_positions_0_y[p] = y;
	    local_prts_positions_0_z[p] = z;
	    local_prts_velocities_x[p] = u;
	    local_prts_velocities_y[p] = v;
	    local_prts_velocities_z[p] = w;
	    local_prts_velocities_0_x[p] = u;
	    local_prts_velocities_0_y[p] = v;
	    local_prts_velocities_0_z[p] = w;
	    local_prts_indexes_0_i[p] = -1;	/// It will be set properly later
	    local_prts_indexes_0_j[p] = -1;	/// It will be set properly later
	    local_prts_indexes_0_k[p] = -1;	/// It will be set properly later
	    prt_count++;

	};

	// GPU data movement methods
	void copyToDeviceParticles() {
            
	    #pragma acc enter data copyin( local_prts_identifiers[0:prt_capacity], local_prts_diameters[0:prt_capacity], local_prts_densities[0:prt_capacity], local_prts_positions_x[0:prt_capacity], local_prts_positions_y[0:prt_capacity], local_prts_positions_z[0:prt_capacity], local_prts_positions_0_x[0:prt_capacity], local_prts_positions_0_y[0:prt_capacity], local_prts_positions_0_z[0:prt_capacity], local_prts_velocities_x[0:prt_capacity], local_prts_velocities_y[0:prt_capacity], local_prts_velocities_z[0:prt_capacity], local_prts_velocities_0_x[0:prt_capacity], local_prts_velocities_0_y[0:prt_capacity], local_prts_velocities_0_z[0:prt_capacity], local_prts_indexes_0_i[0:prt_capacity], local_prts_indexes_0_j[0:prt_capacity], local_prts_indexes_0_k[0:prt_capacity] )
	    
	};

	void updateDeviceParticles() {
            
	    #pragma acc update device( local_prts_identifiers[0:prt_capacity], local_prts_diameters[0:prt_capacity], local_prts_densities[0:prt_capacity], local_prts_positions_x[0:prt_capacity], local_prts_positions_y[0:prt_capacity], local_prts_positions_z[0:prt_capacity], local_prts_positions_0_x[0:prt_capacity], local_prts_positions_0_y[0:prt_capacity], local_prts_positions_0_z[0:prt_capacity], local_prts_velocities_x[0:prt_capacity], local_prts_velocities_y[0:prt_capacity], local_prts_velocities_z[0:prt_capacity], local_prts_velocities_0_x[0:prt_capacity], local_prts_velocities_0_y[0:prt_capacity], local_prts_velocities_0_z[0:prt_capacity], local_prts_indexes_0_i[0:prt_capacity], local_prts_indexes_0_j[0:prt_capacity], local_prts_indexes_0_k[0:prt_capacity] )
	    
	};

	void updateHostParticles() {
            
	    #pragma acc update host( local_prts_identifiers[0:prt_capacity], local_prts_diameters[0:prt_capacity], local_prts_densities[0:prt_capacity], local_prts_positions_x[0:prt_capacity], local_prts_positions_y[0:prt_capacity], local_prts_positions_z[0:prt_capacity], local_prts_positions_0_x[0:prt_capacity], local_prts_positions_0_y[0:prt_capacity], local_prts_positions_0_z[0:prt_capacity], local_prts_velocities_x[0:prt_capacity], local_prts_velocities_y[0:prt_capacity], local_prts_velocities_z[0:prt_capacity], local_prts_velocities_0_x[0:prt_capacity], local_prts_velocities_0_y[0:prt_capacity], local_prts_velocities_0_z[0:prt_capacity], local_prts_indexes_0_i[0:prt_capacity], local_prts_indexes_0_j[0:prt_capacity], local_prts_indexes_0_k[0:prt_capacity] )
	    
	};

        // Particle containers (raw arrays)
        int*    local_prts_identifiers;
        double* local_prts_diameters;
        double* local_prts_densities;
        double* local_prts_positions_x;
        double* local_prts_positions_y;
        double* local_prts_positions_z;
        double* local_prts_positions_0_x;
        double* local_prts_positions_0_y;
        double* local_prts_positions_0_z;
        double* local_prts_velocities_x;
        double* local_prts_velocities_y;
        double* local_prts_velocities_z;
        double* local_prts_velocities_0_x;
        double* local_prts_velocities_0_y;
        double* local_prts_velocities_0_z;
        int* local_prts_indexes_0_i;
        int* local_prts_indexes_0_j;
        int* local_prts_indexes_0_k;

	// Size tracking
	int prt_count      = 0;
        int prt_capacity   = 0;

    protected:


        /// Info about particle distribution
	double diameter_prt;				/// Diameter of particles
	double density_prt;				/// Density of particles
	double number_density_prts;			/// Number density of particles
        int num_prts_total; 		       		/// Total number of particles across all tasks
        std::vector<int> num_prts_local_in_use; 	/// Number of local particles in use
        double buffer_ratio_prts;			/// Buffer ratio of allocated particles per task; e.g., 2, 3, 5

        /// Input/output info
	std::string configuration_file;			/// Configuration file name (YAML language)
	std::string output_data_file_name_prts;		/// Output data file name (HDF5 format)	
        bool generate_xdmf_file_prts;			/// Generate xdmf file reader

        /// Info about MPI
        int my_rank, world_size, np_x, np_y, np_z;

        /// Info about Eulerian domain
        double x_0;					/// Domain start point in x-direction
        double y_0;					/// Domain start point in y-direction
        double z_0;					/// Domain start point in z-direction
        double L_x;					/// Domain size in x-direction
        double L_y;					/// Domain size in x-direction
        double L_z;					/// Domain size in x-direction

        /// The range of x-, y- and z-values that belongs to each task
        std::vector<double> x_min_per_task;
        std::vector<double> x_max_per_task;
        std::vector<double> y_min_per_task;
        std::vector<double> y_max_per_task;
        std::vector<double> z_min_per_task;
        std::vector<double> z_max_per_task;

        /// Boundary conditions
        int bocos_type[6];					/// Array of boundary conditions type
        double bocos_u[6];					/// Array of boundary conditions u
        double bocos_v[6];					/// Array of boundary conditions v
        double bocos_w[6];					/// Array of boundary conditions w
        double bocos_P[6];					/// Array of boundary conditions P
        double bocos_T[6];					/// Array of boundary conditions T

        /// For communication
        void copy_over_received_data(std::vector<char> & recv_buffer, int &num_prts_recv) {

            char * buffer = recv_buffer.data();
            size_t bytes_processed = 0;
            for( int i = 0; i < num_prts_recv; i++ ) {
                assign_from_buffer_prt( num_prts_local_in_use[my_rank] + i, buffer );
                auto size = get_byte_size_prt( num_prts_local_in_use[my_rank] + i );
                buffer += size;
                bytes_processed += size;
                assert( bytes_processed <= recv_buffer.size() );
            }

            // Update the total number of particles in use
            num_prts_local_in_use[my_rank] += num_prts_recv;

        };

        /// Gather a single value from all tasks. Value from task my_rank is stored in values[my_rank]
	std::vector<double> gather_from_tasks(double * value) {
            std::vector<double> values( world_size );
            MPI_Allgather( value, sizeof(double), MPI_BYTE, values.data(), sizeof(double), MPI_BYTE, MPI_COMM_WORLD );
            return values;
	};

};

#endif /*_LAGRANGIAN_POINT_PARTICLES_*/
