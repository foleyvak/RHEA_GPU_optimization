#ifndef _MY_RHEA_
#define _MY_RHEA_

////////// INCLUDES //////////
#include "src/FlowSolverRHEA.hpp"

////////// myRHEA CLASS //////////
class myRHEA : public FlowSolverRHEA {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        myRHEA(const std::string configuration_file) : FlowSolverRHEA(configuration_file) {	/// Parametrized constructor

            /// Set geometrical and mesh parameters
            L = L_x;				
            Delta = L/num_grid_x; 
            cell_volume = Delta*Delta*Delta;
	    domain_volume = L*L*L;

            /// NOTE: For OpenACC compatibility, the following field initializations must be moved to the base class constructor (FlowSolverRHEA.cpp)

            // ---- MOVE THESE LINES TO FlowSolverRHEA.cpp ----
	    /// Set parallel topology of distributed arrays
            w_x_field.setTopology(topo,"w_x");
            w_y_field.setTopology(topo,"w_y");
            w_z_field.setTopology(topo,"w_z");
            w_S_x_field.setTopology(topo,"w_S_x");
            w_S_y_field.setTopology(topo,"w_S_y");
            w_S_z_field.setTopology(topo,"w_S_z");
	    phi_field.setTopology(topo,"phi");
	    phi_source_field.setTopology(topo,"phi_source"); 
            // ---- END COPY ----        

	};

        virtual ~myRHEA() {};									/// Destructor

	////////// SOLVER METHODS //////////
        
        /// Set initial conditions: u, v, w, P and T ... needs to be modified/overwritten according to the problem under consideration
        void setInitialConditions();

        /// Calculate rhou, rhov, rhow and rhoE source terms ... needs to be modified/overwritten according to the problem under consideration
        void calculateSourceTerms();

        /// Temporal hook function ... needs to be modified/overwritten according to the problem under consideration
        void temporalHookFunction();

        /// Tag immersed boundary method ... needs to be modified/overwritten according to the problem under consideration
        void tagImmersedBoundaryMethod();

	/// Solve Poisson equation to obtain the solenoidal part of the modified velocity
        void solveScalarPoisson();

	/// Passot-Pouquet turbulent flow spectrum
        double passotPouquetTurbulentFlowSpectrum(const double &k);

        /// Set initial conditions of particles: x, y, z, u, v, and w ... needs to be modified/overwritten according to the problem under consideration
        void setInitialParticlesPositionsVelocities();

        /// Advance point particles in time: velocity ... needs to be modified/overwritten according to the problem under consideration
        void timeAdvanceVelocityPointParticles();

    protected:
       
        /// Geometrical and mesh parameters
        double L;					/// Size of computational domain (assuming Lx = Ly = Lz)
        double Delta;					/// Grid spacing (assuming uniform mesh) 
        double cell_volume;				/// Volume of grid cell (assuming uniform mesh)
	double domain_volume;				/// Volume of computational domain
 
        /// NOTE: For OpenACC compatibility, the following field declarations must be moved to the base class (FlowSolverRHEA.hpp) 

        // ---- MOVE THESE LINES TO FlowSolverRHEA.hpp ----
	DistributedArray w_x_field;			/// 3-D field of w_x
        DistributedArray w_y_field;			/// 3-D field of w_x
        DistributedArray w_z_field;			/// 3-D field of w_x
        DistributedArray w_S_x_field;			/// 3-D field of w_S_x
        DistributedArray w_S_y_field;			/// 3-D field of w_S_x
        DistributedArray w_S_z_field;			/// 3-D field of w_S_x
	DistributedArray phi_field;			/// 3-D field of phi        
	DistributedArray phi_source_field;		/// 3-D field of phi_source
        // ---- END MOVE ----

    private:

};

#endif /*_MY_RHEA_*/
