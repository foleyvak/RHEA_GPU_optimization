#ifndef _MY_RHEA_
#define _MY_RHEA_

////////// INCLUDES //////////
#include "src/FlowSolverRHEA.hpp"

////////// myRHEA CLASS //////////
class myRHEA : public FlowSolverRHEA {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        myRHEA(const std::string configuration_file) : FlowSolverRHEA(configuration_file) {};	/// Parametrized constructor
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

        /// Set initial conditions of particles: x, y, z, u, v, and w ... needs to be modified/overwritten according to the problem under consideration
        void setInitialParticlesPositionsVelocities();

        /// Advance point particles in time: velocity ... needs to be modified/overwritten according to the problem under consideration
        void timeAdvanceVelocityPointParticles();

    protected:

    private:

};

#endif /*_MY_RHEA_*/
