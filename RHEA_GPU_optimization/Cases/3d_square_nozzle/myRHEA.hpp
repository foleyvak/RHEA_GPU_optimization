#ifndef _MY_RHEA_
#define _MY_RHEA_

////////// INCLUDES //////////
#include "src/FlowSolverRHEA.hpp"

////////// myRHEA CLASS //////////
class myRHEA : public FlowSolverRHEA {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        // myRHEA(const std::string configuration_file) : FlowSolverRHEA(configuration_file) {};/// Parametrized constructor
        myRHEA(const std::string configuration_file);						/// Parametrized constructor
        virtual ~myRHEA() {};									/// Destructor

	////////// SOLVER METHODS //////////
        
        /// Set initial conditions: u, v, w, P and T ... needs to be modified/overwritten according to the problem under consideration
        void setInitialConditions();

        /// Calculate rhou, rhov, rhow and rhoE source terms ... needs to be modified/overwritten according to the problem under consideration
        void calculateSourceTerms();

        /// Temporal hook function ... needs to be modified/overwritten according to the problem under consideration
        void temporalHookFunction();

        /// Calculate inviscid fluxes in x-, y- and z-direction
        void calculateInviscidFluxes();

        /// Calculate viscous fluxes
        void calculateViscousFluxes();

        /// Calculate time step satisfying CFL constraint
        void calculateTimeStep();

        /// Update boundary values: rho, rhou, rhov, rhow and rhoE
        void updateBoundaries();

    protected:
    
        DistributedArray det_Jacobian_field;		/// 3-D field of the determinant of Jacobian 


    private:

};

#endif /*_MY_RHEA_*/
