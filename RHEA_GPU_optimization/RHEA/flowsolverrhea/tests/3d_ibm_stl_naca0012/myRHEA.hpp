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

        /// Evaluates the intersection point "intersec" and its position factor lambda between a line with point "p" and direction vector "ray" and a plane defined by 3 points contained in a vector "tri"
        void planeLineIntersect(const D3 &p, const D3 &ray, const std::vector<D3> &tri, D3 &intersect, double &lambda);

        /// Evaluates whether a point "p" is within a triangle "tri" defined by three points contained in a vector
        bool isInTri(const D3 &p, const std::vector<D3> &tri);

        /// Evaluates whether a point "p" is within a STL defined as a vector of vectors of D3, each D3 corresponding to a STL triangle vertex
        bool isInSTL(const D3 &p, const std::vector<std::vector<D3> > &stl_tris);

    private:

};

#endif /*_MY_RHEA_*/
