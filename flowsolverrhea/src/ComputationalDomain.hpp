#ifndef _ComputationalDomain_
#define _ComputationalDomain_

////////// INCLUDES //////////
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include "MacroParameters.hpp"
#include "DistributedArray.hpp"

/// Forward declaration
class ParallelTopology;

class ComputationalDomain{
    public:
        ComputationalDomain() {};		// ... modified for OpenACC
        ComputationalDomain(double, double, double,double, double,double, double, double, double, int, int, int, bool, std::string);
        
        void printDomain();
        void setBocos(int b[6]);
        int getGNx(){return gNx;};
        int getGNy(){return gNy;};
        int getGNz(){return gNz;};
        double getGlobx(int id){return globx[id];};
        double getGloby(int id){return globy[id];};
        double getGlobz(int id){return globz[id];};

        void set_x(int id,double val){ x[id]=val;};
        void set_y(int id,double val){ y[id]=val;};
        void set_z(int id,double val){ z[id]=val;};

        int getBoco(int bocoid){return bc[bocoid];};
        void calculateLocalGrid(int lNx,int lNy, int lNz);
	
	/// Adjust location to closest grid point
        void adjustLocationToClosestGridPoint(double & x_position, double & y_position, double & z_position, DistributedArray &x_field, DistributedArray &y_field, DistributedArray &z_field, ParallelTopology *topo);
 
        //Local positions
        double *x;
        double *y;
        double *z;


    protected:
        //Global Number of cells in each direction
        int gNx;
        int gNy;
        int gNz;

        //Local number of cells in each direction
        int _lNx_;
        int _lNy_;
        int _lNz_;

        //Boundary conditions
        int bc[6];

        //Positions globally
        double *globx;
        double *globy;
        double *globz;

        //Dimensions of the ComputationalDomain
        double L_x;
        double L_y;
        double L_z;

        //Origin 
        double x_0;
        double y_0;
        double z_0;

        // Stretching factors
        double A_x;
        double A_y;
        double A_z;

        void calculateGlobalGrid();
        void readGlobalGrid(std::string external_mesh_file);

};


#endif
