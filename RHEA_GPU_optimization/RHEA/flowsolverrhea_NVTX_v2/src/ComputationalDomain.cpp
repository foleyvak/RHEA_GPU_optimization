#include "ComputationalDomain.hpp"
#include "ParallelTopology.hpp"

using namespace std;

ComputationalDomain::ComputationalDomain(double sizex, double sizey, double sizez, double orig_x, double orig_y, double orig_z, double stretch_x, double stretch_y, double stretch_z, int ncellsx, int ncellsy, int ncellsz, bool external_mesh, string external_mesh_file) {

    gNx=ncellsx;
    gNy=ncellsy;
    gNz=ncellsz;

    L_x=sizex;
    L_y=sizey;
    L_z=sizez;

    x_0 = orig_x;
    y_0 = orig_y;
    z_0 = orig_z;

    A_x = stretch_x;
    A_y = stretch_y;
    A_z = stretch_z;

    // Position of the Global Cells
    globx = new double[gNx + 2];
    globy = new double[gNy + 2];
    globz = new double[gNz + 2];

    if( external_mesh ) {
        readGlobalGrid( external_mesh_file );
    } else {
        calculateGlobalGrid();
    }

}

void ComputationalDomain::calculateGlobalGrid() {

    double eta_x, eta_y, eta_z;

    // Mesh in x
    for( int i = 0; i < (gNx + 2); i++ ) {
        eta_x = (i - 0.5)/gNx;
        globx[i] = x_0 + L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x;

        if(i == 0) {
            eta_x = ( 1.0 - 0.5 )/gNx;
            globx[i] = x_0 - ( L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x );
        }
        if( i == ( gNx + 1 ) ) {
            eta_x = ( gNx - 0.5 )/gNx;
            globx[i] = x_0 + 2.0*L_x - ( L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x );
        }
    }

    // Mesh in y
    for( int j = 0; j < (gNy + 2); j++ ) {
        eta_y = (j - 0.5)/gNy;
        globy[j] = y_0 + L_y*eta_y + A_y*( 0.5*L_y - L_y*eta_y )*( 1.0 - eta_y )*eta_y;

        if( j == 0 ) {
            eta_y = ( 1.0 - 0.5 )/gNy;
            globy[j] = y_0 - ( L_y*eta_y + A_y*( 0.5*L_y - L_y*eta_y )*( 1.0 - eta_y )*eta_y );
        }
        if( j == (gNy + 1) ) {
            eta_y = ( gNy - 0.5 )/gNy;
            globy[j] = y_0 + 2.0*L_y - ( L_y*eta_y + A_y*( 0.5*L_y - L_y*eta_y )*( 1.0 - eta_y )*eta_y );
        }
    }

    // Mesh in z
    for( int k = 0; k < (gNz + 2); k++ ) {
        eta_z = (k - 0.5)/gNz;
        globz[k] = z_0 + L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z;

        if( k == 0 ) {
            eta_z = ( 1.0 - 0.5 )/gNz;
            globz[k] = z_0 - ( L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z );
        }
        if( k == (gNz + 1) ) {
            eta_z = ( gNz - 0.5 )/gNz;
            globz[k] = z_0 + 2.0*L_z - ( L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z );
        }
    }

}

void ComputationalDomain::readGlobalGrid(string external_mesh_file) {

    /// Create and ifstream object to open the file in read mode 
    ifstream mesh_file( external_mesh_file ); 

    /// Check if file opened successfully 
    if( mesh_file.is_open() ) { 
  
        // File opened successfully
        //cout << "Mesh file opened successfully" << endl;

        /// Create string to read values from file
        string mesh_position;	
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string
        getline( mesh_file, mesh_position );	/// Flush initial string

        /// Mesh in x
        getline( mesh_file, mesh_position );	/// Flush "x-direction points" string
        for( int i = 1; i <= gNx; i++ ) {
            getline( mesh_file, mesh_position );
	    globx[i] = stod( mesh_position ); 
        }
        globx[0]       = x_0 - ( globx[1] - x_0 );
        globx[gNx + 1] = ( x_0 + L_x ) + ( ( x_0 + L_x ) - globx[gNx] );

        /// Mesh in y
        getline( mesh_file, mesh_position );	/// Flush "y-direction points" string
        for( int j = 1; j <= gNy; j++ ) {
            getline( mesh_file, mesh_position );
	    globy[j] = stod( mesh_position ); 
        }
        globy[0]       = y_0 - ( globy[1] - y_0 );
        globy[gNy + 1] = ( y_0 + L_y ) + ( ( y_0 + L_y ) - globy[gNy] );	

        /// Mesh in z
        getline( mesh_file, mesh_position );	/// Flush "z-direction points" string
        for( int k = 1; k <= gNz; k++ ) {
            getline( mesh_file, mesh_position );
	    globz[k] = stod( mesh_position ); 
        }
        globz[0]       = z_0 - ( globz[1] - z_0 );
        globz[gNz + 1] = ( z_0 + L_z ) + ( ( z_0 + L_z ) - globz[gNz] );

        /// Close file 
        mesh_file.close(); 

    } else {

        /// Display error if file was not opened 
        cout << "Error opening mesh file!" << endl; 

    } 

};

void ComputationalDomain::adjustLocationToClosestGridPoint(double & x_position, double & y_position, double & z_position, DistributedArray &x_field, DistributedArray &y_field, DistributedArray &z_field, ParallelTopology *topo) {

    /// Copy original position
    double original_x_position = x_position;
    double original_y_position = y_position;
    double original_z_position = z_position;

    /// Initialize to largest double value
    double min_distance = numeric_limits<double>::max();

    /// All points: locate closest grid point
    double x, y, z, distance;
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                x = x_field[I1D(i,j,k)];
                y = y_field[I1D(i,j,k)];
                z = z_field[I1D(i,j,k)];
                /// Distance
		distance = pow( pow( x - original_x_position, 2.0 ) + pow( y - original_y_position, 2.0 ) + pow( z - original_z_position, 2.0 ), 1.0/2.0 );
                /// If minimum distance, update information
		if( distance < min_distance ) {
		    min_distance = distance;
		    x_position = x;
		    y_position = y;
		    z_position = z;
		}
            }
        }
    }

//cout << original_x_position << " " << original_y_position << " " << original_z_position  << endl;
//cout << x_position << " " << y_position << " " << z_position << "  " << min_distance << endl;

}

void ComputationalDomain::calculateLocalGrid(int lNx, int lNy, int lNz) {

    x = new double[lNx];
    y = new double[lNy];
    z = new double[lNz];

    _lNx_=lNx;
    _lNy_=lNy;
    _lNz_=lNz;

}

void ComputationalDomain::printDomain()
{

    cout<<"Global x: "<<endl;
    for(int i=0;i<gNx+2;i++)
        cout<<i<<" : "<<globx[i]<<" ; ";
    cout<<endl;

    cout<<"Global y: "<<endl;
    for(int i=0;i<gNy+2;i++)
        cout<<i<<" : "<<globy[i]<<" ; ";
    cout<<endl;

    cout<<"Global z: "<<endl;
    for(int i=0;i<gNz+2;i++)
        cout<<i<<" : "<<globz[i]<<" ; ";
    cout<<endl;

    cout<<"Local x: "<<endl;
    for(int i=0;i<_lNx_;i++)
        cout<<i<<" : "<<x[i]<<" ; ";
    cout<<endl;

    cout<<"Local y: "<<endl;
    for(int i=0;i<_lNy_;i++)
        cout<<i<<" : "<<y[i]<<" ; ";
    cout<<endl;

    cout<<"Local z: "<<endl;
    for(int i=0;i<_lNz_;i++)
        cout<<i<<" : "<<z[i]<<" ; ";
    cout<<endl;


}   

void ComputationalDomain::setBocos(int b[6])
{
    bc[_WEST_]  = b[_WEST_];
    bc[_EAST_]  = b[_EAST_];
    bc[_SOUTH_] = b[_SOUTH_];
    bc[_NORTH_] = b[_NORTH_];
    bc[_BACK_]  = b[_BACK_];
    bc[_FRONT_] = b[_FRONT_];
}

