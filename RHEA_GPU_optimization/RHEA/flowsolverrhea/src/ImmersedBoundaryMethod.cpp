#include "ImmersedBoundaryMethod.hpp"

using namespace std;

////////// FIXED PARAMETERS //////////
const double global_zero = 1.0e-12;		// Global zero


////////// BaseImmersedBoundaryMethod CLASS //////////

BaseImmersedBoundaryMethod::BaseImmersedBoundaryMethod() {};
        
BaseImmersedBoundaryMethod::BaseImmersedBoundaryMethod(const string configuration_file) : configuration_file(configuration_file) {
    #pragma acc enter data copyin(this)		// ... added for OpenACC
};

BaseImmersedBoundaryMethod::~BaseImmersedBoundaryMethod() {
    #pragma acc exit data delete(this)		// ... added for OpenACC
};


////////// D3 CLASS //////////

D3 crossProduct(const D3 &a, const D3 &b) {

    D3 r( a.pd[1] * b.pd[2] - a.pd[2] * b.pd[1],
          a.pd[2] * b.pd[0] - a.pd[0] * b.pd[2],
          a.pd[0] * b.pd[1] - a.pd[1] * b.pd[0] );

    return r;
};

D3::D3() : gzero( global_zero ) {

    pd[0] = 0.0;
    pd[1] = 0.0;
    pd[2] = 0.0;
};

D3::D3(const double &x, const double &y, const double &z) : gzero( global_zero ) {

    pd[0] = x;
    pd[1] = y;
    pd[2] = z;

};

D3::~D3() {};
