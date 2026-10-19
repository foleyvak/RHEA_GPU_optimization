#ifndef _IMMERSED_BOUNDARY_METHOD_
#define _IMMERSED_BOUNDARY_METHOD_

////////// INCLUDES //////////
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <complex>
#include <mpi.h>
#include "yaml-cpp/yaml.h"

////////// CLASS DECLARATION //////////
class BaseImmersedBoundaryMethod;			/// Base immersed boundary method (IBM)
class D3;						/// Three-dimensional vector of doubles in Cartesian coordinates

////////// FUNCTION DECLARATION //////////


////////// BaseImmersedBoundaryMethod CLASS //////////
class BaseImmersedBoundaryMethod {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        BaseImmersedBoundaryMethod();						/// Default constructor
        BaseImmersedBoundaryMethod(const std::string configuration_file);	/// Parametrized constructor
        virtual ~BaseImmersedBoundaryMethod();					/// Destructor

	////////// GET FUNCTIONS //////////
	#pragma acc routine
        inline void getVelocityIBM(double u_IBM_, double v_IBM_, double w_IBM_) { u_IBM_ = u_IBM; v_IBM_ = v_IBM; w_IBM_ = w_IBM; };
	#pragma acc routine
        inline double getTemperatureIBM() { return( T_IBM ); };

	////////// SET FUNCTIONS //////////
	#pragma acc routine
	inline void setVelocityIBM(double u_IBM_, double v_IBM_, double w_IBM_) { 
		u_IBM = u_IBM_; v_IBM = v_IBM_; w_IBM = w_IBM_; 
	};
	
	#pragma acc routine
	inline void setTemperatureIBM(double T_IBM_) { 
		T_IBM = T_IBM_; 
	};

	////////// METHODS //////////
       
        /// Read configuration (input) file written in YAML language
        //virtual void readConfigurationFile() {};
        void readConfigurationFile() {};	// ... modified for OpenACC

    protected:

        ////////// PARAMETERS //////////

        /// Velocity and temperature of IBM
        double u_IBM;			/// u-velocity of IBM
        double v_IBM;			/// v-velocity of IBM
        double w_IBM;			/// w-velocity of IBM
        double T_IBM;			/// Temperature of IBM

        /// Model parameters
	std::string configuration_file;						/// Configuration file name (YAML language)

    private:

};

////////// D3 CLASS //////////
class D3 { 

        ////////// FRIEND FUNCTIONS & CLASSES //////////

        /// Cross product
        friend D3 crossProduct(const D3 &a, const D3 &b); 

        /// Product by scalar
        friend inline D3 operator*(double &l, const D3 &a);

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        D3();								/// Default constructor
        D3(const double &x,const double &y,const double &z);		/// Parametrized constructor
        virtual ~D3();							/// Destructor

	////////// GET FUNCTIONS //////////
        inline const double *getCoordinates() const { return pd; };
	
	////////// SET FUNCTIONS //////////
        inline void setCoordinates(const double *pd_) { pd[0] = pd_[0]; pd[1] = pd_[1]; pd[2] = pd_[2]; }; 
	
        ////////// METHODS //////////

	/// Norm-2
        inline double norm2() const { return( sqrt( pd[0]*pd[0] + pd[1]*pd[1] + pd[2]*pd[2] ) ); };
       
	/// Vector normalization
        inline void normalize() { ( *this ) *= 1.0/norm2(); };

        ////////// OPERATORS //////////
        inline D3 operator+(const D3 &a) const;
        inline D3 operator-(const D3 &a) const;
        inline D3 operator*(const double &l) const;
        inline double operator*(const D3 &a) const;
        inline void operator+=(const D3 &a);       
        inline void operator-=(const D3 &a);
        inline void operator*=(const double &l);
        inline D3 operator^(const D3 &b) const;
        inline D3& operator=(const D3 &a); 
        inline bool operator==(const D3 &) const; 
        inline double  operator[](int i) const;
        inline double& operator[](int i);     

    protected:

        double pd[3];
        double gzero;

    private:   

};

inline D3 operator*(double &l, const D3 &a) {

    D3 r( l*a.pd[0], l*a.pd[1], l*a.pd[2] ); 

    return r;                               

};

inline D3 D3::operator+(const D3 &a) const {

    return( D3 ( pd[0] + a.pd[0], pd[1] + a.pd[1], pd[2] + a.pd[2] ) );

};

inline D3 D3::operator-(const D3 &a) const {

    return( D3 ( pd[0] - a.pd[0], pd[1] - a.pd[1], pd[2] - a.pd[2] ) );

};

inline D3 D3::operator*(const double &l) const {

    return( D3 ( pd[0]*l, pd[1]*l, pd[2]*l ) );

};

inline double D3::operator*(const D3 &a) const {  

    return( pd[0]*a.pd[0] + pd[1]*a.pd[1] + pd[2]*a.pd[2] );

};

inline void D3::operator+=(const D3 &a) {  

    pd[0] += a.pd[0];
    pd[1] += a.pd[1];
    pd[2] += a.pd[2];

};

inline void D3::operator-=(const D3 &a) { 

    pd[0] -= a.pd[0];
    pd[1] -= a.pd[1];
    pd[2] -= a.pd[2];

};

inline void D3::operator*=(const double &l) {

    pd[0] *= l;
    pd[1] *= l;
    pd[2] *= l;

};

inline D3 D3::operator^(const D3 &b) const {

    D3 r( pd[1]*b.pd[2] - pd[2]*b.pd[1] ,
          pd[2]*b.pd[0] - pd[0]*b.pd[2] , 
          pd[0]*b.pd[1] - pd[1]*b.pd[0] );

    return r;                           

};

inline D3& D3::operator=(const D3 &a) {

    pd[0] = a.pd[0]; 
    pd[1] = a.pd[1]; 
    pd[2] = a.pd[2]; 

    return *this; 

};

inline bool D3::operator==(const D3 &a) const {              

    if( ( (*this) - a ).norm2() < gzero ) {
        return true;
    } else {
        return false;
    }

};

inline double D3::operator[](int i) const { 

    return pd[i];

};

inline double& D3::operator[](int i) {

    return pd[i];

};

#endif /*_IMMERSED_BOUNDARY_METHOD_*/
