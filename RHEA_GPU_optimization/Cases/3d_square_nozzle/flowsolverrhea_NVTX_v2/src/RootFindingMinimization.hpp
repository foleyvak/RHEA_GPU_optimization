#ifndef _ROOT_FINDING_MINIMIZATION_HPP_
#define _ROOT_FINDING_MINIMIZATION_HPP_

////////// INCLUDES //////////
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>

////////// CLASS DECLARATION //////////
class BaseRootFindingMinimization;			/// BaseRootFindingMinimization
class Brent;						/// Brent
class NewtonRaphson;					/// NewtonRaphson
class Broyden;						/// Broyden
class BFGS;						/// BFGS

////////// FUNCTION DECLARATION //////////


////////// BaseRootFindingMinimization CLASS //////////
class BaseRootFindingMinimization {

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        //BaseRootFindingMinimization();					/// Default constructor
        BaseRootFindingMinimization(const int &n_, double* fvec_);		/// Parametrized constructor
        virtual ~BaseRootFindingMinimization();                         	/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////

        /// Numerical Recipes in C++ shift algorithm
	#pragma acc routine
        inline void shft3(double &a, double &b, double &c, const double &d) { a = b; b = c; c = d; };

	// Numerical Recipes in C++ sign algorithm
	#pragma acc routine
        inline double SIGN(const double &a, const double &b) { return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a); };

	/// Numerical Recipes in C++ SQR algorithm
	#pragma acc routine
        inline double SQR(const double &a) { return a*a; };	

        /// Numerical recipes in C++ linear search
	#pragma acc routine
        void lnsrch(double* xold, double &fold, double* g, double* p, double* x,double &f, double &stpmax, bool &check);

        /// Numerical recipes in C++ LU decomposition method
	#pragma acc routine
	void lubksb(double** a, int* indx, double* b);

        /// Numerical recipes in C++ LU decomposition method
	#pragma acc routine
        void ludcmp(double** a, int* indx, double &d);

        /// Numerical recipes in C++ Jacobian method
	#pragma acc routine
        void fdjac(double* x, double** df);
        
        /// Numerical recipes in C++ QR decomposition method
	#pragma acc routine
        void qrdcmp(double** a, double* c, double* d, bool &sing);
        
        /// Numerical recipes in C++ rsolv method
	#pragma acc routine
        void rsolv(double** a, double* d, double* b);
        
        /// Numerical recipes in C++ qrupdt method
	#pragma acc routine
        void qrupdt(double** r, double** qt, double* u, double* v);
        
        /// Numerical recipes in C++ rotate method
	#pragma acc routine
        void rotate(double** r, double** qt, const int &i, const double &a, const double &b);

        /// Numerical recipes in C++ fmin method
	#pragma acc routine
        double fmin(double* x);

        /// Runs the minimization algorithm, calculating the minimum (fxmin) and its independent variables (xmin)
        //virtual void solve(double &fxmin, double* xmin, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP) = 0;		// ... modified for OpenACC
        
        /// Evaluates the functions (residuals) in position x
        virtual void function_vector(double* x, double* fx) = 0;

    protected:

        ////////// PARAMETERS //////////

	int n;		/// Dimension size
	double* fvec;	/// Vector of functions to be zeroed
	
	double* fdjac_f;
	double* xold;
	double* vv;
	double* g;
	double* p;
	double** fjac;	

    private:

};


////////// Brent CLASS //////////

class Brent : public BaseRootFindingMinimization {

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        //Brent();							/// Default constructor 
        Brent(const int &n_, double* fvec_);				/// Parametrized constructor
        virtual ~Brent();               				/// Destructor    

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////
	#pragma acc routine
        void set_ax_bx_cx(const double &ax_, const double &bx_, const double &cx_);

	////////// METHODS //////////

        /// Runs the minimization algorithm, calculating the minimum (fxmin) and its independent variables (xmin)
	#pragma acc routine
        void solve(double &fxmin, double* xmin, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP);

    protected:

        ////////// PARAMETERS //////////

        /// Bracketing triplet of abscissas ax, bx, cx (such that bx is between ax and cx, and f(bx) is less than both f(ax) and f(cx))
        double ax, bx, cx;

	double* x;
	double* u_;

};


////////// NewtonRaphson CLASS //////////

class NewtonRaphson : public BaseRootFindingMinimization {

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        //NewtonRaphson();						/// Default constructor 
        NewtonRaphson(const int &n_, double* fvec_);			/// Parametrized constructor
        virtual ~NewtonRaphson();               			/// Destructor    
 
	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////

        /// Runs the minimization algorithm, calculating the minimum (fxmin) and its independent variables (xmin)
	#pragma acc routine
        void solve(double &fxmin, double* xmin, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP);

    protected:

        ////////// PARAMETERS //////////
	
	int* indx;
	
};


////////// Broyden CLASS //////////

class Broyden : public BaseRootFindingMinimization {

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        //Broyden();							/// Default constructor 
        Broyden(const int &n_, double* fvec_);				/// Parametrized constructor
        virtual ~Broyden();               				/// Destructor    
 
	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////

        /// Runs the minimization algorithm, calculating the minimum (fxmin) and its independent variables (xmin)
	#pragma acc routine
        void solve(double &fxmin, double* xmin, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP);

    protected:

        ////////// PARAMETERS //////////
	
	double* c;
	double* d;
	double* fvcold;
	double* s;
	double* t;
	double* w;
	double** qt;

};


////////// BFGS CLASS //////////

class BFGS : public BaseRootFindingMinimization {

    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        //BFGS();							/// Default constructor 
        BFGS(const int &n_, double* fvec_);				/// Parametrized constructor
        virtual ~BFGS();               					/// Destructor    
 
	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////

        /// Calculates the gradient of the function
	#pragma acc routine
        void dfunc(double* x, double* g);

        /// Runs the minimization algorithm, calculating the minimum (fxmin) and its independent variables (xmin)
	#pragma acc routine
        void solve(double &fxmin, double* xmin, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP);

    protected:

        ////////// PARAMETERS //////////
	
	double* f1;
	double* f2;
	double* pnew;
	double* xi;
	double* dg;
	double* hdg;
	double** hessin;

};

#endif /*_ROOT_FINDING_MINIMIZATION_HPP_*/
