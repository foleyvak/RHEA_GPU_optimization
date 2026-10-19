#ifndef _TRANSPORT_COEFFICIENTS_
#define _TRANSPORT_COEFFICIENTS_

////////// INCLUDES //////////
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include "yaml-cpp/yaml.h"

////////// CLASS DECLARATION //////////
class BaseTransportCoefficients;				/// Base transport coefficients
class ConstantTransportCoefficients;				/// Constant transport coefficients
class LowPressureGasTransportCoefficients;			/// Low-pressure gas variable transport coefficients
class HighPressureTransportCoefficients;			/// High-pressure transport coefficients
//class CoolPropTransportCoefficients;                            /// CoolProp transport coefficients
class SvdSurrogateTransportCoefficients;                        /// SVD surrogate transport coefficients

////////// FUNCTION DECLARATION //////////


////////// BaseTransportCoefficients CLASS //////////
class BaseTransportCoefficients {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        BaseTransportCoefficients();						/// Default constructor
        BaseTransportCoefficients(const std::string configuration_file);	/// Parametrized constructor
        virtual ~BaseTransportCoefficients();					/// Destructor

	////////// GET FUNCTIONS //////////
        inline double getDynamicViscosity() { return( mu ); };
        inline double getThermalConductivity() { return( kappa ); };

	////////// SET FUNCTIONS //////////
        inline void setDynamicViscosity(double mu_) { mu = mu_; };
        inline void setThermalConductivity(double kappa_) { kappa = kappa_; };

	////////// METHODS //////////
       
        /// Read configuration (input) file written in YAML language
        //virtual void readConfigurationFile() {};
        void readConfigurationFile() {};	// ... modified for OpenACC
 
        /// Calculate dynamic viscosity
        //virtual double calculateDynamicViscosity(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateDynamicViscosity(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate thermal conductivity
        //virtual double calculateThermalConductivity(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateThermalConductivity(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

    protected:

        ////////// PARAMETERS //////////

        /// Transport coefficients
        double mu;							/// Dynamic viscosity [Pa·s]
        double kappa;							/// Thermal conductivity [W/(m·K)]

        /// Thermodynamic properties
        double R_universal = 8.31446261815324;				/// Universal (ideal-gas) gas constant [J/(mol·K)]

        /// Model parameters
	std::string configuration_file;					/// Configuration file name (YAML language)

    private:

};


////////// ConstantTransportCoefficients CLASS //////////
class ConstantTransportCoefficients : public BaseTransportCoefficients {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        ConstantTransportCoefficients();					/// Default constructor
        ConstantTransportCoefficients(const std::string configuration_file);	/// Parametrized constructor
        virtual ~ConstantTransportCoefficients();				/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate dynamic viscosity
	#pragma acc routine
        double calculateDynamicViscosity(const double &P, const double &T, const double &rho);

        /// Calculate thermal conductivity
	#pragma acc routine
        double calculateThermalConductivity(const double &P, const double &T, const double &rho);

    protected:

        ////////// PARAMETERS //////////

    private:

};


////////// LowPressureGasTransportCoefficients CLASS //////////
class LowPressureGasTransportCoefficients : public BaseTransportCoefficients {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        LowPressureGasTransportCoefficients();						/// Default constructor
        LowPressureGasTransportCoefficients(const std::string configuration_file);	/// Parametrized constructor
        virtual ~LowPressureGasTransportCoefficients();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate dynamic viscosity
	#pragma acc routine
        double calculateDynamicViscosity(const double &P, const double &T, const double &rho);

        /// Calculate thermal conductivity
	#pragma acc routine
        double calculateThermalConductivity(const double &P, const double &T, const double &rho);

    protected:

        ////////// PARAMETERS //////////

        /// Transport parameters
        double mu_0;						/// Reference dynamic viscosity [Pa·s]
        double kappa_0;						/// Reference thermal conductivity [W/(m·K)]

        /// Auxiliar coefficients
        double T_0;						/// Reference temperature [K]
        double S_mu;						/// Sutherland constant for mu [K]
        double S_kappa;						/// Sutherland constant for kappa [K]

    private:

};


////////// HighPressureTransportCoefficients CLASS //////////
class HighPressureTransportCoefficients : public BaseTransportCoefficients {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        HighPressureTransportCoefficients();						/// Default constructor
        HighPressureTransportCoefficients(const std::string configuration_file);	/// Parametrized constructor
        virtual ~HighPressureTransportCoefficients();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate dynamic viscosity
	#pragma acc routine
        double calculateDynamicViscosity(const double &P, const double &T, const double &rho);

        /// Calculate thermal conductivity
	#pragma acc routine
        double calculateThermalConductivity(const double &P, const double &T, const double &rho);

        /// Calculate standard thermodynamic variables from NASA 7-coefficient polynomial
	#pragma acc routine
        double calculateMolarStdCpFromNASApolynomials(const double &T);

    protected:

        ////////// PARAMETERS //////////

        /// Transport parameters
        double molecular_weight;				/// Molecular weight [kg/mol]
        double critical_temperature;				/// Critical temperature [K]
        double critical_molar_volume;				/// Critical molar volume [m3/mol]
        double acentric_factor;					/// Acentric factor [-]
        double dipole_moment;					/// Dipole moment [D]
        double association_factor;				/// Association factor [-]
        double NASA_coefficients[15];				/// NASA 7-coefficient polynomial

        /// Auxiliar coefficients
        double adimensional_dipole_moment;						/// Adimensional dipole moment [-]
        double E1_mu, E2_mu, E3_mu, E4_mu, E5_mu, E6_mu, E7_mu, E8_mu, E9_mu, E10_mu;	/// Dynamic viscosity auxiliar coefficients
        double B1_k, B2_k, B3_k, B4_k, B5_k, B6_k, B7_k;				/// Thermal conductivity auxiliar coefficients

    private:

};

/*
////////// CoolPropTransportCoefficients CLASS //////////
class CoolPropTransportCoefficients : public BaseTransportCoefficients {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        CoolPropTransportCoefficients();
        CoolPropTransportCoefficients(const std::string configuration_file);
        virtual ~CoolPropTransportCoefficients();

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

        ////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate dynamic viscosity
        #pragma acc routine
        double calculateDynamicViscosity(const double &P, const double &T, const double &rho);

        /// Calculate thermal conductivity
        #pragma acc routine
        double calculateThermalConductivity(const double &P, const double &T, const double &rho);

    protected:

        ////////// PARAMETERS //////////
        CoolProp::HelmholtzEOSBackend *cp_substance;		/// Helmholtz Energy Equation of State

};
*/

////////// SvdSurrogateTransportCoefficients CLASS //////////
class SvdSurrogateTransportCoefficients : public BaseTransportCoefficients {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        SvdSurrogateTransportCoefficients();
        SvdSurrogateTransportCoefficients(const std::string configuration_file);
        virtual ~SvdSurrogateTransportCoefficients();

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

        ////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate dynamic viscosity
        #pragma acc routine
        double calculateDynamicViscosity(const double &P, const double &T, const double &rho);

        /// Calculate thermal conductivity
        #pragma acc routine
        double calculateThermalConductivity(const double &P, const double &T, const double &rho);

    protected:

        /// Model parameters
	//std::string substance_name;					/// Substance name
	//double substance_molecular_weight;				/// Substance molecular weight
	double substance_P_min;						/// Substance minimum P
	double substance_P_max;						/// Substance maximum P
	double substance_T_min;						/// Substance minimum T
	double substance_T_max;						/// Substance maximum T
	//double substance_rho_min;					/// Substance minimum rho
	//double substance_rho_max;					/// Substance maximum rho
	//double substance_e_min;					/// Substance minimum e
	//double substance_e_max;					/// Substance maximum e	
	int substance_SVD_rank;						/// Substance SVD rank
        double* substance_P_vector;					/// Substance P vector 
        double* substance_T_vector;					/// Substance T vector
        //double* substance_rho_vector;					/// Substance rho vector 
        //double* substance_e_vector;					/// Substance e vector	
        double* substance_T_P_to_mu_matrix_U;				/// Substance T-P to mu matrix U 
        double* substance_T_P_to_mu_matrix_S;				/// Substance T-P to mu matrix S 
        double* substance_T_P_to_mu_matrix_VT;				/// Substance T-P to mu matrix V transpose
        double* substance_T_P_to_kappa_matrix_U;			/// Substance T-P to kappa matrix U 
        double* substance_T_P_to_kappa_matrix_S;			/// Substance T-P to kappa matrix S 
        double* substance_T_P_to_kappa_matrix_VT;			/// Substance T-P to kappa matrix V transpose

};

#endif /*_TRANSPORT_COEFFICIENTS_*/
