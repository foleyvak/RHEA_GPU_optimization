#ifndef _THERMODYNAMIC_MODEL_
#define _THERMODYNAMIC_MODEL_

////////// INCLUDES //////////
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <complex>
#include <limits>
#include <mpi.h>
#include "yaml-cpp/yaml.h"
#include "RootFindingMinimization.hpp"
#if COOLPROP_ACTIVATED_
#include "HelmholtzEOSBackend.h"
#endif


////////// CLASS DECLARATION //////////
class BaseThermodynamicModel;				/// Base thermodynamic model
class IdealGasThermodynamicModel;			/// Ideal-gas thermodynamic model
class StiffenedGasThermodynamicModel;			/// Stiffened-gas thermodynamic model
class PengRobinsonThermodynamicModel;			/// Peng-Robinson (real-fluid) thermodynamic model

#if COOLPROP_ACTIVATED_
class CoolPropThermodynamicModel;			/// CoolProp thermodynamic model
#endif

class SvdSurrogateThermodynamicModel;			/// SVD surrogate thermodynamic model

////////// FUNCTION DECLARATION //////////


////////// BaseThermodynamicModel CLASS //////////
class BaseThermodynamicModel {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        BaseThermodynamicModel();					/// Default constructor
        BaseThermodynamicModel(const std::string configuration_file);	/// Parametrized constructor
        virtual ~BaseThermodynamicModel();				/// Destructor

	////////// GET FUNCTIONS //////////
	inline double getSpecificGasConstant() { return( R_specific ); };
	inline double getMolecularWeight() { return( molecular_weight ); };

	////////// SET FUNCTIONS //////////
        inline void setSpecificGasConstant(double R_specific_) { R_specific = R_specific_; };
        inline void setMolecularWeight(double molecular_weight_) { molecular_weight = molecular_weight_; };

	////////// METHODS //////////
       
        /// Read configuration (input) file written in YAML language
        //virtual void readConfigurationFile() {};
        void readConfigurationFile() {};

        /// Calculate temperature from pressure and density
        //virtual double calculateTemperatureFromPressureDensity(const double &P, const double &rho) = 0;
	#pragma acc routine
        double calculateTemperatureFromPressureDensity(const double &P, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate temperature from pressure and density with initial guess
        //virtual void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {};
	#pragma acc routine
        void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {};	// ... modified for OpenACC

        /// Calculate pressure from temperature and density
        //virtual double calculatePressureFromTemperatureDensity(const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculatePressureFromTemperatureDensity(const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate internal energy from pressure, temperature and density
        //virtual double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate entropy from pressure, temperature and density
        //virtual double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate pressure and temperature from density and internal energy
        //virtual void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {};
	#pragma acc routine
        void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {};	// ... modified for OpenACC
        
        /// Calculate density from pressure and temperature
        //virtual double calculateDensityFromPressureTemperature(const double &P, const double &T) = 0;
	#pragma acc routine
        double calculateDensityFromPressureTemperature(const double &P, const double &T) { return 0.0; };	// ... modified for OpenACC

        /// Calculate density and internal energy from pressure and temperature
        //virtual void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {};
	#pragma acc routine
        void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {};	// ... modified for OpenACC

        /// Calculate specific heat capacities
        //virtual void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho) {};
	#pragma acc routine
        void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho) {};	// ... modified for OpenACC

        /// Calculate heat capacities ratio
        //virtual double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate speed of sound
        //virtual double calculateSoundSpeed(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateSoundSpeed(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

        /// Calculate expansivity & compressibility
        //virtual double calculateVolumeExpansivity(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateVolumeExpansivity(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC
        //virtual double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC
        //virtual double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) = 0;
	#pragma acc routine
        double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) { return 0.0; };	// ... modified for OpenACC

    protected:

        ////////// PARAMETERS //////////

        /// Thermodynamic properties
        double R_universal = 8.31446261815324;					/// Universal (ideal-gas) gas constant [J/(mol·K)]
        double R_specific;							/// Specific gas constant [J/(kg·K)]
        double molecular_weight;						/// Molecular weight [kg/mol]

        /// Model parameters
	std::string configuration_file;						/// Configuration file name (YAML language)
        
    private:

};


////////// IdealGasThermodynamicModel CLASS //////////
class IdealGasThermodynamicModel : public BaseThermodynamicModel {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        IdealGasThermodynamicModel();							/// Default constructor
        IdealGasThermodynamicModel(const std::string configuration_file);		/// Parametrized constructor
        virtual ~IdealGasThermodynamicModel();						/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate temperature from pressure and density
	#pragma acc routine
        double calculateTemperatureFromPressureDensity(const double &P, const double &rho);

        /// Calculate temperature from pressure and density with initial guess
	#pragma acc routine
        void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho);

        /// Calculate pressure from temperature and density
	#pragma acc routine
        double calculatePressureFromTemperatureDensity(const double &T, const double &rho);

        /// Calculate internal energy from pressure, temperature and density
	#pragma acc routine
        double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate entropy from pressure, temperature and density
	#pragma acc routine
        double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate pressure and temperature from density and internal energy
	#pragma acc routine
        void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e);

        /// Calculate density from pressure and temperature
	#pragma acc routine
        double calculateDensityFromPressureTemperature(const double &P, const double &T);

        /// Calculate density and internal energy from pressure and temperature
	#pragma acc routine
        void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T);

        /// Calculate specific heat capacities
	#pragma acc routine
        void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho);

        /// Calculate heat capacities ratio
	#pragma acc routine
        double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho);

        /// Calculate speed of sound
	#pragma acc routine
        double calculateSoundSpeed(const double &P, const double &T, const double &rho);

        /// Calculate expansivity & compressibility
	#pragma acc routine
        double calculateVolumeExpansivity(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho);

    protected:

        ////////// PARAMETERS //////////

        /// Thermodynamic properties
        double gamma;						/// Heat capacities ratio (ideal-gas) [-]

    private:

};


////////// StiffenedGasThermodynamicModel CLASS //////////
class StiffenedGasThermodynamicModel : public BaseThermodynamicModel {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        StiffenedGasThermodynamicModel();							/// Default constructor
        StiffenedGasThermodynamicModel(const std::string configuration_file);			/// Parametrized constructor
        virtual ~StiffenedGasThermodynamicModel();						/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate temperature from pressure and density
	#pragma acc routine
        double calculateTemperatureFromPressureDensity(const double &P, const double &rho);

        /// Calculate temperature from pressure and density with initial guess
	#pragma acc routine
        void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho);

        /// Calculate pressure from temperature and density
	#pragma acc routine
        double calculatePressureFromTemperatureDensity(const double &T, const double &rho);

        /// Calculate internal energy from pressure, temperature and density
	#pragma acc routine
        double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate entropy from pressure, temperature and density
	#pragma acc routine
        double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate pressure and temperature from density and internal energy
	#pragma acc routine
        void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e);

        /// Calculate density from pressure and temperature
	#pragma acc routine
        double calculateDensityFromPressureTemperature(const double &P, const double &T);

        /// Calculate density and internal energy from pressure and temperature
	#pragma acc routine
        void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T);

        /// Calculate specific heat capacities
	#pragma acc routine
        void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho);

        /// Calculate heat capacities ratio
	#pragma acc routine
        double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho);

        /// Calculate speed of sound
	#pragma acc routine
        double calculateSoundSpeed(const double &P, const double &T, const double &rho);

        /// Calculate expansivity & compressibility
	#pragma acc routine
        double calculateVolumeExpansivity(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho);

    protected:

        ////////// PARAMETERS //////////

        /// Thermodynamic properties
        double gamma;						/// Heat capacities ratio [-]
        double P_inf;						/// Pressure infinity (liquid stiffnes) [Pa]
        double e_0;						/// Internal energy zero reference [J/kg]
        double c_v;						/// Specific isochoric heat capacity [J/(kg·K)]

    private:

};


////////// PengRobinsonThermodynamicModel CLASS //////////
class PengRobinsonThermodynamicModel : public BaseThermodynamicModel {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        PengRobinsonThermodynamicModel();							/// Default constructor
        PengRobinsonThermodynamicModel(const std::string configuration_file);			/// Parametrized constructor
        virtual ~PengRobinsonThermodynamicModel();						/// Destructor

	////////// GET FUNCTIONS //////////
        inline double getCriticalPressure() { return( critical_pressure ); };
        inline double getCriticalTemperature() { return( critical_temperature ); };

	////////// SET FUNCTIONS //////////
        inline void setCriticalPressure(double critical_pressure_) { critical_pressure = critical_pressure_; };
        inline void setCriticalTemperature(double critical_temperature_) { critical_temperature = critical_temperature_; };

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate temperature from pressure and density
	#pragma acc routine
        double calculateTemperatureFromPressureDensity(const double &P, const double &rho);

        /// Calculate temperature from pressure and density with initial guess
	#pragma acc routine
        void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho);

        /// Calculate pressure from temperature and density
	#pragma acc routine
        double calculatePressureFromTemperatureDensity(const double &T, const double &rho);

        /// Calculate internal energy from pressure, temperature and density
	#pragma acc routine
        double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate entropy from pressure, temperature and density
	#pragma acc routine
        double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate pressure and temperature from density and internal energy
	#pragma acc routine
        void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e);

        /// Calculate density from pressure and temperature
	#pragma acc routine
        double calculateDensityFromPressureTemperature(const double &P, const double &T);

        /// Calculate density and internal energy from pressure and temperature
	#pragma acc routine
        void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T);

        /// Calculate specific heat capacities
	#pragma acc routine
        void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho);

        /// Calculate heat capacities ratio
	#pragma acc routine
        double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho);

        /// Calculate speed of sound
	#pragma acc routine
        double calculateSoundSpeed(const double &P, const double &T, const double &rho);

        /// Calculate expansivity & compressibility
	#pragma acc routine
        double calculateVolumeExpansivity(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho);

        /// Calculate molar internal energy from pressure, temperature and molar volume
	#pragma acc routine
        double calculateMolarInternalEnergyFromPressureTemperatureMolarVolume(const double &P, const double &T, const double &bar_v);

        /// Calculate molar entropy from pressure, temperature and molar volume
	#pragma acc routine
        double calculateMolarEntropyFromPressureTemperatureMolarVolume(const double &P, const double &T, const double &bar_v);

        /// Calculate attractive-forces a coefficient
	#pragma acc routine
        double calculate_eos_a(const double &T);

        /// Calculate first derivative of attractive-forces a coefficient
	#pragma acc routine
        double calculate_eos_a_first_derivative(const double &T);

        /// Calculate second derivative of attractive-forces a coefficient
	#pragma acc routine
        double calculate_eos_a_second_derivative(const double &T);

        /// Calculate compressibility factor
	#pragma acc routine
        double calculate_Z(const double &P, const double &T, const double &bar_v);

        /// Calculate auxiliar parameters
	#pragma acc routine
        double calculate_A(const double &P, const double &T);
	#pragma acc routine
        double calculate_B(const double &P, const double &T);
	#pragma acc routine
        double calculate_M(const double &Z, const double &B);
	#pragma acc routine
        double calculate_N(const double &eos_a_first_derivative, const double &B);

        /// Calculate standard thermodynamic variables from NASA 7-coefficient polynomial
	#pragma acc routine
        double calculateMolarStdCpFromNASApolynomials(const double &T);
	#pragma acc routine
        double calculateMolarStdEnthalpyFromNASApolynomials(const double &T);
	#pragma acc routine
        double calculateMolarStdEntropyFromNASApolynomials(const double &T);

        /// Calculate high-pressure departure functions
	#pragma acc routine
        double calculateDepartureFunctionMolarCp(const double &P, const double &T, const double &bar_v);
	#pragma acc routine
        double calculateDepartureFunctionMolarCv(const double &P, const double &T, const double &bar_v);
	#pragma acc routine
        double calculateDepartureFunctionMolarEnthalpy(const double &P, const double &T, const double &bar_v);
	#pragma acc routine
        double calculateDepartureFunctionMolarEntropy(const double &P, const double &T, const double &bar_v);

        /// Calculate temperature from pressure and molar volume
	#pragma acc routine
        double calculateTemperatureFromPressureMolarVolume(const double &P, const double &bar_v);

        /// Calculate thermodynamic derivatives
	#pragma acc routine
        double calculateDPDTConstantMolarVolume(const double &T, const double &bar_v);
	#pragma acc routine
        double calculateDPDvConstantTemperature(const double &T, const double &bar_v);

        /// Calculate roots of cubic polynomial
	//#pragma acc routine
        //void calculateRootsCubicPolynomial(std::complex<double> &root_1, std::complex<double> &root_2, std::complex<double> &root_3, double &a, double &b, double &c, double &d);
        
	// ... Modified for OpenACC
	#pragma acc routine
	void calculateRootsCubicPolynomial_(double &root_1_real, double &root_2_real, double &root_3_real, double &root_2_imag, double &root_3_imag, double &a, double &b, double &c, double &d);
    
    	/// Nonlinear solver functions
	#pragma acc routine
	void function_vector2D(double &xmin_1, double &xmin_2, double &fx_1, double &fx_2, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm);
	void fdjac2D(double &xmin_1, double &xmin_2, double &df_11, double &df_12, double &df_21, double &df_22, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm, double &nls_P_r, double &nls_T_r);
	double fmin2D(double &xmin_1, double &xmin_2, double &nls_P_r, double &nls_T_r, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm);
	void lnsrch2D(double &xold_1, double &xold_2, double &fold, double &g_1, double &g_2, double &p_1, double &p_2, double &xmin_1, double &xmin_2, double &f, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm, double &nls_P_r, double &nls_T_r, double &stpmax, bool &check);
	void ludcmp2D(double &a_11, double &a_12, double &a_21, double &a_22, int &indx_1, int &indx_2);
	void lubksb2D(double &fjac_11, double &fjac_12, double &fjac_21, double &fjac_22, int &indx_1, int &indx_2, double &p_1, double &p_2);
	void solve2D(double &fxmin, double &xmin_1, double &xmin_2, const double &P_norm, const double &T_norm, const double &target_rho, const double &target_e, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP);
	
    protected:

        ////////// PARAMETERS //////////

        /// Thermodynamic properties
        double acentric_factor;					/// Acentric factor [-]
        double critical_temperature;				/// Critical temperature [K]
        double critical_pressure;				/// Critical pressure [Pa]
        double critical_molar_volume;				/// Critical molar volume [m3/mol]
        double NASA_coefficients[15];				/// NASA 7-coefficient polynomial

        /// Equation of state (EoS) parameters
        //double eos_a;						/// EoS attractive-forces coefficient
        double eos_b;						/// EoS finite-pack-volume coefficient
        double eos_ac, eos_kappa;				/// EoS dimensionless parameters

        /// Aitken's delta-squared process parameters
        int aitken_max_iter              = 100;			/// Maximum number of iterations
        double aitken_relative_tolerance = 1.0e-5;		/// Relative tolerance

        /// Nonlinear P-T solver parameters
        int nls_PT_max_iter              = 100;			/// Maximum number of iterations
        double nls_PT_relative_tolerance = 1.0e-5;		/// Relative tolerance
        //double nls_PT_STPMX              = 1.0e2;		/// Scaled maximum step length allowed in line searches
        double nls_PT_STPMX              = 1.0e-3;		/// Scaled maximum step length allowed in line searches
        
    private:

};

/*
////////// CoolPropThermodynamicModel CLASS //////////
class CoolPropThermodynamicModel : public BaseThermodynamicModel {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        CoolPropThermodynamicModel();						/// Default constructor
        CoolPropThermodynamicModel(const std::string configuration_file);	/// Parametrized constructor
        virtual ~CoolPropThermodynamicModel();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate temperature from pressure and density
	#pragma acc routine
        double calculateTemperatureFromPressureDensity(const double &P, const double &rho);

        /// Calculate temperature from pressure and density with initial guess
	#pragma acc routine
        void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho);

        /// Calculate pressure from temperature and density
	#pragma acc routine
        double calculatePressureFromTemperatureDensity(const double &T, const double &rho);

        /// Calculate internal energy from pressure, temperature and density
	#pragma acc routine
        double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate entropy from pressure, temperature and density
	#pragma acc routine
        double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate pressure and temperature from density and internal energy
	#pragma acc routine
        void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e);

        /// Calculate density from pressure and temperature
	#pragma acc routine
        double calculateDensityFromPressureTemperature(const double &P, const double &T);

        /// Calculate density and internal energy from pressure and temperature
	#pragma acc routine
        void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T);

        /// Calculate specific heat capacities
	#pragma acc routine
        void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho);

        /// Calculate heat capacities ratio
	#pragma acc routine
        double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho);

        /// Calculate speed of sound
	#pragma acc routine
        double calculateSoundSpeed(const double &P, const double &T, const double &rho);

        /// Calculate expansivity & compressibility
	#pragma acc routine
        double calculateVolumeExpansivity(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho);

    protected:

        ////////// PARAMETERS //////////
        CoolProp::HelmholtzEOSBackend *cp_substance;		/// Helmholtz Energy Equation of State

    private:

};
*/

////////// SvdSurrogateThermodynamicModel CLASS //////////
class SvdSurrogateThermodynamicModel : public BaseThermodynamicModel {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        SvdSurrogateThermodynamicModel();						/// Default constructor
        SvdSurrogateThermodynamicModel(const std::string configuration_file);		/// Parametrized constructor
        virtual ~SvdSurrogateThermodynamicModel();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Read configuration (input) file written in YAML language
        void readConfigurationFile();

        /// Calculate temperature from pressure and density
	#pragma acc routine
        double calculateTemperatureFromPressureDensity(const double &P, const double &rho);

        /// Calculate temperature from pressure and density with initial guess
	#pragma acc routine
        void calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho);

        /// Calculate pressure from temperature and density
	#pragma acc routine
        double calculatePressureFromTemperatureDensity(const double &T, const double &rho);

        /// Calculate internal energy from pressure, temperature and density
	#pragma acc routine
        double calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate entropy from pressure, temperature and density
	#pragma acc routine
        double calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho);

        /// Calculate pressure and temperature from density and internal energy
	#pragma acc routine
        void calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e);

        /// Calculate density from pressure and temperature
	#pragma acc routine
        double calculateDensityFromPressureTemperature(const double &P, const double &T);

        /// Calculate density and internal energy from pressure and temperature
	#pragma acc routine
        void calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T);

        /// Calculate specific heat capacities
	#pragma acc routine
        void calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho);

        /// Calculate heat capacities ratio
	#pragma acc routine
        double calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho);

        /// Calculate speed of sound
	#pragma acc routine
        double calculateSoundSpeed(const double &P, const double &T, const double &rho);

        /// Calculate expansivity & compressibility
	#pragma acc routine
        double calculateVolumeExpansivity(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsothermalCompressibility(const double &P, const double &T, const double &rho);
	#pragma acc routine
        double calculateIsentropicCompressibility(const double &P, const double &T, const double &rho);

    protected:

        /// Model parameters
	//std::string substance_name;					/// Substance name
	double substance_molecular_weight;				/// Substance molecular weight
	double substance_P_min;						/// Substance minimum P
	double substance_P_max;						/// Substance maximum P
	double substance_T_min;						/// Substance minimum T
	double substance_T_max;						/// Substance maximum T
	double substance_rho_min;					/// Substance minimum rho
	double substance_rho_max;					/// Substance maximum rho
	double substance_e_min;						/// Substance minimum e
	double substance_e_max;						/// Substance maximum e
	int substance_SVD_rank;						/// Substance SVD rank
        double* substance_P_vector;					/// Substance P vector 
        double* substance_T_vector;					/// Substance T vector 
        double* substance_rho_vector;					/// Substance rho vector 
        double* substance_e_vector;					/// Substance e vector
        double* substance_rho_P_to_T_matrix_U;				/// Substance rho-P to T matrix U 
        double* substance_rho_P_to_T_matrix_S;				/// Substance rho-P to T matrix S 
        double* substance_rho_P_to_T_matrix_VT;				/// Substance rho-P to T matrix V transpose									
        double* substance_rho_T_to_P_matrix_U;				/// Substance rho-T to P matrix U 
        double* substance_rho_T_to_P_matrix_S;				/// Substance rho-T to P matrix S 
        double* substance_rho_T_to_P_matrix_VT;				/// Substance rho-T to P matrix V transpose
        double* substance_T_P_to_rho_matrix_U;				/// Substance T-P to rho matrix U 
        double* substance_T_P_to_rho_matrix_S;				/// Substance T-P to rho matrix S 
        double* substance_T_P_to_rho_matrix_VT;				/// Substance T-P to rho matrix V transpose
        double* substance_T_P_to_e_matrix_U;				/// Substance T-P to e matrix U 
        double* substance_T_P_to_e_matrix_S;				/// Substance T-P to e matrix S 
        double* substance_T_P_to_e_matrix_VT;				/// Substance T-P to e matrix V transpose
        double* substance_T_P_to_s_matrix_U;				/// Substance T-P to s matrix U 
        double* substance_T_P_to_s_matrix_S;				/// Substance T-P to s matrix S 
        double* substance_T_P_to_s_matrix_VT;				/// Substance T-P to s matrix V transpose	
        double* substance_T_P_to_c_p_matrix_U;				/// Substance T-P to c_p matrix U 
        double* substance_T_P_to_c_p_matrix_S;				/// Substance T-P to c_p matrix S 
        double* substance_T_P_to_c_p_matrix_VT;				/// Substance T-P to c_p matrix V transpose
        double* substance_T_P_to_c_v_matrix_U;				/// Substance T-P to c_v matrix U 
        double* substance_T_P_to_c_v_matrix_S;				/// Substance T-P to c_v matrix S 
        double* substance_T_P_to_c_v_matrix_VT;				/// Substance T-P to c_v matrix V transpose
        double* substance_T_P_to_sos_matrix_U;				/// Substance T-P to sos matrix U 
        double* substance_T_P_to_sos_matrix_S;				/// Substance T-P to sos matrix S 
        double* substance_T_P_to_sos_matrix_VT;				/// Substance T-P to sos matrix V transpose	
        double* substance_T_P_to_alpha_matrix_U;			/// Substance T-P to volume expansivity (alpha) matrix U 
        double* substance_T_P_to_alpha_matrix_S;			/// Substance T-P to volume expansivity (alpha) matrix S 
        double* substance_T_P_to_alpha_matrix_VT;			/// Substance T-P to volume expansivity (alpha) matrix V transpose
        double* substance_T_P_to_kappa_T_matrix_U;			/// Substance T-P to isothermal compressibility (kappa_T) matrix U 
        double* substance_T_P_to_kappa_T_matrix_S;			/// Substance T-P to isothermal compressibility (kappa_T) matrix S 
        double* substance_T_P_to_kappa_T_matrix_VT;			/// Substance T-P to isothermal compressibility (kappa_T) matrix V transpose
        double* substance_T_P_to_kappa_S_matrix_U;			/// Substance T-P to isentropic compressibility (kappa_S) matrix U 
        double* substance_T_P_to_kappa_S_matrix_S;			/// Substance T-P to isentropic compressibility (kappa_S) matrix S 
        double* substance_T_P_to_kappa_S_matrix_VT;			/// Substance T-P to isentropic compressibility (kappa_S) matrix V transpose
        double* substance_e_rho_to_P_matrix_U;				/// Substance e-rho to P matrix U 
        double* substance_e_rho_to_P_matrix_S;				/// Substance e-rho to P matrix S 
        double* substance_e_rho_to_P_matrix_VT;				/// Substance e-rho to P matrix V transpose
        double* substance_e_rho_to_T_matrix_U;				/// Substance e-rho to T matrix U 
        double* substance_e_rho_to_T_matrix_S;				/// Substance e-rho to T matrix S 
        double* substance_e_rho_to_T_matrix_VT;				/// Substance e-rho to T matrix V transpose

    private:

};

#endif /*_THERMODYNAMIC_MODEL_*/
