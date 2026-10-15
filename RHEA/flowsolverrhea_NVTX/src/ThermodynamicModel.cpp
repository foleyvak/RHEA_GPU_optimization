#include "ThermodynamicModel.hpp"

using namespace std;

////////// FIXED PARAMETERS //////////
#define _ACTIVATE_COUT_ 0


////////// BaseThermodynamicModel CLASS //////////

BaseThermodynamicModel::BaseThermodynamicModel() {};
        
BaseThermodynamicModel::BaseThermodynamicModel(const string configuration_file) : configuration_file(configuration_file) {};

BaseThermodynamicModel::~BaseThermodynamicModel() {};


////////// IdealGasThermodynamicModel CLASS //////////

IdealGasThermodynamicModel::IdealGasThermodynamicModel() : BaseThermodynamicModel() {};
        
IdealGasThermodynamicModel::IdealGasThermodynamicModel(const string configuration_file) : BaseThermodynamicModel(configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile();

};

IdealGasThermodynamicModel::~IdealGasThermodynamicModel() {};

void IdealGasThermodynamicModel::readConfigurationFile() {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile(configuration_file);

    /// Fluid & flow properties
    const YAML::Node & fluid_flow_properties = configuration["fluid_flow_properties"];
    if( fluid_flow_properties["substance_name"] ) {

        /// Create YAML object
	string substance_name          = fluid_flow_properties["substance_name"].as<string>();
	string substances_library_file = fluid_flow_properties["substances_library_file"].as<string>();
        YAML::Node substances_library  = YAML::LoadFile( substances_library_file );
        YAML::Node substance;

        #include "substances_selection.txt"

        R_specific = substance["R_specific"].as<double>();
        gamma      = substance["gamma"].as<double>();

    } else {

        R_specific = fluid_flow_properties["R_specific"].as<double>();
        gamma      = fluid_flow_properties["gamma"].as<double>();

    }
    
    /// Calculate molecular weight
    molecular_weight = R_universal/R_specific;

};

double IdealGasThermodynamicModel::calculateTemperatureFromPressureDensity(const double &P, const double &rho) {

    /// Ideal-gas model:
    /// P = rho*R_specific*T is pressure

    double T = P/( rho*R_specific );

    return( T );

};

void IdealGasThermodynamicModel::calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {

    /// Ideal-gas model:
    /// P = rho*R_specific*T is pressure

    /// Initial temperature guess is not needed
    T = P/( rho*R_specific );

};

double IdealGasThermodynamicModel::calculatePressureFromTemperatureDensity(const double &T, const double &rho) {

    /// Ideal-gas model:
    /// P = rho*R_specific*T is pressure

    double P = rho*R_specific*T;

    return( P );

};

double IdealGasThermodynamicModel::calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Ideal-gas model:
    /// e = c_v*T is specific internal energy

    double c_v = R_specific/( gamma - 1.0 );

    double e = c_v*T;

    return( e );

};

double IdealGasThermodynamicModel::calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Ideal-gas model:
    /// s = c_v*ln( P/rho^gamma ) is specific entropy

    double c_v = R_specific/( gamma - 1.0 );

    double s = c_v*log( P/pow( rho, gamma ) );

    return( s );

};

void IdealGasThermodynamicModel::calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {

    /// Ideal-gas model:
    /// P = e*rho*(gamma - 1) is pressure
    /// T = e/c_v is temperature

    double c_v = R_specific/( gamma - 1.0 );

    P = e*rho*( gamma - 1.0 ); 
    T = e/c_v;

};

double IdealGasThermodynamicModel::calculateDensityFromPressureTemperature(const double &P, const double &T) {

    /// Ideal-gas model:
    /// rho = P/( R_specific*T ) is density
    
    double rho = P/( R_specific*T );

    return( rho );

};

void IdealGasThermodynamicModel::calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {

    /// Ideal-gas model:
    /// rho = P/( e*( gamma - 1.0 ) ) is density
    /// e = c_v*T is specific internal energy

    double c_v = R_specific/( gamma - 1.0 );

    e   = c_v*T;
    rho = P/( e*( gamma - 1.0 ) );

};

void IdealGasThermodynamicModel::calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho) {

    /// Ideal-gas model:
    /// c_v = R_specific/(gamma - 1)
    /// c_p = c_v*gamma

    c_v = R_specific/( gamma - 1.0 );
    c_p = c_v*gamma;

};

double IdealGasThermodynamicModel::calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) {

    /// Ideal-gas model:
    /// gamma = c_p/c_v

    return( gamma );

};

double IdealGasThermodynamicModel::calculateSoundSpeed(const double &P, const double &T, const double &rho) {

    /// Ideal-gas model:
    /// sos = sqrt(gamma*P/rho) is speed of sound

    double sos = sqrt( gamma*P/rho );

    return( sos );

};

double IdealGasThermodynamicModel::calculateVolumeExpansivity(const double &P, const double &T, const double &rho) {

    double expansivity = 1.0/T;

    return( expansivity );
  
};

double IdealGasThermodynamicModel::calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) {

    double isothermal_compressibility = 1.0/P;

    return( isothermal_compressibility );
  
};  

double IdealGasThermodynamicModel::calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) {
    
    double isentropic_compressibility = 1.0/( gamma*P );
    
    return( isentropic_compressibility );
  
};


////////// StiffenedGasThermodynamicModel CLASS //////////

StiffenedGasThermodynamicModel::StiffenedGasThermodynamicModel() : BaseThermodynamicModel() {};
        
StiffenedGasThermodynamicModel::StiffenedGasThermodynamicModel(const string configuration_file) : BaseThermodynamicModel(configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile();

};

StiffenedGasThermodynamicModel::~StiffenedGasThermodynamicModel() {};

void StiffenedGasThermodynamicModel::readConfigurationFile() {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile(configuration_file);

    /// Fluid & flow properties
    const YAML::Node & fluid_flow_properties = configuration["fluid_flow_properties"];
    if( fluid_flow_properties["substance_name"] ) {

        /// Create YAML object
	string substance_name          = fluid_flow_properties["substance_name"].as<string>();
	string substances_library_file = fluid_flow_properties["substances_library_file"].as<string>();
        YAML::Node substances_library  = YAML::LoadFile( substances_library_file );
        YAML::Node substance;

        #include "substances_selection.txt"

        R_specific = substance["R_specific"].as<double>();
        gamma      = substance["gamma"].as<double>();
        P_inf      = substance["P_inf"].as<double>();
        e_0        = substance["e_0"].as<double>();
        c_v        = substance["c_v"].as<double>();

    } else {

        R_specific = fluid_flow_properties["R_specific"].as<double>();
        gamma      = fluid_flow_properties["gamma"].as<double>();
        P_inf      = fluid_flow_properties["P_inf"].as<double>();
        e_0        = fluid_flow_properties["e_0"].as<double>();
        c_v        = fluid_flow_properties["c_v"].as<double>();

    }

    /// Calculate molecular weight
    molecular_weight = R_universal/R_specific;

};

double StiffenedGasThermodynamicModel::calculateTemperatureFromPressureDensity(const double &P, const double &rho) {

    /// Stiffened-gas model:
    /// P = (e - e_0)*rho*(gamma - 1) - gamma*P_inf is pressure
    /// T = ((e - e_0) - (P_inf/rho))/c_v is temperature

    double T = ( ( ( P + gamma*P_inf )/( rho*( gamma - 1.0 ) ) ) - ( P_inf/rho ) )/c_v;

    return( T );

};

void StiffenedGasThermodynamicModel::calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {

    /// Stiffened-gas model:
    /// P = (e - e_0)*rho*(gamma - 1) - gamma*P_inf is pressure
    /// T = ((e - e_0) - (P_inf/rho))/c_v is temperature

    T = ( ( ( P + gamma*P_inf )/( rho*( gamma - 1.0 ) ) ) - ( P_inf/rho ) )/c_v;

};

double StiffenedGasThermodynamicModel::calculatePressureFromTemperatureDensity(const double &T, const double &rho) {

    /// Stiffened-gas model:
    /// P = (e - e_0)*rho*(gamma - 1) - gamma*P_inf is pressure
    /// T = ((e - e_0) - (P_inf/rho))/c_v is temperature

    double e = e_0 + ( P_inf/rho ) + c_v*T;
    double P = (e - e_0)*rho*(gamma - 1) - gamma*P_inf;

    return( P );

};

double StiffenedGasThermodynamicModel::calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Stiffened-gas model:
    /// e = c_v*T*((P + gamma*P_inf)/(P + P_inf)) + e_0 is specific internal energy

    double e = c_v*T*( ( P + gamma*P_inf )/( P + P_inf ) ) + e_0;

    return( e );

};

double StiffenedGasThermodynamicModel::calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Stiffened-gas model:
    /// s = c_v*ln( ((P+P_inf)/rho^gamma ) is specific entropy

    double c_v = R_specific/( gamma - 1.0 );

    double s = c_v*log( ( P+P_inf )/pow( rho, gamma ) );

    return( s );

};

void StiffenedGasThermodynamicModel::calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {

    /// Stiffened-gas model:
    /// P = (e - e_0)*rho*(gamma - 1) - gamma*P_inf is pressure
    /// T = ((e - e_0) - (P_inf/rho))/c_v is temperature
    
    P = ( e - e_0 )*rho*( gamma - 1.0 ) - gamma*P_inf; 
    T = ( ( e - e_0 ) - ( P_inf/rho ) )/c_v;

};

double StiffenedGasThermodynamicModel::calculateDensityFromPressureTemperature(const double &P, const double &T) {

    /// Stiffened-gas model:
    /// e = c_v*T*((P + gamma*P_inf)/(P + P_inf)) + e_0 is specific internal energy
    /// rho = P_inf/(e - e_0 - c_v*T) is density

    double e   = c_v*T*( ( P + gamma*P_inf )/( P + P_inf ) ) + e_0;
    double rho = P_inf/( e - e_0 - c_v*T );

    return( rho );

};

void StiffenedGasThermodynamicModel::calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {

    /// Stiffened-gas model:
    /// e = c_v*T*((P + gamma*P_inf)/(P + P_inf)) + e_0 is specific internal energy
    /// rho = P_inf/(e - e_0 - c_v*T) is density

    e   = c_v*T*( ( P + gamma*P_inf )/( P + P_inf ) ) + e_0;
    rho = P_inf/( e - e_0 - c_v*T );

};

void StiffenedGasThermodynamicModel::calculateSpecificHeatCapacities(double &c_v_, double &c_p, const double &P, const double &T, const double &rho) {

    /// Stiffened-gas model:
    /// c_p = c_v*gamma

    c_v_ = c_v;
    c_p  = c_v*gamma;

};

double StiffenedGasThermodynamicModel::calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) {

    /// Stiffened-gas model:
    /// gamma = c_p/c_v

    return( gamma );

};

double StiffenedGasThermodynamicModel::calculateSoundSpeed(const double &P, const double &T, const double &rho) {

    /// Stiffened-gas model:
    /// sos = sqrt(gamma*(P+P_inf)/rho) is speed of sound

    double sos = sqrt( gamma*( P + P_inf )/rho );

    return( sos );

};

double StiffenedGasThermodynamicModel::calculateVolumeExpansivity(const double &P, const double &T, const double &rho) {

    double bar_v = molecular_weight/rho;
    double dP_dT_const_v = molecular_weight*c_v*( gamma - 1.0 )/bar_v;
    double dP_dv_const_T = ( -1.0 )*molecular_weight*c_v*( gamma - 1.0 )*T/( bar_v*bar_v );

    double expansivity = ( -1.0 )*( dP_dT_const_v/( bar_v*dP_dv_const_T ) );

    return( expansivity );
  
};

double StiffenedGasThermodynamicModel::calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) {

    double bar_v = molecular_weight/rho;
    double dP_dv_const_T = ( -1.0 )*molecular_weight*c_v*( gamma - 1.0 )*T/( bar_v*bar_v );

    double isothermal_compressibility = ( -1.0 )/( bar_v*dP_dv_const_T );

    return( isothermal_compressibility );
  
};  

double StiffenedGasThermodynamicModel::calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) {
   
    double bar_v = molecular_weight/rho;
    double dP_dT_const_v = molecular_weight*c_v*( gamma - 1.0 )/bar_v;
    double dP_dv_const_T = ( -1.0 )*molecular_weight*c_v*( gamma - 1.0 )*T/( bar_v*bar_v );

    double isothermal_compressibility = ( -1.0 )/( bar_v*dP_dv_const_T );
    double expansivity                = ( -1.0 )*( dP_dT_const_v/( bar_v*dP_dv_const_T ) );

    double c_p = c_v*gamma;

    double bar_c_p = molecular_weight*c_p;
      
    double isentropic_compressibility = ( isothermal_compressibility - ( ( bar_v*T*pow( expansivity, 2.0 ) )/bar_c_p ) );
    
    return( isentropic_compressibility );
  
};


////////// PengRobinsonThermodynamicModel CLASS //////////

PengRobinsonThermodynamicModel::PengRobinsonThermodynamicModel() : BaseThermodynamicModel() {};
        
PengRobinsonThermodynamicModel::PengRobinsonThermodynamicModel(const string configuration_file) : BaseThermodynamicModel(configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile();

    /// Calculate value of selected variables
    eos_b  = 0.077796*( R_universal*critical_temperature/critical_pressure );
    eos_ac = 0.457236*( pow( R_universal*critical_temperature, 2.0 )/critical_pressure );
    if( acentric_factor > 0.49 ) {
    	eos_kappa = 0.379642 + 1.48503*acentric_factor - 0.164423*pow( acentric_factor, 2.0 ) + 0.016666*pow( acentric_factor, 3.0 );
    } else {
	eos_kappa = 0.37464 + 1.54226*acentric_factor - 0.26992*pow( acentric_factor, 2.0 );
    }

    #pragma acc enter data copyin(this) 
};

PengRobinsonThermodynamicModel::~PengRobinsonThermodynamicModel() {

};

void PengRobinsonThermodynamicModel::readConfigurationFile() {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile(configuration_file);

    /// Fluid & flow properties
    const YAML::Node & fluid_flow_properties = configuration["fluid_flow_properties"];
    if( fluid_flow_properties["substance_name"] ) {

        /// Create YAML object
	string substance_name          = fluid_flow_properties["substance_name"].as<string>();
	string substances_library_file = fluid_flow_properties["substances_library_file"].as<string>();
        YAML::Node substances_library  = YAML::LoadFile( substances_library_file );
        YAML::Node substance;

        #include "substances_selection.txt"

        molecular_weight      = substance["molecular_weight"].as<double>();
        acentric_factor       = substance["acentric_factor"].as<double>();
        critical_temperature  = substance["critical_temperature"].as<double>();
        critical_pressure     = substance["critical_pressure"].as<double>();
        critical_molar_volume = substance["critical_molar_volume"].as<double>();
        NASA_coefficients[0]  = substance["NASA_coefficients"][0].as<double>();
        NASA_coefficients[1]  = substance["NASA_coefficients"][1].as<double>();
        NASA_coefficients[2]  = substance["NASA_coefficients"][2].as<double>();
        NASA_coefficients[3]  = substance["NASA_coefficients"][3].as<double>();
        NASA_coefficients[4]  = substance["NASA_coefficients"][4].as<double>();
        NASA_coefficients[5]  = substance["NASA_coefficients"][5].as<double>();
        NASA_coefficients[6]  = substance["NASA_coefficients"][6].as<double>();
        NASA_coefficients[7]  = substance["NASA_coefficients"][7].as<double>();
        NASA_coefficients[8]  = substance["NASA_coefficients"][8].as<double>();
        NASA_coefficients[9]  = substance["NASA_coefficients"][9].as<double>();
        NASA_coefficients[10] = substance["NASA_coefficients"][10].as<double>();
        NASA_coefficients[11] = substance["NASA_coefficients"][11].as<double>();
        NASA_coefficients[12] = substance["NASA_coefficients"][12].as<double>();
        NASA_coefficients[13] = substance["NASA_coefficients"][13].as<double>();
        NASA_coefficients[14] = substance["NASA_coefficients"][14].as<double>();

    } else {

        molecular_weight      = fluid_flow_properties["molecular_weight"].as<double>();
        acentric_factor       = fluid_flow_properties["acentric_factor"].as<double>();
        critical_temperature  = fluid_flow_properties["critical_temperature"].as<double>();
        critical_pressure     = fluid_flow_properties["critical_pressure"].as<double>();
        critical_molar_volume = fluid_flow_properties["critical_molar_volume"].as<double>();
        NASA_coefficients[0]  = fluid_flow_properties["NASA_coefficients"][0].as<double>();
        NASA_coefficients[1]  = fluid_flow_properties["NASA_coefficients"][1].as<double>();
        NASA_coefficients[2]  = fluid_flow_properties["NASA_coefficients"][2].as<double>();
        NASA_coefficients[3]  = fluid_flow_properties["NASA_coefficients"][3].as<double>();
        NASA_coefficients[4]  = fluid_flow_properties["NASA_coefficients"][4].as<double>();
        NASA_coefficients[5]  = fluid_flow_properties["NASA_coefficients"][5].as<double>();
        NASA_coefficients[6]  = fluid_flow_properties["NASA_coefficients"][6].as<double>();
        NASA_coefficients[7]  = fluid_flow_properties["NASA_coefficients"][7].as<double>();
        NASA_coefficients[8]  = fluid_flow_properties["NASA_coefficients"][8].as<double>();
        NASA_coefficients[9]  = fluid_flow_properties["NASA_coefficients"][9].as<double>();
        NASA_coefficients[10] = fluid_flow_properties["NASA_coefficients"][10].as<double>();
        NASA_coefficients[11] = fluid_flow_properties["NASA_coefficients"][11].as<double>();
        NASA_coefficients[12] = fluid_flow_properties["NASA_coefficients"][12].as<double>();
        NASA_coefficients[13] = fluid_flow_properties["NASA_coefficients"][13].as<double>();
        NASA_coefficients[14] = fluid_flow_properties["NASA_coefficients"][14].as<double>();

    }

};

double PengRobinsonThermodynamicModel::calculateTemperatureFromPressureDensity(const double &P, const double &rho) {

    /// Numerical Recipes in C++, Second Edition.
    /// W.H. Press, S.A. Teulosky, W.T. Vetterling, B.P. Flannery.
    /// 5.1 Series and Their Convergence: Aitken’s delta-squared process.

    // Calculate molar volume
    double bar_v = molecular_weight/rho;

    // Initial temperature guess using ideal-gas model
    double T = P*bar_v/R_universal;

    /// Aitken’s delta-squared process:
    double x_0 = T, x_1, x_2, denominator;
    for(int iter = 0; iter < aitken_max_iter; iter++) { 
        x_1 = ( (bar_v - eos_b )/R_universal )*( P + ( this->calculate_eos_a( x_0 )/( pow( bar_v, 2.0 ) + 2.0*eos_b*bar_v - pow( eos_b, 2.0 ) ) ) );
        x_2 = ( (bar_v - eos_b )/R_universal )*( P + ( this->calculate_eos_a( x_1 )/( pow( bar_v, 2.0 ) + 2.0*eos_b*bar_v - pow( eos_b, 2.0 ) ) ) );

        denominator = x_2 - 2.0*x_1 + x_0;
        //T = x_2 - ( pow( x_2 - x_1, 2.0 ) )/denominator;
        T = x_2 - ( pow( x_2 - x_1, 2.0 ) )/( denominator + 1.0e-10 );
    
        if( abs( ( T - x_2 )/ T ) < aitken_relative_tolerance ) break;	/// If the result is within tolerance, leave the loop!
        x_0 = T;							/// Otherwise, update x_0 to iterate again ...                 
    }

    return( T );

};

void PengRobinsonThermodynamicModel::calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {

    /// Numerical Recipes in C++, Second Edition.
    /// W.H. Press, S.A. Teulosky, W.T. Vetterling, B.P. Flannery.
    /// 5.1 Series and Their Convergence: Aitken’s delta-squared process.

    // Calculate molar volume
    double bar_v = molecular_weight/rho;

    /// Aitken’s delta-squared process:
    double x_0 = T, x_1, x_2, denominator;
    for(int iter = 0; iter < aitken_max_iter; iter++) { 
        x_1 = ( (bar_v - eos_b )/R_universal )*( P + ( this->calculate_eos_a( x_0 )/( pow( bar_v, 2.0 ) + 2.0*eos_b*bar_v - pow( eos_b, 2.0 ) ) ) );
        x_2 = ( (bar_v - eos_b )/R_universal )*( P + ( this->calculate_eos_a( x_1 )/( pow( bar_v, 2.0 ) + 2.0*eos_b*bar_v - pow( eos_b, 2.0 ) ) ) );

        denominator = x_2 - 2.0*x_1 + x_0;
        //T = x_2 - ( pow( x_2 - x_1, 2.0 ) )/( denominator + 1.0e-15 );
        if( abs( denominator/x_0 ) < 1.0e-8 ) {
	    //cout << denominator << "  " << x_0 << "  " << denominator/x_0 << endl;
	    break;
	}
        T = x_2 - ( pow( x_2 - x_1, 2.0 ) )/denominator;

        if( abs( ( T - x_2 )/ T ) < aitken_relative_tolerance ) break;	/// If the result is within tolerance, leave the loop!
        x_0 = T;							/// Otherwise, update x_0 to iterate again ...                 
    }

};

double PengRobinsonThermodynamicModel::calculatePressureFromTemperatureDensity(const double &T, const double &rho) {

    double bar_v = molecular_weight/rho;

    double P = R_universal*T/( bar_v - eos_b ) - this->calculate_eos_a( T )/( bar_v*bar_v + 2.0*eos_b*bar_v - eos_b*eos_b );

    return( P );

};

double PengRobinsonThermodynamicModel::calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double bar_v = molecular_weight/rho;
    double e     = ( 1.0/molecular_weight )*this->calculateMolarInternalEnergyFromPressureTemperatureMolarVolume( P, T, bar_v );

    return( e );

};

double PengRobinsonThermodynamicModel::calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double bar_v = molecular_weight/rho;
    double s     = ( 1.0/molecular_weight )*this->calculateMolarEntropyFromPressureTemperatureMolarVolume( P, T, bar_v );

    return( s );

};

void PengRobinsonThermodynamicModel::calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {

    //double P_norm   = this->critical_pressure;	/// Set pressure normalization factor
    double P_norm    = fabs( P ) + 1.0e-14;		/// Set pressure normalization factor
    //double T_norm   = this->critical_temperature;	/// Set temperature normalization factor
    double T_norm    = fabs( T ) + 1.0e-14;		/// Set temperature normalization factor
    double nls_f     = -1.0;				/// Nonlinear solver residual value
    int nls_num_iter = 0;				/// Number of iterations required to obtain the solution

    /// Calculate P & T from rho & e by means of a nonlinear solver
    double nls_P_unknown = P/P_norm;
    double nls_T_unknown = T/T_norm;
    this->solve2D( nls_f, nls_P_unknown, nls_T_unknown, P_norm, T_norm, rho, e, nls_PT_max_iter, nls_num_iter, nls_PT_relative_tolerance, nls_PT_STPMX );	/// Nonlinear solver
    P = nls_P_unknown*P_norm;										/// Update P & T from nonlinear solver (unnormalized)
    T = nls_T_unknown*T_norm;

};

double PengRobinsonThermodynamicModel::calculateDensityFromPressureTemperature(const double &P, const double &T) {

    /// Auxiliar parameters
    double eos_a  = this->calculate_eos_a( T );
    double eos_en = P*eos_b + R_universal*T;
    double a      = P;
    double b      = P*2.0*eos_b - eos_en;
    double c      = P*( -1.0 )*eos_b*eos_b - eos_en*2.0*eos_b + eos_a;
    double d      = ( -1.0 )*eos_b*( eos_a + eos_en*( -1.0 )*eos_b );

    // Cubic solve to calculate rho
     
    //complex<double> v_1, v_2, v_3;		// ... modified for OpenACC
    //this->calculateRootsCubicPolynomial( v_1, v_2, v_3, a, b, c, d );
    //double rho = molecular_weight/real( v_1 );		/// First root is always real

    double root_1_real, root_2_real, root_3_real, root_2_imag, root_3_imag;
    this->calculateRootsCubicPolynomial_( root_1_real, root_2_real, root_3_real, root_2_imag, root_3_imag, a, b, c, d );
    double rho = molecular_weight/root_1_real;		/// First root is always real

    return( rho );

};

void PengRobinsonThermodynamicModel::calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {

    /// Auxiliar parameters
    double eos_a  = this->calculate_eos_a( T );
    double eos_en = P*eos_b + R_universal*T;
    double a      = P;
    double b      = P*2.0*eos_b - eos_en;
    double c      = P*( -1.0 )*eos_b*eos_b - eos_en*2.0*eos_b + eos_a;
    double d      = ( -1.0 )*eos_b*( eos_a + eos_en*( -1.0 )*eos_b );
    
    /// Cubic solve to calculate rho  
    //complex<double> v_1, v_2, v_3;		// ... modified for OpenACC
    //this->calculateRootsCubicPolynomial( v_1, v_2, v_3, a, b, c, d );
    //rho = molecular_weight/real( v_1 );		/// First root is always real

    double root_1_real, root_2_real, root_3_real, root_2_imag, root_3_imag;
    this->calculateRootsCubicPolynomial_( root_1_real, root_2_real, root_3_real, root_2_imag, root_3_imag, a, b, c, d );
    rho = molecular_weight/root_1_real;		/// First root is always real
    
    /// Calculate e
    double bar_v = molecular_weight/rho;
    e            = ( 1.0/molecular_weight )*this->calculateMolarInternalEnergyFromPressureTemperatureMolarVolume( P, T, bar_v );

};

void PengRobinsonThermodynamicModel::calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho) {
    
    double bar_v       = molecular_weight/rho;
    double std_bar_c_p = this->calculateMolarStdCpFromNASApolynomials( T );
    double std_bar_c_v = std_bar_c_p - R_universal;

    c_v = ( 1.0/molecular_weight )*( std_bar_c_v + this->calculateDepartureFunctionMolarCv( P, T, bar_v ) );
    c_p = ( 1.0/molecular_weight )*( std_bar_c_p + this->calculateDepartureFunctionMolarCp( P, T, bar_v ) );

};

double PengRobinsonThermodynamicModel::calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) {

    double bar_v       = molecular_weight/rho;
    double std_bar_c_p = this->calculateMolarStdCpFromNASApolynomials( T );
    double std_bar_c_v = std_bar_c_p - R_universal;

    double c_v = ( 1.0/molecular_weight )*( std_bar_c_v + this->calculateDepartureFunctionMolarCv( P, T, bar_v ) );
    double c_p = ( 1.0/molecular_weight )*( std_bar_c_p + this->calculateDepartureFunctionMolarCp( P, T, bar_v ) );

    double gamma = c_p/c_v;

    return( gamma );

};

double PengRobinsonThermodynamicModel::calculateSoundSpeed(const double &P, const double &T, const double &rho) {

    double sos = sqrt( 1.0/( rho*this->calculateIsentropicCompressibility( P, T, rho ) ) );

    return( sos );

};

double PengRobinsonThermodynamicModel::calculateVolumeExpansivity(const double &P, const double &T, const double &rho) {

    double bar_v = molecular_weight/rho;
    double dP_dT_const_v = this->calculateDPDTConstantMolarVolume( T, bar_v );
    double dP_dv_const_T = this->calculateDPDvConstantTemperature( T, bar_v );

    double expansivity = ( -1.0 )*( dP_dT_const_v/( bar_v*dP_dv_const_T ) );

    return( expansivity );
  
};

double PengRobinsonThermodynamicModel::calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) {

    double bar_v = molecular_weight/rho;
    double dP_dv_const_T = this->calculateDPDvConstantTemperature( T, bar_v );

    double isothermal_compressibility = ( -1.0 )/( bar_v*dP_dv_const_T );

    return( isothermal_compressibility );
  
};  

double PengRobinsonThermodynamicModel::calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) {

    double bar_v = molecular_weight/rho;
    double isothermal_compressibility = this->calculateIsothermalCompressibility( P, T, rho );
    double expansivity                = this->calculateVolumeExpansivity( P, T, rho );
    double bar_c_p                    = this->calculateMolarStdCpFromNASApolynomials( T ) + this->calculateDepartureFunctionMolarCp( P, T, bar_v );
      
    double isentropic_compressibility = ( isothermal_compressibility - ( ( bar_v*T*pow( expansivity, 2.0 ) )/bar_c_p ) );
    
    return( isentropic_compressibility );
  
};

double PengRobinsonThermodynamicModel::calculateMolarInternalEnergyFromPressureTemperatureMolarVolume(const double &P, const double &T, const double &bar_v) {

    double bar_e = this->calculateMolarStdEnthalpyFromNASApolynomials( T ) + this->calculateDepartureFunctionMolarEnthalpy( P, T, bar_v ) - P*bar_v;

    return( bar_e );

};

double PengRobinsonThermodynamicModel::calculateMolarEntropyFromPressureTemperatureMolarVolume(const double &P, const double &T, const double &bar_v) {

    double bar_s = this->calculateMolarStdEntropyFromNASApolynomials( T ) + this->calculateDepartureFunctionMolarEntropy( P, T, bar_v );

    return( bar_s );

};

double PengRobinsonThermodynamicModel::calculate_eos_a(const double &T) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a = 0.457236*( pow( R_universal*critical_temperature, 2.0 )/critical_pressure )*pow( 1.0 + eos_kappa*( 1.0 - sqrt( T/critical_temperature ) ), 2.0 );

    return( eos_a );

};

double PengRobinsonThermodynamicModel::calculate_eos_a_first_derivative(const double &T) {
   
    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.
 
    double eos_a_first_derivative = eos_kappa*eos_ac*( ( eos_kappa/critical_temperature ) - ( ( 1.0 + eos_kappa )/sqrt( T*critical_temperature ) ) );

    return( eos_a_first_derivative );

};

double PengRobinsonThermodynamicModel::calculate_eos_a_second_derivative(const double &T) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a_second_derivative = ( eos_kappa*eos_ac*( 1.0 + eos_kappa ) )/( 2.0*sqrt( pow( T, 3.0 )*critical_temperature ) );

    return( eos_a_second_derivative );

};

double PengRobinsonThermodynamicModel::calculate_Z(const double &P, const double &T, const double &bar_v) {

    double Z = ( P*bar_v )/( R_universal*T );

    return( Z );

};

double PengRobinsonThermodynamicModel::calculate_A(const double &P, const double &T) {

    double eos_a = calculate_eos_a( T );

    double A = ( eos_a*P )/pow( R_universal*T, 2.0 );

    return( A );

};

double PengRobinsonThermodynamicModel::calculate_B(const double &P, const double &T) {

    double B = ( eos_b*P )/( R_universal*T );

    return( B );

};

double PengRobinsonThermodynamicModel::calculate_M(const double &Z, const double &B) {

    double M = ( Z*Z + 2.0*B*Z - B*B )/( Z - B );

    return( M );

};

double PengRobinsonThermodynamicModel::calculate_N(const double &eos_a_first_derivative, const double &B) {

    double N = eos_a_first_derivative*( B/( eos_b*R_universal ) );

    return( N );

};

double PengRobinsonThermodynamicModel::calculateMolarStdCpFromNASApolynomials(const double &T) {

    double std_bar_c_p = 0.0;

    if( ( T >= 200.0 ) && ( T < 1000.0 ) ) {
        std_bar_c_p = R_universal*( NASA_coefficients[7] + NASA_coefficients[8]*T + NASA_coefficients[9]*pow( T, 2.0 ) + NASA_coefficients[10]*pow( T, 3.0 ) + NASA_coefficients[11]*pow( T, 4.0 ) );
    } else if( ( T >= 1000.0 ) && ( T < 6000.0 ) ) {
        std_bar_c_p = R_universal*( NASA_coefficients[0] + NASA_coefficients[1]*T + NASA_coefficients[2]*pow( T, 2.0 ) + NASA_coefficients[3]*pow( T, 3.0 ) + NASA_coefficients[4]*pow( T, 4.0 ) );
    } else if ( T < 200 ) {
        // Assume constant temperature below T = 200 K	    
        double T_min = 200.0;	    
        
        std_bar_c_p = R_universal*( NASA_coefficients[7] + NASA_coefficients[8]*T_min + NASA_coefficients[9]*pow( T_min, 2.0 ) + NASA_coefficients[10]*pow( T_min, 3.0 ) + NASA_coefficients[11]*pow( T_min, 4.0 ) );
    } else {
#if _ACTIVATE_COUT_
        cout << endl << "NASA 7-coefficient polynomials for std bar c_p. T = " << T << " is above 6000 K." << endl << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
#endif
        int* trash = nullptr; *trash = 42;	// Causes segmentation fault
    }

    return( std_bar_c_p );

};

double PengRobinsonThermodynamicModel::calculateMolarStdEnthalpyFromNASApolynomials(const double &T) {

    double std_bar_h = 0.0;

    if( (T >= 200.0 ) && ( T < 1000.0 ) ) {
	//std_bar_h = R_universal*T*( NASA_coefficients[7] + NASA_coefficients[8]*T/2.0 + NASA_coefficients[9]*pow( T, 2.0 )/3.0 + NASA_coefficients[10]*pow( T, 3.0 )/4.0 + NASA_coefficients[11]*pow( T, 4.0 )/5.0 + NASA_coefficients[12]/T) - R_universal*NASA_coefficients[14];
	std_bar_h = R_universal*T*( NASA_coefficients[7] + NASA_coefficients[8]*T/2.0 + NASA_coefficients[9]*pow( T, 2.0 )/3.0 + NASA_coefficients[10]*pow( T, 3.0 )/4.0 + NASA_coefficients[11]*pow( T, 4.0 )/5.0 + NASA_coefficients[12]/T );
    } else if( ( T >= 1000.0 ) && ( T < 6000.0 ) ) {
	//std_bar_h = R_universal*T*( NASA_coefficients[0] + NASA_coefficients[1]*T/2.0 + NASA_coefficients[2]*pow( T, 2.0 )/3.0 + NASA_coefficients[3]*pow( T, 3.0 )/4.0 + NASA_coefficients[4]*pow( T, 4.0 )/5.0 + NASA_coefficients[5]/T ) - R_universal*NASA_coefficients[14];
	std_bar_h = R_universal*T*( NASA_coefficients[0] + NASA_coefficients[1]*T/2.0 + NASA_coefficients[2]*pow( T, 2.0 )/3.0 + NASA_coefficients[3]*pow( T, 3.0 )/4.0 + NASA_coefficients[4]*pow( T, 4.0 )/5.0 + NASA_coefficients[5]/T );
    } else if( T < 200.0 ) {
	// Assume linear interpolation from T = 200 K 
        double T_min = 200.0;
	    
	//double std_bar_h_min   = R_universal*T_min*( NASA_coefficients[7] + NASA_coefficients[8]*T_min/2.0 + NASA_coefficients[9]*pow( T_min, 2.0 )/3.0 + NASA_coefficients[10]*pow( T_min, 3.0 )/4.0 + NASA_coefficients[11]*pow( T_min, 4.0 )/5.0 + NASA_coefficients[12]/T_min ) - R_universal*NASA_coefficients[14];
	double std_bar_h_min   = R_universal*T_min*( NASA_coefficients[7] + NASA_coefficients[8]*T_min/2.0 + NASA_coefficients[9]*pow( T_min, 2.0 )/3.0 + NASA_coefficients[10]*pow( T_min, 3.0 )/4.0 + NASA_coefficients[11]*pow( T_min, 4.0 )/5.0 + NASA_coefficients[12]/T_min );
	double std_bar_h_slope = R_universal*( NASA_coefficients[7] + NASA_coefficients[8]*T_min + NASA_coefficients[9]*pow( T_min, 2.0 ) + NASA_coefficients[10]*pow( T_min, 3.0 ) + NASA_coefficients[11]*pow( T_min, 4.0 ) );

	std_bar_h = std_bar_h_min + std_bar_h_slope*( T - T_min );
    } else {
#if _ACTIVATE_COUT_
	cout << endl << "NASA 7-coefficient polynomials for std bar h. T = " << T << " is above 6000 K." << endl << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );										
#endif
        int* trash = nullptr; *trash = 42;	// Causes segmentation fault
    }

    return( std_bar_h );

};

double PengRobinsonThermodynamicModel::calculateMolarStdEntropyFromNASApolynomials(const double &T) {

    double std_bar_s = 0.0;

    if( (T >= 200.0 ) && ( T < 1000.0 ) ) {
	std_bar_s = R_universal*( NASA_coefficients[7]*log(T) + NASA_coefficients[8]*T + NASA_coefficients[9]*pow( T, 2.0 )/2.0 + NASA_coefficients[10]*pow( T, 3.0 )/3.0 + NASA_coefficients[11]*pow( T, 4.0 )/4.0 + NASA_coefficients[13] );
    } else if( ( T >= 1000.0 ) && ( T < 6000.0 ) ) {
	std_bar_s = R_universal*( NASA_coefficients[0]*log(T) + NASA_coefficients[1]*T + NASA_coefficients[2]*pow( T, 2.0 )/2.0 + NASA_coefficients[3]*pow( T, 3.0 )/3.0 + NASA_coefficients[4]*pow( T, 4.0 )/5.0 + NASA_coefficients[6] );
    } else if( T < 200.0 ) {
	// Assume linear interpolation from T = 200 K 
        double T_min = 200.0;
	    
	double std_bar_s_min   = R_universal*( NASA_coefficients[7]*log(T_min) + NASA_coefficients[8]*T_min + NASA_coefficients[9]*pow( T_min, 2.0 )/2.0 + NASA_coefficients[10]*pow( T_min, 3.0 )/3.0 + NASA_coefficients[11]*pow( T_min, 4.0 )/4.0 + NASA_coefficients[13] );
	double std_bar_s_slope = R_universal*( NASA_coefficients[7]/T_min + NASA_coefficients[8] + NASA_coefficients[9]*T_min + NASA_coefficients[10]*pow( T_min, 2.0 ) + NASA_coefficients[11]*pow( T_min, 3.0 ) );

	std_bar_s = std_bar_s_min + std_bar_s_slope*( T - T_min );
    } else {
#if _ACTIVATE_COUT_
	cout << endl << "NASA 7-coefficient polynomials for std bar s. T = " << T << " is above 6000 K." << endl << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );										
#endif
        int* trash = nullptr; *trash = 42;	// Causes segmentation fault
    }

    return( std_bar_s );

};

double PengRobinsonThermodynamicModel::calculateDepartureFunctionMolarCp(const double &P, const double &T, const double &bar_v) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a_first_derivative  = this->calculate_eos_a_first_derivative( T );
    double eos_a_second_derivative = this->calculate_eos_a_second_derivative( T );
    double Z                       = this->calculate_Z( P, T, bar_v );
    double A                       = this->calculate_A( P, T );
    double B                       = this->calculate_B( P, T );
    double M                       = this->calculate_M( Z, B );
    double N                       = this->calculate_N( eos_a_first_derivative, B );

    double Delta_bar_c_p = ( ( R_universal*pow( M - N ,2.0 ) )/( pow( M, 2.0 ) - 2.0*A*( Z + B ) ) ) - ( ( T*eos_a_second_derivative )/( 2.0*sqrt(2.0)*eos_b ) )*log( ( Z + ( 1.0 - sqrt( 2.0 ) )*B )/( Z + ( 1.0 + sqrt( 2.0 ) )*B ) ) - R_universal; 

    return( Delta_bar_c_p );
  
};

double PengRobinsonThermodynamicModel::calculateDepartureFunctionMolarCv(const double &P, const double &T, const double &bar_v) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a_second_derivative = this->calculate_eos_a_second_derivative( T );
    double Z                       = this->calculate_Z( P, T, bar_v );
    double B                       = this->calculate_B( P, T );

    double Delta_bar_c_v = ( -1.0 )*( ( T*eos_a_second_derivative )/( 2.0*sqrt( 2.0 )*eos_b ) )*log( ( Z + ( 1.0 - sqrt( 2.0 ) )*B )/( Z + ( 1.0 + sqrt( 2.0 ) )*B ) );

    return( Delta_bar_c_v );
  
  };

double PengRobinsonThermodynamicModel::calculateDepartureFunctionMolarEnthalpy(const double &P, const double &T, const double &bar_v) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a                  = this->calculate_eos_a( T );
    double eos_a_first_derivative = this->calculate_eos_a_first_derivative( T );
    double Z                      = this->calculate_Z( P, T, bar_v );
    double B                      = this->calculate_B( P, T );

    double Delta_bar_h = R_universal*T*( Z - 1.0 ) + ( ( eos_a - eos_a_first_derivative*T )/( 2.0*sqrt( 2.0 )*eos_b ) )*log( ( Z + ( 1.0 - sqrt( 2.0 ) )*B )/( Z + ( 1.0 + sqrt( 2.0 ) )*B ) );

    return( Delta_bar_h );
  
};

double PengRobinsonThermodynamicModel::calculateDepartureFunctionMolarEntropy(const double &P, const double &T, const double &bar_v) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double Z                      = this->calculate_Z( P, T, bar_v );
    double A                      = this->calculate_A( P, T );
    double B                      = this->calculate_B( P, T );

    double Delta_bar_s = R_universal*( log( Z - B ) + ( A/( 2.0*sqrt( 2.0 )*B ) )*( eos_kappa*sqrt( T/critical_temperature )/( 1.0 + eos_kappa*( 1.0 - sqrt( T/critical_temperature ) ) ) )*log( ( Z + ( 1.0 - sqrt( 2.0 ) )*B )/( Z + ( 1.0 + sqrt( 2.0 ) )*B ) ) );

    return( Delta_bar_s );
  
};

double PengRobinsonThermodynamicModel::calculateTemperatureFromPressureMolarVolume(const double &P, const double &bar_v) {

    /// Numerical Recipes in C++, Second Edition.
    /// W.H. Press, S.A. Teulosky, W.T. Vetterling, B.P. Flannery.
    /// 5.1 Series and Their Convergence: Aitken’s delta-squared process.

    // Initial temperature guess using ideal-gas model
    double T = P*bar_v/R_universal;

    /// Aitken’s delta-squared process:
    double x_0 = T, x_1, x_2, denominator;
    for(int iter = 0; iter < aitken_max_iter; iter++) { 
        x_1 = ( (bar_v - eos_b )/R_universal )*( P + ( this->calculate_eos_a( x_0 )/( pow( bar_v, 2.0 ) + 2.0*eos_b*bar_v - pow( eos_b, 2.0 ) ) ) );
        x_2 = ( (bar_v - eos_b )/R_universal )*( P + ( this->calculate_eos_a( x_1 )/( pow( bar_v, 2.0 ) + 2.0*eos_b*bar_v - pow( eos_b, 2.0 ) ) ) );

        denominator = x_2 - 2.0*x_1 + x_0;
        T = x_2 - ( pow( x_2 - x_1, 2.0 ) )/denominator;
    
        if( abs( ( T - x_2 )/ T ) < aitken_relative_tolerance ) break;	/// If the result is within tolerance, leave the loop!
        x_0 = T;							/// Otherwise, update x_0 to iterate again ...                 
    }

    return( T );

};

double PengRobinsonThermodynamicModel::calculateDPDTConstantMolarVolume(const double &T, const double &bar_v) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a_first_derivative = this->calculate_eos_a_first_derivative( T );

    double dP_dT_const_v = ( R_universal/( bar_v - eos_b ) ) - ( eos_a_first_derivative/( bar_v*bar_v + 2.0*bar_v*eos_b - eos_b*eos_b ) );

    return( dP_dT_const_v );
  
};

double PengRobinsonThermodynamicModel::calculateDPDvConstantTemperature(const double &T, const double &bar_v) {

    /// Peng-Robinson model:
    /// D. Y. Peng, D. B. Robinson.
    /// A new two-constant equation of state.
    /// Industrial and Engineering Chemistry: Fundamentals, 15, 59-64, 1976.

    double eos_a = this->calculate_eos_a( T );
    
    double dP_dv_const_T = ( -1.0 )*( ( R_universal*T )/pow( bar_v - eos_b, 2.0 ) ) + ( eos_a*( 2.0*bar_v + 2.0*eos_b ) )/pow( pow( bar_v, 2.0 ) + 2.0*bar_v*eos_b - pow( eos_b, 2.0 ), 2.0 );

    return( dP_dv_const_T );
  
};  

/*void PengRobinsonThermodynamicModel::calculateRootsCubicPolynomial(complex<double> &root_1, complex<double> &root_2, complex<double> &root_3, double &a, double &b, double &c, double &d) {

    if( a == 0 ) {	/// End if a == 0
#if _ACTIVATE_COUT_
        cout << "The coefficient of the cube of x is 0. Please use the utility for a SECOND degree quadratic. No further action taken." << endl;
#endif
        return;
    }
    if( d == 0 ) {	/// End if d == 0
#if _ACTIVATE_COUT_
        cout << "One root is 0. Now divide through by x and use the utility for a SECOND degree quadratic to solve the resulting equation for the other two roots. No further action taken." << endl;
#endif
        return;
    }

    b /= a;
    c /= a;
    d /= a;

    double disc, q, r, dum1, s, t, term1, r13;
    q = (3.0*c - (b*b))/9.0;
    r = -(27.0*d) + b*(9.0*c - 2.0*(b*b));
    r /= 54.0;
    disc = q*q*q + r*r;

    root_1.imag( 0 );	/// The first root is always real
    term1 = (b/3.0);

    if( disc > 0 ) {	/// One root real, two are complex
        s = r + sqrt(disc);
        s = ((s < 0) ? -pow(-s, (1.0/3.0)) : pow(s, (1.0/3.0)));
        t = r - sqrt(disc);
        t = ((t < 0) ? -pow(-t, (1.0/3.0)) : pow(t, (1.0/3.0)));
        root_1.real( -term1 + s + t );
        term1 += (s + t)/2.0;
        root_2.real( -term1 );
        root_3.real( -term1 );
        term1 = sqrt(3.0)*(-t + s)/2;
        root_2.imag( term1 );
        root_3.imag( -term1 );
        return;
    }	/// End if (disc > 0)

#if _ACTIVATE_COUT_
    cout << endl;
    cout << "The PengRobinsonThermodynamicModel::calculateRootsCubicPolynomial has found more than one real root." << endl;
    cout << "Vapor-liquid equilibrium conditions must be solved." << endl;
    cout << "Another option is to avoid calculating rho from P and T." << endl;
    cout << endl;
#endif

    /// The remaining options are all real
    root_2.imag( 0 );
    root_3.imag( 0 );
    if(disc == 0) {	/// All roots real, at least two are equal
        r13 = ((r < 0) ? -pow(-r,(1.0/3.0)) : pow(r,(1.0/3.0)));
        root_1.real( -term1 + 2.0*r13 );
        root_2.real( -(r13 + term1) );
        root_3.real( -(r13 + term1) );
        return;
    }	/// End if (disc == 0)

    /// Only option left is that all roots are real and unequal (to get here, q < 0)
    q = -q;
    dum1 = q*q*q;
    dum1 = acos(r/sqrt(dum1));
    r13 = 2.0*sqrt(q);
    root_1.real( -term1 + r13*cos(dum1/3.0) );
    root_2.real( -term1 + r13*cos((dum1 + 2.0*3.141592654)/3.0) );
    root_3.real( -term1 + r13*cos((dum1 + 4.0*3.141592654)/3.0) );

    return;

};*/


void PengRobinsonThermodynamicModel::calculateRootsCubicPolynomial_(double &root_1_real, double &root_2_real, double &root_3_real, double &root_2_imag, double &root_3_imag, double &a, double &b, double &c, double &d) {

    if (a == 0.0) {
        // Coefficient for x^3 is zero. No further action.
        return;
    }
    if (d == 0.0) {
        // One root is zero. No further action.
        return;
    }

    // Normalize coefficients
    b /= a;
    c /= a;
    d /= a;

    // Intermediate calculations
    double q = (3.0 * c - (b * b)) / 9.0;
    double r = -(27.0 * d) + b * (9.0 * c - 2.0 * (b * b));
    r /= 54.0;
    double disc = q * q * q + r * r;

    // First root is always real
    root_1_real = 0.0;
    root_2_imag = 0.0;
    root_3_imag = 0.0;
    double term1 = b / 3.0;

    if (disc > 0.0) {
        // One real root, two complex roots
        double s = r + sqrt(disc);
        s = (s < 0.0) ? -cbrt(-s) : cbrt(s);
        double t = r - sqrt(disc);
        t = (t < 0.0) ? -cbrt(-t) : cbrt(t);

        root_1_real = -term1 + s + t;
        term1 += (s + t) / 2.0;
        root_2_real = -term1;
        root_3_real = -term1;

        double imag_part = sqrt(3.0) * (s - t) / 2.0;
        root_2_imag = imag_part;
        root_3_imag = -imag_part;

        return;
    }

    // All roots are real
    if (disc == 0.0) {
        // Triple or double roots
        double r13 = (r < 0.0) ? -cbrt(-r) : cbrt(r);
        root_1_real = -term1 + 2.0 * r13;
        root_2_real = -(r13 + term1);
        root_3_real = -(r13 + term1);

        return;
    }

    // All roots are real and unequal
    double q_abs = -q;
    double dum1 = q_abs * q_abs * q_abs;
    double theta = acos(r / sqrt(dum1));
    double r13 = 2.0 * sqrt(q_abs);

    root_1_real = -term1 + r13 * cos(theta / 3.0);
    root_2_real = -term1 + r13 * cos((theta + 2.0 * 3.141592654) / 3.0);
    root_3_real = -term1 + r13 * cos((theta + 4.0 * 3.141592654) / 3.0);

    return;

};

/// Evaluates the functions (residuals) in position x
void PengRobinsonThermodynamicModel::function_vector2D(double &xmin_1, double &xmin_2, double &fx_1, double &fx_2, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm) {

    /// Set state to x
    double P = xmin_1*P_norm;		// Unnormalize pressure
    double T = xmin_2*T_norm;		// Unnormalize temperature
    	
    /// For single-component systems:
    /// P will not oscillate for supercritical thermodynamic states
    /// P will oscillate for subcritical thermodynamic states
    /// ... small oscillations close to the critical point (slightly subcritical)
    /// ... large oscillations far from the critical point (notably subcritical) -> in that case, use two-phase solver
    double molecular_weight = this->getMolecularWeight(); 
    double target_molar_v   = molecular_weight/target_rho;
    double guess_rho = this->calculateDensityFromPressureTemperature( P, T );
    double guess_e   = this->calculateMolarInternalEnergyFromPressureTemperatureMolarVolume( P, T, target_molar_v )/molecular_weight;

    /// Compute fx (residuals)
    fx_1 = ( guess_rho - target_rho )/( fabs( target_rho ) + 1.0e-14 );		/// function normalized
    fx_2 = ( guess_e - target_e )/( fabs( target_e ) + 1.0e-14 );		/// function normalized

};

/// Numerical recipes in C++ Jacobian method
void PengRobinsonThermodynamicModel::fdjac2D(double &xmin_1, double &xmin_2, double &df_11, double &df_12, double &df_21, double &df_22, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm, double &nls_P_r, double &nls_T_r) {

    /// Globally Convergent Methods for Nonlinear Systems of Equations: fdjac method
    /// W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
    /// Numerical recipes in C++.
    /// Cambridge University Press, 2001.

    //const double EPS = 1.0e-8;
    const double EPS = 1.0e-6;
    double h, temp;
    double f_1, f_2;				/// Residuals after perturbation

    /// First column
    temp = xmin_1;
    h = EPS*fabs( temp );
    if( h == 0.0 ) h = EPS;
    xmin_1 = temp + h;
    h = xmin_1 - temp;	
    function_vector2D(xmin_1, xmin_2, f_1, f_2, target_rho, target_e, P_norm, T_norm);
    xmin_1 = temp;
    df_11 = (f_1 - nls_P_r)/h;
    df_21 = (f_2 - nls_T_r)/h;
    
    /// Second column
    temp = xmin_2;
    h = EPS*fabs( temp );
    if( h == 0.0 ) h = EPS;
    xmin_2 = temp + h;
    h = xmin_2 - temp;
    function_vector2D(xmin_1, xmin_2, f_1, f_2, target_rho, target_e, P_norm, T_norm);
    xmin_2 = temp;
    df_12 = (f_1 - nls_P_r)/h;
    df_22 = (f_2 - nls_T_r)/h;

};

/// Numerical recipes in C++ fmin method
double PengRobinsonThermodynamicModel::fmin2D(double &xmin_1, double &xmin_2, double &nls_P_r, double &nls_T_r, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm) {

    /// Globally Convergent Methods for Nonlinear Systems of Equations: fmin method
    /// W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
    /// Numerical recipes in C++.
    /// Cambridge University Press, 2001.

    function_vector2D(xmin_1, xmin_2, nls_P_r, nls_T_r, target_rho, target_e, P_norm, T_norm);
    double sum = ( nls_P_r*nls_P_r + nls_T_r*nls_T_r );

    return( 0.5*sum );

};

/// Numerical recipes in C++ linear search
void PengRobinsonThermodynamicModel::lnsrch2D(double &xold_1, double &xold_2, double &fold, double &g_1, double &g_2, double &p_1, double &p_2, double &xmin_1, double &xmin_2, double &f, const double &target_rho, const double &target_e, const double &P_norm, const double &T_norm, double &nls_P_r, double &nls_T_r, double &stpmax, bool &check) {

    /// Globally Convergent Methods for Nonlinear Systems of Equations: linear search
    /// W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
    /// Numerical recipes in C++.
    /// Cambridge University Press, 2001.

    const double ALF = 1.0e-4, TOLX = std::numeric_limits<double>::epsilon();
    double a, alam, alam2 = 0.0, alamin, b, disc, f2 = 0.0;
    double rhs1, rhs2, slope = 0.0, sum = 0.0, temp, test, tmplam;

    check = false;
    sum = sqrt( p_1*p_1 + p_2*p_2 );
    if ( sum>stpmax ){
        p_1 *= stpmax/sum;
        p_2 *= stpmax/sum;
    }
    slope = g_1*p_1 + g_2*p_2;
    temp = fabs(p_1)/std::max( fabs(xold_1), 1.0 );
    test = temp;
    temp = fabs(p_2)/std::max( fabs(xold_2), 1.0 );
    if (temp > test) test = temp;
    alamin = TOLX/test;

    alam = 1.0;
    for(;;) {
        xmin_1 = xold_1 + alam*p_1;
        xmin_2 = xold_2 + alam*p_2;
        f = fmin2D( xmin_1, xmin_2, nls_P_r, nls_T_r, target_rho, target_e, P_norm, T_norm );
        if( alam < alamin ) {
            xmin_1 = xold_1;
            xmin_2 = xold_2;
            check = true;
            return;
        } else if( f <= fold + ALF*alam*slope ) {
            return;
        } else {
            if( alam == 1.0 ) {
                tmplam = -slope/( 2.0*( f - fold - slope ) );
            } else {
                rhs1 = f - fold - alam*slope;
                rhs2 = f2 - fold - alam2*slope;
                a = ( rhs1/( alam*alam ) - rhs2/( alam2*alam2 ) )/( alam - alam2 );
                b = ( -alam2*rhs1/( alam*alam ) + alam*rhs2/( alam2*alam2 ) )/( alam - alam2 );
                if( a == 0.0 ) {
                    tmplam = -slope/(2.0*b);
                } else {
                    disc = b*b - 3.0*a*slope;
        	    if( disc < 0.0 ) tmplam = 0.5*alam;
        	    else if( b <= 0.0 ) tmplam = ( -b + sqrt( disc ) )/( 3.0*a );
        	    else tmplam = -slope/( b + sqrt( disc ) );
                }
                if( tmplam > 0.5*alam ) tmplam = 0.5*alam;
            }
        }
        alam2 = alam;
        f2 = f;
        alam = std::max( tmplam, 0.1*alam );
    }

};

void PengRobinsonThermodynamicModel::ludcmp2D(double &a_11, double &a_12, double &a_21, double &a_22, int &indx_1, int &indx_2) {

    /// Globally Convergent Methods for Nonlinear Systems of Equations: ludcmp method
    /// W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
    /// Numerical recipes in C++.
    /// Cambridge University Press, 2001.

    /// ludcmp method expanded for n=2

    const double TINY=1.0e-20;
    int imax=0;
    double big,dum,sum,temp;
    double vv_1, vv_2;

    double d=1.0;
    
    big = std::max(fabs(a_11), fabs(a_12));
    vv_1 = 1.0 / big;
    big = std::max(fabs(a_21), fabs(a_22));
    vv_2 = 1.0 / big;

    // j = 0
    // i = 0 → nothing
    // i = 1
    sum = a_21;
    a_21 = sum;
    
    big = 0.0;
    // test row 0
    sum = a_11; // since no k loop
    if ((dum = vv_1 * fabs(sum)) >= big) {
        big = dum;
        imax = 0;
    }
    // test row 1
    sum = a_21; // since no k loop
    if ((dum = vv_2 * fabs(sum)) >= big) {
        big = dum;
        imax = 1;
    }
    
    if (imax != 0) {
        // swap row 0 and row 1
        temp = a_11; a_11 = a_21; a_21 = temp;
        temp = a_12; a_12 = a_22; a_22 = temp;
        d = -d;
        std::swap(vv_1, vv_2);
    }
    indx_1 = imax;
    if (a_11 == 0.0) a_11 = TINY;
    
    dum = 1.0 / a_11;
    a_21 *= dum;
    
    // j = 1
    // i = 0
    sum = a_12;
    a_12 = sum;
    // i = 1
    sum = a_22 - a_21 * a_12;
    a_22 = sum;
    
    big = 0.0;
    sum = a_22;
    if ((dum = vv_2 * fabs(sum)) >= big) {
        big = dum;
        imax = 1;
    } else {
        imax = 1; // always 1 since j=1
    }
    
    if (imax != 1) {
        // (won’t happen in 2x2 case, but kept for consistency)
        temp = a_12; a_12 = a_22; a_22 = temp;
        d = -d;
    }
    indx_2 = imax;
    if (a_22 == 0.0) a_22 = TINY;

};

void PengRobinsonThermodynamicModel::lubksb2D(double &fjac_11, double &fjac_12, double &fjac_21, double &fjac_22, int &indx_1, int &indx_2, double &p_1, double &p_2) {

    /// Globally Convergent Methods for Nonlinear Systems of Equations: lubksb method
    /// W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
    /// Numerical recipes in C++.
    /// Cambridge University Press, 2001.

    int ii=0,ip;
    double sum;

    ip = indx_1;
    if (ip == 0) {
        sum = p_1;
        p_1 = p_1;
    } else {
        sum = p_2;
        p_2 = p_1;
    }
    if (ii != 0) {
    } else if (sum != 0.0) {
        ii = 1;
    }
    p_1 = sum;
    ip = indx_2;
    if (ip == 1) {
        sum = p_2;
        p_2 = p_2;
    } else {
        sum = p_1;
        p_1 = p_2;
    }
    if (ii != 0) {
        for (int j = ii - 1; j < 1; j++) {
            sum -= fjac_21 * p_1;
        }
    } else if (sum != 0.0) {
        ii = 2;
    }
    p_2 = sum;
    
    // Back substitution
    // i = 1
    sum = p_2;
    p_2 = sum / fjac_22;
    sum = p_1;
    sum -= fjac_12 * p_2;
    p_1 = sum / fjac_11;

};

/// Runs the minimization algorithm, calculating the minimum (fxmin) and its independent variables (xmin)
void PengRobinsonThermodynamicModel::solve2D(double &fxmin, double &xmin_1, double &xmin_2, const double &P_norm, const double &T_norm, const double &target_rho, const double &target_e, const int &max_iter, int &iter, const double &tolerance, const double &MX_STP) {

    /// Newton-Raphson method using approximated derivatives:
    /// W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
    /// Numerical recipes in C++.
    /// Cambridge University Press, 2001.

    //const int MAXITS=200;
    const int MAXITS=max_iter;
    //const double TOLF=1.0e-8, TOLX=std::numeric_limits<double>::epsilon(), STPMX=100.0, TOLMIN=1.0e-12;
    const double TOLF=tolerance, TOLX=std::numeric_limits<double>::epsilon(), STPMX=MX_STP, TOLMIN=1.0e-12;
    bool check;
    double den,f,fold,stpmax,sum,temp,test;
    double nls_P_r,nls_T_r;		/// Residuals 

    int indx_1, indx_2;
    double g_1, g_2, p_1, p_2, xold_1, xold_2;
    double fjac_11 = 0.0;
    double fjac_12 = 0.0;
    double fjac_21 = 0.0;
    double fjac_22 = 0.0;

    f = fmin2D( xmin_1, xmin_2, nls_P_r, nls_T_r, target_rho, target_e, P_norm, T_norm );

    test = 0.0;
    if ( fabs(nls_P_r) > test ) fxmin = test = fabs(nls_P_r);
    if ( fabs(nls_T_r) > test ) fxmin = test = fabs(nls_T_r);
    if( test < 0.01*TOLF ) {
      check = false;
      return;
    }
    sum = ( xmin_1*xmin_1 + xmin_2*xmin_2 );
    stpmax = STPMX*std::max( sqrt( sum ), 2.0 );
    
    for(iter = 0; iter < MAXITS; iter++) {
      	
      fdjac2D(xmin_1, xmin_2, fjac_11, fjac_12, fjac_21, fjac_22, target_rho, target_e, P_norm, T_norm, nls_P_r, nls_T_r);
      
      g_1 = fjac_11*nls_P_r + fjac_21*nls_T_r; 
      g_2 = fjac_12*nls_P_r + fjac_22*nls_T_r; 
      
      xold_1 = xmin_1; xold_2 = xmin_2;
      fold = f;
      p_1 = - nls_P_r; p_2 = - nls_T_r;

      ludcmp2D(fjac_11, fjac_12, fjac_21, fjac_22, indx_1, indx_2);
      lubksb2D(fjac_11, fjac_12, fjac_21, fjac_22, indx_1, indx_2, p_1, p_2);
      lnsrch2D(xold_1,xold_2,fold,g_1,g_2,p_1,p_2,xmin_1,xmin_2,f,target_rho,target_e,P_norm,T_norm,nls_P_r,nls_T_r,stpmax,check);
      test = 0.0;
      if ( fabs(nls_P_r) > test ) { fxmin = test = fabs(nls_P_r); }
      if ( fabs(nls_T_r) > test ) { fxmin = test = fabs(nls_T_r); }
      if( test < TOLF ) {
         check = false;
#if _ACTIVATE_COUT_
         cout << "Newton-Raphson's minimization TOLF error: " << test << " ( iteration: " << iter << " )" << endl;
#endif
	   return;
	 }
	 if(check) {
	     test = 0.0;
	     den = std::max( f, 1.0 );
	     temp = fabs( g_1 )*std::max( fabs(xmin_1), 1.0 )/den;
	     if ( temp > test) test = temp;
	     temp = fabs( g_2 )*std::max( fabs(xmin_2), 1.0 )/den;
	     if ( temp > test) test = temp;
	     check = ( test < TOLMIN );
	     return;
	 }
	 test = 0.0;
	 temp = ( fabs( xmin_1 - xold_1 ) )/std::max( fabs(xmin_1), 1.0 );	
	 if( temp > test ) fxmin = test = temp;
	 temp = ( fabs( xmin_2 - xold_2 ) )/std::max( fabs(xmin_2), 1.0 );	
	 if( temp > test ) fxmin = test = temp;
	 if( test < TOLX ) {
#if _ACTIVATE_COUT_
	     cout << "Newton-Raphson's minimization TOLX error: " << test << " ( iteration: " << iter << " )" << endl;
#endif
             return;
         }
    }
#if _ACTIVATE_COUT_
    cout << "MAXITS exceeded in Newton-Raphson" << endl;
#endif

};

/*
////////// CoolPropThermodynamicModel CLASS //////////

CoolPropThermodynamicModel::CoolPropThermodynamicModel() : BaseThermodynamicModel() {};
        
CoolPropThermodynamicModel::CoolPropThermodynamicModel(const string configuration_file) : BaseThermodynamicModel(configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile();

};

CoolPropThermodynamicModel::~CoolPropThermodynamicModel() {

    /// Free cp_substance
    if( cp_substance != NULL ) free( cp_substance );

};

void CoolPropThermodynamicModel::readConfigurationFile() {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile(configuration_file);

    /// Fluid & flow properties
    const YAML::Node & fluid_flow_properties = configuration["fluid_flow_properties"];

    /// Create YAML object
    string substance_name = fluid_flow_properties["substance_name"].as<string>();

    /// Initialize CoolProp state
    cp_substance = new CoolProp::HelmholtzEOSBackend( substance_name );

    /// Extract molecular weight from CoolProp
    molecular_weight = cp_substance->molar_mass();
    
    /// Calculate specific gas constant
    R_specific = R_universal/molecular_weight;

};

double CoolPropThermodynamicModel::calculateTemperatureFromPressureDensity(const double &P, const double &rho) {

    /// CoolProp requires state update before extracting a property
    /// DmassP_INPUTS specifies that inputs are mass density (rho) and pressure (P)
    cp_substance->update(CoolProp::DmassP_INPUTS, rho, P);

    return( cp_substance->T() );

};

void CoolPropThermodynamicModel::calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {

    /// CoolProp requires state update before extracting a property
    /// DmassP_INPUTS specifies that inputs are mass density (rho) and pressure (P)
    cp_substance->update(CoolProp::DmassP_INPUTS, rho, P);

    T = cp_substance->T();

};

double CoolPropThermodynamicModel::calculatePressureFromTemperatureDensity(const double &T, const double &rho) {

    /// CoolProp requires state update before extracting a property
    /// DmassT_INPUTS specifies that inputs are mass density (rho) and temperature (T)
    cp_substance->update(CoolProp::DmassT_INPUTS, rho, T);

    return( cp_substance->p() );

};

double CoolPropThermodynamicModel::calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->umass() );

};

double CoolPropThermodynamicModel::calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->smass() );

};

void CoolPropThermodynamicModel::calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {

    /// CoolProp requires state update before extracting properties
    /// DmassUmass_INPUTS specifies that inputs are mass density (rho) and specific internal energy (e)
    cp_substance->update(CoolProp::DmassUmass_INPUTS, rho, e);

    P = cp_substance->p();
    T = cp_substance->T();

};

double CoolPropThermodynamicModel::calculateDensityFromPressureTemperature(const double &P, const double &T) {

    /// CoolProp requires state update before extracting a property
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->rhomass() );

};

void CoolPropThermodynamicModel::calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {

    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    rho = cp_substance->rhomass();
    e = cp_substance->umass();

};

void CoolPropThermodynamicModel::calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho) {

    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    c_v = cp_substance->cvmass();
    c_p = cp_substance->cpmass();

};

double CoolPropThermodynamicModel::calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) {

    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->cpmass()/cp_substance->cvmass() );

};

double CoolPropThermodynamicModel::calculateSoundSpeed(const double &P, const double &T, const double &rho) {

    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->speed_sound() );

};

double CoolPropThermodynamicModel::calculateVolumeExpansivity(const double &P, const double &T, const double &rho) {

    /// Volumetric expansion coefficient (isobaric):
    /// alpha = (1/V) * (dV/dT)_P = -(1/rho) * (drho/dT)_P
    
    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->isobaric_expansion_coefficient() );

};

double CoolPropThermodynamicModel::calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) {

    /// Isothermal compressibility:
    /// kappa_T = -(1/V) * (dV/dP)_T = (1/rho) * (drho/dP)_T
    
    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);

    return( cp_substance->isothermal_compressibility() );

};

double CoolPropThermodynamicModel::calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) {

    /// Isentropic compressibility:
    /// kappa_S = -(1/V) * (dV/dP)_S = (1/rho) * (drho/dP)_S = kappa_T/gamma
    
    /// CoolProp requires state update before extracting properties
    /// PT_INPUTS specifies that inputs are pressure (P) and temperature (T)
    cp_substance->update(CoolProp::PT_INPUTS, P, T);    

    // Calculate isothermal compressibility
    double kappa_T = cp_substance->isothermal_compressibility();

    // Calculate heat capacity ratio
    double gamma = cp_substance->cpmass()/cp_substance->cvmass();

    return( kappa_T/gamma );

};
*/

////////// SvdSurrogateThermodynamicModel CLASS //////////

SvdSurrogateThermodynamicModel::SvdSurrogateThermodynamicModel() : BaseThermodynamicModel() {};
        
SvdSurrogateThermodynamicModel::SvdSurrogateThermodynamicModel(const string configuration_file) : BaseThermodynamicModel(configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile();

};

SvdSurrogateThermodynamicModel::~SvdSurrogateThermodynamicModel() {

    delete[] substance_P_vector;	
    delete[] substance_T_vector;
    delete[] substance_rho_vector;	
    delete[] substance_e_vector;
    delete[] substance_rho_P_to_T_matrix_U;
    delete[] substance_rho_P_to_T_matrix_S;
    delete[] substance_rho_P_to_T_matrix_VT;
    delete[] substance_rho_T_to_P_matrix_U;
    delete[] substance_rho_T_to_P_matrix_S;
    delete[] substance_rho_T_to_P_matrix_VT;
    delete[] substance_T_P_to_rho_matrix_U;
    delete[] substance_T_P_to_rho_matrix_S;
    delete[] substance_T_P_to_rho_matrix_VT;
    delete[] substance_T_P_to_e_matrix_U;
    delete[] substance_T_P_to_e_matrix_S;
    delete[] substance_T_P_to_e_matrix_VT;
    delete[] substance_T_P_to_s_matrix_U;
    delete[] substance_T_P_to_s_matrix_S;
    delete[] substance_T_P_to_s_matrix_VT;
    delete[] substance_T_P_to_c_p_matrix_U;
    delete[] substance_T_P_to_c_p_matrix_S;
    delete[] substance_T_P_to_c_p_matrix_VT;
    delete[] substance_T_P_to_c_v_matrix_U;
    delete[] substance_T_P_to_c_v_matrix_S;
    delete[] substance_T_P_to_c_v_matrix_VT;
    delete[] substance_T_P_to_sos_matrix_U;
    delete[] substance_T_P_to_sos_matrix_S;
    delete[] substance_T_P_to_sos_matrix_VT;
    delete[] substance_T_P_to_alpha_matrix_U;
    delete[] substance_T_P_to_alpha_matrix_S;
    delete[] substance_T_P_to_alpha_matrix_VT;
    delete[] substance_T_P_to_kappa_T_matrix_U;
    delete[] substance_T_P_to_kappa_T_matrix_S;
    delete[] substance_T_P_to_kappa_T_matrix_VT;
    delete[] substance_T_P_to_kappa_S_matrix_U;
    delete[] substance_T_P_to_kappa_S_matrix_S;
    delete[] substance_T_P_to_kappa_S_matrix_VT;
    delete[] substance_e_rho_to_P_matrix_U;
    delete[] substance_e_rho_to_P_matrix_S;
    delete[] substance_e_rho_to_P_matrix_VT;
    delete[] substance_e_rho_to_T_matrix_U;
    delete[] substance_e_rho_to_T_matrix_S;
    delete[] substance_e_rho_to_T_matrix_VT;

};

void SvdSurrogateThermodynamicModel::readConfigurationFile() {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile( configuration_file );

    /// Fluid & flow properties
    const YAML::Node & fluid_flow_properties = configuration["fluid_flow_properties"];
    if( fluid_flow_properties["substance_surrogate_file"] ) {

        /// Create YAML object
        string substance_surrogate_file = fluid_flow_properties["substance_surrogate_file"].as<string>();
        YAML::Node substance_surrogate  = YAML::LoadFile( substance_surrogate_file );
    
	/// Substance metadata
        const YAML::Node & substance_metadata = substance_surrogate["substance_metadata"];
        //substance_name  = substance_metadata["substance_name"].as<string>();
        substance_molecular_weight = substance_metadata["substance_molecular_weight"].as<double>();
        substance_P_min = substance_metadata["substance_P_min"].as<double>();
        substance_P_max = substance_metadata["substance_P_max"].as<double>();
        substance_T_min = substance_metadata["substance_T_min"].as<double>();
        substance_T_max = substance_metadata["substance_T_max"].as<double>();
        substance_rho_min = substance_metadata["substance_rho_min"].as<double>();
        substance_rho_max = substance_metadata["substance_rho_max"].as<double>();
        substance_e_min = substance_metadata["substance_e_min"].as<double>();
        substance_e_max = substance_metadata["substance_e_max"].as<double>();
        substance_SVD_rank = substance_metadata["substance_SVD_rank"].as<int>();

	/// Allocate matrices data
        substance_P_vector = new double[substance_SVD_rank];
        substance_T_vector = new double[substance_SVD_rank];
        substance_rho_vector = new double[substance_SVD_rank];
        substance_e_vector = new double[substance_SVD_rank];
        substance_rho_P_to_T_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_rho_P_to_T_matrix_S = new double[substance_SVD_rank];
        substance_rho_P_to_T_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_rho_T_to_P_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_rho_T_to_P_matrix_S = new double[substance_SVD_rank];
        substance_rho_T_to_P_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];	
        substance_T_P_to_rho_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_rho_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_rho_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_e_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_e_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_e_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];	
        substance_T_P_to_s_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_s_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_s_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];	
        substance_T_P_to_c_p_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_c_p_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_c_p_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_c_v_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_c_v_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_c_v_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_sos_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_sos_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_sos_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];	
        substance_T_P_to_alpha_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_alpha_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_alpha_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_kappa_T_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_kappa_T_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_kappa_T_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_kappa_S_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_T_P_to_kappa_S_matrix_S = new double[substance_SVD_rank];
        substance_T_P_to_kappa_S_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_e_rho_to_P_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_e_rho_to_P_matrix_S = new double[substance_SVD_rank];
        substance_e_rho_to_P_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];
        substance_e_rho_to_T_matrix_U = new double[substance_SVD_rank*substance_SVD_rank];
        substance_e_rho_to_T_matrix_S = new double[substance_SVD_rank];
        substance_e_rho_to_T_matrix_VT = new double[substance_SVD_rank*substance_SVD_rank];

	/// Substance SVD data
        const YAML::Node & substance_SVD_data = substance_surrogate["substance_SVD_data"];
	for(int d = 0; d < (substance_SVD_rank*substance_SVD_rank); ++d) {
            substance_rho_P_to_T_matrix_U[d] = substance_SVD_data["substance_rho_P_to_T_matrix_U"][d].as<double>(); 
            substance_rho_P_to_T_matrix_VT[d] = substance_SVD_data["substance_rho_P_to_T_matrix_VT"][d].as<double>();
            substance_rho_T_to_P_matrix_U[d] = substance_SVD_data["substance_rho_T_to_P_matrix_U"][d].as<double>(); 
            substance_rho_T_to_P_matrix_VT[d] = substance_SVD_data["substance_rho_T_to_P_matrix_VT"][d].as<double>();
            substance_T_P_to_rho_matrix_U[d] = substance_SVD_data["substance_T_P_to_rho_matrix_U"][d].as<double>(); 
            substance_T_P_to_rho_matrix_VT[d] = substance_SVD_data["substance_T_P_to_rho_matrix_VT"][d].as<double>();
            substance_T_P_to_e_matrix_U[d] = substance_SVD_data["substance_T_P_to_e_matrix_U"][d].as<double>(); 
            substance_T_P_to_e_matrix_VT[d] = substance_SVD_data["substance_T_P_to_e_matrix_VT"][d].as<double>();
            substance_T_P_to_s_matrix_U[d] = substance_SVD_data["substance_T_P_to_s_matrix_U"][d].as<double>(); 
            substance_T_P_to_s_matrix_VT[d] = substance_SVD_data["substance_T_P_to_s_matrix_VT"][d].as<double>();	    
            substance_T_P_to_c_p_matrix_U[d] = substance_SVD_data["substance_T_P_to_c_p_matrix_U"][d].as<double>(); 
            substance_T_P_to_c_p_matrix_VT[d] = substance_SVD_data["substance_T_P_to_c_p_matrix_VT"][d].as<double>();
            substance_T_P_to_c_v_matrix_U[d] = substance_SVD_data["substance_T_P_to_c_v_matrix_U"][d].as<double>(); 
            substance_T_P_to_c_v_matrix_VT[d] = substance_SVD_data["substance_T_P_to_c_v_matrix_VT"][d].as<double>();
            substance_T_P_to_sos_matrix_U[d] = substance_SVD_data["substance_T_P_to_sos_matrix_U"][d].as<double>(); 
            substance_T_P_to_sos_matrix_VT[d] = substance_SVD_data["substance_T_P_to_sos_matrix_VT"][d].as<double>();
            substance_T_P_to_alpha_matrix_U[d] = substance_SVD_data["substance_T_P_to_alpha_matrix_U"][d].as<double>(); 
            substance_T_P_to_alpha_matrix_VT[d] = substance_SVD_data["substance_T_P_to_alpha_matrix_VT"][d].as<double>();
            substance_T_P_to_kappa_T_matrix_U[d] = substance_SVD_data["substance_T_P_to_kappa_T_matrix_U"][d].as<double>(); 
            substance_T_P_to_kappa_T_matrix_VT[d] = substance_SVD_data["substance_T_P_to_kappa_T_matrix_VT"][d].as<double>();
            substance_T_P_to_kappa_S_matrix_U[d] = substance_SVD_data["substance_T_P_to_kappa_S_matrix_U"][d].as<double>(); 
            substance_T_P_to_kappa_S_matrix_VT[d] = substance_SVD_data["substance_T_P_to_kappa_S_matrix_VT"][d].as<double>();
            substance_e_rho_to_P_matrix_U[d] = substance_SVD_data["substance_e_rho_to_P_matrix_U"][d].as<double>(); 
            substance_e_rho_to_P_matrix_VT[d] = substance_SVD_data["substance_e_rho_to_P_matrix_VT"][d].as<double>();
            substance_e_rho_to_T_matrix_U[d] = substance_SVD_data["substance_e_rho_to_T_matrix_U"][d].as<double>(); 
            substance_e_rho_to_T_matrix_VT[d] = substance_SVD_data["substance_e_rho_to_T_matrix_VT"][d].as<double>();
	}
	for(int d = 0; d < substance_SVD_rank; ++d) {
            substance_P_vector[d] = substance_SVD_data["substance_P_vector"][d].as<double>();
            substance_T_vector[d] = substance_SVD_data["substance_T_vector"][d].as<double>();
            substance_rho_vector[d] = substance_SVD_data["substance_rho_vector"][d].as<double>();
            substance_e_vector[d] = substance_SVD_data["substance_e_vector"][d].as<double>();
            substance_rho_P_to_T_matrix_S[d] = substance_SVD_data["substance_rho_P_to_T_matrix_S"][d].as<double>();
            substance_rho_T_to_P_matrix_S[d] = substance_SVD_data["substance_rho_T_to_P_matrix_S"][d].as<double>();
            substance_T_P_to_rho_matrix_S[d] = substance_SVD_data["substance_T_P_to_rho_matrix_S"][d].as<double>();
            substance_T_P_to_e_matrix_S[d] = substance_SVD_data["substance_T_P_to_e_matrix_S"][d].as<double>();
            substance_T_P_to_s_matrix_S[d] = substance_SVD_data["substance_T_P_to_s_matrix_S"][d].as<double>();
            substance_T_P_to_c_p_matrix_S[d] = substance_SVD_data["substance_T_P_to_c_p_matrix_S"][d].as<double>();
            substance_T_P_to_c_v_matrix_S[d] = substance_SVD_data["substance_T_P_to_c_v_matrix_S"][d].as<double>();
            substance_T_P_to_sos_matrix_S[d] = substance_SVD_data["substance_T_P_to_sos_matrix_S"][d].as<double>();
            substance_T_P_to_alpha_matrix_S[d] = substance_SVD_data["substance_T_P_to_alpha_matrix_S"][d].as<double>();
            substance_T_P_to_kappa_T_matrix_S[d] = substance_SVD_data["substance_T_P_to_kappa_T_matrix_S"][d].as<double>();
            substance_T_P_to_kappa_S_matrix_S[d] = substance_SVD_data["substance_T_P_to_kappa_S_matrix_S"][d].as<double>();
            substance_e_rho_to_P_matrix_S[d] = substance_SVD_data["substance_e_rho_to_P_matrix_S"][d].as<double>();
            substance_e_rho_to_T_matrix_S[d] = substance_SVD_data["substance_e_rho_to_T_matrix_S"][d].as<double>();
	}

    }
    
    /// Calculate specific gas constant
    R_specific = R_universal/molecular_weight;

};

double SvdSurrogateThermodynamicModel::calculateTemperatureFromPressureDensity(const double &P, const double &rho) {

    /// Clipping for robust execution
    double rho_aux = rho, P_aux = P;
    if( rho < substance_rho_min ) { rho_aux = substance_rho_min; } else if( rho > substance_rho_max ) { rho_aux = substance_rho_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find density interval [i, i+1]
    double delta_i = ( substance_rho_vector[1] - substance_rho_vector[0] ) + 1.0e-10;
    int i = (int)( ( rho_aux - substance_rho_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_rho = ( rho_aux - substance_rho_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double temperature = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for density from U (row i and row i+1)
        u_k = ( 1.0 - w_rho )*substance_rho_P_to_T_matrix_U[u_row1 + k] + w_rho*substance_rho_P_to_T_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_rho_P_to_T_matrix_VT[vt_row + j] + w_P*substance_rho_P_to_T_matrix_VT[vt_row + j + 1];
        /// Update temperature value
        temperature += substance_rho_P_to_T_matrix_S[k]*u_k*vt_k;
    }

    return( temperature );

};

void SvdSurrogateThermodynamicModel::calculateTemperatureFromPressureDensityWithInitialGuess(double &T, const double &P, const double &rho) {

    /// Clipping for robust execution
    double rho_aux = rho, P_aux = P;
    if( rho < substance_rho_min ) { rho_aux = substance_rho_min; } else if( rho > substance_rho_max ) { rho_aux = substance_rho_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find density interval [i, i+1]
    double delta_i = ( substance_rho_vector[1] - substance_rho_vector[0] ) + 1.0e-10;
    int i = (int)( ( rho_aux - substance_rho_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_rho = ( rho_aux - substance_rho_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double temperature = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for density from U (row i and row i+1)
        u_k = ( 1.0 - w_rho )*substance_rho_P_to_T_matrix_U[u_row1 + k] + w_rho*substance_rho_P_to_T_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_rho_P_to_T_matrix_VT[vt_row + j] + w_P*substance_rho_P_to_T_matrix_VT[vt_row + j + 1];
        /// Update temperature value
        temperature += substance_rho_P_to_T_matrix_S[k]*u_k*vt_k;
    }
    T = temperature;

};

double SvdSurrogateThermodynamicModel::calculatePressureFromTemperatureDensity(const double &T, const double &rho) {

    /// Clipping for robust execution
    double rho_aux = rho, T_aux = T;
    if( rho < substance_rho_min ) { rho_aux = substance_rho_min; } else if( rho > substance_rho_max ) { rho_aux = substance_rho_max; }
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
   
    /// Find density interval [i, i+1]
    double delta_i = ( substance_rho_vector[1] - substance_rho_vector[0] ) + 1.0e-10;
    int i = (int)( ( rho_aux - substance_rho_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_rho = ( rho_aux - substance_rho_vector[i] )/delta_i;

    /// Find temperature interval [j, j+1]
    double delta_j = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int j = (int)( ( T_aux - substance_T_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_T = ( T_aux - substance_T_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double pressure = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for density from U (row i and row i+1)
        u_k = ( 1.0 - w_rho )*substance_rho_T_to_P_matrix_U[u_row1 + k] + w_rho*substance_rho_T_to_P_matrix_U[u_row2 + k];
        /// Mode k for temperature from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_T )*substance_rho_T_to_P_matrix_VT[vt_row + j] + w_T*substance_rho_T_to_P_matrix_VT[vt_row + j + 1];
        /// Update pressure value
        pressure += substance_rho_T_to_P_matrix_S[k]*u_k*vt_k;
    }

    return( pressure );

};

double SvdSurrogateThermodynamicModel::calculateInternalEnergyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double specific_internal_energy = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_e_matrix_U[u_row1 + k] + w_T*substance_T_P_to_e_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_e_matrix_VT[vt_row + j] + w_P*substance_T_P_to_e_matrix_VT[vt_row + j + 1];
        /// Update specific internal energy value
        specific_internal_energy += substance_T_P_to_e_matrix_S[k]*u_k*vt_k;
    }

    return( specific_internal_energy );

};

double SvdSurrogateThermodynamicModel::calculateEntropyFromPressureTemperatureDensity(const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double specific_entropy = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_s_matrix_U[u_row1 + k] + w_T*substance_T_P_to_s_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_s_matrix_VT[vt_row + j] + w_P*substance_T_P_to_s_matrix_VT[vt_row + j + 1];
        /// Update specific entropy value
        specific_entropy += substance_T_P_to_s_matrix_S[k]*u_k*vt_k;
    }

    return( specific_entropy );

};

void SvdSurrogateThermodynamicModel::calculatePressureTemperatureFromDensityInternalEnergy(double &P, double &T, const double &rho, const double &e) {

    /// Clipping for robust execution
    double e_aux = e, rho_aux = rho;
    if( e < substance_e_min ) {	e_aux = substance_e_min; } else if( e > substance_e_max ) { e_aux = substance_e_max; }
    if( rho < substance_rho_min ) { rho_aux = substance_rho_min; } else if( rho > substance_rho_max ) { rho_aux = substance_rho_max; }
   
    /// Find specific internal energy interval [i, i+1]
    double delta_i = ( substance_e_vector[1] - substance_e_vector[0] ) + 1.0e-10;
    int i = (int)( ( e_aux - substance_e_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_e = ( e_aux - substance_e_vector[i] )/delta_i;

    /// Find density interval [j, j+1]
    double delta_j = ( substance_rho_vector[1] - substance_rho_vector[0] ) + 1.0e-10;
    int j = (int)( ( rho_aux - substance_rho_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_rho = ( rho_aux - substance_rho_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double pressure = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for specific internal energy from U (row i and row i+1)
        u_k = ( 1.0 - w_e )*substance_e_rho_to_P_matrix_U[u_row1 + k] + w_e*substance_e_rho_to_P_matrix_U[u_row2 + k];
        /// Mode k for density from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_rho )*substance_e_rho_to_P_matrix_VT[vt_row + j] + w_rho*substance_e_rho_to_P_matrix_VT[vt_row + j + 1];
        /// Update pressure value
        pressure += substance_e_rho_to_P_matrix_S[k]*u_k*vt_k;
    }
    P = pressure;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    vt_row = -1;
    double temperature = 0.0;
    u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for specific internal energy from U (row i and row i+1)
        u_k = ( 1.0 - w_e )*substance_e_rho_to_T_matrix_U[u_row1 + k] + w_e*substance_e_rho_to_T_matrix_U[u_row2 + k];
        /// Mode k for density from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_rho )*substance_e_rho_to_T_matrix_VT[vt_row + j] + w_rho*substance_e_rho_to_T_matrix_VT[vt_row + j + 1];
        /// Update temperature value
        temperature += substance_e_rho_to_T_matrix_S[k]*u_k*vt_k;
    }
    T = temperature;

};

double SvdSurrogateThermodynamicModel::calculateDensityFromPressureTemperature(const double &P, const double &T) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double density = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_rho_matrix_U[u_row1 + k] + w_T*substance_T_P_to_rho_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_rho_matrix_VT[vt_row + j] + w_P*substance_T_P_to_rho_matrix_VT[vt_row + j + 1];
        /// Update density value
        density += substance_T_P_to_rho_matrix_S[k]*u_k*vt_k;
    }

    return( density );

};

void SvdSurrogateThermodynamicModel::calculateDensityInternalEnergyFromPressureTemperature(double &rho, double &e, const double &P, const double &T) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double density = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_rho_matrix_U[u_row1 + k] + w_T*substance_T_P_to_rho_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_rho_matrix_VT[vt_row + j] + w_P*substance_T_P_to_rho_matrix_VT[vt_row + j + 1];
        /// Update density value
        density += substance_T_P_to_rho_matrix_S[k]*u_k*vt_k;
    }
    rho = density;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    vt_row = -1;
    double specific_internal_energy = 0.0;
    u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_e_matrix_U[u_row1 + k] + w_T*substance_T_P_to_e_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_e_matrix_VT[vt_row + j] + w_P*substance_T_P_to_e_matrix_VT[vt_row + j + 1];
        /// Update specific internal energy value
        specific_internal_energy += substance_T_P_to_e_matrix_S[k]*u_k*vt_k;
    }
    e = specific_internal_energy;

};

void SvdSurrogateThermodynamicModel::calculateSpecificHeatCapacities(double &c_v, double &c_p, const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double specific_isochoric_heat_capacity = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_c_v_matrix_U[u_row1 + k] + w_T*substance_T_P_to_c_v_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_c_v_matrix_VT[vt_row + j] + w_P*substance_T_P_to_c_v_matrix_VT[vt_row + j + 1];
        /// Update specific_isochoric_heat_capacity value
        specific_isochoric_heat_capacity += substance_T_P_to_c_v_matrix_S[k]*u_k*vt_k;
    }
    c_v = specific_isochoric_heat_capacity;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    vt_row = -1;
    double specific_isobaric_heat_capacity = 0.0;
    u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_c_p_matrix_U[u_row1 + k] + w_T*substance_T_P_to_c_p_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_c_p_matrix_VT[vt_row + j] + w_P*substance_T_P_to_c_p_matrix_VT[vt_row + j + 1];
        /// Update specific internal energy value
        specific_isobaric_heat_capacity += substance_T_P_to_c_p_matrix_S[k]*u_k*vt_k;
    }
    c_p = specific_isobaric_heat_capacity;

};

double SvdSurrogateThermodynamicModel::calculateHeatCapacitiesRatio(const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double specific_isochoric_heat_capacity = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_c_v_matrix_U[u_row1 + k] + w_T*substance_T_P_to_c_v_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_c_v_matrix_VT[vt_row + j] + w_P*substance_T_P_to_c_v_matrix_VT[vt_row + j + 1];
        /// Update specific_isochoric_heat_capacity value
        specific_isochoric_heat_capacity += substance_T_P_to_c_v_matrix_S[k]*u_k*vt_k;
    }

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    vt_row = -1;
    double specific_isobaric_heat_capacity = 0.0;
    u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_c_p_matrix_U[u_row1 + k] + w_T*substance_T_P_to_c_p_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_c_p_matrix_VT[vt_row + j] + w_P*substance_T_P_to_c_p_matrix_VT[vt_row + j + 1];
        /// Update specific internal energy value
        specific_isobaric_heat_capacity += substance_T_P_to_c_p_matrix_S[k]*u_k*vt_k;
    }

    return( specific_isobaric_heat_capacity/specific_isochoric_heat_capacity );

};

double SvdSurrogateThermodynamicModel::calculateSoundSpeed(const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double sound_speed = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_sos_matrix_U[u_row1 + k] + w_T*substance_T_P_to_sos_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_sos_matrix_VT[vt_row + j] + w_P*substance_T_P_to_sos_matrix_VT[vt_row + j + 1];
        /// Update sound_speed value
        sound_speed += substance_T_P_to_sos_matrix_S[k]*u_k*vt_k;
    }

    return( sound_speed );

};

double SvdSurrogateThermodynamicModel::calculateVolumeExpansivity(const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double volume_expansivity = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_alpha_matrix_U[u_row1 + k] + w_T*substance_T_P_to_alpha_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_alpha_matrix_VT[vt_row + j] + w_P*substance_T_P_to_alpha_matrix_VT[vt_row + j + 1];
        /// Update volume_expansivity value
        volume_expansivity += substance_T_P_to_alpha_matrix_S[k]*u_k*vt_k;
    }

    return( volume_expansivity );
  
};

double SvdSurrogateThermodynamicModel::calculateIsothermalCompressibility(const double &P, const double &T, const double &rho) {

    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double isothermal_compressibility = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_kappa_T_matrix_U[u_row1 + k] + w_T*substance_T_P_to_kappa_T_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_kappa_T_matrix_VT[vt_row + j] + w_P*substance_T_P_to_kappa_T_matrix_VT[vt_row + j + 1];
        /// Update isothermal_compressibility value
        isothermal_compressibility += substance_T_P_to_kappa_T_matrix_S[k]*u_k*vt_k;
    }

    return( isothermal_compressibility );
  
};  

double SvdSurrogateThermodynamicModel::calculateIsentropicCompressibility(const double &P, const double &T, const double &rho) {
    
    /// Clipping for robust execution
    double T_aux = T, P_aux = P;
    if( T < substance_T_min ) {	T_aux = substance_T_min; } else if( T > substance_T_max ) { T_aux = substance_T_max; }
    if( P < substance_P_min ) {	P_aux = substance_P_min; } else if( P > substance_P_max ) { P_aux = substance_P_max; }
   
    /// Find temperature interval [i, i+1]
    double delta_i = ( substance_T_vector[1] - substance_T_vector[0] ) + 1.0e-10;
    int i = (int)( ( T_aux - substance_T_vector[0])/delta_i );
    i = ( i < 0) ? 0 : ( i > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : i );
    double w_T = ( T_aux - substance_T_vector[i] )/delta_i;

    /// Find pressure interval [j, j+1]
    double delta_j = ( substance_P_vector[1] - substance_P_vector[0] ) + 1.0e-10;
    int j = (int)( ( P_aux - substance_P_vector[0])/delta_j );
    j = ( j < 0) ? 0 : ( j > ( substance_SVD_rank - 2 ) ? ( substance_SVD_rank - 2 ) : j );
    double w_P = ( P_aux - substance_P_vector[j] )/delta_j;

    /// Pre-calculate 1D array row offsets
    int u_row1 = i*substance_SVD_rank;
    int u_row2 = (i + 1)*substance_SVD_rank;

    /// Accumulate interpolated SVD reconstruction across rank substance_SVD_rank
    int vt_row = -1;
    double isentropic_compressibility = 0.0, u_k = 0.0, vt_k = 0.0;
    for( int k = 0; k < substance_SVD_rank; ++k ) {
        /// Mode k for temperature from U (row i and row i+1)
        u_k = ( 1.0 - w_T )*substance_T_P_to_kappa_S_matrix_U[u_row1 + k] + w_T*substance_T_P_to_kappa_S_matrix_U[u_row2 + k];
        /// Mode k for pressure from VT (row k, column j and j+1)
        vt_row = k*substance_SVD_rank;
        vt_k = ( 1.0 - w_P )*substance_T_P_to_kappa_S_matrix_VT[vt_row + j] + w_P*substance_T_P_to_kappa_S_matrix_VT[vt_row + j + 1];
        /// Update isentropic_compressibility value
        isentropic_compressibility += substance_T_P_to_kappa_S_matrix_S[k]*u_k*vt_k;
    }

    return( isentropic_compressibility );
  
};
