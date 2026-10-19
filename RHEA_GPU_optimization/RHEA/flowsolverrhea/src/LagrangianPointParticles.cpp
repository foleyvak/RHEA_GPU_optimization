#include "LagrangianPointParticles.hpp"

using namespace std;

////////// FIXED PARAMETERS //////////
const double pi = 2.0*asin( 1.0 );				/// pi number (fixed)
#define PI 3.14159265358979323846				/// OpenACC


////////// BaseLagrangianPointParticle CLASS //////////

BaseLagrangianPointParticle::BaseLagrangianPointParticle() {};
        
BaseLagrangianPointParticle::BaseLagrangianPointParticle(const string configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile( configuration_file );

    /// Default initialization of identifier and position ... they will be overwritten later
    identifier_prt = -1;
    position_prt[0] = 0.0; position_prt[1] = 0.0; position_prt[2] = 0.0;
    position_0_prt[0] = 0.0; position_0_prt[1] = 0.0; position_0_prt[2] = 0.0;

    /// Default initialization of velocity to zero ... it can be modified in the future 
    velocity_prt[0] = 0.0; velocity_prt[1] = 0.0; velocity_prt[2] = 0.0;
    velocity_0_prt[0] = 0.0; velocity_0_prt[1] = 0.0; velocity_0_prt[2] = 0.0;

};

BaseLagrangianPointParticle::~BaseLagrangianPointParticle() {};

void BaseLagrangianPointParticle::readConfigurationFile( const string configuration_file ) {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile( configuration_file );

    /// Lagrangian point particles
    const YAML::Node & lagrangian_point_particles = configuration["lagrangian_point_particles"];
    diameter_prt = lagrangian_point_particles["diameter_particles"].as<double>();
    density_prt  = lagrangian_point_particles["density_particles"].as<double>();

};

double BaseLagrangianPointParticle::calculateVolumePrt() {

    /// Assuming spherical particle
    double volume_prt = ( PI/6.0 )*pow( diameter_prt, 3.0 );

    return volume_prt;

};

double BaseLagrangianPointParticle::calculateMassPrt() {

    /// Assuming spherical particle
    double mass_prt = ( PI/6.0 )*density_prt*pow( diameter_prt, 3.0 );

    return mass_prt;

};

double BaseLagrangianPointParticle::calculateRelaxationTimePrt(const double &mu_fluid) {

    /// Assuming spherical particle
    double relaxation_time_prt = ( density_prt*pow( diameter_prt, 2.0 ) )/( 18.0*mu_fluid );

    return relaxation_time_prt;

};

double BaseLagrangianPointParticle::calculateDensityRatioBetaPrt(const double &rho_fluid) {

    /// Assuming point particle
    double density_ratio_prt = ( 3.0*rho_fluid )/( rho_fluid + 2.0*density_prt );

    return density_ratio_prt;

};

////////// DistributedPointParticles CLASS //////////

DistributedPointParticles::DistributedPointParticles() {};
        
DistributedPointParticles::DistributedPointParticles(const string configuration_file) {

    /// Initialize MPI stuff	
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /// Read configuration (input) file
    this->readConfigurationFile( configuration_file );

    /// Total number of particles
    num_prts_total = std::round( L_x*L_y*L_z*number_density_prts );

    /// Resize local number of particles in use
    num_prts_local_in_use.resize( world_size, 0 );

};

DistributedPointParticles::~DistributedPointParticles() {
    freeParticleArrays(); 
};

void DistributedPointParticles::readConfigurationFile( const string configuration_file ) {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile( configuration_file );

    /// Problem parameters
    const YAML::Node & problem_parameters = configuration["problem_parameters"];
    x_0 = problem_parameters["x_0"].as<double>();
    y_0 = problem_parameters["y_0"].as<double>();
    z_0 = problem_parameters["z_0"].as<double>();
    L_x = problem_parameters["L_x"].as<double>();
    L_y = problem_parameters["L_y"].as<double>();
    L_z = problem_parameters["L_z"].as<double>();

    /// Boundary conditions
    std::string dummy_type_boco;
    const YAML::Node & boundary_conditions = configuration["boundary_conditions"];
    /// West
    dummy_type_boco = boundary_conditions["west_bc"][0].as<std::string>();
    if( dummy_type_boco == "DIRICHLET" ) {
        bocos_type[_WEST_] = _DIRICHLET_;
    } else if( dummy_type_boco == "NEUMANN" ) {
        bocos_type[_WEST_] = _NEUMANN_;
    } else if( dummy_type_boco == "PERIODIC" ) {
        bocos_type[_WEST_] = _PERIODIC_;
    } else if( dummy_type_boco == "SUBSONIC_INFLOW" ) {
        bocos_type[_WEST_] = _SUBSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUBSONIC_OUTFLOW" ) {
        bocos_type[_WEST_] = _SUBSONIC_OUTFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_INFLOW" ) {
        bocos_type[_WEST_] = _SUPERSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_OUTFLOW" ) {
        bocos_type[_WEST_] = _SUPERSONIC_OUTFLOW_;
    } else {
        std::cout << "West boundary condition not available!" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_WEST_] = boundary_conditions["west_bc"][1].as<double>();
    bocos_v[_WEST_] = boundary_conditions["west_bc"][2].as<double>();
    bocos_w[_WEST_] = boundary_conditions["west_bc"][3].as<double>();
    bocos_P[_WEST_] = boundary_conditions["west_bc"][4].as<double>();
    bocos_T[_WEST_] = boundary_conditions["west_bc"][5].as<double>();
    /// East
    dummy_type_boco = boundary_conditions["east_bc"][0].as<std::string>();
    if( dummy_type_boco == "DIRICHLET" ) {
        bocos_type[_EAST_] = _DIRICHLET_;
    } else if( dummy_type_boco == "NEUMANN" ) {
        bocos_type[_EAST_] = _NEUMANN_;
    } else if( dummy_type_boco == "PERIODIC" ) {
        bocos_type[_EAST_] = _PERIODIC_;
    } else if( dummy_type_boco == "SUBSONIC_INFLOW" ) {
        bocos_type[_EAST_] = _SUBSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUBSONIC_OUTFLOW" ) {
        bocos_type[_EAST_] = _SUBSONIC_OUTFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_INFLOW" ) {
        bocos_type[_EAST_] = _SUPERSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_OUTFLOW" ) {
        bocos_type[_EAST_] = _SUPERSONIC_OUTFLOW_;
    } else {
        std::cout << "East boundary condition not available!" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_EAST_] = boundary_conditions["east_bc"][1].as<double>();
    bocos_v[_EAST_] = boundary_conditions["east_bc"][2].as<double>();
    bocos_w[_EAST_] = boundary_conditions["east_bc"][3].as<double>();
    bocos_P[_EAST_] = boundary_conditions["east_bc"][4].as<double>();
    bocos_T[_EAST_] = boundary_conditions["east_bc"][5].as<double>();
    /// South
    dummy_type_boco = boundary_conditions["south_bc"][0].as<std::string>();
    if( dummy_type_boco == "DIRICHLET" ) {
        bocos_type[_SOUTH_] = _DIRICHLET_;
    } else if( dummy_type_boco == "NEUMANN" ) {
        bocos_type[_SOUTH_] = _NEUMANN_;
    } else if( dummy_type_boco == "PERIODIC" ) {
        bocos_type[_SOUTH_] = _PERIODIC_;
    } else if( dummy_type_boco == "SUBSONIC_INFLOW" ) {
        bocos_type[_SOUTH_] = _SUBSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUBSONIC_OUTFLOW" ) {
        bocos_type[_SOUTH_] = _SUBSONIC_OUTFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_INFLOW" ) {
        bocos_type[_SOUTH_] = _SUPERSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_OUTFLOW" ) {
        bocos_type[_SOUTH_] = _SUPERSONIC_OUTFLOW_;
    } else {
        std::cout << "South boundary condition not available!" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_SOUTH_] = boundary_conditions["south_bc"][1].as<double>();
    bocos_v[_SOUTH_] = boundary_conditions["south_bc"][2].as<double>();
    bocos_w[_SOUTH_] = boundary_conditions["south_bc"][3].as<double>();
    bocos_P[_SOUTH_] = boundary_conditions["south_bc"][4].as<double>();
    bocos_T[_SOUTH_] = boundary_conditions["south_bc"][5].as<double>();
    /// North
    dummy_type_boco = boundary_conditions["north_bc"][0].as<std::string>();
    if( dummy_type_boco == "DIRICHLET" ) {
        bocos_type[_NORTH_] = _DIRICHLET_;
    } else if( dummy_type_boco == "NEUMANN" ) {
        bocos_type[_NORTH_] = _NEUMANN_;
    } else if( dummy_type_boco == "PERIODIC" ) {
        bocos_type[_NORTH_] = _PERIODIC_;
    } else if( dummy_type_boco == "SUBSONIC_INFLOW" ) {
        bocos_type[_NORTH_] = _SUBSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUBSONIC_OUTFLOW" ) {
        bocos_type[_NORTH_] = _SUBSONIC_OUTFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_INFLOW" ) {
        bocos_type[_NORTH_] = _SUPERSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_OUTFLOW" ) {
        bocos_type[_NORTH_] = _SUPERSONIC_OUTFLOW_;
    } else {
        std::cout << "North boundary condition not available!" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_NORTH_] = boundary_conditions["north_bc"][1].as<double>();
    bocos_v[_NORTH_] = boundary_conditions["north_bc"][2].as<double>();
    bocos_w[_NORTH_] = boundary_conditions["north_bc"][3].as<double>();
    bocos_P[_NORTH_] = boundary_conditions["north_bc"][4].as<double>();
    bocos_T[_NORTH_] = boundary_conditions["north_bc"][5].as<double>();
    /// Back
    dummy_type_boco = boundary_conditions["back_bc"][0].as<std::string>();
    if( dummy_type_boco == "DIRICHLET" ) {
        bocos_type[_BACK_] = _DIRICHLET_;
    } else if( dummy_type_boco == "NEUMANN" ) {
        bocos_type[_BACK_] = _NEUMANN_;
    } else if( dummy_type_boco == "PERIODIC" ) {
        bocos_type[_BACK_] = _PERIODIC_;
    } else if( dummy_type_boco == "SUBSONIC_INFLOW" ) {
        bocos_type[_BACK_] = _SUBSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUBSONIC_OUTFLOW" ) {
        bocos_type[_BACK_] = _SUBSONIC_OUTFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_INFLOW" ) {
        bocos_type[_BACK_] = _SUPERSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_OUTFLOW" ) {
        bocos_type[_BACK_] = _SUPERSONIC_OUTFLOW_;
    } else {
        std::cout << "Back boundary condition not available!" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_BACK_] = boundary_conditions["back_bc"][1].as<double>();
    bocos_v[_BACK_] = boundary_conditions["back_bc"][2].as<double>();
    bocos_w[_BACK_] = boundary_conditions["back_bc"][3].as<double>();
    bocos_P[_BACK_] = boundary_conditions["back_bc"][4].as<double>();
    bocos_T[_BACK_] = boundary_conditions["back_bc"][5].as<double>();
    /// Front
    dummy_type_boco = boundary_conditions["front_bc"][0].as<std::string>();
    if( dummy_type_boco == "DIRICHLET" ) {
        bocos_type[_FRONT_] = _DIRICHLET_;
    } else if( dummy_type_boco == "NEUMANN" ) {
        bocos_type[_FRONT_] = _NEUMANN_;
    } else if( dummy_type_boco == "PERIODIC" ) {
        bocos_type[_FRONT_] = _PERIODIC_;
    } else if( dummy_type_boco == "SUBSONIC_INFLOW" ) {
        bocos_type[_FRONT_] = _SUBSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUBSONIC_OUTFLOW" ) {
        bocos_type[_FRONT_] = _SUBSONIC_OUTFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_INFLOW" ) {
        bocos_type[_FRONT_] = _SUPERSONIC_INFLOW_;
    } else if( dummy_type_boco == "SUPERSONIC_OUTFLOW" ) {
        bocos_type[_FRONT_] = _SUPERSONIC_OUTFLOW_;
    } else {
        std::cout << "Front boundary condition not available!" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_FRONT_] = boundary_conditions["front_bc"][1].as<double>();
    bocos_v[_FRONT_] = boundary_conditions["front_bc"][2].as<double>();
    bocos_w[_FRONT_] = boundary_conditions["front_bc"][3].as<double>();
    bocos_P[_FRONT_] = boundary_conditions["front_bc"][4].as<double>();
    bocos_T[_FRONT_] = boundary_conditions["front_bc"][5].as<double>();

    /// Lagrangian point particles
    const YAML::Node & lagrangian_point_particles = configuration["lagrangian_point_particles"];
    diameter_prt               = lagrangian_point_particles["diameter_particles"].as<double>();
    density_prt                = lagrangian_point_particles["density_particles"].as<double>();
    number_density_prts        = lagrangian_point_particles["number_density_particles"].as<double>();
    buffer_ratio_prts          = lagrangian_point_particles["buffer_ratio_particles"].as<double>();
    output_data_file_name_prts = lagrangian_point_particles["output_data_file_name_particles"].as<std::string>();
    generate_xdmf_file_prts    = lagrangian_point_particles["generate_xdmf_file_particles"].as<bool>();

    /// Parallelization scheme
    const YAML::Node & parallelization_scheme = configuration["parallelization_scheme"];
    np_x = parallelization_scheme["np_x"].as<int>();
    np_y = parallelization_scheme["np_y"].as<int>();
    np_z = parallelization_scheme["np_z"].as<int>();

};

void DistributedPointParticles::set_subdomains_distributed_prts( double &xmin_local, double &xmax_local, double &ymin_local, double &ymax_local, double &zmin_local, double &zmax_local ) {

    /// Skip if there are no particles
    if( this->get_num_prts_total() < 1 ) return;

    /// Get the min and max x-, y- and z-positions that each of the tasks is responsible for
    x_min_per_task = std::vector<double>( np_x, 0.0 );
    x_max_per_task = std::vector<double>( np_x, 0.0 );
    y_min_per_task = std::vector<double>( np_y, 0.0 );
    y_max_per_task = std::vector<double>( np_y, 0.0 );
    z_min_per_task = std::vector<double>( np_z, 0.0 );
    z_max_per_task = std::vector<double>( np_z, 0.0 );

    /// Fetch these values
    x_min_per_task = gather_from_tasks( &xmin_local );
    x_max_per_task = gather_from_tasks( &xmax_local );
    y_min_per_task = gather_from_tasks( &ymin_local );
    y_max_per_task = gather_from_tasks( &ymax_local );
    z_min_per_task = gather_from_tasks( &zmin_local );
    z_max_per_task = gather_from_tasks( &zmax_local );

};

/// Calculate volume of particle iprt
#pragma acc routine
double DistributedPointParticles::calculate_volume_prt(int iprt) {

    /// Assuming spherical particle
    double volume_prt = ( PI/6.0 )*pow( local_prts_diameters[iprt], 3.0 );

    return volume_prt;

};

/// Calculate mass of particle iprt
#pragma acc routine
double DistributedPointParticles::calculate_mass_prt(int iprt) {

    /// Assuming spherical particle
    double mass_prt = ( PI/6.0 )*local_prts_densities[iprt]*pow( local_prts_diameters[iprt], 3.0 );

    return mass_prt;

};

/// Calculate relaxation time of particle iprt
#pragma acc routine
double DistributedPointParticles::calculate_relaxation_time_prt(int iprt, const double &mu_fluid) {

    /// Assuming spherical particle
    double relaxation_time_prt = ( local_prts_densities[iprt]*pow( local_prts_diameters[iprt], 2.0 ) )/( 18.0*mu_fluid );

    return relaxation_time_prt;

};

/// Calculate density ratio beta of particle iprt
double DistributedPointParticles::calculate_density_ratio_beta_prt(int iprt, const double &rho_fluid) {

    /// Assuming point particle
    double density_ratio_prt = ( 3.0*rho_fluid )/( rho_fluid + 2.0*local_prts_densities[iprt] );

    return density_ratio_prt;

};
