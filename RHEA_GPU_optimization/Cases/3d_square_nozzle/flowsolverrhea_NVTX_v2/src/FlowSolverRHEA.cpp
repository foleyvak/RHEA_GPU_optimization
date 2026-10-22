#include "FlowSolverRHEA.hpp"

#ifdef USE_NVTX
    #include <nvtx3/nvToolsExt.h>     /// header-only NVTX3 (use <nvToolsExt.h> and link -lnvToolsExt for NVTX v2)
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP()      nvtxRangePop()
#else
    #define NVTX_PUSH(name)
    #define NVTX_POP()
#endif

using namespace std;


////////// COMPILATION DIRECTIVES //////////
#define _PRESSURE_BASED_WAVE_SPEED_ESTIMATES_ 0		/// Select approach for estimating wave speeds

////////// FIXED PARAMETERS //////////
const double epsilon        = 1.0e-10;			/// Small epsilon number (fixed)
//const double pi             = 2.0*asin(1.0);		/// pi number (fixed)
const double pi             = 3.14159265358979323846;   /// pi number (fixed) ... added for OpenACC
const int cout_precision    = 5;		        /// Output precision (fixed)
const int fstream_precision = 15;	                /// Fstream precision (fixed)


////////// PRAGMA DIRECTIVES //////////


////////// FlowSolverRHEA CLASS //////////

FlowSolverRHEA::FlowSolverRHEA() {};

FlowSolverRHEA::FlowSolverRHEA(const string name_configuration_file, function<void(FlowSolverRHEA*)> custom_mesh_filler) : configuration_file(name_configuration_file) {

    /// Read configuration (input) file
    this->readConfigurationFile();
	
    /// Set value of selected variables
    current_time      = 0.0;	/// Current time (restart will overwrite it)
    current_time_iter = 0;	/// Current time iteration (restart will overwrite it)
    averaging_time    = 0.0;	/// Current averaging time (restart will overwrite it)

    /// Construct (initialize) thermodynamic model
    /*if( thermodynamic_model == "IDEAL_GAS" ) {
        thermodynamics = new IdealGasThermodynamicModel( configuration_file );
    } else if( thermodynamic_model == "STIFFENED_GAS" ) {
        thermodynamics = new StiffenedGasThermodynamicModel( configuration_file );
    } else if( thermodynamic_model == "PENG_ROBINSON" ) {
        thermodynamics = new PengRobinsonThermodynamicModel( configuration_file );
#if COOLPROP_ACTIVATED_	
    } else if( thermodynamic_model == "COOLPROP" ) {
        thermodynamics = new CoolPropThermodynamicModel( configuration_file );
#endif
    } else if( thermodynamic_model == "SVD_SURROGATE" ) {
        thermodynamics = new SvdSurrogateThermodynamicModel( configuration_file );	
    } else {
        cout << "Thermodynamic model not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }*/
    thermodynamics = new _ThermodynamicModel_( configuration_file );	// ... modified for OpenACC

    /// Construct (initialize) transport coefficients model
    /*if( transport_coefficients_model == "CONSTANT" ) {
        transport_coefficients = new ConstantTransportCoefficients( configuration_file );
    } else if( transport_coefficients_model == "LOW_PRESSURE_GAS" ) {
        transport_coefficients = new LowPressureGasTransportCoefficients( configuration_file );
    } else if( transport_coefficients_model == "HIGH_PRESSURE" ) {
        transport_coefficients = new HighPressureTransportCoefficients( configuration_file );
#if COOLPROP_ACTIVATED_
    } else if( transport_coefficients_model == "COOLPROP" ) {
        transport_coefficients = new CoolPropTransportCoefficients( configuration_file );
#endif
    } else if( transport_coefficients_model == "SVD_SURROGATE" ) {
        transport_coefficients = new SvdSurrogateTransportCoefficients( configuration_file );	
    } else {
        cout << "Transport coefficients model not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }*/
    transport_coefficients = new _TransportCoefficientsModel_( configuration_file );	// ... modified for OpenACC

    /// Construct (initialize) Riemann solver
    /*if( riemann_solver_scheme == "DIVERGENCE" ) {
        riemann_solver = new DivergenceFluxApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "MURMAN-ROE" ) {
        riemann_solver = new MurmanRoeFluxApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "KGP" ) {
        riemann_solver = new KgpFluxApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "SHIMA" ) {
        riemann_solver = new ShimaFluxApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "HLL" ) {
        riemann_solver = new HllApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "HLLC" ) {
        riemann_solver = new HllcApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "HLLC+" ) {
        riemann_solver = new HllcPlusApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "ECKEP" ) {
        riemann_solver = new EckepFluxApproximateRiemannSolver();
    } else if( riemann_solver_scheme == "HES" ) {
        riemann_solver = new HesFluxApproximateRiemannSolver();	
    } else {
        cout << "Riemann solver not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }*/
    riemann_solver = new _RiemannSolverScheme_();	// ... modified for OpenACC

    /// Construct (initialize) Runge-Kutta method
    /*if( runge_kutta_time_scheme == "RK1" ) {
        runge_kutta_method = new RungeKutta1Method();
    } else if( runge_kutta_time_scheme == "SSP-RK2" ) {
        runge_kutta_method = new StrongStabilityPreservingRungeKutta2Method();
    } else if( runge_kutta_time_scheme == "SSP-RK3" ) {
        runge_kutta_method = new StrongStabilityPreservingRungeKutta3Method();
    } else {
        cout << "Runge-Kutta time scheme not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }*/
    runge_kutta_method = new _ExplicitRungeKuttaMethod_();	// ... modified for OpenACC
    rk_number_stages = runge_kutta_method->getNumberRungeKuttaStages();

    /// Construct (initialize) immersed boundary method
    /*if( activate_immersed_boundary_method ) {
        immersed_boundary_method = new BaseImmersedBoundaryMethod( configuration_file );
    }*/
    immersed_boundary_method = new _ImmersedBoundaryMethod_( configuration_file ); // ... modified for OpenACC

    /// Construct (initialize) computational domain
    mesh = new ComputationalDomain(L_x, L_y, L_z, x_0, y_0, z_0, A_x, A_y, A_z, num_grid_x, num_grid_y, num_grid_z, external_mesh, external_mesh_file);

    /// Set boundary conditions to computational domain
    mesh->setBocos(bocos_type);

    /// Construct (initialize) parallel topology
    topo = new ParallelTopology(mesh, np_x, np_y, np_z);
    //if(topo->getRank() == 0) mesh->printDomain();
    //for(int p = 0; p < np_x*np_y*np_z; p++) topo->printCommSchemeToFile(p);

    /// Set local mesh values for I1D macro
    _lNx_ = topo->getlNx();
    _lNy_ = topo->getlNy();
    _lNz_ = topo->getlNz();
    _ls_  = _lNx_*_lNy_*_lNz_;

    #pragma acc enter data copyin(this, thermodynamics, transport_coefficients, riemann_solver, runge_kutta_method, immersed_boundary_method, mesh, topo)

    /// Set parallel topology of mesh coordinates
    x_field.setTopology(topo,"x");
    y_field.setTopology(topo,"y");
    z_field.setTopology(topo,"z");

    /// Set parallel topology of primitive, conserved, thermodynamic and thermophysical variables	
    rho_field.setTopology(topo,"rho");
    u_field.setTopology(topo,"u");
    v_field.setTopology(topo,"v");
    w_field.setTopology(topo,"w");
    E_field.setTopology(topo,"E");
    s_field.setTopology(topo,"s");
    rhou_field.setTopology(topo,"rhou");
    rhov_field.setTopology(topo,"rhov");
    rhow_field.setTopology(topo,"rhow");
    rhoE_field.setTopology(topo,"rhoE");
    P_field.setTopology(topo,"P");
    T_field.setTopology(topo,"T");
    sos_field.setTopology(topo,"sos");
    mu_field.setTopology(topo,"mu");
    kappa_field.setTopology(topo,"kappa");
    c_v_field.setTopology(topo,"c_v");
    c_p_field.setTopology(topo,"c_p");

    /// Set parallel topology of time-integration variables	
    rho_0_field.setTopology(topo,"rho_0");
    rhou_0_field.setTopology(topo,"rhou_0");
    rhov_0_field.setTopology(topo,"rhov_0");
    rhow_0_field.setTopology(topo,"rhow_0");
    rhoE_0_field.setTopology(topo,"rhoE_0");    
    P_0_field.setTopology(topo,"P_0");    

    /// Set parallel topology of inviscid fluxes	
    rho_inv_flux.setTopology(topo,"rho_inv");
    rhou_inv_flux.setTopology(topo,"rhou_inv");
    rhov_inv_flux.setTopology(topo,"rhov_inv");
    rhow_inv_flux.setTopology(topo,"rhow_inv");
    rhoE_inv_flux.setTopology(topo,"rhoE_inv");
    P_inv_flux.setTopology(topo,"P_inv");

    /// Set parallel topology of viscous fluxes	
    rhou_vis_flux.setTopology(topo,"rhou_vis");
    rhov_vis_flux.setTopology(topo,"rhov_vis");
    rhow_vis_flux.setTopology(topo,"rhow_vis");
    rhoE_vis_flux.setTopology(topo,"rhoE_vis");
    work_vis_rhoe_flux.setTopology(topo,"work_vis_rhoe");
    P_vis_flux.setTopology(topo,"P_vis");

    /// Set parallel topology of source terms
    f_rhou_field.setTopology(topo,"f_rhou");
    f_rhov_field.setTopology(topo,"f_rhov");
    f_rhow_field.setTopology(topo,"f_rhow");
    f_rhoE_field.setTopology(topo,"f_rhoE");

    /// Set parallel topology of time-averaged quantities
    avg_rho_field.setTopology(topo,"avg_rho");
    avg_rhou_field.setTopology(topo,"avg_rhou");
    avg_rhov_field.setTopology(topo,"avg_rhov");
    avg_rhow_field.setTopology(topo,"avg_rhow");
    avg_rhoE_field.setTopology(topo,"avg_rhoE");
    avg_rhoP_field.setTopology(topo,"avg_rhoP");
    avg_rhoT_field.setTopology(topo,"avg_rhoT");
    avg_u_field.setTopology(topo,"avg_u");
    avg_v_field.setTopology(topo,"avg_v");
    avg_w_field.setTopology(topo,"avg_w");
    avg_E_field.setTopology(topo,"avg_E");
    avg_s_field.setTopology(topo,"avg_s");
    avg_P_field.setTopology(topo,"avg_P");
    avg_T_field.setTopology(topo,"avg_T");
    avg_sos_field.setTopology(topo,"avg_sos");
    avg_mu_field.setTopology(topo,"avg_mu");
    avg_kappa_field.setTopology(topo,"avg_kappa");
    avg_c_v_field.setTopology(topo,"avg_c_v");
    avg_c_p_field.setTopology(topo,"avg_c_p");
    rmsf_rho_field.setTopology(topo,"rmsf_rho");
    rmsf_rhou_field.setTopology(topo,"rmsf_rhou");
    rmsf_rhov_field.setTopology(topo,"rmsf_rhov");
    rmsf_rhow_field.setTopology(topo,"rmsf_rhow");
    rmsf_rhoE_field.setTopology(topo,"rmsf_rhoE");
    rmsf_u_field.setTopology(topo,"rmsf_u");
    rmsf_v_field.setTopology(topo,"rmsf_v");
    rmsf_w_field.setTopology(topo,"rmsf_w");
    rmsf_E_field.setTopology(topo,"rmsf_E");
    rmsf_s_field.setTopology(topo,"rmsf_s");
    rmsf_P_field.setTopology(topo,"rmsf_P");
    rmsf_T_field.setTopology(topo,"rmsf_T");
    rmsf_sos_field.setTopology(topo,"rmsf_sos");
    rmsf_mu_field.setTopology(topo,"rmsf_mu");
    rmsf_kappa_field.setTopology(topo,"rmsf_kappa");
    rmsf_c_v_field.setTopology(topo,"rmsf_c_v");
    rmsf_c_p_field.setTopology(topo,"rmsf_c_p");
    favre_uffuff_field.setTopology(topo,"favre_uffuff");
    favre_uffvff_field.setTopology(topo,"favre_uffvff");
    favre_uffwff_field.setTopology(topo,"favre_uffwff");
    favre_vffvff_field.setTopology(topo,"favre_vffvff");
    favre_vffwff_field.setTopology(topo,"favre_vffwff");
    favre_wffwff_field.setTopology(topo,"favre_wffwff");
    favre_uffEff_field.setTopology(topo,"favre_uffEff");
    favre_vffEff_field.setTopology(topo,"favre_vffEff");
    favre_wffEff_field.setTopology(topo,"favre_wffEff");

    /// Set parallel topology of immersed boundary method (IBM) quantities
    tag_IBM_field.setTopology(topo,"tag_IBM");
    u_IBM_field.setTopology(topo,"u_IBM");
    v_IBM_field.setTopology(topo,"v_IBM");
    w_IBM_field.setTopology(topo,"w_IBM");
    T_IBM_field.setTopology(topo,"T_IBM");

    // Added for cylindrical coordinates
    det_Jacobian_field.setTopology(topo,"det_Jacobian_field");

    /// Fill mesh x, y, z, delta_x, delta_y, delta_z fields
    //this->fillMeshCoordinatesSizesFields();
    if (custom_mesh_filler != nullptr) {
        /// Run the injected CHILD logic, passing 'this' so it can access the fields
        custom_mesh_filler(this); 
    } else {
        /// Run the standard FATHER logic
        this->fillMeshCoordinatesSizesFields();
    }

    /// Initialize distributed point particles
    point_particles = new DistributedPointParticles( configuration_file );
    double xmin_local = 0.5*( x_field[I1D(0,0,0)] + x_field[I1D(1,0,0)] ); double xmax_local = 0.5*( x_field[I1D(_lNx_-2,0,0)] + x_field[I1D(_lNx_-1,0,0)] ); 
    double ymin_local = 0.5*( y_field[I1D(0,0,0)] + y_field[I1D(0,1,0)] ); double ymax_local = 0.5*( y_field[I1D(0,_lNy_-2,0)] + y_field[I1D(0,_lNy_-1,0)] ); 
    double zmin_local = 0.5*( z_field[I1D(0,0,0)] + z_field[I1D(0,0,1)] ); double zmax_local = 0.5*( z_field[I1D(0,0,_lNz_-2)] + z_field[I1D(0,0,_lNz_-1)] ); 
    point_particles->set_subdomains_distributed_prts( xmin_local, xmax_local, ymin_local, ymax_local, zmin_local, zmax_local );

    /// Construct (initialize) writer/reader
    char char_array[ output_data_file_name.length() + 1 ]; 
    strcpy( char_array, output_data_file_name.c_str() );
    writer_reader = new WriteReadHDF5( topo, char_array, generate_xdmf_file );
    writer_reader->addAttributeDouble( "Time");
    writer_reader->addAttributeInt( "Iteration" );
    writer_reader->addAttributeDouble( "AveragingTime" );
    writer_reader->addField(&x_field);
    writer_reader->addField(&y_field);
    writer_reader->addField(&z_field);
    writer_reader->addField(&rho_field);
    writer_reader->addField(&u_field);
    writer_reader->addField(&v_field);
    writer_reader->addField(&w_field);
    writer_reader->addField(&E_field);
    writer_reader->addField(&s_field);
    writer_reader->addField(&P_field);
    writer_reader->addField(&T_field);
    writer_reader->addField(&sos_field);
    writer_reader->addField(&mu_field);
    writer_reader->addField(&kappa_field);
    writer_reader->addField(&c_v_field);
    writer_reader->addField(&c_p_field);
    writer_reader->addField(&avg_rho_field);
    writer_reader->addField(&avg_rhou_field);
    writer_reader->addField(&avg_rhov_field);
    writer_reader->addField(&avg_rhow_field);
    writer_reader->addField(&avg_rhoE_field);
    writer_reader->addField(&avg_rhoP_field);
    writer_reader->addField(&avg_rhoT_field);
    writer_reader->addField(&avg_u_field);
    writer_reader->addField(&avg_v_field);
    writer_reader->addField(&avg_w_field);
    writer_reader->addField(&avg_E_field);
    writer_reader->addField(&avg_s_field);
    writer_reader->addField(&avg_P_field);
    writer_reader->addField(&avg_T_field);
    writer_reader->addField(&avg_sos_field);
    writer_reader->addField(&avg_mu_field);
    writer_reader->addField(&avg_kappa_field);
    writer_reader->addField(&avg_c_v_field);
    writer_reader->addField(&avg_c_p_field);
    writer_reader->addField(&rmsf_rho_field);
    writer_reader->addField(&rmsf_rhou_field);
    writer_reader->addField(&rmsf_rhov_field);
    writer_reader->addField(&rmsf_rhow_field);
    writer_reader->addField(&rmsf_rhoE_field);
    writer_reader->addField(&rmsf_u_field);
    writer_reader->addField(&rmsf_v_field);
    writer_reader->addField(&rmsf_w_field);
    writer_reader->addField(&rmsf_E_field);
    writer_reader->addField(&rmsf_s_field);
    writer_reader->addField(&rmsf_P_field);
    writer_reader->addField(&rmsf_T_field);
    writer_reader->addField(&rmsf_sos_field);
    writer_reader->addField(&rmsf_mu_field);
    writer_reader->addField(&rmsf_kappa_field);
    writer_reader->addField(&rmsf_c_v_field);
    writer_reader->addField(&rmsf_c_p_field);
    writer_reader->addField(&favre_uffuff_field);
    writer_reader->addField(&favre_uffvff_field);
    writer_reader->addField(&favre_uffwff_field);
    writer_reader->addField(&favre_vffvff_field);
    writer_reader->addField(&favre_vffwff_field);
    writer_reader->addField(&favre_wffwff_field);
    writer_reader->addField(&favre_uffEff_field);
    writer_reader->addField(&favre_vffEff_field);
    writer_reader->addField(&favre_wffEff_field);
    writer_reader->addField(&tag_IBM_field);
    //writer_reader->addField(&u_IBM_field);
    //writer_reader->addField(&v_IBM_field);
    //writer_reader->addField(&w_IBM_field);
    //writer_reader->addField(&T_IBM_field);
    
    /// Adjust location of 2d data output slices to corresponding closest grid point
    for(int dos = 0; dos < number_two_dimensional_data_output_slices; ++dos) {
        mesh->adjustLocationToClosestGridPoint(dos_x_positions[dos], dos_y_positions[dos], dos_z_positions[dos], x_field, y_field, z_field, topo);
    }
   
    /// Construct (initialize) temporal point probes
    TemporalPointProbe temporal_point_probe(mesh, topo);
    temporal_point_probes.resize( number_temporal_point_probes );
    for(int tpp = 0; tpp < number_temporal_point_probes; ++tpp) {
        /// Set parameters of temporal point probe
	temporal_point_probe.setPositionX( tpp_x_positions[tpp] );
	temporal_point_probe.setPositionY( tpp_y_positions[tpp] );
	temporal_point_probe.setPositionZ( tpp_z_positions[tpp] );
	temporal_point_probe.setOutputFileName( tpp_output_file_names[tpp] );
        /// Insert temporal point probe to vector
        temporal_point_probes[tpp] = temporal_point_probe;
	/// Locate closest grid point to probe
	temporal_point_probes[tpp].locateClosestGridPointToProbe(x_field, y_field, z_field);
    }	   

    /// Construct (initialize) timers
    timers = new ParallelTimer( print_timers );
    timers->createTimer( "execute" );
    timers->createTimer( "time_iteration_loop" );
    timers->createTimer( "calculate_time_step" );
    timers->createTimer( "output_solver_state" );
    timers->createTimer( "time_advance_point_particles" );
    timers->createTimer( "rk_iteration_loop" );
    timers->createTimer( "calculate_thermophysical_properties" );
    timers->createTimer( "calculate_inviscid_fluxes" );
    timers->createTimer( "calculate_viscous_fluxes" );
    timers->createTimer( "calculate_source_terms" );
    timers->createTimer( "immersed_boundary_method" );
    timers->createTimer( "time_advance_conserved_variables" );
    timers->createTimer( "conserved_to_primitive_variables" );
    timers->createTimer( "calculate_thermodynamics_from_primitive_variables" );
    timers->createTimer( "update_boundaries" );
    timers->createTimer( "temporal_hook_function" );
    timers->createTimer( "update_time_averaged_quantities" );
    timers->createTimer( "update_previous_state_conserved_variables" );
    
    #pragma acc enter data copyin(point_particles)
};

FlowSolverRHEA::~FlowSolverRHEA() {

    /// Free thermodynamics, transport_coefficients, riemann_solver, runge_kutta_method
    /// immersed_boundary_method, mesh, topo, point_particles, writer_reader and timers
    if( thermodynamics != NULL ) free( thermodynamics );
    if( transport_coefficients != NULL ) free( transport_coefficients );
    if( riemann_solver != NULL ) free( riemann_solver );
    if( runge_kutta_method != NULL ) free( runge_kutta_method );
    if( immersed_boundary_method != NULL ) free( immersed_boundary_method );
    if( mesh != NULL ) free( mesh );	
    if( topo != NULL ) free( topo );
    if( point_particles != NULL ) free( point_particles );
    if( writer_reader != NULL ) free( writer_reader );
    if( timers != NULL ) free( timers );

    #pragma acc exit data delete(this, thermodynamics, transport_coefficients, riemann_solver, runge_kutta_method, immersed_boundary_method, mesh, topo, point_particles)

};

void FlowSolverRHEA::readConfigurationFile() {

    /// Create YAML object
    YAML::Node configuration = YAML::LoadFile( configuration_file );

    /// Fluid properties
    const YAML::Node & fluid_flow_properties = configuration["fluid_flow_properties"];
    thermodynamic_model          = fluid_flow_properties["thermodynamic_model"].as<string>();
    transport_coefficients_model = fluid_flow_properties["transport_coefficients_model"].as<string>();

    /// Problem parameters
    const YAML::Node & problem_parameters = configuration["problem_parameters"];
    x_0          = problem_parameters["x_0"].as<double>();
    y_0          = problem_parameters["y_0"].as<double>();
    z_0          = problem_parameters["z_0"].as<double>();
    L_x          = problem_parameters["L_x"].as<double>();
    L_y          = problem_parameters["L_y"].as<double>();
    L_z          = problem_parameters["L_z"].as<double>();
    final_time   = problem_parameters["final_time"].as<double>();

    /// Computational parameters
    const YAML::Node & computational_parameters = configuration["computational_parameters"];
    num_grid_x                         = computational_parameters["num_grid_x"].as<int>();
    num_grid_y                         = computational_parameters["num_grid_y"].as<int>();
    num_grid_z                         = computational_parameters["num_grid_z"].as<int>();
    A_x                                = computational_parameters["A_x"].as<double>();
    A_y                                = computational_parameters["A_y"].as<double>();
    A_z                                = computational_parameters["A_z"].as<double>();
    external_mesh                      = computational_parameters["external_mesh"].as<bool>();
    external_mesh_file                 = computational_parameters["external_mesh_file"].as<string>();
    CFL                                = computational_parameters["CFL"].as<double>();
    riemann_solver_scheme              = computational_parameters["riemann_solver_scheme"].as<string>();
    runge_kutta_time_scheme            = computational_parameters["runge_kutta_time_scheme"].as<string>();
    transport_pressure_scheme          = computational_parameters["transport_pressure_scheme"].as<bool>();
    artificial_compressibility_method  = computational_parameters["artificial_compressibility_method"].as<bool>();
    epsilon_acm                        = computational_parameters["epsilon_acm"].as<double>();
    final_time_iter                    = computational_parameters["final_time_iter"].as<int>();

    /// Boundary conditions
    string dummy_type_boco;
    const YAML::Node & boundary_conditions = configuration["boundary_conditions"];
    /// West
    dummy_type_boco = boundary_conditions["west_bc"][0].as<string>();
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
        cout << "West boundary condition not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_WEST_] = boundary_conditions["west_bc"][1].as<double>();
    bocos_v[_WEST_] = boundary_conditions["west_bc"][2].as<double>();
    bocos_w[_WEST_] = boundary_conditions["west_bc"][3].as<double>();
    bocos_P[_WEST_] = boundary_conditions["west_bc"][4].as<double>();
    bocos_T[_WEST_] = boundary_conditions["west_bc"][5].as<double>();
    /// East
    dummy_type_boco = boundary_conditions["east_bc"][0].as<string>();
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
        cout << "East boundary condition not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_EAST_] = boundary_conditions["east_bc"][1].as<double>();
    bocos_v[_EAST_] = boundary_conditions["east_bc"][2].as<double>();
    bocos_w[_EAST_] = boundary_conditions["east_bc"][3].as<double>();
    bocos_P[_EAST_] = boundary_conditions["east_bc"][4].as<double>();
    bocos_T[_EAST_] = boundary_conditions["east_bc"][5].as<double>();
    /// South
    dummy_type_boco = boundary_conditions["south_bc"][0].as<string>();
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
        cout << "South boundary condition not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_SOUTH_] = boundary_conditions["south_bc"][1].as<double>();
    bocos_v[_SOUTH_] = boundary_conditions["south_bc"][2].as<double>();
    bocos_w[_SOUTH_] = boundary_conditions["south_bc"][3].as<double>();
    bocos_P[_SOUTH_] = boundary_conditions["south_bc"][4].as<double>();
    bocos_T[_SOUTH_] = boundary_conditions["south_bc"][5].as<double>();
    /// North
    dummy_type_boco = boundary_conditions["north_bc"][0].as<string>();
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
        cout << "North boundary condition not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_NORTH_] = boundary_conditions["north_bc"][1].as<double>();
    bocos_v[_NORTH_] = boundary_conditions["north_bc"][2].as<double>();
    bocos_w[_NORTH_] = boundary_conditions["north_bc"][3].as<double>();
    bocos_P[_NORTH_] = boundary_conditions["north_bc"][4].as<double>();
    bocos_T[_NORTH_] = boundary_conditions["north_bc"][5].as<double>();
    /// Back
    dummy_type_boco = boundary_conditions["back_bc"][0].as<string>();
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
        cout << "Back boundary condition not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_BACK_] = boundary_conditions["back_bc"][1].as<double>();
    bocos_v[_BACK_] = boundary_conditions["back_bc"][2].as<double>();
    bocos_w[_BACK_] = boundary_conditions["back_bc"][3].as<double>();
    bocos_P[_BACK_] = boundary_conditions["back_bc"][4].as<double>();
    bocos_T[_BACK_] = boundary_conditions["back_bc"][5].as<double>();
    /// Front
    dummy_type_boco = boundary_conditions["front_bc"][0].as<string>();
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
        cout << "Front boundary condition not available!" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    bocos_u[_FRONT_] = boundary_conditions["front_bc"][1].as<double>();
    bocos_v[_FRONT_] = boundary_conditions["front_bc"][2].as<double>();
    bocos_w[_FRONT_] = boundary_conditions["front_bc"][3].as<double>();
    bocos_P[_FRONT_] = boundary_conditions["front_bc"][4].as<double>();
    bocos_T[_FRONT_] = boundary_conditions["front_bc"][5].as<double>();

    /// Immersed boundary method
    const YAML::Node & immersed_boundary_method = configuration["immersed_boundary_method"];
    activate_immersed_boundary_method = immersed_boundary_method["activate_immersed_boundary_method"].as<bool>();

    /// Print/Write/Read file parameters
    const YAML::Node & print_write_read_parameters = configuration["print_write_read_parameters"];
    print_frequency_iter          = print_write_read_parameters["print_frequency_iter"].as<int>();
    output_data_file_name         = print_write_read_parameters["output_data_file_name"].as<string>();
    output_frequency_iter         = print_write_read_parameters["output_frequency_iter"].as<int>();
    generate_xdmf_file            = print_write_read_parameters["generate_xdmf_file"].as<bool>();
    use_restart                   = print_write_read_parameters["use_restart"].as<bool>();
    restart_data_file             = print_write_read_parameters["restart_data_file"].as<string>();
    time_averaging_active         = print_write_read_parameters["time_averaging_active"].as<bool>();
    reset_time_averaging          = print_write_read_parameters["reset_time_averaging"].as<bool>();

    /// 2D data output slices
    const YAML::Node & two_dimensional_data_output_slices = configuration["two_dimensional_data_output_slices"];
    number_two_dimensional_data_output_slices = two_dimensional_data_output_slices["number_two_dimensional_data_output_slices"].as<int>();
    dos_normal_directions.resize( number_two_dimensional_data_output_slices );
    dos_x_positions.resize( number_two_dimensional_data_output_slices );
    dos_y_positions.resize( number_two_dimensional_data_output_slices );
    dos_z_positions.resize( number_two_dimensional_data_output_slices );
    dos_output_frequency_iters.resize( number_two_dimensional_data_output_slices );
    dos_generate_xdmf_files.resize( number_two_dimensional_data_output_slices );
    dos_output_file_names.resize( number_two_dimensional_data_output_slices );
    string dos_yaml_input_name;
    for(int dos = 0; dos < number_two_dimensional_data_output_slices; ++dos) {
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_normal_direction";
        dos_normal_directions[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<string>();
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_x_position";
        dos_x_positions[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<double>();
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_y_position";
        dos_y_positions[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<double>();
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_z_position";
        dos_z_positions[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<double>();
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_output_frequency_iter";
        dos_output_frequency_iters[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<int>();
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_generate_xdmf_file";
        dos_generate_xdmf_files[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<bool>();
	dos_yaml_input_name = "slice_" + to_string( dos + 1 ) + "_output_data_file_name";
        dos_output_file_names[dos] = two_dimensional_data_output_slices[dos_yaml_input_name].as<string>();
    }

    /// Temporal point probes
    const YAML::Node & temporal_point_probes = configuration["temporal_point_probes"];
    number_temporal_point_probes = temporal_point_probes["number_temporal_point_probes"].as<int>();
    tpp_x_positions.resize( number_temporal_point_probes );
    tpp_y_positions.resize( number_temporal_point_probes );
    tpp_z_positions.resize( number_temporal_point_probes );
    tpp_output_frequency_iters.resize( number_temporal_point_probes );
    tpp_output_file_names.resize( number_temporal_point_probes );
    string tpp_yaml_input_name;
    for(int tpp = 0; tpp < number_temporal_point_probes; ++tpp) {
	tpp_yaml_input_name = "probe_" + to_string( tpp + 1 ) + "_x_position";
        tpp_x_positions[tpp] = temporal_point_probes[tpp_yaml_input_name].as<double>();
	tpp_yaml_input_name = "probe_" + to_string( tpp + 1 ) + "_y_position";
        tpp_y_positions[tpp] = temporal_point_probes[tpp_yaml_input_name].as<double>();
	tpp_yaml_input_name = "probe_" + to_string( tpp + 1 ) + "_z_position";
        tpp_z_positions[tpp] = temporal_point_probes[tpp_yaml_input_name].as<double>();
	tpp_yaml_input_name = "probe_" + to_string( tpp + 1 ) + "_output_frequency_iter";
        tpp_output_frequency_iters[tpp] = temporal_point_probes[tpp_yaml_input_name].as<int>();
	tpp_yaml_input_name = "probe_" + to_string( tpp + 1 ) + "_output_data_file_name";
        tpp_output_file_names[tpp] = temporal_point_probes[tpp_yaml_input_name].as<string>();
    }	    

    /// Lagrangian point particles
    const YAML::Node & lagrangian_point_particles = configuration["lagrangian_point_particles"];
    activate_pure_tracer_particles      = lagrangian_point_particles["activate_pure_tracer_particles"].as<bool>();
    activate_two_way_coupling_particles = lagrangian_point_particles["activate_two_way_coupling_particles"].as<bool>();
    buffer_ratio_particles              = lagrangian_point_particles["buffer_ratio_particles"].as<double>();
    output_frequency_iter_particles     = lagrangian_point_particles["output_frequency_iter_particles"].as<int>();
    use_restart_particles               = lagrangian_point_particles["use_restart_particles"].as<bool>();
    restart_data_file_particles         = lagrangian_point_particles["restart_data_file_particles"].as<string>();

    /// Timers information
    const YAML::Node & timers_information = configuration["timers_information"];
    print_timers = timers_information["print_timers"].as<bool>();
    timers_information_file = timers_information["timers_information_file"].as<string>();

    /// Parallelization scheme
    const YAML::Node & parallelization_scheme = configuration["parallelization_scheme"];
    np_x = parallelization_scheme["np_x"].as<int>();
    np_y = parallelization_scheme["np_y"].as<int>();
    np_z = parallelization_scheme["np_z"].as<int>();

};

void FlowSolverRHEA::fillMeshCoordinatesSizesFields() {

    /// All (inner, halo, boundary) points: x, y and z
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                x_field[I1D(i,j,k)] = mesh->x[i];
                y_field[I1D(i,j,k)] = mesh->y[j];
                z_field[I1D(i,j,k)] = mesh->z[k];
            }
        }
    }

    /// Update halo values (do not activate!)
    //x_field.update();
    //y_field.update();
    //z_field.update();

    #pragma acc update device(x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_])

};

void FlowSolverRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// All (inner, halo, boundary): u, v, w, P and T
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] = 0.0;
                v_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = 0.0;
                P_field[I1D(i,j,k)] = 0.0;
                T_field[I1D(i,j,k)] = 0.0;
            }
        }
    }

    /// Update halo values
    u_field.update();
    v_field.update();
    w_field.update();
    P_field.update();
    T_field.update();

};

void FlowSolverRHEA::initializeFromRestart() {

    /// Read from file to restart solver: data, time and time iteration
    char char_restart_data_file[ restart_data_file.length() + 1 ]; 
    strcpy( char_restart_data_file, restart_data_file.c_str() );
    writer_reader->read( char_restart_data_file );
    current_time      = writer_reader->getAttributeDouble( "Time" );
    current_time_iter = writer_reader->getAttributeInt( "Iteration" );
    averaging_time    = writer_reader->getAttributeDouble( "AveragingTime" );
    if( reset_time_averaging ) {

	/// Reset time averaging
        averaging_time = 0.0;

	/// Reset avg and fluctuating fields
        avg_rho_field       = 0.0;
        avg_rhou_field      = 0.0;
        avg_rhov_field      = 0.0;
        avg_rhow_field      = 0.0;
        avg_rhoE_field      = 0.0;
        avg_rhoP_field      = 0.0;
        avg_rhoT_field      = 0.0;
        avg_u_field         = 0.0;
        avg_v_field         = 0.0;
        avg_w_field         = 0.0;
        avg_E_field         = 0.0;
        avg_s_field         = 0.0;
        avg_P_field         = 0.0;
        avg_T_field         = 0.0;
        avg_sos_field       = 0.0;
        avg_mu_field        = 0.0;
        avg_kappa_field     = 0.0;
        avg_c_v_field       = 0.0;
        avg_c_p_field       = 0.0;
        rmsf_rho_field      = 0.0;
        rmsf_rhou_field     = 0.0;
        rmsf_rhov_field     = 0.0;
        rmsf_rhow_field     = 0.0;
        rmsf_rhoE_field     = 0.0;
        rmsf_u_field        = 0.0;
        rmsf_v_field        = 0.0;
        rmsf_w_field        = 0.0;
        rmsf_E_field        = 0.0;
        rmsf_s_field        = 0.0;
        rmsf_P_field        = 0.0;
        rmsf_T_field        = 0.0;
        rmsf_sos_field      = 0.0;
        rmsf_mu_field       = 0.0;
        rmsf_kappa_field    = 0.0;
        rmsf_c_v_field      = 0.0;
        rmsf_c_p_field      = 0.0;
        favre_uffuff_field = 0.0;
        favre_uffvff_field = 0.0;
        favre_uffwff_field = 0.0;
        favre_vffvff_field = 0.0;
        favre_vffwff_field = 0.0;
        favre_wffwff_field = 0.0;
        favre_uffEff_field = 0.0;
        favre_vffEff_field = 0.0;
        favre_wffEff_field = 0.0;

    }
    
    #pragma acc update device(averaging_time, rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], avg_rho_field.vector[0:_ls_], avg_rhou_field.vector[0:_ls_], avg_rhov_field.vector[0:_ls_], avg_rhow_field.vector[0:_ls_], avg_rhoE_field.vector[0:_ls_], avg_rhoP_field.vector[0:_ls_], avg_rhoT_field.vector[0:_ls_], avg_u_field.vector[0:_ls_], avg_v_field.vector[0:_ls_], avg_w_field.vector[0:_ls_], avg_E_field.vector[0:_ls_], avg_s_field.vector[0:_ls_], avg_P_field.vector[0:_ls_], avg_T_field.vector[0:_ls_], avg_sos_field.vector[0:_ls_], avg_mu_field.vector[0:_ls_], avg_kappa_field.vector[0:_ls_], avg_c_v_field.vector[0:_ls_], avg_c_p_field.vector[0:_ls_], rmsf_rho_field.vector[0:_ls_], rmsf_rhou_field.vector[0:_ls_], rmsf_rhov_field.vector[0:_ls_], rmsf_rhow_field.vector[0:_ls_], rmsf_rhoE_field.vector[0:_ls_], rmsf_u_field.vector[0:_ls_], rmsf_v_field.vector[0:_ls_], rmsf_w_field.vector[0:_ls_], rmsf_E_field.vector[0:_ls_], rmsf_s_field.vector[0:_ls_], rmsf_P_field.vector[0:_ls_], rmsf_T_field.vector[0:_ls_], rmsf_sos_field.vector[0:_ls_], rmsf_mu_field.vector[0:_ls_], rmsf_kappa_field.vector[0:_ls_], rmsf_c_v_field.vector[0:_ls_], rmsf_c_p_field.vector[0:_ls_], favre_uffuff_field.vector[0:_ls_], favre_uffvff_field.vector[0:_ls_], favre_uffwff_field.vector[0:_ls_], favre_vffvff_field.vector[0:_ls_], favre_vffwff_field.vector[0:_ls_], favre_wffwff_field.vector[0:_ls_], favre_uffEff_field.vector[0:_ls_], favre_vffEff_field.vector[0:_ls_], favre_wffEff_field.vector[0:_ls_], tag_IBM_field.vector[0:_ls_]) 

    /// Update halo values
    rho_field.update();
    u_field.update();
    v_field.update();
    w_field.update();
    E_field.update();
    s_field.update();
    P_field.update();
    T_field.update();
    sos_field.update();
    mu_field.update();
    kappa_field.update();
    c_v_field.update();
    c_p_field.update();
    avg_rho_field.update();
    avg_rhou_field.update();
    avg_rhov_field.update();
    avg_rhow_field.update();
    avg_rhoE_field.update();
    avg_rhoP_field.update();
    avg_rhoT_field.update();
    avg_u_field.update();
    avg_v_field.update();
    avg_w_field.update();
    avg_E_field.update();
    avg_s_field.update();
    avg_P_field.update();
    avg_T_field.update();
    avg_sos_field.update();
    avg_mu_field.update();
    avg_kappa_field.update();
    avg_c_v_field.update();
    avg_c_p_field.update();
    rmsf_rho_field.update();
    rmsf_rhou_field.update();
    rmsf_rhov_field.update();
    rmsf_rhow_field.update();
    rmsf_rhoE_field.update();
    rmsf_u_field.update();
    rmsf_v_field.update();
    rmsf_w_field.update();
    rmsf_E_field.update();
    rmsf_s_field.update();
    rmsf_P_field.update();
    rmsf_T_field.update();
    rmsf_sos_field.update();
    rmsf_mu_field.update();
    rmsf_kappa_field.update();
    rmsf_c_v_field.update();
    rmsf_c_p_field.update();
    favre_uffuff_field.update();
    favre_uffvff_field.update();
    favre_uffwff_field.update();
    favre_vffvff_field.update();
    favre_vffwff_field.update();
    favre_wffwff_field.update();
    favre_uffEff_field.update();
    favre_vffEff_field.update();
    favre_wffEff_field.update();
    tag_IBM_field.update();
    //u_IBM_field.update();
    //v_IBM_field.update();
    //w_IBM_field.update();
    //T_IBM_field.update();

#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], avg_rho_field.vector[0:_ls_], avg_rhou_field.vector[0:_ls_], avg_rhov_field.vector[0:_ls_], avg_rhow_field.vector[0:_ls_], avg_rhoE_field.vector[0:_ls_], avg_rhoP_field.vector[0:_ls_], avg_rhoT_field.vector[0:_ls_], avg_u_field.vector[0:_ls_], avg_v_field.vector[0:_ls_], avg_w_field.vector[0:_ls_], avg_E_field.vector[0:_ls_], avg_s_field.vector[0:_ls_], avg_P_field.vector[0:_ls_], avg_T_field.vector[0:_ls_], avg_sos_field.vector[0:_ls_], avg_mu_field.vector[0:_ls_], avg_kappa_field.vector[0:_ls_], avg_c_v_field.vector[0:_ls_], avg_c_p_field.vector[0:_ls_], rmsf_rho_field.vector[0:_ls_], rmsf_rhou_field.vector[0:_ls_], rmsf_rhov_field.vector[0:_ls_], rmsf_rhow_field.vector[0:_ls_], rmsf_rhoE_field.vector[0:_ls_], rmsf_u_field.vector[0:_ls_], rmsf_v_field.vector[0:_ls_], rmsf_w_field.vector[0:_ls_], rmsf_E_field.vector[0:_ls_], rmsf_s_field.vector[0:_ls_], rmsf_P_field.vector[0:_ls_], rmsf_T_field.vector[0:_ls_], rmsf_sos_field.vector[0:_ls_], rmsf_mu_field.vector[0:_ls_], rmsf_kappa_field.vector[0:_ls_], rmsf_c_v_field.vector[0:_ls_], rmsf_c_p_field.vector[0:_ls_], favre_uffuff_field.vector[0:_ls_], favre_uffvff_field.vector[0:_ls_], favre_uffwff_field.vector[0:_ls_], favre_vffvff_field.vector[0:_ls_], favre_vffwff_field.vector[0:_ls_], favre_wffwff_field.vector[0:_ls_], favre_uffEff_field.vector[0:_ls_], favre_vffEff_field.vector[0:_ls_], favre_wffEff_field.vector[0:_ls_], tag_IBM_field.vector[0:_ls_])
#endif

    /// Fill mesh x, y, z, delta_x, delta_y, delta_z fields
    this->fillMeshCoordinatesSizesFields();

};

void FlowSolverRHEA::initializeThermodynamics() {

    /// All (inner, halo, boundary): rho, E, s, sos, c_v and c_p
    double rho, e, ke, c_v, c_p;
    //#pragma acc update host(rho_field.vector[0:_ls_],E_field.vector[0:_ls_],s_field.vector[0:_ls_],sos_field.vector[0:_ls_],c_v_field.vector[0:_ls_],c_p_field.vector[0:_ls_])	
    #pragma acc parallel loop collapse(3) present(this, rho_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], P_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_]) private(e, rho, ke, c_v, c_p)
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho, e, P_field[I1D(i,j,k)], T_field[I1D(i,j,k)] );
                rho_field[I1D(i,j,k)] = rho;
                ke                    = 0.5*( pow( u_field[I1D(i,j,k)], 2.0 ) + pow( v_field[I1D(i,j,k)], 2.0 ) + pow( w_field[I1D(i,j,k)], 2.0 ) );
                E_field[I1D(i,j,k)]   = e + ke;
                s_field[I1D(i,j,k)]   = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(rho_field.vector[0:_ls_],E_field.vector[0:_ls_],s_field.vector[0:_ls_],sos_field.vector[0:_ls_],c_v_field.vector[0:_ls_],c_p_field.vector[0:_ls_])	
#endif
    /// Update halo values
    rho_field.update();
    E_field.update();
    s_field.update();
    sos_field.update();
    c_v_field.update();
    c_p_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(rho_field.vector[0:_ls_],E_field.vector[0:_ls_],s_field.vector[0:_ls_],sos_field.vector[0:_ls_],c_v_field.vector[0:_ls_],c_p_field.vector[0:_ls_])	
#endif

};

void FlowSolverRHEA::primitiveToConservedVariables() {
    
    /// All (inner, halo, boundary): rhou, rhov, rhow and rhoE
    #pragma acc parallel loop collapse(3) present(this, rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_])
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                rhou_field[I1D(i,j,k)] = rho_field[I1D(i,j,k)]*u_field[I1D(i,j,k)]; 
                rhov_field[I1D(i,j,k)] = rho_field[I1D(i,j,k)]*v_field[I1D(i,j,k)]; 
                rhow_field[I1D(i,j,k)] = rho_field[I1D(i,j,k)]*w_field[I1D(i,j,k)]; 
                rhoE_field[I1D(i,j,k)] = rho_field[I1D(i,j,k)]*E_field[I1D(i,j,k)]; 
            }
        }
    }

    /// Update halo values
    //rhou_field.update();
    //rhov_field.update();
    //rhow_field.update();
    //rhoE_field.update();

};

void FlowSolverRHEA::conservedToPrimitiveVariables() {

    /// All (inner, halo, boundary) points: u, v, w and E
    //#pragma acc kernels loop collapse(3) independent 
    #pragma acc parallel loop collapse(3) present(this, u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], rho_field.vector[0:_ls_])	
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] = rhou_field[I1D(i,j,k)]/rho_field[I1D(i,j,k)]; 
                v_field[I1D(i,j,k)] = rhov_field[I1D(i,j,k)]/rho_field[I1D(i,j,k)]; 
                w_field[I1D(i,j,k)] = rhow_field[I1D(i,j,k)]/rho_field[I1D(i,j,k)]; 
                E_field[I1D(i,j,k)] = rhoE_field[I1D(i,j,k)]/rho_field[I1D(i,j,k)]; 
            }
        }
    }

    /// Update halo values
    //u_field.update();
    //v_field.update();
    //w_field.update();
    //E_field.update();

};

void FlowSolverRHEA::calculateThermodynamicsFromPrimitiveVariables() {
    
    if( transport_pressure_scheme ) {

	/// All (inner, halo, boundary) points: T, E, s, rhoE, sos, c_v and c_p
        double T, e, ke, c_v, c_p = 0.0;
	#pragma acc parallel loop collapse(3) private(T, e, ke, c_v, c_p) present(this, T_field.vector[0:_ls_], P_field.vector[0:_ls_], rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics)	
        for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
	            T = T_field[I1D(i,j,k)]; 
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T, P_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    T_field[I1D(i,j,k)]    = T;
		    P_field[I1D(i,j,k)]    = thermodynamics->calculatePressureFromTemperatureDensity( T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );	/// Added to improve pressure accuracy
                    e  = thermodynamics->calculateInternalEnergyFromPressureTemperatureDensity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] ); 
                    ke = 0.5*( pow( u_field[I1D(i,j,k)], 2.0 ) + pow( v_field[I1D(i,j,k)], 2.0 ) + pow( w_field[I1D(i,j,k)], 2.0 ) ); 
                    E_field[I1D(i,j,k)]    = e + ke;
                    s_field[I1D(i,j,k)]    = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    rhoE_field[I1D(i,j,k)] = rho_field[I1D(i,j,k)]*E_field[I1D(i,j,k)];
                    sos_field[I1D(i,j,k)]  = thermodynamics->calculateSoundSpeed( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    c_v_field[I1D(i,j,k)]  = c_v;
                    c_p_field[I1D(i,j,k)]  = c_p;
                }
            }
        }

        /// Update halo values
        //T_field.update();
        //E_field.update();
        //s_field.update();
        //rhoE_field.update();
        //sos_field.update();
        //c_v_field.update();
        //c_p_field.update();
		    
    } else {

        /// All (inner, halo, boundary) points: P, T, sos, c_v and c_p
        double ke, e, P, T, c_v, c_p = 0.0;
	//#pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],E_field.vector[0:_ls_],s_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_],rho_field.vector[0:_ls_],c_p_field.vector[0:_ls_],c_v_field.vector[0:_ls_],sos_field.vector[0:_ls_])
        #pragma acc parallel loop collapse(3) private(ke,e,c_v,P,T,c_p) present(this, topo, thermodynamics, u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], rho_field.vector[0:_ls_], c_p_field.vector[0:_ls_], c_v_field.vector[0:_ls_], sos_field.vector[0:_ls_])
	for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                    ke = 0.5*( pow( u_field[I1D(i,j,k)], 2.0 ) + pow( v_field[I1D(i,j,k)], 2.0 ) + pow( w_field[I1D(i,j,k)], 2.0 ) ); 
                    e  = E_field[I1D(i,j,k)] - ke;
                    if( artificial_compressibility_method ) {
		        P = P_thermo;			/// Initial pressure guess
                    } else {
                        P = P_field[I1D(i,j,k)];	/// Initial pressure guess
		    }
                    T = T_field[I1D(i,j,k)]; 		/// Initial temperature guess
                    thermodynamics->calculatePressureTemperatureFromDensityInternalEnergy( P, T, rho_field[I1D(i,j,k)], e );
                    P_field[I1D(i,j,k)]   = P; 
                    T_field[I1D(i,j,k)]   = T; 
		    P_field[I1D(i,j,k)]   = thermodynamics->calculatePressureFromTemperatureDensity( T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );	/// Added to improve pressure accuracy
                    s_field[I1D(i,j,k)]   = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                    c_v_field[I1D(i,j,k)] = c_v;
                    c_p_field[I1D(i,j,k)] = c_p;
                }
            }
        }

        /// Update halo values
	//#pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_],E_field.vector[0:_ls_],s_field.vector[0:_ls_],P_field.vector[0:_ls_],T_field.vector[0:_ls_],rho_field.vector[0:_ls_],c_p_field.vector[0:_ls_],c_v_field.vector[0:_ls_],sos_field.vector[0:_ls_])
        //P_field.update();
        //T_field.update();
        //s_field.update();
        //sos_field.update();
        //c_v_field.update();
        //c_p_field.update();
        
    }

};

void FlowSolverRHEA::updateBoundaries() {

    /// General form: w_g*phi_g + w_in*phi_in = phi_b
    /// phi_g is ghost cell value
    /// phi_in is inner cell value
    /// phi_b is boundary value/flux
    /// w_g is ghost cell weight
    /// w_in is inner cell weight

    /// Declare weights and ghost & inner values
    double wg_g = 0.0, wg_in = 0.0;
    double u_g, v_g, w_g, P_g, T_g, rho_g, e_g, ke_g, E_g;
    double u_in, v_in, w_in, P_in, T_in;
    double Delta_g;

    /// Parameters iterative solver for subsonic NSCBC
    int max_iter = 10;
    double rel_tol = 1.0e-5;
    
    /// West boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, x_field.vector[0:_ls_])  
    for(int i = topo->iter_bound[_WEST_][_INIX_]; i <= topo->iter_bound[_WEST_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_WEST_][_INIY_]; j <= topo->iter_bound[_WEST_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_WEST_][_INIZ_]; k <= topo->iter_bound[_WEST_][_ENDZ_]; k++) {
                if( ( bocos_type[_WEST_] == _DIRICHLET_ ) or ( bocos_type[_WEST_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_WEST_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) - x_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( x_field[I1D(i+1,j,k)] - 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
                if( bocos_type[_WEST_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i+1,j,k)];
                v_in = v_field[I1D(i+1,j,k)];
                w_in = w_field[I1D(i+1,j,k)];
                P_in = P_field[I1D(i+1,j,k)];
                T_in = T_field[I1D(i+1,j,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_WEST_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_WEST_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_WEST_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_WEST_] == _DIRICHLET_ ) and ( bocos_P[_WEST_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_WEST_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_WEST_] == _DIRICHLET_ ) and ( bocos_T[_WEST_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_WEST_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_WEST_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in_in = x_field[I1D(i+2,j,k)] - x_field[I1D(i+1,j,k)];
		    double rho_in = rho_field[I1D(i+1,j,k)]; 
		    double sos_in = sos_field[I1D(i+1,j,k)];
		    double drho_dx_in_in = ( rho_field[I1D(i+2,j,k)] - rho_field[I1D(i+1,j,k)] )/Delta_in_in;
		    double du_dx_in_in   = (   u_field[I1D(i+2,j,k)] -   u_field[I1D(i+1,j,k)] )/Delta_in_in;
		    double dP_dx_in_in   = (   P_field[I1D(i+2,j,k)] -   P_field[I1D(i+1,j,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dx_in_in - rho_in*sos_in*du_dx_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_WEST_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
                    bool aitken_converged = false;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter && !aitken_converged; ite++ ) {
                        /// Aitken's x_1
                        double drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        double du_dx_g   = (   u_in -   u_g )/Delta_g;
                        double dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        du_dx_g   = (   u_in -   u_g )/Delta_g;
                        dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            aitken_converged = true;		/// Converged: loop condition will stop further iterations
			} else {
				x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
			}
		    }
		} else if( bocos_type[_WEST_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in = x_field[I1D(i+2,j,k)] - x_field[I1D(i+1,j,k)];
		    double rho_in = rho_field[I1D(i+1,j,k)]; 
		    double sos_in = sos_field[I1D(i+1,j,k)];
		    double Ma_in  = u_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_x;
		    double drho_dx_in = ( rho_field[I1D(i+2,j,k)] - rho_field[I1D(i+1,j,k)] )/Delta_in;
		    double du_dx_in   = (   u_field[I1D(i+2,j,k)] -   u_field[I1D(i+1,j,k)] )/Delta_in;
		    double dv_dx_in   = (   v_field[I1D(i+2,j,k)] -   v_field[I1D(i+1,j,k)] )/Delta_in;
		    double dw_dx_in   = (   w_field[I1D(i+2,j,k)] -   w_field[I1D(i+1,j,k)] )/Delta_in;
		    double dP_dx_in   = (   P_field[I1D(i+2,j,k)] -   P_field[I1D(i+1,j,k)] )/Delta_in;
		    double lambda_5_in = u_in + sos_in;
                    double L_1_lambda_1_in = dP_dx_in - rho_in*sos_in*du_dx_in;
                    double L_2_lambda_2_in = sos_in*sos_in*drho_dx_in - dP_dx_in;
                    double L_3_lambda_3_in = dv_dx_in;
                    double L_4_lambda_4_in = dw_dx_in;
		    double L_5_lambda_5_in = K_in*( P_in - bocos_P[_WEST_] )/lambda_5_in;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_3_dx_in = L_3_lambda_3_in;
                    double dQ_4_dx_in = L_4_lambda_4_in;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_WEST_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i+1,j,k)]; 
		    double sos_in = sos_field[I1D(i+1,j,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_WEST_], bocos_T[_WEST_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_WEST_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_WEST_] - wg_in*T_in )/wg_g;
		    double drho_dx_g = ( rho_field[I1D(i+1,j,k)] - rho_g )/Delta_g;
		    double du_dx_g   = (   u_field[I1D(i+1,j,k)] - u_g   )/Delta_g;
		    double dv_dx_g   = (   v_field[I1D(i+1,j,k)] - v_g   )/Delta_g;
		    double dw_dx_g   = (   w_field[I1D(i+1,j,k)] - w_g   )/Delta_g;
		    double dP_dx_g   = (   P_field[I1D(i+1,j,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dx_g - rho_in*sos_in*du_dx_g;
                    double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                    double L_3_lambda_3_g = dv_dx_g;
                    double L_4_lambda_4_g = dw_dx_g;
                    double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_3_dx_in = L_3_lambda_3_g;
                    double dQ_4_dx_in = L_4_lambda_4_g;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );			
		} else if( bocos_type[_WEST_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// East boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, x_field.vector[0:_ls_]) 
    for(int i = topo->iter_bound[_EAST_][_INIX_]; i <= topo->iter_bound[_EAST_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_EAST_][_INIY_]; j <= topo->iter_bound[_EAST_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_EAST_][_INIZ_]; k <= topo->iter_bound[_EAST_][_ENDZ_]; k++) {
                if( ( bocos_type[_EAST_] == _DIRICHLET_ ) or ( bocos_type[_EAST_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_EAST_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( x_field[I1D(i+1,j,k)] - 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( 0.5*( x_field[I1D(i,j,k)] + x_field[I1D(i+1,j,k)] ) - x_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
                if( bocos_type[_EAST_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i-1,j,k)];
                v_in = v_field[I1D(i-1,j,k)];
                w_in = w_field[I1D(i-1,j,k)];
                P_in = P_field[I1D(i-1,j,k)];
                T_in = T_field[I1D(i-1,j,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_EAST_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_EAST_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_EAST_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_EAST_] == _DIRICHLET_ ) and ( bocos_P[_EAST_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_EAST_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_EAST_] == _DIRICHLET_ ) and ( bocos_T[_EAST_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_EAST_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_EAST_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = x_field[I1D(i-1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in_in = x_field[I1D(i-2,j,k)] - x_field[I1D(i-1,j,k)];
		    double rho_in = rho_field[I1D(i-1,j,k)]; 
		    double sos_in = sos_field[I1D(i-1,j,k)];
		    double drho_dx_in_in = ( rho_field[I1D(i-2,j,k)] - rho_field[I1D(i-1,j,k)] )/Delta_in_in;
		    double du_dx_in_in   = (   u_field[I1D(i-2,j,k)] -   u_field[I1D(i-1,j,k)] )/Delta_in_in;
		    double dP_dx_in_in   = (   P_field[I1D(i-2,j,k)] -   P_field[I1D(i-1,j,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dx_in_in - rho_in*sos_in*du_dx_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_EAST_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
                    bool aitken_converged = false;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter && !aitken_converged; ite++ ) {
                        /// Aitken's x_1
                        double drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        double du_dx_g   = (   u_in -   u_g )/Delta_g;
                        double dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dx_g = ( rho_in - rho_g )/Delta_g;
                        du_dx_g   = (   u_in -   u_g )/Delta_g;
                        dP_dx_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                        L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                        dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dx_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            aitken_converged = true;		/// Converged: loop condition will stop further iterations
			} else {
				x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
			}
		    }
		} else if( bocos_type[_EAST_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = x_field[I1D(i-1,j,k)] - x_field[I1D(i,j,k)];
                    double Delta_in = x_field[I1D(i-2,j,k)] - x_field[I1D(i-1,j,k)];
		    double rho_in = rho_field[I1D(i-1,j,k)]; 
		    double sos_in = sos_field[I1D(i-1,j,k)];
		    double Ma_in  = u_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_x;
		    double drho_dx_in = ( rho_field[I1D(i-2,j,k)] - rho_field[I1D(i-1,j,k)] )/Delta_in;
		    double du_dx_in   = (   u_field[I1D(i-2,j,k)] -   u_field[I1D(i-1,j,k)] )/Delta_in;
		    double dv_dx_in   = (   v_field[I1D(i-2,j,k)] -   v_field[I1D(i-1,j,k)] )/Delta_in;
		    double dw_dx_in   = (   w_field[I1D(i-2,j,k)] -   w_field[I1D(i-1,j,k)] )/Delta_in;
		    double dP_dx_in   = (   P_field[I1D(i-2,j,k)] -   P_field[I1D(i-1,j,k)] )/Delta_in;
		    double lambda_1_in = u_in - sos_in;
		    double L_1_lambda_1_in = K_in*( P_in - bocos_P[_EAST_] )/lambda_1_in;
                    double L_2_lambda_2_in = sos_in*sos_in*drho_dx_in - dP_dx_in;
                    double L_3_lambda_3_in = dv_dx_in;
                    double L_4_lambda_4_in = dw_dx_in;
                    double L_5_lambda_5_in = dP_dx_in + rho_in*sos_in*du_dx_in;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_3_dx_in = L_3_lambda_3_in;
                    double dQ_4_dx_in = L_4_lambda_4_in;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_EAST_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = x_field[I1D(i-1,j,k)] - x_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i-1,j,k)]; 
		    double sos_in = sos_field[I1D(i-1,j,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_EAST_], bocos_T[_EAST_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_EAST_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_EAST_] - wg_in*T_in )/wg_g;
		    double drho_dx_g = ( rho_field[I1D(i-1,j,k)] - rho_g )/Delta_g;
		    double du_dx_g   = (   u_field[I1D(i-1,j,k)] - u_g   )/Delta_g;
		    double dv_dx_g   = (   v_field[I1D(i-1,j,k)] - v_g   )/Delta_g;
		    double dw_dx_g   = (   w_field[I1D(i-1,j,k)] - w_g   )/Delta_g;
		    double dP_dx_g   = (   P_field[I1D(i-1,j,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dx_g - rho_in*sos_in*du_dx_g;
                    double L_2_lambda_2_g = sos_in*sos_in*drho_dx_g - dP_dx_g;
                    double L_3_lambda_3_g = dv_dx_g;
                    double L_4_lambda_4_g = dw_dx_g;
                    double L_5_lambda_5_g = dP_dx_g + rho_in*sos_in*du_dx_g;
                    double dQ_1_dx_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dx_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_3_dx_in = L_3_lambda_3_g;
                    double dQ_4_dx_in = L_4_lambda_4_g;
                    double dQ_5_dx_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dx_in;
                    u_g   = u_in   - Delta_g*dQ_2_dx_in;
                    v_g   = v_in   - Delta_g*dQ_3_dx_in;
                    w_g   = w_in   - Delta_g*dQ_4_dx_in;
                    P_g   = P_in   - Delta_g*dQ_5_dx_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_EAST_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;		
            }
        }
    }

    /// South boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, y_field.vector[0:_ls_]) 
    for(int i = topo->iter_bound[_SOUTH_][_INIX_]; i <= topo->iter_bound[_SOUTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_SOUTH_][_INIY_]; j <= topo->iter_bound[_SOUTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_SOUTH_][_INIZ_]; k <= topo->iter_bound[_SOUTH_][_ENDZ_]; k++) {
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) or ( bocos_type[_SOUTH_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_SOUTH_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) - y_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( y_field[I1D(i,j+1,k)] - 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
                if( bocos_type[_SOUTH_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j+1,k)];
                v_in = v_field[I1D(i,j+1,k)];
                w_in = w_field[I1D(i,j+1,k)];
                P_in = P_field[I1D(i,j+1,k)];
                T_in = T_field[I1D(i,j+1,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_SOUTH_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_SOUTH_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_SOUTH_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_P[_SOUTH_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_SOUTH_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_SOUTH_] == _DIRICHLET_ ) and ( bocos_T[_SOUTH_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_SOUTH_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_SOUTH_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in_in = y_field[I1D(i,j+2,k)] - y_field[I1D(i,j+1,k)];
		    double rho_in = rho_field[I1D(i,j+1,k)]; 
		    double sos_in = sos_field[I1D(i,j+1,k)];
		    double drho_dy_in_in = ( rho_field[I1D(i,j+2,k)] - rho_field[I1D(i,j+1,k)] )/Delta_in_in;
		    double dv_dy_in_in   = (   v_field[I1D(i,j+2,k)] -   v_field[I1D(i,j+1,k)] )/Delta_in_in;
		    double dP_dy_in_in   = (   P_field[I1D(i,j+2,k)] -   P_field[I1D(i,j+1,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dy_in_in - rho_in*sos_in*dv_dy_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_SOUTH_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
                    bool aitken_converged = false;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter && !aitken_converged; ite++ ) {
                        /// Aitken's x_1
                        double drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        double dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        double dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            aitken_converged = true;		/// Converged: loop condition will stop further iterations
			} else {
				x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
			}
		    }
		} else if( bocos_type[_SOUTH_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in = y_field[I1D(i,j+2,k)] - y_field[I1D(i,j+1,k)];
		    double rho_in = rho_field[I1D(i,j+1,k)]; 
		    double sos_in = sos_field[I1D(i,j+1,k)];
		    double Ma_in  = v_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_y;
		    double drho_dy_in = ( rho_field[I1D(i,j+2,k)] - rho_field[I1D(i,j+1,k)] )/Delta_in;
		    double du_dy_in   = (   u_field[I1D(i,j+2,k)] -   u_field[I1D(i,j+1,k)] )/Delta_in;
		    double dv_dy_in   = (   v_field[I1D(i,j+2,k)] -   v_field[I1D(i,j+1,k)] )/Delta_in;
		    double dw_dy_in   = (   w_field[I1D(i,j+2,k)] -   w_field[I1D(i,j+1,k)] )/Delta_in;
		    double dP_dy_in   = (   P_field[I1D(i,j+2,k)] -   P_field[I1D(i,j+1,k)] )/Delta_in;
		    double lambda_5_in = v_in + sos_in;
                    double L_1_lambda_1_in = dP_dy_in - rho_in*sos_in*dv_dy_in;
                    double L_2_lambda_2_in = du_dy_in;
                    double L_3_lambda_3_in = sos_in*sos_in*drho_dy_in - dP_dy_in;
                    double L_4_lambda_4_in = dw_dy_in;
		    double L_5_lambda_5_in = K_in*( P_in - bocos_P[_SOUTH_] )/lambda_5_in;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dy_in = L_2_lambda_2_in;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_4_dy_in = L_4_lambda_4_in;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_SOUTH_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j+1,k)]; 
		    double sos_in = sos_field[I1D(i,j+1,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_SOUTH_], bocos_T[_SOUTH_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_SOUTH_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_SOUTH_] - wg_in*T_in )/wg_g;
		    double drho_dy_g = ( rho_field[I1D(i,j+1,k)] - rho_g )/Delta_g;
		    double du_dy_g   = (   u_field[I1D(i,j+1,k)] - u_g   )/Delta_g;
		    double dv_dy_g   = (   v_field[I1D(i,j+1,k)] - v_g   )/Delta_g;
		    double dw_dy_g   = (   w_field[I1D(i,j+1,k)] - w_g   )/Delta_g;
		    double dP_dy_g   = (   P_field[I1D(i,j+1,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dy_g - rho_in*sos_in*dv_dy_g;
                    double L_2_lambda_2_g = du_dy_g;
                    double L_3_lambda_3_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                    double L_4_lambda_4_g = dw_dy_g;
                    double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dy_in = L_2_lambda_2_g;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_4_dy_in = L_4_lambda_4_g;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_SOUTH_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// North boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, y_field.vector[0:_ls_]) 
    for(int i = topo->iter_bound[_NORTH_][_INIX_]; i <= topo->iter_bound[_NORTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_NORTH_][_INIY_]; j <= topo->iter_bound[_NORTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_NORTH_][_INIZ_]; k <= topo->iter_bound[_NORTH_][_ENDZ_]; k++) {
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) or ( bocos_type[_NORTH_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_NORTH_] == _SUPERSONIC_INFLOW_ ) ) { 
                    wg_g  = 1.0 - ( y_field[I1D(i,j+1,k)] - 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( 0.5*( y_field[I1D(i,j,k)] + y_field[I1D(i,j+1,k)] ) - y_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
                if( bocos_type[_NORTH_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j-1,k)];
                v_in = v_field[I1D(i,j-1,k)];
                w_in = w_field[I1D(i,j-1,k)];
                P_in = P_field[I1D(i,j-1,k)];
                T_in = T_field[I1D(i,j-1,k)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_NORTH_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_NORTH_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_NORTH_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) and ( bocos_P[_NORTH_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_NORTH_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_NORTH_] == _DIRICHLET_ ) and ( bocos_T[_NORTH_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_NORTH_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_NORTH_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = y_field[I1D(i,j-1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in_in = y_field[I1D(i,j-2,k)] - y_field[I1D(i,j-1,k)];
		    double rho_in = rho_field[I1D(i,j-1,k)]; 
		    double sos_in = sos_field[I1D(i,j-1,k)];
		    double drho_dy_in_in = ( rho_field[I1D(i,j-2,k)] - rho_field[I1D(i,j-1,k)] )/Delta_in_in;
		    double dv_dy_in_in   = (   v_field[I1D(i,j-2,k)] -   v_field[I1D(i,j-1,k)] )/Delta_in_in;
		    double dP_dy_in_in   = (   P_field[I1D(i,j-2,k)] -   P_field[I1D(i,j-1,k)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dy_in_in - rho_in*sos_in*dv_dy_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_NORTH_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
                    bool aitken_converged = false;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter && !aitken_converged; ite++ ) {
                        /// Aitken's x_1
                        double drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        double dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        double dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dy_g = ( rho_in - rho_g )/Delta_g;
                        dv_dy_g   = (   v_in -   v_g )/Delta_g;
                        dP_dy_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                        L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                        dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dy_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            aitken_converged = true;		/// Converged: loop condition will stop further iterations
			} else {
				x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
			}
		    }
		} else if( bocos_type[_NORTH_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = y_field[I1D(i,j-1,k)] - y_field[I1D(i,j,k)];
                    double Delta_in = y_field[I1D(i,j-2,k)] - y_field[I1D(i,j-1,k)];
		    double rho_in = rho_field[I1D(i,j-1,k)]; 
		    double sos_in = sos_field[I1D(i,j-1,k)];
		    double Ma_in  = v_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_y;
		    double drho_dy_in = ( rho_field[I1D(i,j-2,k)] - rho_field[I1D(i,j-1,k)] )/Delta_in;
		    double du_dy_in   = (   u_field[I1D(i,j-2,k)] -   u_field[I1D(i,j-1,k)] )/Delta_in;
		    double dv_dy_in   = (   v_field[I1D(i,j-2,k)] -   v_field[I1D(i,j-1,k)] )/Delta_in;
		    double dw_dy_in   = (   w_field[I1D(i,j-2,k)] -   w_field[I1D(i,j-1,k)] )/Delta_in;
		    double dP_dy_in   = (   P_field[I1D(i,j-2,k)] -   P_field[I1D(i,j-1,k)] )/Delta_in;
		    double lambda_1_in = v_in - sos_in;
		    double L_1_lambda_1_in = K_in*( P_in - bocos_P[_NORTH_] )/lambda_1_in;
                    double L_2_lambda_2_in = du_dy_in;
                    double L_3_lambda_3_in = sos_in*sos_in*drho_dy_in - dP_dy_in;
                    double L_4_lambda_4_in = dw_dy_in;
                    double L_5_lambda_5_in = dP_dy_in + rho_in*sos_in*dv_dy_in;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dy_in = L_2_lambda_2_in;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_4_dy_in = L_4_lambda_4_in;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_NORTH_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = y_field[I1D(i,j-1,k)] - y_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j-1,k)]; 
		    double sos_in = sos_field[I1D(i,j-1,k)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_NORTH_], bocos_T[_NORTH_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_NORTH_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_NORTH_] - wg_in*T_in )/wg_g;
		    double drho_dy_g = ( rho_field[I1D(i,j-1,k)] - rho_g )/Delta_g;
		    double du_dy_g   = (   u_field[I1D(i,j-1,k)] - u_g   )/Delta_g;
		    double dv_dy_g   = (   v_field[I1D(i,j-1,k)] - v_g   )/Delta_g;
		    double dw_dy_g   = (   w_field[I1D(i,j-1,k)] - w_g   )/Delta_g;
		    double dP_dy_g   = (   P_field[I1D(i,j-1,k)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dy_g - rho_in*sos_in*dv_dy_g;
                    double L_2_lambda_2_g = du_dy_g;
                    double L_3_lambda_3_g = sos_in*sos_in*drho_dy_g - dP_dy_g;
                    double L_4_lambda_4_g = dw_dy_g;
                    double L_5_lambda_5_g = dP_dy_g + rho_in*sos_in*dv_dy_g;
                    double dQ_1_dy_in = ( 1.0/( sos_in*sos_in ) )*( L_3_lambda_3_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dy_in = L_2_lambda_2_g;
                    double dQ_3_dy_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_4_dy_in = L_4_lambda_4_g;
                    double dQ_5_dy_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dy_in;
                    u_g   = u_in   - Delta_g*dQ_2_dy_in;
                    v_g   = v_in   - Delta_g*dQ_3_dy_in;
                    w_g   = w_in   - Delta_g*dQ_4_dy_in;
                    P_g   = P_in   - Delta_g*dQ_5_dy_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_NORTH_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// Back boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, z_field.vector[0:_ls_])  
    for(int i = topo->iter_bound[_BACK_][_INIX_]; i <= topo->iter_bound[_BACK_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_BACK_][_INIY_]; j <= topo->iter_bound[_BACK_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_BACK_][_INIZ_]; k <= topo->iter_bound[_BACK_][_ENDZ_]; k++) {
                if( ( bocos_type[_BACK_] == _DIRICHLET_ ) or ( bocos_type[_BACK_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_BACK_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) - z_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( z_field[I1D(i,j,k+1)] - 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
                if( bocos_type[_BACK_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j,k+1)];
                v_in = v_field[I1D(i,j,k+1)];
                w_in = w_field[I1D(i,j,k+1)];
                P_in = P_field[I1D(i,j,k+1)];
                T_in = T_field[I1D(i,j,k+1)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_BACK_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_BACK_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_BACK_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_BACK_] == _DIRICHLET_ ) and ( bocos_P[_BACK_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_BACK_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_BACK_] == _DIRICHLET_ ) and ( bocos_T[_BACK_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_BACK_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_BACK_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];
                    double Delta_in_in = z_field[I1D(i,j,k+2)] - z_field[I1D(i,j,k+1)];
		    double rho_in = rho_field[I1D(i,j,k+1)]; 
		    double sos_in = sos_field[I1D(i,j,k+1)];
		    double drho_dz_in_in = ( rho_field[I1D(i,j,k+2)] - rho_field[I1D(i,j,k+1)] )/Delta_in_in;
		    double dw_dz_in_in   = (   w_field[I1D(i,j,k+2)] -   w_field[I1D(i,j,k+1)] )/Delta_in_in;
		    double dP_dz_in_in   = (   P_field[I1D(i,j,k+2)] -   P_field[I1D(i,j,k+1)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dz_in_in - rho_in*sos_in*dw_dz_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_BACK_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
                    bool aitken_converged = false;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter && !aitken_converged; ite++ ) {
                        /// Aitken's x_1
                        double drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        double dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        double dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            aitken_converged = true;		/// Converged: loop condition will stop further iterations
			} else {
				x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
			}
		    }
		} else if( bocos_type[_BACK_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];
                    double Delta_in = z_field[I1D(i,j,k+2)] - z_field[I1D(i,j,k+1)];
		    double rho_in = rho_field[I1D(i,j,k+1)]; 
		    double sos_in = sos_field[I1D(i,j,k+1)];
		    double Ma_in  = w_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_z;
		    double drho_dz_in = ( rho_field[I1D(i,j,k+2)] - rho_field[I1D(i,j,k+1)] )/Delta_in;
		    double du_dz_in   = (   u_field[I1D(i,j,k+2)] -   u_field[I1D(i,j,k+1)] )/Delta_in;
		    double dv_dz_in   = (   v_field[I1D(i,j,k+2)] -   v_field[I1D(i,j,k+1)] )/Delta_in;
		    double dw_dz_in   = (   w_field[I1D(i,j,k+2)] -   w_field[I1D(i,j,k+1)] )/Delta_in;
		    double dP_dz_in   = (   P_field[I1D(i,j,k+2)] -   P_field[I1D(i,j,k+1)] )/Delta_in;
		    double lambda_5_in = w_in + sos_in;
                    double L_1_lambda_1_in = dP_dz_in - rho_in*sos_in*dw_dz_in;
                    double L_2_lambda_2_in = du_dz_in;
                    double L_3_lambda_3_in = dv_dz_in;
                    double L_4_lambda_4_in = sos_in*sos_in*drho_dz_in - dP_dz_in;
		    double L_5_lambda_5_in = K_in*( P_in - bocos_P[_BACK_] )/lambda_5_in;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dz_in = L_2_lambda_2_in;
                    double dQ_3_dz_in = L_3_lambda_3_in;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_BACK_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j,k+1)]; 
		    double sos_in = sos_field[I1D(i,j,k+1)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_BACK_], bocos_T[_BACK_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_BACK_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_BACK_] - wg_in*T_in )/wg_g;
		    double drho_dz_g = ( rho_field[I1D(i,j,k+1)] - rho_g )/Delta_g;
		    double du_dz_g   = (   u_field[I1D(i,j,k+1)] - u_g   )/Delta_g;
		    double dv_dz_g   = (   v_field[I1D(i,j,k+1)] - v_g   )/Delta_g;
		    double dw_dz_g   = (   w_field[I1D(i,j,k+1)] - w_g   )/Delta_g;
		    double dP_dz_g   = (   P_field[I1D(i,j,k+1)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dz_g - rho_in*sos_in*dw_dz_g;
                    double L_2_lambda_2_g = du_dz_g;
                    double L_3_lambda_3_g = dv_dz_g;
                    double L_4_lambda_4_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                    double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dz_in = L_2_lambda_2_g;
                    double dQ_3_dz_in = L_3_lambda_3_g;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_BACK_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }

    /// Front boundary points: rho, rhou, rhov, rhow and rhoE
    //#pragma acc parallel loop collapse(3) private(rho_g, T_g, P_g, e_g)
    #pragma acc parallel loop collapse(3) private (rho_g, P_g, T_g,e_g,u_g,v_g,w_g,E_g,ke_g) present(this, mu_field.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], thermodynamics, topo, z_field.vector[0:_ls_])  
    for(int i = topo->iter_bound[_FRONT_][_INIX_]; i <= topo->iter_bound[_FRONT_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_FRONT_][_INIY_]; j <= topo->iter_bound[_FRONT_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_FRONT_][_INIZ_]; k <= topo->iter_bound[_FRONT_][_ENDZ_]; k++) {
                if( ( bocos_type[_FRONT_] == _DIRICHLET_ ) or ( bocos_type[_FRONT_] == _SUBSONIC_INFLOW_ ) or ( bocos_type[_FRONT_] == _SUPERSONIC_INFLOW_ ) ) {
                    wg_g  = 1.0 - ( z_field[I1D(i,j,k+1)] - 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = 1.0 - ( 0.5*( z_field[I1D(i,j,k)] + z_field[I1D(i,j,k+1)] ) - z_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
                if( bocos_type[_FRONT_] == _NEUMANN_ ) {
                    wg_g  = (  1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                    wg_in = ( -1.0 )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] );
                }
		/// Get/calculate inner values
                u_in = u_field[I1D(i,j,k-1)];
                v_in = v_field[I1D(i,j,k-1)];
                w_in = w_field[I1D(i,j,k-1)];
                P_in = P_field[I1D(i,j,k-1)];
                T_in = T_field[I1D(i,j,k-1)];	
		/// Calculate ghost primitive variables
                u_g = ( bocos_u[_FRONT_] - wg_in*u_in )/wg_g;
                v_g = ( bocos_v[_FRONT_] - wg_in*v_in )/wg_g;
                w_g = ( bocos_w[_FRONT_] - wg_in*w_in )/wg_g;
                if( ( bocos_type[_FRONT_] == _DIRICHLET_ ) and ( bocos_P[_FRONT_] < 0.0 ) ) {
                    P_g = P_in;
                } else {
                    P_g = ( bocos_P[_FRONT_] - wg_in*P_in )/wg_g;
                }
                if( ( bocos_type[_FRONT_] == _DIRICHLET_ ) and ( bocos_T[_FRONT_] < 0.0 ) ) {
                    T_g = T_in;
                } else {
                    T_g = ( bocos_T[_FRONT_] - wg_in*T_in )/wg_g;
                }
                if( bocos_type[_FRONT_] == _SUBSONIC_INFLOW_ ) {
                    double Delta_g     = z_field[I1D(i,j,k-1)] - z_field[I1D(i,j,k)];
                    double Delta_in_in = z_field[I1D(i,j,k-2)] - z_field[I1D(i,j,k-1)];
		    double rho_in = rho_field[I1D(i,j,k-1)]; 
		    double sos_in = sos_field[I1D(i,j,k-1)];
		    double drho_dz_in_in = ( rho_field[I1D(i,j,k-2)] - rho_field[I1D(i,j,k-1)] )/Delta_in_in;
		    double dw_dz_in_in   = (   w_field[I1D(i,j,k-2)] -   w_field[I1D(i,j,k-1)] )/Delta_in_in;
		    double dP_dz_in_in   = (   P_field[I1D(i,j,k-2)] -   P_field[I1D(i,j,k-1)] )/Delta_in_in;
                    double L_1_lambda_1_in_in = dP_dz_in_in - rho_in*sos_in*dw_dz_in_in;
                    rho_g = rho_field[I1D(i,j,k)];
		    T_g   = ( bocos_T[_FRONT_] - wg_in*T_in )/wg_g;
                    P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
		    /// Aitken’s delta-squared process:
                    double x_0_Aitken = rho_g;
                    bool aitken_converged = false;
		    #pragma acc loop seq
                    for( int ite = 0; ite < max_iter && !aitken_converged; ite++ ) {
                        /// Aitken's x_1
                        double drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        double dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        double dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        double L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_1_Aitken = rho_g;
                        /// Aitken's x_2
                        drho_dz_g = ( rho_in - rho_g )/Delta_g;
                        dw_dz_g   = (   w_in -   w_g )/Delta_g;
                        dP_dz_g   = (   P_in -   P_g )/Delta_g;
                        L_2_lambda_2_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                        L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                        dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_2_lambda_2_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_in_in ) );
                        rho_g = rho_in - Delta_g*dQ_1_dz_in;
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        double x_2_Aitken = rho_g;
                        /// Aitken's iteration
                        double denominator = x_2_Aitken - 2.0*x_1_Aitken + x_0_Aitken;
                        rho_g = x_2_Aitken - ( pow( x_2_Aitken - x_1_Aitken, 2.0 )/( denominator + epsilon ) );
                        P_g   = thermodynamics->calculatePressureFromTemperatureDensity( T_g, rho_g );
                        ///cout << ite << "  " << rho_g << "  " << x_0_Aitken << "  " << x_1_Aitken << "  " << x_2_Aitken << endl;
                        /// Aitken's convergence
                        if( abs( (rho_g - x_2_Aitken )/rho_g ) < rel_tol ) {
                            aitken_converged = true;		/// Converged: loop condition will stop further iterations
			} else {
				x_0_Aitken = rho_g;	/// Otherwise, update x_0 to iterate again ...
			}
		    }
		} else if( bocos_type[_FRONT_] == _SUBSONIC_OUTFLOW_ ) {
                    double Delta_g  = z_field[I1D(i,j,k-1)] - z_field[I1D(i,j,k)];
                    double Delta_in = z_field[I1D(i,j,k-2)] - z_field[I1D(i,j,k-1)];
		    double rho_in = rho_field[I1D(i,j,k-1)]; 
		    double sos_in = sos_field[I1D(i,j,k-1)];
		    double Ma_in  = w_in/sos_in;
		    double K_in   = 0.25*sos_in*( 1.0 - Ma_in*Ma_in )/L_z;
		    double drho_dz_in = ( rho_field[I1D(i,j,k-2)] - rho_field[I1D(i,j,k-1)] )/Delta_in;
		    double du_dz_in   = (   u_field[I1D(i,j,k-2)] -   u_field[I1D(i,j,k-1)] )/Delta_in;
		    double dv_dz_in   = (   v_field[I1D(i,j,k-2)] -   v_field[I1D(i,j,k-1)] )/Delta_in;
		    double dw_dz_in   = (   w_field[I1D(i,j,k-2)] -   w_field[I1D(i,j,k-1)] )/Delta_in;
		    double dP_dz_in   = (   P_field[I1D(i,j,k-2)] -   P_field[I1D(i,j,k-1)] )/Delta_in;
		    double lambda_1_in = w_in - sos_in;
		    double L_1_lambda_1_in = K_in*( P_in - bocos_P[_FRONT_] )/lambda_1_in;
                    double L_2_lambda_2_in = du_dz_in;
                    double L_3_lambda_3_in = dv_dz_in;
                    double L_4_lambda_4_in = sos_in*sos_in*drho_dz_in - dP_dz_in;
                    double L_5_lambda_5_in = dP_dz_in + rho_in*sos_in*dw_dz_in;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_in + 0.5*( L_5_lambda_5_in + L_1_lambda_1_in ) );
                    double dQ_2_dz_in = L_2_lambda_2_in;
                    double dQ_3_dz_in = L_3_lambda_3_in;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_in - L_1_lambda_1_in );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_in + L_1_lambda_1_in );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_FRONT_] == _SUPERSONIC_INFLOW_ ) {
		    double Delta_g = z_field[I1D(i,j,k-1)] - z_field[I1D(i,j,k)];
		    double rho_in = rho_field[I1D(i,j,k-1)]; 
		    double sos_in = sos_field[I1D(i,j,k-1)];
                    double rho_b = thermodynamics->calculateDensityFromPressureTemperature( bocos_P[_FRONT_], bocos_T[_FRONT_] );
                    rho_g = ( rho_b - wg_in*rho_in )/wg_g;
                    P_g   = ( bocos_P[_FRONT_] - wg_in*P_in )/wg_g;
                    T_g   = ( bocos_T[_FRONT_] - wg_in*T_in )/wg_g;
		    double drho_dz_g = ( rho_field[I1D(i,j,k-1)] - rho_g )/Delta_g;
		    double du_dz_g   = (   u_field[I1D(i,j,k-1)] - u_g   )/Delta_g;
		    double dv_dz_g   = (   v_field[I1D(i,j,k-1)] - v_g   )/Delta_g;
		    double dw_dz_g   = (   w_field[I1D(i,j,k-1)] - w_g   )/Delta_g;
		    double dP_dz_g   = (   P_field[I1D(i,j,k-1)] - P_g   )/Delta_g;
		    double L_1_lambda_1_g = dP_dz_g - rho_in*sos_in*dw_dz_g;
                    double L_2_lambda_2_g = du_dz_g;
                    double L_3_lambda_3_g = dv_dz_g;
                    double L_4_lambda_4_g = sos_in*sos_in*drho_dz_g - dP_dz_g;
                    double L_5_lambda_5_g = dP_dz_g + rho_in*sos_in*dw_dz_g;
                    double dQ_1_dz_in = ( 1.0/( sos_in*sos_in ) )*( L_4_lambda_4_g + 0.5*( L_5_lambda_5_g + L_1_lambda_1_g ) );
                    double dQ_2_dz_in = L_2_lambda_2_g;
                    double dQ_3_dz_in = L_3_lambda_3_g;
                    double dQ_4_dz_in = ( 1.0/( 2.0*rho_in*sos_in ) )*( L_5_lambda_5_g - L_1_lambda_1_g );
                    double dQ_5_dz_in = 0.5*( L_5_lambda_5_g + L_1_lambda_1_g );
                    rho_g = rho_in - Delta_g*dQ_1_dz_in;
                    u_g   = u_in   - Delta_g*dQ_2_dz_in;
                    v_g   = v_in   - Delta_g*dQ_3_dz_in;
                    w_g   = w_in   - Delta_g*dQ_4_dz_in;
                    P_g   = P_in   - Delta_g*dQ_5_dz_in;
                    T_g   = T_field[I1D(i,j,k)];
                    thermodynamics->calculateTemperatureFromPressureDensityWithInitialGuess( T_g, P_g, rho_g );
		} else if( bocos_type[_FRONT_] == _SUPERSONIC_OUTFLOW_ ) {
                    u_g = u_in;
                    v_g = v_in;
                    w_g = w_in;
                    P_g = P_in;
                    T_g = T_in;
                }
                thermodynamics->calculateDensityInternalEnergyFromPressureTemperature( rho_g, e_g, P_g, T_g );
                ke_g = 0.5*( u_g*u_g + v_g*v_g + w_g*w_g );
                E_g  = e_g + ke_g;
		/// Update ghost conserved variables
                rho_field[I1D(i,j,k)]  = rho_g;
                rhou_field[I1D(i,j,k)] = rho_g*u_g;
                rhov_field[I1D(i,j,k)] = rho_g*v_g;
                rhow_field[I1D(i,j,k)] = rho_g*w_g;
                rhoE_field[I1D(i,j,k)] = rho_g*E_g;
		/// Update u, v, w, E, P, T and sos variables
                u_field[I1D(i,j,k)] = u_g;
                v_field[I1D(i,j,k)] = v_g;
                w_field[I1D(i,j,k)] = w_g;
                E_field[I1D(i,j,k)] = E_g;
                P_field[I1D(i,j,k)] = P_g;
                T_field[I1D(i,j,k)] = T_g;
		/// Update s, sos, c_v and c_p
		double c_v, c_p;
		if( artificial_compressibility_method ) {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_thermo, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_g, rho_g );
		} else {
                    s_field[I1D(i,j,k)] = thermodynamics->calculateEntropyFromPressureTemperatureDensity( P_g, T_g, rho_g );
                    sos_field[I1D(i,j,k)] = thermodynamics->calculateSoundSpeed( P_g, T_g, rho_g );
                    thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_g, T_g, rho_g );
		}
                c_v_field[I1D(i,j,k)] = c_v;
                c_p_field[I1D(i,j,k)] = c_p;
            }
        }
    }
    
    /// Update halo values
    //#pragma acc update host(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_])
    //rho_field.update();
    //rhou_field.update();
    //rhov_field.update();
    //rhow_field.update();
    //rhoE_field.update();
    //u_field.update();
    u_field.fillEdgeCornerBoundaries();
    //v_field.update();
    v_field.fillEdgeCornerBoundaries();
    //w_field.update();
    w_field.fillEdgeCornerBoundaries();
    //E_field.update();
    //s_field.update();
    //P_field.update();
    //T_field.update();
    //sos_field.update();
    //c_v_field.update();
    //c_p_field.update();
    //#pragma acc update device(u_field.vector[0:_ls_],v_field.vector[0:_ls_],w_field.vector[0:_ls_])

};
       
void FlowSolverRHEA::setInitialParticlesPositionsVelocities() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// If empty (default), particles have already been initialized in random positions uniformly and with zero velocity

};

bool FlowSolverRHEA::pointParticleIsContainedWithinFluidCell(const double &x_position_particle, const double &y_position_particle, const double &z_position_particle, const int &local_index_i, const int &local_index_j, const int &local_index_k) {

    double isContained = false;

    if( ( x_position_particle >= 0.5*( x_field[I1D(local_index_i-1,local_index_j,local_index_k)] + x_field[I1D(local_index_i,local_index_j,local_index_k)] ) ) and ( x_position_particle < 0.5*( x_field[I1D(local_index_i,local_index_j,local_index_k)] + x_field[I1D(local_index_i+1,local_index_j,local_index_k)] ) ) and ( y_position_particle >= 0.5*( y_field[I1D(local_index_i,local_index_j-1,local_index_k)] + y_field[I1D(local_index_i,local_index_j,local_index_k)] ) ) and ( y_position_particle < 0.5*( y_field[I1D(local_index_i,local_index_j,local_index_k)] + y_field[I1D(local_index_i,local_index_j+1,local_index_k)] ) ) and ( z_position_particle >= 0.5*( z_field[I1D(local_index_i,local_index_j,local_index_k-1)] + z_field[I1D(local_index_i,local_index_j,local_index_k)] ) ) and ( z_position_particle < 0.5*( z_field[I1D(local_index_i,local_index_j,local_index_k)] + z_field[I1D(local_index_i,local_index_j,local_index_k+1)] ) ) ) {
        isContained = true;
    }	

    return isContained;

};

#pragma acc routine seq
double FlowSolverRHEA::trilinearInterpolation(const double &x, const double &y, const double &z, const double &xm, const double &xc, const double &xp, const double &ym, const double &yc, const double &yp, const double &zm, const double &zc, const double &zp, const double &fmmm, const double &fmmc, const double &fmmp, const double &fmcm, const double &fmcc, const double &fmcp, const double &fmpm, const double &fmpc, const double &fmpp, const double &fcmm, const double &fcmc, const double &fcmp, const double &fccm, const double &fccc, const double &fccp, const double &fcpm, const double &fcpc, const double &fcpp, const double &fpmm, const double &fpmc, const double &fpmp, const double &fpcm, const double &fpcc, const double &fpcp, const double &fppm, const double &fppc, const double &fppp ) {

    // Set interpolation stencil based on particle position
    double x0, x1, y0, y1, z0, z1, f000, f100, f010, f110, f001, f101, f011, f111;
    if( x <= xc ) { 
        x0 = xm; x1 = xc;
        if( y <= yc ) {
            y0 = ym; y1 = yc;
            if( z <= zc ) {
                z0 = zm; z1 = zc;
		f000 = fmmm; f100 = fcmm; f010 = fmcm; f110 = fccm; f001 = fmmc, f101 = fcmc; f011 = fmcc; f111 = fccc; 
	    } else {
	        z0 = zc; z1 = zp;
		f000 = fmmc; f100 = fcmc; f010 = fmcc; f110 = fccc; f001 = fmmp, f101 = fcmp; f011 = fmcp; f111 = fccp; 
	    }
	} else {
	    y0 = yc; y1 = yp;
            if( z <= zc ) {
                z0 = zm; z1 = zc;
		f000 = fmcm; f100 = fccm; f010 = fmpm; f110 = fcpm; f001 = fmcc, f101 = fccc; f011 = fmpc; f111 = fcpc; 
	    } else {
	        z0 = zc; z1 = zp;
		f000 = fmcc; f100 = fccc; f010 = fmpc; f110 = fcpc; f001 = fmcp, f101 = fccp; f011 = fmpp; f111 = fcpp; 
	    }	    
	}
    } else {
	x0 = xc; x1 = xp;
        if( y <= yc ) {
            y0 = ym; y1 = yc;
            if( z <= zc ) {
                z0 = zm; z1 = zc;
		f000 = fcmm; f100 = fpmm; f010 = fccm; f110 = fpcm; f001 = fcmc, f101 = fpmc; f011 = fccc; f111 = fpcc; 
	    } else {
	        z0 = zc; z1 = zp;
		f000 = fcmc; f100 = fpmc; f010 = fccc; f110 = fpcc; f001 = fcmp, f101 = fpmp; f011 = fccp; f111 = fpcp; 
	    }
	} else {
	    y0 = yc; y1 = yp;
            if( z <= zc ) {
                z0 = zm; z1 = zc;
		f000 = fccm; f100 = fpcm; f010 = fcpm; f110 = fppm; f001 = fccc, f101 = fpcc; f011 = fcpc; f111 = fppc; 
	    } else {
	        z0 = zc; z1 = zp;
		f000 = fccc; f100 = fpcc; f010 = fcpc; f110 = fppc; f001 = fccp, f101 = fpcp; f011 = fcpp; f111 = fppp; 
	    }	    
	}
    }

    // Calculate interpolation factors
    double tx = ( x - x0 )/( x1 - x0 );
    double ty = ( y - y0 )/( y1 - y0 );
    double tz = ( z - z0 )/( z1 - z0 );

    // Interpolate along x for each (y, z)
    double c00 = f000*( 1.0 - tx ) + f100*tx;
    double c01 = f001*( 1.0 - tx ) + f101*tx;
    double c10 = f010*( 1.0 - tx ) + f110*tx;
    double c11 = f011*( 1.0 - tx ) + f111*tx;

    // Interpolate along y for each z
    double c0 = c00*( 1.0 - ty ) + c10*ty;
    double c1 = c01*( 1.0 - ty ) + c11*ty;

    // Interpolate along z
    double c = c0*( 1.0 - tz ) + c1*tz;

    return c;

};

void FlowSolverRHEA::updateLagrangianEulerianMeshIndexes0(const int &my_rank) {

    // Get number particles in use
    number_particles_local_in_use = point_particles->get_num_prts_local_in_use( my_rank );

    int i_local_index, j_local_index, k_local_index;
    double x_position_particle, y_position_particle, z_position_particle;
    for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

        /// Initialize indexes
        i_local_index = -1;
        j_local_index = -1;
        k_local_index = -1;

        /// Obtain Lagrangian position 0
        x_position_particle = point_particles->local_prts_positions_0_x[p];
        y_position_particle = point_particles->local_prts_positions_0_y[p];
        z_position_particle = point_particles->local_prts_positions_0_z[p];
 
        /// Search in x-direction
        for( int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; ++i ) {
            if( x_position_particle >= 0.5*( x_field[I1D(i-1,0,0)] + x_field[I1D(i,0,0)] ) && x_position_particle < 0.5*( x_field[I1D(i,0,0)] + x_field[I1D(i+1,0,0)] ) ) {
                i_local_index = i;
                break;
            }
        }

        /// Search in y-direction
        for( int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; ++j ) {
            if( y_position_particle >= 0.5*( y_field[I1D(0,j-1,0)] + y_field[I1D(0,j,0)] ) && y_position_particle < 0.5*( y_field[I1D(0,j,0)] + y_field[I1D(0,j+1,0)] ) ) {
                j_local_index = j;
                break;
            }
        }

        /// Search in z-direction
        for( int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; ++k ) {
            if( z_position_particle >= 0.5*( z_field[I1D(0,0,k-1)] + z_field[I1D(0,0,k)] ) && z_position_particle < 0.5*( z_field[I1D(0,0,k)] + z_field[I1D(0,0,k+1)] ) ) {
                k_local_index = k;
                break;
            }
        }

        /// Set indexes
	point_particles->local_prts_indexes_0_i[p] = i_local_index;
	point_particles->local_prts_indexes_0_j[p] = j_local_index;
	point_particles->local_prts_indexes_0_k[p] = k_local_index;

    }

};

#pragma acc routine seq
void FlowSolverRHEA::obtainLagrangianEulerianVelocityDynamicViscosityValues(double &u_velocity_fluid_particle, double &v_velocity_fluid_particle, double &w_velocity_fluid_particle, double &dynamic_viscosity_fluid, const int &index_particle) {

    /// Obtain Lagrangian-Eulerian indexes 0
    int i_local_index = point_particles->local_prts_indexes_0_i[index_particle];		/// Local index i
    int j_local_index = point_particles->local_prts_indexes_0_j[index_particle];		/// Local index j
    int k_local_index = point_particles->local_prts_indexes_0_k[index_particle];		/// Local index k

    /// Obtain Lagrangian position 0
    double x_position_particle = point_particles->local_prts_positions_0_x[index_particle];	/// Position x
    double y_position_particle = point_particles->local_prts_positions_0_y[index_particle];	/// Position y
    double z_position_particle = point_particles->local_prts_positions_0_z[index_particle];	/// Position z
    
    /// Interpolate (trilinear) values: u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid 
    u_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index)], u_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index)], u_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index,k_local_index-1)], u_field[I1D(i_local_index,j_local_index,k_local_index)], u_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index)], u_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index)], u_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
    v_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index)], v_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index)], v_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index,k_local_index-1)], v_field[I1D(i_local_index,j_local_index,k_local_index)], v_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index)], v_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index)], v_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
    w_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index)], w_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index)], w_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index,k_local_index-1)], w_field[I1D(i_local_index,j_local_index,k_local_index)], w_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index)], w_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index)], w_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
    dynamic_viscosity_fluid = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index,k_local_index)], mu_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );

};

void FlowSolverRHEA::timeAdvancePointParticles(const int &my_rank) {

    /// Skip if there are no particles
    if( point_particles->get_num_prts_total() < 1 ) return;

    /// Communicate particles
    point_particles->updateHostParticles();
    point_particles->communicate_prts();
    number_particles_local_in_use = point_particles->get_num_prts_local_in_use( my_rank );
    point_particles->updateDeviceParticles();

    /// Explicit Euler time-integration of particles position
    delta_t = this->delta_t;
    int capacity = point_particles->get_prt_capacity();  
    #pragma acc parallel loop collapse(1) present(this, point_particles, point_particles->local_prts_positions_x[0:capacity], point_particles->local_prts_positions_y[0:capacity], point_particles->local_prts_positions_z[0:capacity], point_particles->local_prts_positions_0_x[0:capacity], point_particles->local_prts_positions_0_y[0:capacity], point_particles->local_prts_positions_0_z[0:capacity], point_particles->local_prts_velocities_x[0:capacity], point_particles->local_prts_velocities_y[0:capacity], point_particles->local_prts_velocities_z[0:capacity], point_particles->local_prts_velocities_0_x[0:capacity], point_particles->local_prts_velocities_0_y[0:capacity], point_particles->local_prts_velocities_0_z[0:capacity]) copyin(delta_t)
    for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

	point_particles->local_prts_positions_0_x[p] = point_particles->local_prts_positions_x[p];
    	point_particles->local_prts_positions_0_y[p] = point_particles->local_prts_positions_y[p];
    	point_particles->local_prts_positions_0_z[p] = point_particles->local_prts_positions_z[p];
	point_particles->local_prts_velocities_0_x[p] = point_particles->local_prts_velocities_x[p];
        point_particles->local_prts_velocities_0_y[p] = point_particles->local_prts_velocities_y[p];
        point_particles->local_prts_velocities_0_z[p] = point_particles->local_prts_velocities_z[p];
    
        // Euler forward position update
	point_particles->local_prts_positions_x[p] = point_particles->local_prts_positions_x[p] + delta_t*point_particles->local_prts_velocities_x[p];
	point_particles->local_prts_positions_y[p] = point_particles->local_prts_positions_y[p] + delta_t*point_particles->local_prts_velocities_y[p];
	point_particles->local_prts_positions_z[p] = point_particles->local_prts_positions_z[p] + delta_t*point_particles->local_prts_velocities_z[p];
    }
   
    /// Update Lagrangian-Eulerian Mesh indexes 0
    #pragma acc update host(point_particles->local_prts_positions_0_x[0:capacity], point_particles->local_prts_positions_0_y[0:capacity], point_particles->local_prts_positions_0_z[0:capacity])   
    this->updateLagrangianEulerianMeshIndexes0( my_rank );
    #pragma acc update device ( point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity]  )
 
    if( activate_pure_tracer_particles ) {

        int i_local_index, j_local_index, k_local_index;
        double x_position_particle, y_position_particle, z_position_particle;
        double u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle;
	int capacity = point_particles->get_prt_capacity();
        #pragma acc parallel loop collapse (1) private( i_local_index, j_local_index, k_local_index, x_position_particle, y_position_particle, z_position_particle, u_velocity_fluid_particle, w_velocity_fluid_particle,v_velocity_fluid_particle, i_local_index, j_local_index, k_local_index ) present( this, mesh, point_particles, u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], point_particles->local_prts_positions_0_x[0:capacity], point_particles->local_prts_positions_0_y[0:capacity], point_particles->local_prts_positions_0_z[0:capacity], point_particles->local_prts_velocities_x[0:capacity], point_particles->local_prts_velocities_y[0:capacity], point_particles->local_prts_velocities_z[0:capacity], point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity] ) 
        for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

            /// Obtain Lagrangian-Eulerian indexes 0
            i_local_index = point_particles->local_prts_indexes_0_i[p];
            j_local_index = point_particles->local_prts_indexes_0_j[p];
            k_local_index = point_particles->local_prts_indexes_0_k[p];

            /// Obtain Lagrangian position 0
            x_position_particle = point_particles->local_prts_positions_0_x[p];
            y_position_particle = point_particles->local_prts_positions_0_y[p];
            z_position_particle = point_particles->local_prts_positions_0_z[p];
    
            /// Interpolate (trilinear) values: u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle
            u_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index)], u_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index)], u_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index,k_local_index-1)], u_field[I1D(i_local_index,j_local_index,k_local_index)], u_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index)], u_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index)], u_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
            v_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index)], v_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index)], v_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index,k_local_index-1)], v_field[I1D(i_local_index,j_local_index,k_local_index)], v_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index)], v_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index)], v_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
            w_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index)], w_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index)], w_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index,k_local_index-1)], w_field[I1D(i_local_index,j_local_index,k_local_index)], w_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index)], w_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index)], w_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );

             /// Update velocity
	     point_particles->local_prts_velocities_x[p] = u_velocity_fluid_particle;
             point_particles->local_prts_velocities_y[p] = v_velocity_fluid_particle;
             point_particles->local_prts_velocities_z[p] = w_velocity_fluid_particle;

        }

    } else {

        /// Explicit Euler time-integration of particles velocity
        this->timeAdvanceVelocityPointParticles();

    }

};

void FlowSolverRHEA::timeAdvanceVelocityPointParticles() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Explicit Euler time-integration of particles velocity
    int i_local_index, j_local_index, k_local_index;
    double x_position_particle, y_position_particle, z_position_particle;
    double u_velocity_particle, v_velocity_particle, w_velocity_particle;
    double u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid, relaxation_time_particle;
    delta_t = this->delta_t;
    int capacity = point_particles->get_prt_capacity();
    #pragma acc parallel loop collapse (1) private(  i_local_index, j_local_index, k_local_index, x_position_particle, y_position_particle, z_position_particle, u_velocity_particle, v_velocity_particle, w_velocity_particle,  u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid, relaxation_time_particle ) present( this, mesh, point_particles, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], mu_field.vector[0:_ls_], point_particles->local_prts_densities[0:capacity], point_particles->local_prts_diameters[0:capacity], point_particles->local_prts_positions_0_x[0:capacity], point_particles->local_prts_positions_0_y[0:capacity], point_particles->local_prts_positions_0_z[0:capacity], point_particles->local_prts_velocities_x[0:capacity], point_particles->local_prts_velocities_y[0:capacity], point_particles->local_prts_velocities_z[0:capacity], point_particles->local_prts_velocities_0_x[0:capacity], point_particles->local_prts_velocities_0_y[0:capacity], point_particles->local_prts_velocities_0_z[0:capacity], point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity] ) copyin( delta_t )
    for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

        /// Obtain Lagrangian-Eulerian indexes 0
        i_local_index = point_particles->local_prts_indexes_0_i[p];
        j_local_index = point_particles->local_prts_indexes_0_j[p];
        k_local_index = point_particles->local_prts_indexes_0_k[p];

        /// Obtain Lagrangian position 0
        x_position_particle = point_particles->local_prts_positions_0_x[p];
        y_position_particle = point_particles->local_prts_positions_0_y[p];
        z_position_particle = point_particles->local_prts_positions_0_z[p];

        /// Obtain Lagrangian velocities 0
	u_velocity_particle = point_particles->local_prts_velocities_0_x[p];
        v_velocity_particle = point_particles->local_prts_velocities_0_y[p];
        w_velocity_particle = point_particles->local_prts_velocities_0_z[p];

        /// Interpolate (trilinear) values: u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid 
        u_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index)], u_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index)], u_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index,k_local_index-1)], u_field[I1D(i_local_index,j_local_index,k_local_index)], u_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index)], u_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index)], u_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
        v_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index)], v_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index)], v_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index,k_local_index-1)], v_field[I1D(i_local_index,j_local_index,k_local_index)], v_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index)], v_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index)], v_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
        w_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index)], w_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index)], w_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index,k_local_index-1)], w_field[I1D(i_local_index,j_local_index,k_local_index)], w_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index)], w_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index)], w_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
        dynamic_viscosity_fluid = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index,k_local_index)], mu_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );

        /// Calculate relaxation time particle
        //relaxation_time_particle = point_particles->calculate_relaxation_time_prt( p, dynamic_viscosity_fluid );
	relaxation_time_particle = ( point_particles->local_prts_densities[p]*pow( point_particles->local_prts_diameters[p], 2.0 ) )/( 18.0*dynamic_viscosity_fluid );

        /// Update particle velocity
        u_velocity_particle = u_velocity_particle + delta_t*( u_velocity_fluid_particle - u_velocity_particle )/relaxation_time_particle;
        v_velocity_particle = v_velocity_particle + delta_t*( v_velocity_fluid_particle - v_velocity_particle )/relaxation_time_particle;
        w_velocity_particle = w_velocity_particle + delta_t*( w_velocity_fluid_particle - w_velocity_particle )/relaxation_time_particle;
	point_particles->local_prts_velocities_x[p] = u_velocity_particle;
        point_particles->local_prts_velocities_y[p] = v_velocity_particle;
        point_particles->local_prts_velocities_z[p] = w_velocity_particle;

    }

};

void FlowSolverRHEA::updatePreviousStateConservedVariables() {

    /// All (inner, halo, boundary) points: rho_0, rhou_0 rhov_0, rhow_0, rhoE_0 and P_0
    #pragma acc parallel loop collapse(3) present(this, rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], P_field.vector[0:_ls_], rho_0_field.vector[0:_ls_], rhou_0_field.vector[0:_ls_], rhov_0_field.vector[0:_ls_], rhow_0_field.vector[0:_ls_], rhoE_0_field.vector[0:_ls_], P_0_field.vector[0:_ls_]) 
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                rho_0_field[I1D(i,j,k)]  = rho_field[I1D(i,j,k)]; 
                rhou_0_field[I1D(i,j,k)] = rhou_field[I1D(i,j,k)]; 
                rhov_0_field[I1D(i,j,k)] = rhov_field[I1D(i,j,k)]; 
                rhow_0_field[I1D(i,j,k)] = rhow_field[I1D(i,j,k)]; 
                rhoE_0_field[I1D(i,j,k)] = rhoE_field[I1D(i,j,k)]; 
                P_0_field[I1D(i,j,k)]    = P_field[I1D(i,j,k)]; 
            }
        }
    }

    /// Update halo values
    //rho_0_field.update();
    //rhou_0_field.update();
    //rhov_0_field.update();
    //rhow_0_field.update();
    //rhoE_0_field.update();
    //P_0_field.update();

};

void FlowSolverRHEA::calculateTimeStep() {

    /// Inviscid time step size for explicit schemes:
    /// E. F. Toro.
    /// Riemann solvers and numerical methods for fluid dynamics.
    /// Springer, 2009.

    /// Viscous time step size for explicit schemes:
    /// E. Turkel, R.C. Swanson, V. N. Vatsa, J.A. White.
    /// Multigrid for hypersonic viscous two- and three-dimensional flows.
    /// NASA Contractor Report 187603, 1991.

    /// Initialize to largest double value
    double local_delta_t = numeric_limits<double>::max();

    /// Inner points: find minimum (local) delta_t
    double sos, c_p;
    double delta_x, delta_y, delta_z;
    double S_x, S_y, S_z;
    #pragma acc parallel loop collapse(3) independent private(delta_x,delta_y,delta_z, S_x,S_y,S_z,c_p,sos) reduction(min:local_delta_t) present(this, rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], sos_field.vector[0:_ls_], c_p_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_])  
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Speed of sound
		sos = sos_field[I1D(i,j,k)];
                /// Heat capacities
                //thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
		c_p = c_p_field[I1D(i,j,k)];
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                /// x-direction inviscid, viscous & thermal terms
                S_x           = abs( u_field[I1D(i,j,k)] ) + sos;
                local_delta_t = min( local_delta_t, CFL*delta_x/S_x );											/// acoustic scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*pow( delta_x, 2.0 )/max( mu_field[I1D(i,j,k)], epsilon ) );		/// viscous scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*c_p*pow( delta_x, 2.0 )/max( kappa_field[I1D(i,j,k)], epsilon ) );	/// thermal diffusivity scale
                /// y-direction inviscid, viscous & thermal terms
                S_y           = abs( v_field[I1D(i,j,k)] ) + sos;
                local_delta_t = min( local_delta_t, CFL*delta_y/S_y );											/// acoustic scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*pow( delta_y, 2.0 )/max( mu_field[I1D(i,j,k)], epsilon ) );		/// viscous scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*c_p*pow( delta_y, 2.0 )/max( kappa_field[I1D(i,j,k)], epsilon ) );	/// thermal diffusivity scale
                /// z-direction inviscid, viscous & thermal terms
                S_z           = abs( w_field[I1D(i,j,k)] ) + sos;
                local_delta_t = min( local_delta_t, CFL*delta_z/S_z );											/// acoustic scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*pow( delta_z, 2.0 )/max( mu_field[I1D(i,j,k)], epsilon ) );		/// viscous scale
                local_delta_t = min( local_delta_t, CFL*rho_field[I1D(i,j,k)]*c_p*pow( delta_z, 2.0 )/max( kappa_field[I1D(i,j,k)], epsilon ) );	/// thermal diffusivity scale
            }
        }
    }

    /// Iterate through local particles in use
    if( ( point_particles->get_num_prts_total() > 0 ) and !activate_pure_tracer_particles ) {

        int i_local_index, j_local_index, k_local_index;
        double u_velocity_particle, v_velocity_particle, w_velocity_particle;   
	int capacity = point_particles->get_prt_capacity();
        #pragma acc parallel loop collapse (1) private(i_local_index, j_local_index, k_local_index, u_velocity_particle, v_velocity_particle, w_velocity_particle, delta_z, delta_y, delta_x) present(this, mesh, topo, point_particles, x_field.vector[0:_ls_], v_field.vector[0:_ls_], z_field.vector[0:_ls_], point_particles->local_prts_velocities_0_x[0:capacity], point_particles->local_prts_velocities_0_y[0:capacity], point_particles->local_prts_velocities_0_z[0:capacity], point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity]) reduction(min:local_delta_t)
        for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

            /// Obtain Lagrangian-Eulerian indexes 0
            i_local_index = point_particles->local_prts_indexes_0_i[p];		/// Local index i
            j_local_index = point_particles->local_prts_indexes_0_j[p];		/// Local index j
            k_local_index = point_particles->local_prts_indexes_0_k[p];		/// Local index k

            /// Obtain velocity of point particle
            u_velocity_particle = point_particles->local_prts_velocities_0_x[p];
            v_velocity_particle = point_particles->local_prts_velocities_0_y[p];
            w_velocity_particle = point_particles->local_prts_velocities_0_z[p];

            /// Geometric stuff
            delta_x = 0.5*( x_field[I1D(i_local_index+1,j_local_index,k_local_index)] - x_field[I1D(i_local_index-1,j_local_index,k_local_index)] ); 
            delta_y = 0.5*( y_field[I1D(i_local_index,j_local_index+1,k_local_index)] - y_field[I1D(i_local_index,j_local_index-1,k_local_index)] ); 
            delta_z = 0.5*( z_field[I1D(i_local_index,j_local_index,k_local_index+1)] - z_field[I1D(i_local_index,j_local_index,k_local_index-1)] );

            /// x-direction particle term
            local_delta_t = min( local_delta_t, CFL*delta_x/max( abs( u_velocity_particle ), epsilon ) );	/// Particle scale
            /// y-direction particle term
            local_delta_t = min( local_delta_t, CFL*delta_y/max( abs( v_velocity_particle ), epsilon ) );	/// Particle scale
            /// z-direction particle term
            local_delta_t = min( local_delta_t, CFL*delta_z/max( abs( w_velocity_particle ), epsilon ) );	/// Particle scale
        
        }

    }

    /// Find minimum (global) delta_t
    double global_delta_t;
    MPI_Allreduce(&local_delta_t, &global_delta_t, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    /// Set new time step
    delta_t = global_delta_t;
   
};

void FlowSolverRHEA::calculateTransportCoefficients() {
    
    /// All (inner, halo, boundary) points: mu and kappa
    //#pragma acc kernels loop collapse(3) independent
    #pragma acc parallel loop collapse(3) present(this, mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], P_field.vector[0:_ls_], rho_field.vector[0:_ls_], T_field.vector[0:_ls_])
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                mu_field[I1D(i,j,k)]    = transport_coefficients->calculateDynamicViscosity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                kappa_field[I1D(i,j,k)] = transport_coefficients->calculateThermalConductivity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
            }
        }
    }

    /// Update halo values
    //mu_field.update();
    //kappa_field.update();

};

void FlowSolverRHEA::imposeImmersedBoundaryMethod() {

    //this->tagImmersedBoundaryMethod();
    this->reconstructImmersedBoundaryMethod();
    this->forceImmersedBoundaryMethod();

};

void FlowSolverRHEA::tagImmersedBoundaryMethod() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Set velocity & temperature of IBM
    immersed_boundary_method->setVelocityIBM( 0.0, 0.0, 0.0 );			// [m/s]
    immersed_boundary_method->setTemperatureIBM( 300.0 );			// [K]
    
    /// Set tags of IBM (tagging)

    /// All (inner, halo, boundary): tag_IBM
    //#pragma acc parallel loop collapse(3) 
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                tag_IBM_field[I1D(i,j,k)] = 0.0;
		//tag_IBM_field[I1D(i,j,k)] = 0.5;
		//tag_IBM_field[I1D(i,j,k)] = 1.0;
            }
        }
    }
    
    /// Update halo values
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(tag_IBM_field.vector[0:_ls_])
#endif
    tag_IBM_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(tag_IBM_field.vector[0:_ls_])
#endif

};

void FlowSolverRHEA::reconstructImmersedBoundaryMethod() {

    /// Get velocity & temperature of IBM (reconstructing)
    double u_IBM = 0.0, v_IBM = 0.0, w_IBM = 0.0;
    immersed_boundary_method->getVelocityIBM( u_IBM, v_IBM, w_IBM ); 
    double T_IBM = immersed_boundary_method->getTemperatureIBM(); 
    
    /// All (inner, halo, boundary): u_IBM, v_IBM, w_IBM, T_IBM
    double tag_IBM, tag_IBM_tolerance = 1.0e-5;
    //#pragma acc parallel loop collapse(3) 
    #pragma acc parallel loop collapse(3) present(this,tag_IBM_field.vector[0:_ls_],u_IBM_field.vector[0:_ls_], v_IBM_field.vector[0:_ls_], w_IBM_field.vector[0:_ls_],T_IBM_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], T_field.vector[0:_ls_], immersed_boundary_method) copyin (u_IBM,w_IBM,v_IBM, T_IBM) private (tag_IBM,tag_IBM_tolerance) 
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                /// Check tag of IBM
		tag_IBM = tag_IBM_field[I1D(i,j,k)];
                /// Internal IBM point
		if( tag_IBM > ( 1.0 - tag_IBM_tolerance ) ) {
                    /// Insert IBM velocity & temperature into IBM velocity & temperature
                    u_IBM_field[I1D(i,j,k)] = u_IBM;
                    v_IBM_field[I1D(i,j,k)] = v_IBM;
                    w_IBM_field[I1D(i,j,k)] = w_IBM;
                    T_IBM_field[I1D(i,j,k)] = T_IBM;
                /// External IBM point
		} else if( tag_IBM < ( 0.0 + tag_IBM_tolerance ) ) {
                    /// Insert flow velocity & temperature into IBM velocity & temperature
                    u_IBM_field[I1D(i,j,k)] = u_field[I1D(i,j,k)];
                    v_IBM_field[I1D(i,j,k)] = v_field[I1D(i,j,k)];
                    w_IBM_field[I1D(i,j,k)] = w_field[I1D(i,j,k)];
                    T_IBM_field[I1D(i,j,k)] = T_field[I1D(i,j,k)];
                /// Interface IBM point
		} else {
                    /// Insert flow-IBM interpolated velocity & temperature into IBM velocity & temperature
                    u_IBM_field[I1D(i,j,k)] = tag_IBM*u_IBM + ( 1.0 - tag_IBM )*u_field[I1D(i,j,k)];
                    v_IBM_field[I1D(i,j,k)] = tag_IBM*v_IBM + ( 1.0 - tag_IBM )*v_field[I1D(i,j,k)];
                    w_IBM_field[I1D(i,j,k)] = tag_IBM*w_IBM + ( 1.0 - tag_IBM )*w_field[I1D(i,j,k)];
                    T_IBM_field[I1D(i,j,k)] = tag_IBM*T_IBM + ( 1.0 - tag_IBM )*T_field[I1D(i,j,k)];
		}
            }
        }
    }

    /// Update halo values
    //u_IBM_field.update();
    //v_IBM_field.update();
    //w_IBM_field.update();
    //T_IBM_field.update();

};

void FlowSolverRHEA::forceImmersedBoundaryMethod() {

    /// Set penalty method of IBM (forcing)

    /// IBM forcing parameter
    const double K_IBM = ( -1.0 )*0.5*delta_t;
    
    /// Inner points: f_rhou, f_rhov, f_rhow and f_rhoE
    double ke, e_IBM, ke_IBM, f_rhoE_;
    delta_t = this->delta_t;
    #pragma acc parallel loop collapse(3) present(this, mu_field.vector[0:_ls_], f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_], rho_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], E_field.vector[0:_ls_], u_IBM_field.vector[0:_ls_], v_IBM_field.vector[0:_ls_], w_IBM_field.vector[0:_ls_], T_IBM_field.vector[0:_ls_]) private(ke, e_IBM,ke_IBM,f_rhoE_)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                f_rhou_field[I1D(i,j,k)] += rho_field[I1D(i,j,k)]*( u_field[I1D(i,j,k)] - u_IBM_field[I1D(i,j,k)] )/K_IBM;
                f_rhov_field[I1D(i,j,k)] += rho_field[I1D(i,j,k)]*( v_field[I1D(i,j,k)] - v_IBM_field[I1D(i,j,k)] )/K_IBM;
                f_rhow_field[I1D(i,j,k)] += rho_field[I1D(i,j,k)]*( w_field[I1D(i,j,k)] - w_IBM_field[I1D(i,j,k)] )/K_IBM;
                e_IBM = thermodynamics->calculateInternalEnergyFromPressureTemperatureDensity( P_field[I1D(i,j,k)], T_IBM_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
		f_rhoE_ = f_rhoE_field[I1D(i,j,k)];
		if( transport_pressure_scheme ) {
                    ke = 0.5*( pow( u_field[I1D(i,j,k)], 2.0 ) + pow( v_field[I1D(i,j,k)], 2.0 ) + pow( w_field[I1D(i,j,k)], 2.0 ) );
                    f_rhoE_field[I1D(i,j,k)] = f_rhoE_+ rho_field[I1D(i,j,k)]*( ( E_field[I1D(i,j,k)] - ke ) - e_IBM )/K_IBM;
                } else {
                    ke_IBM = 0.5*( pow( u_IBM_field[I1D(i,j,k)], 2.0 ) + pow( v_IBM_field[I1D(i,j,k)], 2.0 ) + pow( w_IBM_field[I1D(i,j,k)], 2.0 ) );
                    f_rhoE_field[I1D(i,j,k)] = f_rhoE_+  rho_field[I1D(i,j,k)]*( E_field[I1D(i,j,k)] - ( e_IBM + ke_IBM ) )/K_IBM; 
                }
            }
        }
    }

    /// Update halo values
    //f_rhou_field.update();
    //f_rhov_field.update();
    //f_rhow_field.update();
    //f_rhoE_field.update();
   
};

void FlowSolverRHEA::calculateSourceTerms() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    /// Inner points: f_rhou, f_rhov, f_rhow and f_rhoE
    #pragma acc parallel loop collapse(3) present(this, f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_], topo)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                f_rhou_field[I1D(i,j,k)] = 0.0;
                f_rhov_field[I1D(i,j,k)] = 0.0;
                f_rhow_field[I1D(i,j,k)] = 0.0;
                f_rhoE_field[I1D(i,j,k)] = 0.0;
                //f_rhoE_field[I1D(i,j,k)] = ( -1.0 )*( f_rhou_field[I1D(i,j,k)]*u_field[I1D(i,j,k)] + f_rhov_field[I1D(i,j,k)]*v_field[I1D(i,j,k)] + f_rhow_field[I1D(i,j,k)]*w_field[I1D(i,j,k)] );
            }
        }
    }

    /// Update halo values
    //f_rhou_field.update();
    //f_rhov_field.update();
    //f_rhow_field.update();
    //f_rhoE_field.update();

};

void FlowSolverRHEA::calculateInviscidFluxes() {

    /// Unsplit method for Euler equations:
    /// E. F. Toro.
    /// Riemann solvers and numerical methods for fluid dynamics.
    /// Springer, 2009.

    /// Inner points: rho, rhou, rhov, rhow and rhoE
    int index_LLL, index_LL, index_L, index_R, index_RR, index_RRR, var_type;
    double delta_x, delta_y, delta_z;
    double rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, P_rhouvw_LLL, T_LLL, a_LLL;
    double rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, P_rhouvw_LL, T_LL, a_LL;
    double rho_L, u_L, v_L, w_L, E_L, s_L, P_L, P_rhouvw_L, T_L, a_L;
    double rho_R, u_R, v_R, w_R, E_R, s_R, P_R, P_rhouvw_R, T_R, a_R;
    double rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, P_rhouvw_RR, T_RR, a_RR;
    double rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, P_rhouvw_RRR, T_RRR, a_RRR;
    double rho_F_p, rho_F_m, rhou_F_p, rhou_F_m, rhov_F_p;
    double rhov_F_m, rhow_F_p, rhow_F_m, rhoE_F_p, rhoE_F_m;
    #pragma acc parallel loop collapse(3) private(index_LLL, index_LL, index_L, index_R, index_RR, index_RRR, var_type, delta_x, delta_y, delta_z, rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, P_rhouvw_RRR, T_RRR, a_RRR, rho_F_p, rho_F_m, rhou_F_p, rhou_F_m, rhov_F_p, rhov_F_m, rhow_F_p, rhow_F_m, rhoE_F_p, rhoE_F_m) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], rho_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], sos_field.vector[0:_ls_], rho_inv_flux.vector[0:_ls_], rhou_inv_flux.vector[0:_ls_], rhov_inv_flux.vector[0:_ls_], rhow_inv_flux.vector[0:_ls_], rhoE_inv_flux.vector[0:_ls_] )  
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                /// x-direction i+1/2
                index_LLL = i - 2; index_LL = i - 1; index_L = i;
		if( i <= ( topo->iter_common[_INNER_][_INIX_] ) ) { index_LLL = i; index_LL = i; index_L = i; }
                index_R = i + 1; index_RR = i + 2; index_RRR = i + 3;
                if( i >= ( topo->iter_common[_INNER_][_ENDX_] - 1 ) ) { index_R = i + 1; index_RR = i + 1; index_RRR = i + 1; }
                rho_LLL = rho_field[I1D(index_LLL,j,k)]; rho_RRR = rho_field[I1D(index_RRR,j,k)]; 
                rho_LL  = rho_field[I1D(index_LL,j,k)];  rho_RR  = rho_field[I1D(index_RR,j,k)]; 
                rho_L   = rho_field[I1D(index_L,j,k)];   rho_R   = rho_field[I1D(index_R,j,k)];
                u_LLL   = u_field[I1D(index_LLL,j,k)];   u_RRR   = u_field[I1D(index_RRR,j,k)]; 
                u_LL    = u_field[I1D(index_LL,j,k)];    u_RR    = u_field[I1D(index_RR,j,k)]; 
                u_L     = u_field[I1D(index_L,j,k)];     u_R     = u_field[I1D(index_R,j,k)];
                v_LLL   = v_field[I1D(index_LLL,j,k)];   v_RRR   = v_field[I1D(index_RRR,j,k)]; 
                v_LL    = v_field[I1D(index_LL,j,k)];    v_RR    = v_field[I1D(index_RR,j,k)]; 
                v_L     = v_field[I1D(index_L,j,k)];     v_R     = v_field[I1D(index_R,j,k)];
                w_LLL   = w_field[I1D(index_LLL,j,k)];   w_RRR   = w_field[I1D(index_RRR,j,k)]; 
                w_LL    = w_field[I1D(index_LL,j,k)];    w_RR    = w_field[I1D(index_RR,j,k)]; 
                w_L     = w_field[I1D(index_L,j,k)];     w_R     = w_field[I1D(index_R,j,k)];
                E_LLL   = E_field[I1D(index_LLL,j,k)];   E_RRR   = E_field[I1D(index_RRR,j,k)]; 
                E_LL    = E_field[I1D(index_LL,j,k)];    E_RR    = E_field[I1D(index_RR,j,k)]; 
                E_L     = E_field[I1D(index_L,j,k)];     E_R     = E_field[I1D(index_R,j,k)];
                s_LLL   = s_field[I1D(index_LLL,j,k)];   s_RRR   = s_field[I1D(index_RRR,j,k)]; 
                s_LL    = s_field[I1D(index_LL,j,k)];    s_RR    = s_field[I1D(index_RR,j,k)]; 
                s_L     = s_field[I1D(index_L,j,k)];     s_R     = s_field[I1D(index_R,j,k)];		
                P_LLL   = P_field[I1D(index_LLL,j,k)];   P_RRR   = P_field[I1D(index_RRR,j,k)]; 
                P_LL    = P_field[I1D(index_LL,j,k)];    P_RR    = P_field[I1D(index_RR,j,k)]; 
                P_L     = P_field[I1D(index_L,j,k)];     P_R     = P_field[I1D(index_R,j,k)];
                T_LLL   = T_field[I1D(index_LLL,j,k)];   T_RRR   = T_field[I1D(index_RRR,j,k)]; 
                T_LL    = T_field[I1D(index_LL,j,k)];    T_RR    = T_field[I1D(index_RR,j,k)]; 
                T_L     = T_field[I1D(index_L,j,k)];     T_R     = T_field[I1D(index_R,j,k)];
                a_LLL   = sos_field[I1D(index_LLL,j,k)]; a_RRR   = sos_field[I1D(index_RRR,j,k)]; 
                a_LL    = sos_field[I1D(index_LL,j,k)];  a_RR    = sos_field[I1D(index_RR,j,k)]; 
                a_L     = sos_field[I1D(index_L,j,k)];   a_R     = sos_field[I1D(index_R,j,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                /// rho
                var_type = 0;
                rho_F_p  = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhou
                var_type = 1;
                rhou_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhov
                var_type = 2;
                rhov_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhow
                var_type = 3;
                rhow_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// x-direction i-1/2
                index_LLL = i - 3; index_LL = i - 2; index_L = i - 1;
		if( i <= ( topo->iter_common[_INNER_][_INIX_] + 1 ) ) { index_LLL = i - 1; index_LL = i - 1; index_L = i - 1; }
                index_R = i; index_RR = i + 1; index_RRR = i + 2;
                if( i >= ( topo->iter_common[_INNER_][_ENDX_] ) ) { index_R = i; index_RR = i; index_RRR = i; }
                rho_LLL = rho_field[I1D(index_LLL,j,k)]; rho_RRR = rho_field[I1D(index_RRR,j,k)]; 
                rho_LL  = rho_field[I1D(index_LL,j,k)];  rho_RR  = rho_field[I1D(index_RR,j,k)]; 
                rho_L   = rho_field[I1D(index_L,j,k)];   rho_R   = rho_field[I1D(index_R,j,k)];
                u_LLL   = u_field[I1D(index_LLL,j,k)];   u_RRR   = u_field[I1D(index_RRR,j,k)]; 
                u_LL    = u_field[I1D(index_LL,j,k)];    u_RR    = u_field[I1D(index_RR,j,k)]; 
                u_L     = u_field[I1D(index_L,j,k)];     u_R     = u_field[I1D(index_R,j,k)];
                v_LLL   = v_field[I1D(index_LLL,j,k)];   v_RRR   = v_field[I1D(index_RRR,j,k)]; 
                v_LL    = v_field[I1D(index_LL,j,k)];    v_RR    = v_field[I1D(index_RR,j,k)]; 
                v_L     = v_field[I1D(index_L,j,k)];     v_R     = v_field[I1D(index_R,j,k)];
                w_LLL   = w_field[I1D(index_LLL,j,k)];   w_RRR   = w_field[I1D(index_RRR,j,k)]; 
                w_LL    = w_field[I1D(index_LL,j,k)];    w_RR    = w_field[I1D(index_RR,j,k)]; 
                w_L     = w_field[I1D(index_L,j,k)];     w_R     = w_field[I1D(index_R,j,k)];
                E_LLL   = E_field[I1D(index_LLL,j,k)];   E_RRR   = E_field[I1D(index_RRR,j,k)]; 
                E_LL    = E_field[I1D(index_LL,j,k)];    E_RR    = E_field[I1D(index_RR,j,k)]; 
                E_L     = E_field[I1D(index_L,j,k)];     E_R     = E_field[I1D(index_R,j,k)];
                s_LLL   = s_field[I1D(index_LLL,j,k)];   s_RRR   = s_field[I1D(index_RRR,j,k)]; 
                s_LL    = s_field[I1D(index_LL,j,k)];    s_RR    = s_field[I1D(index_RR,j,k)]; 
                s_L     = s_field[I1D(index_L,j,k)];     s_R     = s_field[I1D(index_R,j,k)];		
                P_LLL   = P_field[I1D(index_LLL,j,k)];   P_RRR   = P_field[I1D(index_RRR,j,k)]; 
                P_LL    = P_field[I1D(index_LL,j,k)];    P_RR    = P_field[I1D(index_RR,j,k)]; 
                P_L     = P_field[I1D(index_L,j,k)];     P_R     = P_field[I1D(index_R,j,k)];
                T_LLL   = T_field[I1D(index_LLL,j,k)];   T_RRR   = T_field[I1D(index_RRR,j,k)]; 
                T_LL    = T_field[I1D(index_LL,j,k)];    T_RR    = T_field[I1D(index_RR,j,k)]; 
                T_L     = T_field[I1D(index_L,j,k)];     T_R     = T_field[I1D(index_R,j,k)];
                a_LLL   = sos_field[I1D(index_LLL,j,k)]; a_RRR   = sos_field[I1D(index_RRR,j,k)]; 
                a_LL    = sos_field[I1D(index_LL,j,k)];  a_RR    = sos_field[I1D(index_RR,j,k)]; 
                a_L     = sos_field[I1D(index_L,j,k)];   a_R     = sos_field[I1D(index_R,j,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                /// rho
                var_type = 0;
                rho_F_m  = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhou
                var_type = 1;
                rhou_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhov
                var_type = 2;
                rhov_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhow
                var_type = 3;
                rhow_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, u_LLL, v_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, u_LL, v_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, u_L, v_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, u_R, v_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, u_RR, v_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, u_RRR, v_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_x, var_type );
                /// Fluxes x-direction
                rho_inv_flux[I1D(i,j,k)]  = ( rho_F_p - rho_F_m )/delta_x;
                rhou_inv_flux[I1D(i,j,k)] = ( rhou_F_p - rhou_F_m )/delta_x;
                rhov_inv_flux[I1D(i,j,k)] = ( rhov_F_p - rhov_F_m )/delta_x;
                rhow_inv_flux[I1D(i,j,k)] = ( rhow_F_p - rhow_F_m )/delta_x;
                rhoE_inv_flux[I1D(i,j,k)] = ( rhoE_F_p - rhoE_F_m )/delta_x;
                /// y-direction j+1/2
                index_LLL = j - 2; index_LL = j - 1; index_L = j;
		if( j <= ( topo->iter_common[_INNER_][_INIY_] ) ) { index_LLL = j; index_LL = j; index_L = j; }
                index_R = j + 1; index_RR = j + 2; index_RRR = j + 3;
                if( j >= ( topo->iter_common[_INNER_][_ENDY_] - 1 ) ) { index_R = j + 1; index_RR = j + 1; index_RRR = j + 1; }
                rho_LLL = rho_field[I1D(i,index_LLL,k)]; rho_RRR = rho_field[I1D(i,index_RRR,k)]; 
                rho_LL  = rho_field[I1D(i,index_LL,k)];  rho_RR  = rho_field[I1D(i,index_RR,k)]; 
                rho_L   = rho_field[I1D(i,index_L,k)];   rho_R   = rho_field[I1D(i,index_R,k)];
                u_LLL   = u_field[I1D(i,index_LLL,k)];   u_RRR   = u_field[I1D(i,index_RRR,k)]; 
                u_LL    = u_field[I1D(i,index_LL,k)];    u_RR    = u_field[I1D(i,index_RR,k)]; 
                u_L     = u_field[I1D(i,index_L,k)];     u_R     = u_field[I1D(i,index_R,k)];
                v_LLL   = v_field[I1D(i,index_LLL,k)];   v_RRR   = v_field[I1D(i,index_RRR,k)]; 
                v_LL    = v_field[I1D(i,index_LL,k)];    v_RR    = v_field[I1D(i,index_RR,k)]; 
                v_L     = v_field[I1D(i,index_L,k)];     v_R     = v_field[I1D(i,index_R,k)];
                w_LLL   = w_field[I1D(i,index_LLL,k)];   w_RRR   = w_field[I1D(i,index_RRR,k)]; 
                w_LL    = w_field[I1D(i,index_LL,k)];    w_RR    = w_field[I1D(i,index_RR,k)]; 
                w_L     = w_field[I1D(i,index_L,k)];     w_R     = w_field[I1D(i,index_R,k)];
                E_LLL   = E_field[I1D(i,index_LLL,k)];   E_RRR   = E_field[I1D(i,index_RRR,k)]; 
                E_LL    = E_field[I1D(i,index_LL,k)];    E_RR    = E_field[I1D(i,index_RR,k)]; 
                E_L     = E_field[I1D(i,index_L,k)];     E_R     = E_field[I1D(i,index_R,k)];
                s_LLL   = s_field[I1D(i,index_LLL,k)];   s_RRR   = s_field[I1D(i,index_RRR,k)]; 
                s_LL    = s_field[I1D(i,index_LL,k)];    s_RR    = s_field[I1D(i,index_RR,k)]; 
                s_L     = s_field[I1D(i,index_L,k)];     s_R     = s_field[I1D(i,index_R,k)];		
                P_LLL   = P_field[I1D(i,index_LLL,k)];   P_RRR   = P_field[I1D(i,index_RRR,k)]; 
                P_LL    = P_field[I1D(i,index_LL,k)];    P_RR    = P_field[I1D(i,index_RR,k)]; 
                P_L     = P_field[I1D(i,index_L,k)];     P_R     = P_field[I1D(i,index_R,k)];
                T_LLL   = T_field[I1D(i,index_LLL,k)];   T_RRR   = T_field[I1D(i,index_RRR,k)]; 
                T_LL    = T_field[I1D(i,index_LL,k)];    T_RR    = T_field[I1D(i,index_RR,k)]; 
                T_L     = T_field[I1D(i,index_L,k)];     T_R     = T_field[I1D(i,index_R,k)];
                a_LLL   = sos_field[I1D(i,index_LLL,k)]; a_RRR   = sos_field[I1D(i,index_RRR,k)]; 
                a_LL    = sos_field[I1D(i,index_LL,k)];  a_RR    = sos_field[I1D(i,index_RR,k)]; 
                a_L     = sos_field[I1D(i,index_L,k)];   a_R     = sos_field[I1D(i,index_R,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                /// rho
                var_type = 0;
                rho_F_p  = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhou
                var_type = 2;
                rhou_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhov
                var_type = 1;
                rhov_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhow
                var_type = 3;
                rhow_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// y-direction j-1/2
                index_LLL = j - 3; index_LL = j - 2; index_L = j - 1;
		if( j <= ( topo->iter_common[_INNER_][_INIY_] + 1 ) ) { index_LLL = j - 1; index_LL = j - 1; index_L = j - 1; }
                index_R = j; index_RR = j + 1; index_RRR = j + 2;
                if( j >= ( topo->iter_common[_INNER_][_ENDY_] ) ) { index_R = j; index_RR = j; index_RRR = j; }
                rho_LLL = rho_field[I1D(i,index_LLL,k)]; rho_RRR = rho_field[I1D(i,index_RRR,k)]; 
                rho_LL  = rho_field[I1D(i,index_LL,k)];  rho_RR  = rho_field[I1D(i,index_RR,k)]; 
                rho_L   = rho_field[I1D(i,index_L,k)];   rho_R   = rho_field[I1D(i,index_R,k)];
                u_LLL   = u_field[I1D(i,index_LLL,k)];   u_RRR   = u_field[I1D(i,index_RRR,k)]; 
                u_LL    = u_field[I1D(i,index_LL,k)];    u_RR    = u_field[I1D(i,index_RR,k)]; 
                u_L     = u_field[I1D(i,index_L,k)];     u_R     = u_field[I1D(i,index_R,k)];
                v_LLL   = v_field[I1D(i,index_LLL,k)];   v_RRR   = v_field[I1D(i,index_RRR,k)]; 
                v_LL    = v_field[I1D(i,index_LL,k)];    v_RR    = v_field[I1D(i,index_RR,k)]; 
                v_L     = v_field[I1D(i,index_L,k)];     v_R     = v_field[I1D(i,index_R,k)];
                w_LLL   = w_field[I1D(i,index_LLL,k)];   w_RRR   = w_field[I1D(i,index_RRR,k)]; 
                w_LL    = w_field[I1D(i,index_LL,k)];    w_RR    = w_field[I1D(i,index_RR,k)]; 
                w_L     = w_field[I1D(i,index_L,k)];     w_R     = w_field[I1D(i,index_R,k)];
                E_LLL   = E_field[I1D(i,index_LLL,k)];   E_RRR   = E_field[I1D(i,index_RRR,k)]; 
                E_LL    = E_field[I1D(i,index_LL,k)];    E_RR    = E_field[I1D(i,index_RR,k)]; 
                E_L     = E_field[I1D(i,index_L,k)];     E_R     = E_field[I1D(i,index_R,k)];
                s_LLL   = s_field[I1D(i,index_LLL,k)];   s_RRR   = s_field[I1D(i,index_RRR,k)]; 
                s_LL    = s_field[I1D(i,index_LL,k)];    s_RR    = s_field[I1D(i,index_RR,k)]; 
                s_L     = s_field[I1D(i,index_L,k)];     s_R     = s_field[I1D(i,index_R,k)];		
                P_LLL   = P_field[I1D(i,index_LLL,k)];   P_RRR   = P_field[I1D(i,index_RRR,k)]; 
                P_LL    = P_field[I1D(i,index_LL,k)];    P_RR    = P_field[I1D(i,index_RR,k)]; 
                P_L     = P_field[I1D(i,index_L,k)];     P_R     = P_field[I1D(i,index_R,k)];
                T_LLL   = T_field[I1D(i,index_LLL,k)];   T_RRR   = T_field[I1D(i,index_RRR,k)]; 
                T_LL    = T_field[I1D(i,index_LL,k)];    T_RR    = T_field[I1D(i,index_RR,k)]; 
                T_L     = T_field[I1D(i,index_L,k)];     T_R     = T_field[I1D(i,index_R,k)];
                a_LLL   = sos_field[I1D(i,index_LLL,k)]; a_RRR   = sos_field[I1D(i,index_RRR,k)]; 
                a_LL    = sos_field[I1D(i,index_LL,k)];  a_RR    = sos_field[I1D(i,index_RR,k)]; 
                a_L     = sos_field[I1D(i,index_L,k)];   a_R     = sos_field[I1D(i,index_R,k)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                /// rho
                var_type = 0;
                rho_F_m  = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhou
                var_type = 2;
                rhou_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhov
                var_type = 1;
                rhov_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhow
                var_type = 3;
                rhow_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, v_LLL, u_LLL, w_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, v_LL, u_LL, w_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, v_L, u_L, w_L, E_L, s_L, P_L, T_L, a_L, rho_R, v_R, u_R, w_R, E_R, s_R, P_R, T_R, a_R, rho_RR, v_RR, u_RR, w_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, v_RRR, u_RRR, w_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_y, var_type );
                /// Fluxes y-direction
                rho_inv_flux[I1D(i,j,k)]  += ( rho_F_p - rho_F_m )/delta_y;
                rhou_inv_flux[I1D(i,j,k)] += ( rhou_F_p - rhou_F_m )/delta_y;
                rhov_inv_flux[I1D(i,j,k)] += ( rhov_F_p - rhov_F_m )/delta_y;
                rhow_inv_flux[I1D(i,j,k)] += ( rhow_F_p - rhow_F_m )/delta_y;
                rhoE_inv_flux[I1D(i,j,k)] += ( rhoE_F_p - rhoE_F_m )/delta_y;
		/// z-direction k+1/2
                index_LLL = k - 2; index_LL = k - 1; index_L = k;
		if( k <= ( topo->iter_common[_INNER_][_INIZ_] ) ) { index_LLL = k; index_LL = k; index_L = k; }
                index_R = k + 1; index_RR = k + 2; index_RRR = k + 3;
                if( k >= ( topo->iter_common[_INNER_][_ENDZ_] - 1 ) ) { index_R = k + 1; index_RR = k + 1; index_RRR = k + 1; }
                rho_LLL = rho_field[I1D(i,j,index_LLL)]; rho_RRR = rho_field[I1D(i,j,index_RRR)]; 
                rho_LL  = rho_field[I1D(i,j,index_LL)];  rho_RR  = rho_field[I1D(i,j,index_RR)]; 
                rho_L   = rho_field[I1D(i,j,index_L)];   rho_R   = rho_field[I1D(i,j,index_R)];
                u_LLL   = u_field[I1D(i,j,index_LLL)];   u_RRR   = u_field[I1D(i,j,index_RRR)]; 
                u_LL    = u_field[I1D(i,j,index_LL)];    u_RR    = u_field[I1D(i,j,index_RR)]; 
                u_L     = u_field[I1D(i,j,index_L)];     u_R     = u_field[I1D(i,j,index_R)];
                v_LLL   = v_field[I1D(i,j,index_LLL)];   v_RRR   = v_field[I1D(i,j,index_RRR)]; 
                v_LL    = v_field[I1D(i,j,index_LL)];    v_RR    = v_field[I1D(i,j,index_RR)]; 
                v_L     = v_field[I1D(i,j,index_L)];     v_R     = v_field[I1D(i,j,index_R)];
                w_LLL   = w_field[I1D(i,j,index_LLL)];   w_RRR   = w_field[I1D(i,j,index_RRR)]; 
                w_LL    = w_field[I1D(i,j,index_LL)];    w_RR    = w_field[I1D(i,j,index_RR)]; 
                w_L     = w_field[I1D(i,j,index_L)];     w_R     = w_field[I1D(i,j,index_R)];
                E_LLL   = E_field[I1D(i,j,index_LLL)];   E_RRR   = E_field[I1D(i,j,index_RRR)]; 
                E_LL    = E_field[I1D(i,j,index_LL)];    E_RR    = E_field[I1D(i,j,index_RR)]; 
                E_L     = E_field[I1D(i,j,index_L)];     E_R     = E_field[I1D(i,j,index_R)];
                s_LLL   = s_field[I1D(i,j,index_LLL)];   s_RRR   = s_field[I1D(i,j,index_RRR)]; 
                s_LL    = s_field[I1D(i,j,index_LL)];    s_RR    = s_field[I1D(i,j,index_RR)]; 
                s_L     = s_field[I1D(i,j,index_L)];     s_R     = s_field[I1D(i,j,index_R)];		
                P_LLL   = P_field[I1D(i,j,index_LLL)];   P_RRR   = P_field[I1D(i,j,index_RRR)]; 
                P_LL    = P_field[I1D(i,j,index_LL)];    P_RR    = P_field[I1D(i,j,index_RR)]; 
                P_L     = P_field[I1D(i,j,index_L)];     P_R     = P_field[I1D(i,j,index_R)];
                T_LLL   = T_field[I1D(i,j,index_LLL)];   T_RRR   = T_field[I1D(i,j,index_RRR)]; 
                T_LL    = T_field[I1D(i,j,index_LL)];    T_RR    = T_field[I1D(i,j,index_RR)]; 
                T_L     = T_field[I1D(i,j,index_L)];     T_R     = T_field[I1D(i,j,index_R)];
                a_LLL   = sos_field[I1D(i,j,index_LLL)]; a_RRR   = sos_field[I1D(i,j,index_RRR)]; 
                a_LL    = sos_field[I1D(i,j,index_LL)];  a_RR    = sos_field[I1D(i,j,index_RR)]; 
                a_L     = sos_field[I1D(i,j,index_L)];   a_R     = sos_field[I1D(i,j,index_R)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                /// rho
                var_type = 0;
                rho_F_p  = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhou
                var_type = 3;
                rhou_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhov
                var_type = 2;
                rhov_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhow
                var_type = 1;
                rhow_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_p = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// z-direction k-1/2
                index_LLL = k - 3; index_LL = k - 2; index_L = k - 1;
		if( k <= ( topo->iter_common[_INNER_][_INIZ_] + 1 ) ) { index_LLL = k - 1; index_LL = k - 1; index_L = k - 1; }
                index_R = k; index_RR = k + 1; index_RRR = k + 2;
                if( k >= ( topo->iter_common[_INNER_][_ENDZ_] ) ) { index_R = k; index_RR = k; index_RRR = k; }
                rho_LLL = rho_field[I1D(i,j,index_LLL)]; rho_RRR = rho_field[I1D(i,j,index_RRR)]; 
                rho_LL  = rho_field[I1D(i,j,index_LL)];  rho_RR  = rho_field[I1D(i,j,index_RR)]; 
                rho_L   = rho_field[I1D(i,j,index_L)];   rho_R   = rho_field[I1D(i,j,index_R)];
                u_LLL   = u_field[I1D(i,j,index_LLL)];   u_RRR   = u_field[I1D(i,j,index_RRR)]; 
                u_LL    = u_field[I1D(i,j,index_LL)];    u_RR    = u_field[I1D(i,j,index_RR)]; 
                u_L     = u_field[I1D(i,j,index_L)];     u_R     = u_field[I1D(i,j,index_R)];
                v_LLL   = v_field[I1D(i,j,index_LLL)];   v_RRR   = v_field[I1D(i,j,index_RRR)]; 
                v_LL    = v_field[I1D(i,j,index_LL)];    v_RR    = v_field[I1D(i,j,index_RR)]; 
                v_L     = v_field[I1D(i,j,index_L)];     v_R     = v_field[I1D(i,j,index_R)];
                w_LLL   = w_field[I1D(i,j,index_LLL)];   w_RRR   = w_field[I1D(i,j,index_RRR)]; 
                w_LL    = w_field[I1D(i,j,index_LL)];    w_RR    = w_field[I1D(i,j,index_RR)]; 
                w_L     = w_field[I1D(i,j,index_L)];     w_R     = w_field[I1D(i,j,index_R)];
                E_LLL   = E_field[I1D(i,j,index_LLL)];   E_RRR   = E_field[I1D(i,j,index_RRR)]; 
                E_LL    = E_field[I1D(i,j,index_LL)];    E_RR    = E_field[I1D(i,j,index_RR)]; 
                E_L     = E_field[I1D(i,j,index_L)];     E_R     = E_field[I1D(i,j,index_R)];
                s_LLL   = s_field[I1D(i,j,index_LLL)];   s_RRR   = s_field[I1D(i,j,index_RRR)]; 
                s_LL    = s_field[I1D(i,j,index_LL)];    s_RR    = s_field[I1D(i,j,index_RR)]; 
                s_L     = s_field[I1D(i,j,index_L)];     s_R     = s_field[I1D(i,j,index_R)];		
                P_LLL   = P_field[I1D(i,j,index_LLL)];   P_RRR   = P_field[I1D(i,j,index_RRR)]; 
                P_LL    = P_field[I1D(i,j,index_LL)];    P_RR    = P_field[I1D(i,j,index_RR)]; 
                P_L     = P_field[I1D(i,j,index_L)];     P_R     = P_field[I1D(i,j,index_R)];
                T_LLL   = T_field[I1D(i,j,index_LLL)];   T_RRR   = T_field[I1D(i,j,index_RRR)]; 
                T_LL    = T_field[I1D(i,j,index_LL)];    T_RR    = T_field[I1D(i,j,index_RR)]; 
                T_L     = T_field[I1D(i,j,index_L)];     T_R     = T_field[I1D(i,j,index_R)];
                a_LLL   = sos_field[I1D(i,j,index_LLL)]; a_RRR   = sos_field[I1D(i,j,index_RRR)]; 
                a_LL    = sos_field[I1D(i,j,index_LL)];  a_RR    = sos_field[I1D(i,j,index_RR)]; 
                a_L     = sos_field[I1D(i,j,index_L)];   a_R     = sos_field[I1D(i,j,index_R)];
                P_rhouvw_LLL = P_LLL - P_thermo;         P_rhouvw_RRR = P_RRR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_LL  = P_LL - P_thermo;          P_rhouvw_RR  = P_RR - P_thermo;		/// P_Thermo = 0.0 when ACM is deactivated
                P_rhouvw_L   = P_L - P_thermo;           P_rhouvw_R   = P_R - P_thermo;			/// P_Thermo = 0.0 when ACM is deactivated
                /// rho
                var_type = 0;
                rho_F_m  = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhou
                var_type = 3;
                rhou_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhov
                var_type = 2;
                rhov_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhow
                var_type = 1;
                rhow_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_rhouvw_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_rhouvw_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_rhouvw_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_rhouvw_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_rhouvw_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_rhouvw_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// rhoE
                var_type = 4;
                rhoE_F_m = riemann_solver->calculateIntercellFlux( rho_LLL, w_LLL, v_LLL, u_LLL, E_LLL, s_LLL, P_LLL, T_LLL, a_LLL, rho_LL, w_LL, v_LL, u_LL, E_LL, s_LL, P_LL, T_LL, a_LL, rho_L, w_L, v_L, u_L, E_L, s_L, P_L, T_L, a_L, rho_R, w_R, v_R, u_R, E_R, s_R, P_R, T_R, a_R, rho_RR, w_RR, v_RR, u_RR, E_RR, s_RR, P_RR, T_RR, a_RR, rho_RRR, w_RRR, v_RRR, u_RRR, E_RRR, s_RRR, P_RRR, T_RRR, a_RRR, delta_z, var_type );
                /// Fluxes z-direction
                rho_inv_flux[I1D(i,j,k)]  += ( rho_F_p - rho_F_m )/delta_z;
                rhou_inv_flux[I1D(i,j,k)] += ( rhou_F_p - rhou_F_m )/delta_z;
                rhov_inv_flux[I1D(i,j,k)] += ( rhov_F_p - rhov_F_m )/delta_z;
                rhow_inv_flux[I1D(i,j,k)] += ( rhow_F_p - rhow_F_m )/delta_z;
                rhoE_inv_flux[I1D(i,j,k)] += ( rhoE_F_p - rhoE_F_m )/delta_z;
            }
        }
    }

    /// Update halo values
    //rho_inv_flux.update();
    //rhou_inv_flux.update();
    //rhov_inv_flux.update();
    //rhow_inv_flux.update();
    //rhoE_inv_flux.update();

};

void FlowSolverRHEA::calculateViscousFluxes() {

    /// Second-order central finite differences for derivatives:
    /// P. Moin.
    /// Fundamentals of engineering numerical analysis.
    /// Cambridge University Press, 2010.

    /// Inner points: rho_vis_flux, rhou_vis_flux, rhov_vis_flux, rhow_vis_flux, rhoE_vis_flux and work_vis_rhoe_flux 
    double delta_x, delta_y, delta_z;
    double d_u_x, d_u_y, d_u_z, d_v_x, d_v_y, d_v_z, d_w_x, d_w_y, d_w_z;
    double d_T_x, d_T_y, d_T_z;
    double d_mu_x, d_mu_y, d_mu_z, d_kappa_x, d_kappa_y, d_kappa_z;
    double div_uvw, tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
    double div_tau_x, div_tau_y, div_tau_z;
    double div_q, div_uvw_tau_rhoe, div_uvw_tau_rhoke, div_uvw_tau_rhoE; 
    #pragma acc parallel loop collapse(3) private(delta_x, delta_y, delta_z, d_u_x, d_u_y, d_u_z, d_v_x, d_v_y, d_v_z, d_w_x, d_w_y, d_w_z, d_T_x, d_T_y, d_T_z, d_mu_x, d_mu_y, d_mu_z, d_kappa_x, d_kappa_y, d_kappa_z, div_uvw, tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz, div_tau_x, div_tau_y, div_tau_z, div_q, div_uvw_tau_rhoe, div_uvw_tau_rhoke, div_uvw_tau_rhoE) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], T_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], rhou_vis_flux.vector[0:_ls_], rhov_vis_flux.vector[0:_ls_], rhow_vis_flux.vector[0:_ls_], rhoE_vis_flux.vector[0:_ls_], work_vis_rhoe_flux.vector[0:_ls_])  
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                /// Velocity derivatives
                d_u_x = ( u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_u_y = ( u_field[I1D(i,j+1,k)] - u_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_u_z = ( u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                d_v_x = ( v_field[I1D(i+1,j,k)] - v_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_v_y = ( v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_v_z = ( v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                d_w_x = ( w_field[I1D(i+1,j,k)] - w_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_w_y = ( w_field[I1D(i,j+1,k)] - w_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_w_z = ( w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                /// Temperature derivatives
                d_T_x = ( T_field[I1D(i+1,j,k)] - T_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_T_y = ( T_field[I1D(i,j+1,k)] - T_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_T_z = ( T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                /// Transport coefficients derivatives
                d_mu_x    = ( mu_field[I1D(i+1,j,k)] - mu_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_mu_y    = ( mu_field[I1D(i,j+1,k)] - mu_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_mu_z    = ( mu_field[I1D(i,j,k+1)] - mu_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                d_kappa_x = ( kappa_field[I1D(i+1,j,k)] - kappa_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_kappa_y = ( kappa_field[I1D(i,j+1,k)] - kappa_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_kappa_z = ( kappa_field[I1D(i,j,k+1)] - kappa_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                /// Divergence of velocity
                div_uvw = d_u_x + d_v_y + d_w_z;
                /// Viscous stresses ( symmetric tensor )
                tau_xx = 2.0*mu_field[I1D(i,j,k)]*( d_u_x - ( div_uvw/3.0 ) );
                tau_xy = mu_field[I1D(i,j,k)]*( d_u_y + d_v_x );
                tau_xz = mu_field[I1D(i,j,k)]*( d_u_z + d_w_x );
                tau_yy = 2.0*mu_field[I1D(i,j,k)]*( d_v_y - ( div_uvw/3.0 ) );
                tau_yz = mu_field[I1D(i,j,k)]*( d_v_z + d_w_y );
                tau_zz = 2.0*mu_field[I1D(i,j,k)]*( d_w_z - ( div_uvw/3.0 ) );
                /// Divergence of viscous stresses
                div_tau_x = mu_field[I1D(i,j,k)]*( ( 1.00/delta_x )*( ( u_field[I1D(i+1,j,k)] - u_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] )
                                                                    - ( u_field[I1D(i,j,k)] - u_field[I1D(i-1,j,k)] )/( x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)] ) )
                                                 + ( 1.00/delta_y )*( ( u_field[I1D(i,j+1,k)] - u_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] )
                                                                    - ( u_field[I1D(i,j,k)] - u_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)] ) )
                                                 + ( 1.00/delta_z )*( ( u_field[I1D(i,j,k+1)] - u_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] )
                                                                    - ( u_field[I1D(i,j,k)] - u_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)] ) ) )
              + ( 1.0/3.0 )*mu_field[I1D(i,j,k)]*( ( 1.00/delta_x )*( ( u_field[I1D(i+1,j,k)] - u_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] )
                                                                    - ( u_field[I1D(i,j,k)] - u_field[I1D(i-1,j,k)] )/( x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)] ) )
                                                 + ( 0.25/delta_x )*( ( v_field[I1D(i+1,j+1,k)] - v_field[I1D(i+1,j-1,k)] )/delta_y
                                                                    - ( v_field[I1D(i-1,j+1,k)] - v_field[I1D(i-1,j-1,k)] )/delta_y )
                                                 + ( 0.25/delta_x )*( ( w_field[I1D(i+1,j,k+1)] - w_field[I1D(i+1,j,k-1)] )/delta_z
                                                                    - ( w_field[I1D(i-1,j,k+1)] - w_field[I1D(i-1,j,k-1)] )/delta_z ) )
	                  + ( d_mu_x*tau_xx + d_mu_y*tau_xy + d_mu_z*tau_xz )/max( mu_field[I1D(i,j,k)], epsilon );
                div_tau_y = mu_field[I1D(i,j,k)]*( ( 1.00/delta_x )*( ( v_field[I1D(i+1,j,k)] - v_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] )
                                                                    - ( v_field[I1D(i,j,k)] - v_field[I1D(i-1,j,k)] )/( x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)] ) )
                                                 + ( 1.00/delta_y )*( ( v_field[I1D(i,j+1,k)] - v_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] )
                                                                    - ( v_field[I1D(i,j,k)] - v_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)] ) )
                                                 + ( 1.00/delta_z )*( ( v_field[I1D(i,j,k+1)] - v_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] )
                                                                    - ( v_field[I1D(i,j,k)] - v_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)] ) ) )
              + ( 1.0/3.0 )*mu_field[I1D(i,j,k)]*( ( 0.25/delta_y )*( ( u_field[I1D(i+1,j+1,k)] - u_field[I1D(i-1,j+1,k)] )/delta_x
                                                                    - ( u_field[I1D(i+1,j-1,k)] - u_field[I1D(i-1,j-1,k)] )/delta_x )
                                                 + ( 1.00/delta_y )*( ( v_field[I1D(i,j+1,k)] - v_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] )
                                                                    - ( v_field[I1D(i,j,k)] - v_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)] ) )
                                                 + ( 0.25/delta_y )*( ( w_field[I1D(i,j+1,k+1)] - w_field[I1D(i,j+1,k-1)] )/delta_z
                                                                    - ( w_field[I1D(i,j-1,k+1)] - w_field[I1D(i,j-1,k-1)] )/delta_z ) )
	                  + ( d_mu_x*tau_xy + d_mu_y*tau_yy + d_mu_z*tau_yz )/max( mu_field[I1D(i,j,k)], epsilon );
                div_tau_z = mu_field[I1D(i,j,k)]*( ( 1.00/delta_x )*( ( w_field[I1D(i+1,j,k)] - w_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] )
                                                                    - ( w_field[I1D(i,j,k)] - w_field[I1D(i-1,j,k)] )/( x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)] ) )
                                                 + ( 1.00/delta_y )*( ( w_field[I1D(i,j+1,k)] - w_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] )
                                                                    - ( w_field[I1D(i,j,k)] - w_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)] ) )
                                                 + ( 1.00/delta_z )*( ( w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] )
                                                                    - ( w_field[I1D(i,j,k)] - w_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)] ) ) )
              + ( 1.0/3.0 )*mu_field[I1D(i,j,k)]*( ( 0.25/delta_z )*( ( u_field[I1D(i+1,j,k+1)] - u_field[I1D(i-1,j,k+1)] )/delta_x
                                                                    - ( u_field[I1D(i+1,j,k-1)] - u_field[I1D(i-1,j,k-1)] )/delta_x )
                                                 + ( 0.25/delta_z )*( ( v_field[I1D(i,j+1,k+1)] - v_field[I1D(i,j-1,k+1)] )/delta_y
                                                                    - ( v_field[I1D(i,j+1,k-1)] - v_field[I1D(i,j-1,k-1)] )/delta_y )
                                                 + ( 1.00/delta_z )*( ( w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] )
                                                                    - ( w_field[I1D(i,j,k)] - w_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)] ) ) )
	                  + ( d_mu_x*tau_xz + d_mu_y*tau_yz + d_mu_z*tau_zz )/max( mu_field[I1D(i,j,k)], epsilon );
                /// Fourier term
                div_q = ( -1.0 )*kappa_field[I1D(i,j,k)]*( ( 1.0/delta_x )*( ( T_field[I1D(i+1,j,k)] - T_field[I1D(i,j,k)] )/( x_field[I1D(i+1,j,k)] - x_field[I1D(i,j,k)] )
                                                                           - ( T_field[I1D(i,j,k)] - T_field[I1D(i-1,j,k)] )/( x_field[I1D(i,j,k)] - x_field[I1D(i-1,j,k)] ) )
                                                         + ( 1.0/delta_y )*( ( T_field[I1D(i,j+1,k)] - T_field[I1D(i,j,k)] )/( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j,k)] )
                                                                           - ( T_field[I1D(i,j,k)] - T_field[I1D(i,j-1,k)] )/( y_field[I1D(i,j,k)] - y_field[I1D(i,j-1,k)] ) )
                                                         + ( 1.0/delta_z )*( ( T_field[I1D(i,j,k+1)] - T_field[I1D(i,j,k)] )/( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k)] )
                                                                           - ( T_field[I1D(i,j,k)] - T_field[I1D(i,j,k-1)] )/( z_field[I1D(i,j,k)] - z_field[I1D(i,j,k-1)] ) ) )
		      - d_kappa_x*d_T_x - d_kappa_y*d_T_y - d_kappa_z*d_T_z;
                /// Work of viscous stresses for internal energy
                div_uvw_tau_rhoe = tau_xx*d_u_x + tau_xy*d_u_y + tau_xz*d_u_z
                                 + tau_xy*d_v_x + tau_yy*d_v_y + tau_yz*d_v_z
                                 + tau_xz*d_w_x + tau_yz*d_w_y + tau_zz*d_w_z;
                /// Work of viscous stresses for kinetic energy
                div_uvw_tau_rhoke = u_field[I1D(i,j,k)]*div_tau_x + v_field[I1D(i,j,k)]*div_tau_y + w_field[I1D(i,j,k)]*div_tau_z;
                /// Work of viscous stresses for total energy
                div_uvw_tau_rhoE = div_uvw_tau_rhoe + div_uvw_tau_rhoke; 
                /// Viscous fluxes
                rhou_vis_flux[I1D(i,j,k)]      = div_tau_x;
                rhov_vis_flux[I1D(i,j,k)]      = div_tau_y;
                rhow_vis_flux[I1D(i,j,k)]      = div_tau_z;
                rhoE_vis_flux[I1D(i,j,k)]      = ( -1.0 )*div_q + div_uvw_tau_rhoE;
                work_vis_rhoe_flux[I1D(i,j,k)] = ( -1.0 )*div_q + div_uvw_tau_rhoe;
            }
        }
    }

    /// Update halo values
    //rhou_vis_flux.update();
    //rhov_vis_flux.update();
    //rhow_vis_flux.update();
    //rhoE_vis_flux.update();
    //work_vis_rhoe_flux.update();

};

void FlowSolverRHEA::timeAdvanceConservedVariables() {

    /// If transporting pressure instead of total energy, advance pressure
    if( transport_pressure_scheme ) {
        this->timeAdvancePressure();
    }

    delta_t = this->delta_t;
    
    /// Coefficients of explicit Runge-Kutta stages
    double rk_a = 0.0, rk_b = 0.0, rk_c = 0.0;
    runge_kutta_method->setStageCoefficients(rk_a,rk_b,rk_c,rk_time_stage);    

    /// Particles-fluid two-way coupling (Eulerian-Lagrangian)
    if( ( point_particles->get_num_prts_total() ) > 0 && activate_two_way_coupling_particles ) {
        /// Iterate through local particles in use
        int i_local_index = -1, j_local_index = -1, k_local_index = -1;
        double x_position_particle = 0.0, y_position_particle = 0.0, z_position_particle = 0.0;
        double u_velocity_fluid_particle = 0.0, v_velocity_fluid_particle = 0.0, w_velocity_fluid_particle = 0.0;
        double dynamic_viscosity_fluid = 0.0, mass_particle = 0.0, relaxation_time_particle = 0.0;
        double delta_x = 0.0, delta_y = 0.0, delta_z = 0.0, volume = 0.0;
        double f_twc_rhou = 0.0, f_twc_rhov = 0.0, f_twc_rhow = 0.0;
        int capacity = point_particles->get_prt_capacity(); 
        #pragma acc parallel loop collapse (1) private( i_local_index, j_local_index, k_local_index, x_position_particle, y_position_particle, z_position_particle, u_velocity_fluid_particle, v_velocity_fluid_particle, w_velocity_fluid_particle, dynamic_viscosity_fluid, relaxation_time_particle, delta_x, delta_y, delta_z, volume, f_twc_rhou, f_twc_rhov, f_twc_rhow ) present( this, mesh, point_particles, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], mu_field.vector[0:_ls_], f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], point_particles->local_prts_densities[0:capacity], point_particles->local_prts_diameters[0:capacity], point_particles->local_prts_positions_0_x[0:capacity], point_particles->local_prts_positions_0_y[0:capacity], point_particles->local_prts_positions_0_z[0:capacity], point_particles->local_prts_velocities_x[0:capacity], point_particles->local_prts_velocities_y[0:capacity], point_particles->local_prts_velocities_z[0:capacity], point_particles->local_prts_velocities_0_x[0:capacity], point_particles->local_prts_velocities_0_y[0:capacity], point_particles->local_prts_velocities_0_z[0:capacity], point_particles->local_prts_indexes_0_i[0:capacity], point_particles->local_prts_indexes_0_j[0:capacity], point_particles->local_prts_indexes_0_k[0:capacity] )         
	for( int p = 0; p < this->number_particles_local_in_use; p++ ) {

            /// Obtain Lagrangian-Eulerian indexes 0
            i_local_index = point_particles->local_prts_indexes_0_i[p];	/// Local index i
            j_local_index = point_particles->local_prts_indexes_0_j[p];	/// Local index j
            k_local_index = point_particles->local_prts_indexes_0_k[p];	/// Local index k

            /// Geometric stuff
            delta_x = 0.5*( x_field[I1D(i_local_index+1,j_local_index,k_local_index)] - x_field[I1D(i_local_index-1,j_local_index,k_local_index)] ); 
            delta_y = 0.5*( y_field[I1D(i_local_index,j_local_index+1,k_local_index)] - y_field[I1D(i_local_index,j_local_index-1,k_local_index)] ); 
            delta_z = 0.5*( z_field[I1D(i_local_index,j_local_index,k_local_index+1)] - z_field[I1D(i_local_index,j_local_index,k_local_index-1)] );
	    volume = delta_x*delta_y*delta_z;

            /// Obtain Lagrangian position 0
            x_position_particle = point_particles->local_prts_positions_0_x[p];
            y_position_particle = point_particles->local_prts_positions_0_y[p];
            z_position_particle = point_particles->local_prts_positions_0_z[p];

            /// Obtain Lagrangian-Eulerian values
            u_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index,k_local_index)], u_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index-1,k_local_index)], u_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index,j_local_index,k_local_index-1)], u_field[I1D(i_local_index,j_local_index,k_local_index)], u_field[I1D(i_local_index,j_local_index,k_local_index+1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index,j_local_index+1,k_local_index)], u_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index,k_local_index)], u_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], u_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
            v_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index,k_local_index)], v_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index-1,k_local_index)], v_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index,j_local_index,k_local_index-1)], v_field[I1D(i_local_index,j_local_index,k_local_index)], v_field[I1D(i_local_index,j_local_index,k_local_index+1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index,j_local_index+1,k_local_index)], v_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index,k_local_index)], v_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], v_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
            w_velocity_fluid_particle = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index,k_local_index)], w_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index-1,k_local_index)], w_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index,j_local_index,k_local_index-1)], w_field[I1D(i_local_index,j_local_index,k_local_index)], w_field[I1D(i_local_index,j_local_index,k_local_index+1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index,j_local_index+1,k_local_index)], w_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index,k_local_index)], w_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], w_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );
            dynamic_viscosity_fluid = this->trilinearInterpolation( x_position_particle, y_position_particle, z_position_particle, x_field[I1D(i_local_index-1,j_local_index,k_local_index)], x_field[I1D(i_local_index,j_local_index,k_local_index)], x_field[I1D(i_local_index+1,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index-1,k_local_index)], y_field[I1D(i_local_index,j_local_index,k_local_index)], y_field[I1D(i_local_index,j_local_index+1,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index-1)], z_field[I1D(i_local_index,j_local_index,k_local_index)], z_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index-1,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index,k_local_index)], mu_field[I1D(i_local_index,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index,j_local_index+1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index-1,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index,k_local_index+1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index-1)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index)], mu_field[I1D(i_local_index+1,j_local_index+1,k_local_index+1)] );

            //mass_particle = point_particles->calculate_mass_prt( p );
	    mass_particle = ( pi/6.0 )*point_particles->local_prts_densities[p]*pow( point_particles->local_prts_diameters[p], 3.0 );
            //relaxation_time_particle = point_particles->calculate_relaxation_time_prt( p, dynamic_viscosity_fluid );
	    relaxation_time_particle = ( point_particles->local_prts_densities[p]*pow( point_particles->local_prts_diameters[p], 2.0 ) )/( 18.0*dynamic_viscosity_fluid );

            /// Calculate two-way coupling momentum forces
            f_twc_rhou = ( mass_particle/relaxation_time_particle )*( point_particles->local_prts_velocities_0_x[p] - u_velocity_fluid_particle )/volume;
            f_twc_rhov = ( mass_particle/relaxation_time_particle )*( point_particles->local_prts_velocities_0_y[p] - v_velocity_fluid_particle )/volume;
            f_twc_rhow = ( mass_particle/relaxation_time_particle )*( point_particles->local_prts_velocities_0_z[p] - w_velocity_fluid_particle )/volume;
            
            /// Update momentum source terms
            #pragma acc atomic update
            f_rhou_field[I1D(i_local_index,j_local_index,k_local_index)] += f_twc_rhou;
	    #pragma acc atomic update
            f_rhov_field[I1D(i_local_index,j_local_index,k_local_index)] += f_twc_rhov;
	    #pragma acc atomic update
            f_rhow_field[I1D(i_local_index,j_local_index,k_local_index)] += f_twc_rhow;

        }
    }

    /// Inner points: rho, rhou, rhov, rhow and rhoE
    //double f_twc_rhou = 0.0, f_twc_rhov = 0.0, f_twc_rhow = 0.0;
    double f_rhouvw = 0.0, rho_rhs_flux = 0.0, rhou_rhs_flux = 0.0, rhov_rhs_flux = 0.0, rhow_rhs_flux = 0.0, rhoE_rhs_flux = 0.0;
    #pragma acc parallel loop collapse(3) present(this, f_rhou_field.vector[0:_ls_], f_rhov_field.vector[0:_ls_], f_rhow_field.vector[0:_ls_], f_rhoE_field.vector[0:_ls_], rho_inv_flux.vector[0:_ls_], rhou_inv_flux.vector[0:_ls_], rhov_inv_flux.vector[0:_ls_], rhow_inv_flux.vector[0:_ls_], rhoE_inv_flux.vector[0:_ls_], rhou_vis_flux.vector[0:_ls_], rhov_vis_flux.vector[0:_ls_], rhow_vis_flux.vector[0:_ls_], rhoE_vis_flux.vector[0:_ls_], rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], rho_0_field.vector[0:_ls_], rhou_0_field.vector[0:_ls_], rhov_0_field.vector[0:_ls_], rhow_0_field.vector[0:_ls_], rhoE_0_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Work of momentum sources
                f_rhouvw = f_rhou_field[I1D(i,j,k)]*u_field[I1D(i,j,k)] + f_rhov_field[I1D(i,j,k)]*v_field[I1D(i,j,k)] + f_rhow_field[I1D(i,j,k)]*w_field[I1D(i,j,k)];
                /// Sum right-hand-side (RHS) fluxes
                rho_rhs_flux  = ( -1.0 )*rho_inv_flux[I1D(i,j,k)]; 
                rhou_rhs_flux = ( -1.0 )*rhou_inv_flux[I1D(i,j,k)] + rhou_vis_flux[I1D(i,j,k)] + f_rhou_field[I1D(i,j,k)]; 
                rhov_rhs_flux = ( -1.0 )*rhov_inv_flux[I1D(i,j,k)] + rhov_vis_flux[I1D(i,j,k)] + f_rhov_field[I1D(i,j,k)]; 
                rhow_rhs_flux = ( -1.0 )*rhow_inv_flux[I1D(i,j,k)] + rhow_vis_flux[I1D(i,j,k)] + f_rhow_field[I1D(i,j,k)]; 
                rhoE_rhs_flux = ( -1.0 )*rhoE_inv_flux[I1D(i,j,k)] + rhoE_vis_flux[I1D(i,j,k)] + f_rhoE_field[I1D(i,j,k)] + f_rhouvw;
                /// Runge-Kutta step
                rho_field[I1D(i,j,k)]  = rk_a*rho_0_field[I1D(i,j,k)]  + rk_b*rho_field[I1D(i,j,k)]  + rk_c*delta_t*rho_rhs_flux;
                rhou_field[I1D(i,j,k)] = rk_a*rhou_0_field[I1D(i,j,k)] + rk_b*rhou_field[I1D(i,j,k)] + rk_c*delta_t*rhou_rhs_flux;
                rhov_field[I1D(i,j,k)] = rk_a*rhov_0_field[I1D(i,j,k)] + rk_b*rhov_field[I1D(i,j,k)] + rk_c*delta_t*rhov_rhs_flux;
                rhow_field[I1D(i,j,k)] = rk_a*rhow_0_field[I1D(i,j,k)] + rk_b*rhow_field[I1D(i,j,k)] + rk_c*delta_t*rhow_rhs_flux;
                rhoE_field[I1D(i,j,k)] = rk_a*rhoE_0_field[I1D(i,j,k)] + rk_b*rhoE_field[I1D(i,j,k)] + rk_c*delta_t*rhoE_rhs_flux;
	    }
        }
    }
    
    ///// Attention! Communications performed only at the last stage of the Runge-Kutta to improve computational performance
    ///// ... temporal integration is first-order at points connecting partitions
    //if( rk_time_stage == rk_number_stages ) {
    
        /// Update halo values
#if _GPU_AWARE_MPI_DEACTIVATED_ 
	#pragma acc update host(rho_field.vector[0:_ls_],rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_],rhoE_field.vector[0:_ls_])
#endif
        rho_field.update();
        rhou_field.update();
        rhov_field.update();
        rhow_field.update();
        rhoE_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
	#pragma acc update device(rho_field.vector[0:_ls_],rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_],rhoE_field.vector[0:_ls_])
#endif

    //}

};

void FlowSolverRHEA::timeAdvancePressure() {

    /// Inner points: P_inv_flux, P_vis_flux
    double delta_x, delta_y, delta_z;
    double d_P_x, d_P_y, d_P_z, d_u_x, d_v_y, d_w_z;
    double div_uvw;
    //#pragma acc kernels loop collapse(3) independent
    #pragma acc parallel loop collapse(3) private(delta_x, delta_y, delta_z, d_P_x, d_P_y, d_P_z, d_u_x, d_v_y, d_w_z, div_uvw) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], P_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], rho_field.vector[0:_ls_], sos_field.vector[0:_ls_], P_inv_flux.vector[0:_ls_], P_vis_flux.vector[0:_ls_], T_field.vector[0:_ls_], c_v_field.vector[0:_ls_], work_vis_rhoe_flux.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                /// Pressure and velocity derivatives
                d_P_x = ( P_field[I1D(i+1,j,k)] - P_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_P_y = ( P_field[I1D(i,j+1,k)] - P_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_P_z = ( P_field[I1D(i,j,k+1)] - P_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                d_u_x = ( u_field[I1D(i+1,j,k)] - u_field[I1D(i-1,j,k)] )/( 2.0*delta_x );
                d_v_y = ( v_field[I1D(i,j+1,k)] - v_field[I1D(i,j-1,k)] )/( 2.0*delta_y );
                d_w_z = ( w_field[I1D(i,j,k+1)] - w_field[I1D(i,j,k-1)] )/( 2.0*delta_z );
                /// Divergence of velocity
                div_uvw = d_u_x + d_v_y + d_w_z;
                /// Inviscid flux
		if( artificial_compressibility_method ) {
		    P_inv_flux[I1D(i,j,k)] = u_field[I1D(i,j,k)]*d_P_x + v_field[I1D(i,j,k)]*d_P_y + w_field[I1D(i,j,k)]*d_P_z + rho_field[I1D(i,j,k)]*pow( alpha_acm*sos_field[I1D(i,j,k)], 2.0 )*div_uvw;
                } else {
		    P_inv_flux[I1D(i,j,k)] = u_field[I1D(i,j,k)]*d_P_x + v_field[I1D(i,j,k)]*d_P_y + w_field[I1D(i,j,k)]*d_P_z + rho_field[I1D(i,j,k)]*pow( sos_field[I1D(i,j,k)], 2.0 )*div_uvw;
                }
                /// Viscous flux
		P_vis_flux[I1D(i,j,k)] = ( thermodynamics->calculateVolumeExpansivity( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] )/( rho_field[I1D(i,j,k)]*c_v_field[I1D(i,j,k)]*thermodynamics->calculateIsothermalCompressibility( P_field[I1D(i,j,k)], T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] ) ) )*work_vis_rhoe_flux[I1D(i,j,k)];
	    }
        }
    }

    /// Coefficients of explicit Runge-Kutta stages
    double rk_a = 0.0, rk_b = 0.0, rk_c = 0.0;
    runge_kutta_method->setStageCoefficients(rk_a,rk_b,rk_c,rk_time_stage);    
    
    /// Inner points: P
    double P_rhs_flux = 0.0;
    //#pragma acc kernels loop collapse(3) independent
    #pragma acc parallel loop collapse(3) private(P_rhs_flux) present(this, P_inv_flux.vector[0:_ls_], P_vis_flux.vector[0:_ls_], f_rhoE_field.vector[0:_ls_], P_field.vector[0:_ls_], P_0_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Sum right-hand-side (RHS) fluxes
                P_rhs_flux = ( -1.0 )*P_inv_flux[I1D(i,j,k)] + P_vis_flux[I1D(i,j,k)] + f_rhoE_field[I1D(i,j,k)]; 
                /// Runge-Kutta step
                P_field[I1D(i,j,k)] = rk_a*P_0_field[I1D(i,j,k)] + rk_b*P_field[I1D(i,j,k)] + rk_c*delta_t*P_rhs_flux;
	    }
        }
    }
    
    /// Update halo values	
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update host(P_field.vector[0:_ls_])
#endif
    P_field.update();
#if _GPU_AWARE_MPI_DEACTIVATED_ 
    #pragma acc update device(P_field.vector[0:_ls_])
#endif

};

void FlowSolverRHEA::updateHost() {

    if( updated_cpu == false ) {
    	#pragma acc update host(rho_field.vector[0:_ls_], rhou_field.vector[0:_ls_], rhov_field.vector[0:_ls_], rhow_field.vector[0:_ls_], rhoE_field.vector[0:_ls_], u_field.vector[0:_ls_], v_field.vector[0:_ls_], w_field.vector[0:_ls_], E_field.vector[0:_ls_], s_field.vector[0:_ls_], P_field.vector[0:_ls_], T_field.vector[0:_ls_], sos_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], avg_rho_field.vector[0:_ls_], avg_rhou_field.vector[0:_ls_], avg_rhov_field.vector[0:_ls_], avg_rhow_field.vector[0:_ls_], avg_rhoE_field.vector[0:_ls_], avg_rhoP_field.vector[0:_ls_], avg_rhoT_field.vector[0:_ls_], avg_u_field.vector[0:_ls_], avg_v_field.vector[0:_ls_], avg_w_field.vector[0:_ls_], avg_E_field.vector[0:_ls_], avg_s_field.vector[0:_ls_], avg_P_field.vector[0:_ls_], avg_T_field.vector[0:_ls_], avg_sos_field.vector[0:_ls_], avg_mu_field.vector[0:_ls_], avg_kappa_field.vector[0:_ls_], avg_c_v_field.vector[0:_ls_], avg_c_p_field.vector[0:_ls_], rmsf_rho_field.vector[0:_ls_], rmsf_rhou_field.vector[0:_ls_], rmsf_rhov_field.vector[0:_ls_], rmsf_rhow_field.vector[0:_ls_], rmsf_rhoE_field.vector[0:_ls_], rmsf_u_field.vector[0:_ls_], rmsf_v_field.vector[0:_ls_], rmsf_w_field.vector[0:_ls_], rmsf_E_field.vector[0:_ls_], rmsf_s_field.vector[0:_ls_], rmsf_P_field.vector[0:_ls_], rmsf_T_field.vector[0:_ls_], rmsf_sos_field.vector[0:_ls_], rmsf_mu_field.vector[0:_ls_], rmsf_kappa_field.vector[0:_ls_], rmsf_c_v_field.vector[0:_ls_], rmsf_c_p_field.vector[0:_ls_], favre_uffuff_field.vector[0:_ls_], favre_uffvff_field.vector[0:_ls_], favre_uffwff_field.vector[0:_ls_], favre_vffvff_field.vector[0:_ls_], favre_vffwff_field.vector[0:_ls_], favre_wffwff_field.vector[0:_ls_], favre_uffEff_field.vector[0:_ls_], favre_vffEff_field.vector[0:_ls_], favre_wffEff_field.vector[0:_ls_])
    	updated_cpu = true;
    }
};

void FlowSolverRHEA::outputCurrentStateData() {
	
    /// Write to file current solver state, time, time iteration and averaging time
    if( updated_cpu == false ) this->updateHost();
    writer_reader->setAttribute( "Time", current_time );
    writer_reader->setAttribute( "Iteration", current_time_iter );
    writer_reader->setAttribute( "AveragingTime", averaging_time );
    writer_reader->write( current_time_iter );

};

void FlowSolverRHEA::output2dSlicesCurrentStateData() {

    /// Iterate through 2d data output slices
    for(int dos = 0; dos < number_two_dimensional_data_output_slices; ++dos) {
        /// Write to file current 2d data output slice solver state, time, time iteration and averaging time (if criterion satisfied)
        if( current_time_iter%dos_output_frequency_iters[dos] == 0 ) {
	    if( updated_cpu == false ) this->updateHost();
            writer_reader->setAttribute( "Time", current_time );
            writer_reader->setAttribute( "Iteration", current_time_iter );
            writer_reader->setAttribute( "AveragingTime", averaging_time );
            char char_array_1[ dos_output_file_names[dos].length() + 1 ]; 
            strcpy( char_array_1, dos_output_file_names[dos].c_str() );
            char char_array_2[ dos_normal_directions[dos].length() + 1 ]; 
            strcpy( char_array_2, dos_normal_directions[dos].c_str() );
            writer_reader->write2dDataOutputSlice( x_field, y_field, z_field, mesh, topo, char_array_1, char_array_2, dos_x_positions[dos], dos_y_positions[dos], dos_z_positions[dos], dos_generate_xdmf_files[dos], current_time_iter );
        }
    }

};

void FlowSolverRHEA::outputTemporalPointProbesData() {

    /// Initialize MPI stuff
    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /// Iterate through temporal point probes
    for(int tpp = 0; tpp < number_temporal_point_probes; ++tpp) {
        /// Write temporal point probe data to file (if criterion satisfied)
        if( current_time_iter%tpp_output_frequency_iters[tpp] == 0 ) {
	    if( updated_cpu == false ) this->updateHost();
	    /// Owner rank writes to file
            if( temporal_point_probes[tpp].getGlobalOwnerRank() == my_rank ) {
                int i_index, j_index, k_index;
                /// Get local indices i, j, k
		i_index = temporal_point_probes[tpp].getLocalIndexI(); 
		j_index = temporal_point_probes[tpp].getLocalIndexJ(); 
		k_index = temporal_point_probes[tpp].getLocalIndexK();
                /// Generate header string
                string output_header_string; 
	        output_header_string  = "# t [s], avg_t [s], x [m], y [m], z [m], rho [kg/m3], u [m/s], v [m/s], w [m/s], E [J/kg], s [J/(kg·K)], P [Pa], T [K], sos [m/s], mu [Pa·s], kappa [W/(m·K)], c_v [J/(kg·K)], c_p [J/(kg·K)]";
	        output_header_string += ", avg_rho [kg/m3], avg_rhou [kg/(s·m2)], avg_rhov [kg/(s·m2)], avg_rhow [kg/(s·m2)], avg_rhoE [J/m3], avg_rhoP [kg2/(m4·s2)], avg_rhoT [(kg·K)/m3]";
	        output_header_string += ", avg_u [m/s], avg_v [m/s], avg_w [m/s], avg_E [J/kg], avg_s [J/(kg·K)], avg_P [Pa], avg_T [K], avg_sos [m/s], avg_mu [Pa·s], avg_kappa [W/(m·K)], avg_c_v [J/(kg·K)], avg_c_p [J/(kg·K)]";
	        output_header_string += ", rmsf_rho [kg/m3], rmsf_rhou [kg/(s·m2)], rmsf_rhov [kg/(s·m2)], rmsf_rhow [kg/(s·m2)], rmsf_rhoE [J/m3]";
	        output_header_string += ", rmsf_u [m/s], rmsf_v [m/s], rmsf_w [m/s], rmsf_E [J/kg], rmsf_S [J/(kg·K)], rmsf_P [Pa], rmsf_T [K], rmsf_sos [m/s], rmsf_mu [Pa·s], rmsf_kappa [W/(m·K)], rmsf_c_v [J/(kg·K)], rmsf_c_p [J/(kg·K)]";
	        output_header_string += ", favre_uffuff [m2/s2], favre_uffvff [m2/s2], favre_uffwff [m2/s2], favre_vffvff [m2/s2], favre_vffwff [m2/s2], favre_wffwff [m2/s2]";
	        output_header_string += ", favre_uffEff [(m·J)/(s·kg)], favre_vffEff [(m·J)/(s·kg)], favre_wffEff [(m·J)/(s·kg)]";
	        output_header_string += ", tag_IBM [-]";
                /// Generate data string
                ostringstream sstr; sstr.precision( fstream_precision ); sstr << fixed;
                sstr << current_time;
                sstr << "," << averaging_time;
                sstr << "," << x_field[I1D(i_index,j_index,k_index)];
                sstr << "," << y_field[I1D(i_index,j_index,k_index)];
                sstr << "," << z_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rho_field[I1D(i_index,j_index,k_index)];
                sstr << "," << u_field[I1D(i_index,j_index,k_index)];
                sstr << "," << v_field[I1D(i_index,j_index,k_index)];
                sstr << "," << w_field[I1D(i_index,j_index,k_index)];
                sstr << "," << E_field[I1D(i_index,j_index,k_index)];
                sstr << "," << s_field[I1D(i_index,j_index,k_index)];
                sstr << "," << P_field[I1D(i_index,j_index,k_index)];
                sstr << "," << T_field[I1D(i_index,j_index,k_index)];
                sstr << "," << sos_field[I1D(i_index,j_index,k_index)];
                sstr << "," << mu_field[I1D(i_index,j_index,k_index)];
                sstr << "," << kappa_field[I1D(i_index,j_index,k_index)];
                sstr << "," << c_v_field[I1D(i_index,j_index,k_index)];
                sstr << "," << c_p_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rho_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rhou_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rhov_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rhow_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rhoE_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rhoP_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_rhoT_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_u_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_v_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_w_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_E_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_s_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_P_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_T_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_sos_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_mu_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_kappa_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_c_v_field[I1D(i_index,j_index,k_index)];
                sstr << "," << avg_c_p_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_rho_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_rhou_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_rhov_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_rhow_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_rhoE_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_u_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_v_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_w_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_E_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_s_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_P_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_T_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_sos_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_mu_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_kappa_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_c_v_field[I1D(i_index,j_index,k_index)];
                sstr << "," << rmsf_c_p_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_uffuff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_uffvff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_uffwff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_vffvff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_vffwff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_wffwff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_uffEff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_vffEff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << favre_wffEff_field[I1D(i_index,j_index,k_index)];
                sstr << "," << tag_IBM_field[I1D(i_index,j_index,k_index)];
                //sstr << "," << u_IBM_field[I1D(i_index,j_index,k_index)];
                //sstr << "," << v_IBM_field[I1D(i_index,j_index,k_index)];
                //sstr << "," << w_IBM_field[I1D(i_index,j_index,k_index)];
                //sstr << "," << T_IBM_field[I1D(i_index,j_index,k_index)];
                string output_data_string = sstr.str();
                /// Write (header string) data string to file
                temporal_point_probes[tpp].writeDataStringToOutputFile(output_header_string, output_data_string);
	    }
	}
    }	    

};

void FlowSolverRHEA::updateTimeAveragedQuantities() {

    /// All (inner, boundary & halo) points: first- and second-order time statistics of flow quantities 
    //#pragma acc parallel loop collapse (3)
    #pragma acc parallel loop collapse (3) present( rho_field.vector[:_ls_], rhou_field.vector[:_ls_], rhov_field.vector[:_ls_], rhow_field.vector[:_ls_], rhoE_field.vector[:_ls_], u_field.vector[:_ls_], v_field.vector[:_ls_], w_field.vector[:_ls_], E_field.vector[:_ls_], s_field.vector[:_ls_], P_field.vector[:_ls_], T_field.vector[:_ls_], sos_field.vector[:_ls_], mu_field.vector[:_ls_], kappa_field.vector[:_ls_], c_v_field.vector[:_ls_], c_p_field.vector[:_ls_], avg_rho_field.vector[:_ls_], avg_rhou_field.vector[:_ls_], avg_rhov_field.vector[:_ls_], avg_rhow_field.vector[:_ls_], avg_rhoE_field.vector[:_ls_], avg_rhoP_field.vector[:_ls_], avg_rhoT_field.vector[:_ls_], avg_u_field.vector[:_ls_], avg_v_field.vector[:_ls_], avg_w_field.vector[:_ls_], avg_E_field.vector[:_ls_], avg_s_field.vector[:_ls_], avg_P_field.vector[:_ls_], avg_T_field.vector[:_ls_], avg_sos_field.vector[:_ls_], avg_mu_field.vector[:_ls_], avg_kappa_field.vector[:_ls_], avg_c_v_field.vector[:_ls_], avg_c_p_field.vector[:_ls_], rmsf_rho_field.vector[:_ls_], rmsf_rhou_field.vector[:_ls_], rmsf_rhov_field.vector[:_ls_], rmsf_rhow_field.vector[:_ls_], rmsf_rhoE_field.vector[:_ls_], rmsf_u_field.vector[:_ls_], rmsf_v_field.vector[:_ls_], rmsf_w_field.vector[:_ls_], rmsf_E_field.vector[:_ls_], rmsf_s_field.vector[:_ls_], rmsf_P_field.vector[:_ls_], rmsf_T_field.vector[:_ls_], rmsf_sos_field.vector[:_ls_], rmsf_mu_field.vector[:_ls_], rmsf_kappa_field.vector[:_ls_], rmsf_c_v_field.vector[:_ls_], rmsf_c_p_field.vector[:_ls_], favre_uffuff_field.vector[:_ls_], favre_uffvff_field.vector[:_ls_], favre_uffwff_field.vector[:_ls_], favre_vffvff_field.vector[:_ls_], favre_vffwff_field.vector[:_ls_], favre_wffwff_field.vector[:_ls_], favre_uffEff_field.vector[:_ls_], favre_vffEff_field.vector[:_ls_], favre_wffEff_field.vector[:_ls_])	
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                /// Time-averaged quantities
                avg_rho_field[I1D(i,j,k)]   = updateTimeMeanQuantity(rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_rhou_field[I1D(i,j,k)]  = updateTimeMeanQuantity(rhou_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_rhov_field[I1D(i,j,k)]  = updateTimeMeanQuantity(rhov_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_rhow_field[I1D(i,j,k)]  = updateTimeMeanQuantity(rhow_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_rhoE_field[I1D(i,j,k)]  = updateTimeMeanQuantity(rhoE_field[I1D(i,j,k)],avg_rhoE_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_rhoP_field[I1D(i,j,k)]  = updateTimeMeanQuantity(rho_field[I1D(i,j,k)]*P_field[I1D(i,j,k)],avg_rhoP_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_rhoT_field[I1D(i,j,k)]  = updateTimeMeanQuantity(rho_field[I1D(i,j,k)]*T_field[I1D(i,j,k)],avg_rhoT_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_u_field[I1D(i,j,k)]     = updateTimeMeanQuantity(u_field[I1D(i,j,k)],avg_u_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_v_field[I1D(i,j,k)]     = updateTimeMeanQuantity(v_field[I1D(i,j,k)],avg_v_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_w_field[I1D(i,j,k)]     = updateTimeMeanQuantity(w_field[I1D(i,j,k)],avg_w_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_E_field[I1D(i,j,k)]     = updateTimeMeanQuantity(E_field[I1D(i,j,k)],avg_E_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_s_field[I1D(i,j,k)]     = updateTimeMeanQuantity(s_field[I1D(i,j,k)],avg_s_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_P_field[I1D(i,j,k)]     = updateTimeMeanQuantity(P_field[I1D(i,j,k)],avg_P_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_T_field[I1D(i,j,k)]     = updateTimeMeanQuantity(T_field[I1D(i,j,k)],avg_T_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_sos_field[I1D(i,j,k)]   = updateTimeMeanQuantity(sos_field[I1D(i,j,k)],avg_sos_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_mu_field[I1D(i,j,k)]    = updateTimeMeanQuantity(mu_field[I1D(i,j,k)],avg_mu_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_kappa_field[I1D(i,j,k)] = updateTimeMeanQuantity(kappa_field[I1D(i,j,k)],avg_kappa_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_c_v_field[I1D(i,j,k)]   = updateTimeMeanQuantity(c_v_field[I1D(i,j,k)],avg_c_v_field[I1D(i,j,k)],delta_t,averaging_time);
                avg_c_p_field[I1D(i,j,k)]   = updateTimeMeanQuantity(c_p_field[I1D(i,j,k)],avg_c_p_field[I1D(i,j,k)],delta_t,averaging_time);

                /// Root-mean-square-fluctuation quantities
                rmsf_rho_field[I1D(i,j,k)]   = updateTimeRmsfQuantity(rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],rmsf_rho_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_rhou_field[I1D(i,j,k)]  = updateTimeRmsfQuantity(rhou_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],rmsf_rhou_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_rhov_field[I1D(i,j,k)]  = updateTimeRmsfQuantity(rhov_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],rmsf_rhov_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_rhow_field[I1D(i,j,k)]  = updateTimeRmsfQuantity(rhow_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],rmsf_rhow_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_rhoE_field[I1D(i,j,k)]  = updateTimeRmsfQuantity(rhoE_field[I1D(i,j,k)],avg_rhoE_field[I1D(i,j,k)],rmsf_rhoE_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_u_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(u_field[I1D(i,j,k)],avg_u_field[I1D(i,j,k)],rmsf_u_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_v_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(v_field[I1D(i,j,k)],avg_v_field[I1D(i,j,k)],rmsf_v_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_w_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(w_field[I1D(i,j,k)],avg_w_field[I1D(i,j,k)],rmsf_w_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_E_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(E_field[I1D(i,j,k)],avg_E_field[I1D(i,j,k)],rmsf_E_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_s_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(s_field[I1D(i,j,k)],avg_s_field[I1D(i,j,k)],rmsf_s_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_P_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(P_field[I1D(i,j,k)],avg_P_field[I1D(i,j,k)],rmsf_P_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_T_field[I1D(i,j,k)]     = updateTimeRmsfQuantity(T_field[I1D(i,j,k)],avg_T_field[I1D(i,j,k)],rmsf_T_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_sos_field[I1D(i,j,k)]   = updateTimeRmsfQuantity(sos_field[I1D(i,j,k)],avg_sos_field[I1D(i,j,k)],rmsf_sos_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_mu_field[I1D(i,j,k)]    = updateTimeRmsfQuantity(mu_field[I1D(i,j,k)],avg_mu_field[I1D(i,j,k)],rmsf_mu_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_kappa_field[I1D(i,j,k)] = updateTimeRmsfQuantity(kappa_field[I1D(i,j,k)],avg_kappa_field[I1D(i,j,k)],rmsf_kappa_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_c_v_field[I1D(i,j,k)]   = updateTimeRmsfQuantity(c_v_field[I1D(i,j,k)],avg_c_v_field[I1D(i,j,k)],rmsf_c_v_field[I1D(i,j,k)],delta_t,averaging_time);
                rmsf_c_p_field[I1D(i,j,k)]   = updateTimeRmsfQuantity(c_p_field[I1D(i,j,k)],avg_c_p_field[I1D(i,j,k)],rmsf_c_p_field[I1D(i,j,k)],delta_t,averaging_time);

                /// Favre-averaged quantities
		favre_uffuff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(u_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],u_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_uffuff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_uffvff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(u_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],v_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_uffvff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_uffwff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(u_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],w_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_uffwff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_vffvff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(v_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],v_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_vffvff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_vffwff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(v_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],w_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_vffwff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_wffwff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(w_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],w_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_wffwff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_uffEff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(u_field[I1D(i,j,k)],avg_rhou_field[I1D(i,j,k)],E_field[I1D(i,j,k)],avg_rhoE_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_uffEff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_vffEff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(v_field[I1D(i,j,k)],avg_rhov_field[I1D(i,j,k)],E_field[I1D(i,j,k)],avg_rhoE_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_vffEff_field[I1D(i,j,k)],delta_t,averaging_time); 
		favre_wffEff_field[I1D(i,j,k)] = updateTimeFavreAveragedQuantity(w_field[I1D(i,j,k)],avg_rhow_field[I1D(i,j,k)],E_field[I1D(i,j,k)],avg_rhoE_field[I1D(i,j,k)],rho_field[I1D(i,j,k)],avg_rho_field[I1D(i,j,k)],favre_wffEff_field[I1D(i,j,k)],delta_t,averaging_time); 
            }
        }
    }

    /// Update averaging time
    #pragma acc parallel
    averaging_time += delta_t;
    #pragma acc update host(averaging_time)   

    /// Update halo values
    //avg_rho_field.update();
    //avg_rhou_field.update();
    //avg_rhov_field.update();
    //avg_rhow_field.update();
    //avg_rhoE_field.update();
    //avg_rhoP_field.update();
    //avg_rhoT_field.update();
    //avg_u_field.update();
    //avg_v_field.update();
    //avg_w_field.update();
    //avg_E_field.update();
    //avg_s_field.update();
    //avg_P_field.update();
    //avg_T_field.update();
    //avg_sos_field.update();
    //avg_mu_field.update();
    //avg_kappa_field.update();
    //avg_c_v_field.update();
    //avg_c_p_field.update();
    //rmsf_rho_field.update();
    //rmsf_rhou_field.update();
    //rmsf_rhov_field.update();
    //rmsf_rhow_field.update();
    //rmsf_rhoE_field.update();
    //rmsf_u_field.update();
    //rmsf_v_field.update();
    //rmsf_w_field.update();
    //rmsf_E_field.update();
    //rmsf_s_field.update();
    //rmsf_P_field.update();
    //rmsf_T_field.update();
    //rmsf_sos_field.update();
    //rmsf_mu_field.update();
    //rmsf_kappa_field.update();
    //rmsf_c_v_field.update();
    //rmsf_c_p_field.update();
    //favre_uffuff_field.update();
    //favre_uffvff_field.update();
    //favre_uffwff_field.update();
    //favre_vffvff_field.update();
    //favre_vffwff_field.update();
    //favre_wffwff_field.update();

};

double FlowSolverRHEA::updateTimeMeanQuantity(const double &quantity, const double &mean_quantity, const double &delta_t, const double &averaging_time) {

    double updated_mean_quantity = ( mean_quantity*averaging_time + quantity*delta_t )/( averaging_time + delta_t ); 

    return( updated_mean_quantity );

};

double FlowSolverRHEA::updateTimeRmsfQuantity(const double &quantity, const double &mean_quantity, const double &rmsf_quantity, const double &delta_t, const double &averaging_time) {

    double updated_rmsf_quantity = sqrt( ( pow( rmsf_quantity, 2.0 )*averaging_time + pow( quantity - mean_quantity, 2.0 )*delta_t )/( averaging_time + delta_t ) ); 

    return( updated_rmsf_quantity );

};

//double FlowSolverRHEA::updateTimeReynoldsAveragedQuantity(const double &quantity_1, const double &mean_quantity_1, const double &quantity_2, const double &mean_quantity_2, const double &reynolds_averaged_quantity, const double &delta_t, const double &averaging_time) {
//
//    double fluctuating_quantity_1         = quantity_1 - mean_quantity_1;
//    double fluctuating_quantity_2         = quantity_2 - mean_quantity_2;
//    double product_fluctuating_quantities = fluctuating_quantity_1*fluctuating_quantity_2;
//
//    double updated_reynolds_averaged_quantity = ( reynolds_averaged_quantity*averaging_time + product_fluctuating_quantities*delta_t )/( averaging_time + delta_t ); 
//
//    return( updated_reynolds_averaged_quantity );
//
//};

double FlowSolverRHEA::updateTimeFavreAveragedQuantity(const double &quantity_1, const double &mean_rho_quantity_1, const double &quantity_2, const double &mean_rho_quantity_2, const double &rho, const double &mean_rho, const double &favre_averaged_quantity, const double &delta_t, const double &averaging_time) {

    //double old_mean_rho = ( ( averaging_time + delta_t )*mean_rho - rho*delta_t )/max( averaging_time, epsilon );
    double old_mean_rho = ( ( averaging_time + delta_t )*mean_rho - rho*delta_t )/max( averaging_time, 1.0e-10 );	// ... modified for OpenACC

    double favre_mean_quantity_1              = mean_rho_quantity_1/mean_rho;
    double favre_mean_quantity_2              = mean_rho_quantity_2/mean_rho;
    double fluctuating_quantity_1             = quantity_1 - favre_mean_quantity_1;
    double fluctuating_quantity_2             = quantity_2 - favre_mean_quantity_2;
    double product_rho_fluctuating_quantities = rho*fluctuating_quantity_1*fluctuating_quantity_2;

    double updated_favre_averaged_quantity = ( ( old_mean_rho*favre_averaged_quantity*averaging_time + product_rho_fluctuating_quantities*delta_t )/( averaging_time + delta_t ) )/mean_rho; 

    return( updated_favre_averaged_quantity );

};

void FlowSolverRHEA::temporalHookFunction() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

};

double FlowSolverRHEA::calculateVolumeAveragedPressure() {

    /// Inner points: P
    double local_sum_VP = 0.0;
    double local_sum_V  = 0.0;
    double volume       = 0.0;
    double delta_x, delta_y, delta_z; 
    //#pragma acc kernels loop collapse(3) independent reduction(+:local_sum_VP,local_sum_V)    
    #pragma acc parallel loop collapse(3) reduction(+:local_sum_VP,local_sum_V) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], P_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                /// Calculate volume
                volume = delta_x*delta_y*delta_z; 
                /// Sum V*P values
                local_sum_VP += volume*P_field[I1D(i,j,k)];
                /// Sum V values
                local_sum_V += volume;
	    }
        }
    }		    

    /// Communicate local values to obtain global & average values
    double global_sum_VP;
    MPI_Allreduce(&local_sum_VP, &global_sum_VP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_V;
    MPI_Allreduce(&local_sum_V, &global_sum_V, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_avg_P = global_sum_VP/global_sum_V;

    return( global_avg_P );

};

double FlowSolverRHEA::calculateAlphaArtificialCompressibilityMethod() {
    
    const double P_threshold = 1.0e-5*P_thermo;
#if 0	/// L1-norm
    /// Inner points: P
    double local_sum_num = 0.0;
    double local_sum_den = 0.0;
    double delta_x, delta_y, delta_z, volume = 0.0;    
    #pragma acc kernels loop collapse(3) independent reduction(+:local_sum_num,local_sum_den)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                volume  = delta_x*delta_y*delta_z;
                /// Update values
                local_sum_num += volume*abs( P_field[I1D(i,j,k)] );
                local_sum_den += volume*( max( abs( P_field[I1D(i,j,k)] - P_thermo ), P_threshold ) );
	    }
        }
    }		    

    /// Communicate local values to obtain global values
    double global_sum_num;
    MPI_Allreduce(&local_sum_num, &global_sum_num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_den;
    MPI_Allreduce(&local_sum_den, &global_sum_den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_alpha = sqrt( 1.0 + epsilon_acm*global_sum_num/global_sum_den );
#endif

#if 1	/// L2-norm
    /// Inner points: P
    double local_sum_num = 0.0;
    double local_sum_den = 0.0;
    double delta_x, delta_y, delta_z, volume = 0.0;    
    //#pragma acc kernels loop collapse(3) independent reduction(+:local_sum_num,local_sum_den)
    #pragma acc parallel loop collapse(3) reduction(+:local_sum_num,local_sum_den) present(this, x_field.vector[0:_ls_], y_field.vector[0:_ls_], z_field.vector[0:_ls_], P_field.vector[0:_ls_])
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                volume  = delta_x*delta_y*delta_z;
                /// Update values
                local_sum_num += pow( volume*P_field[I1D(i,j,k)], 2.0 );
                local_sum_den += pow( volume*( max( abs( P_field[I1D(i,j,k)] - P_thermo ), P_threshold ) ), 2.0 ); 
	    }
        }
    }		    

    /// Communicate local values to obtain global values
    double global_sum_num;
    MPI_Allreduce(&local_sum_num, &global_sum_num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_den;
    MPI_Allreduce(&local_sum_den, &global_sum_den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_alpha = sqrt( 1.0 + epsilon_acm*sqrt( global_sum_num )/sqrt( global_sum_den ) );
#endif

#if 0	/// infinity-norm
    /// Initialize to largest double value
    double local_alpha = numeric_limits<double>::max();

    /// Inner points: P
    double alpha_aux;
    #pragma acc kernels loop collapse(3) independent reduction(+:local_sum_num,local_sum_den)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Update value
                alpha_aux   = sqrt( 1.0 + ( P_field[I1D(i,j,k)]*epsilon_acm )/( max( abs( P_field[I1D(i,j,k)] - P_thermo ), P_threshold ) ) );
                local_alpha = min( local_alpha, alpha_aux );
	    }
        }
    }		    

    /// Communicate local value to obtain global value
    double global_alpha;
    MPI_Allreduce(&local_alpha, &global_alpha, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

    return( global_alpha );

};

void FlowSolverRHEA::calculateArtificiallyModifiedThermodynamics() {
            
    /// All (inner, halo, boundary) points: sos, c_v, c_p
    double c_v, c_p;
    #pragma acc parallel loop collapse(3) private(c_v, c_p) present(this, rho_field.vector[0:_ls_], T_field.vector[0:_ls_], c_v_field.vector[0:_ls_], c_p_field.vector[0:_ls_], sos_field.vector[0:_ls_])
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                sos_field[I1D(i,j,k)]  = ( 1.0/max( alpha_acm, epsilon ) )*thermodynamics->calculateSoundSpeed( P_thermo, T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                thermodynamics->calculateSpecificHeatCapacities( c_v, c_p, P_thermo, T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                c_v_field[I1D(i,j,k)]  = c_v;
                c_p_field[I1D(i,j,k)]  = c_p;
            }
        }
    }

    /// Update halo values
    //sos_field.update();
    //c_v_field.update();
    //c_p_field.update();

};

void FlowSolverRHEA::calculateArtificiallyModifiedTransportCoefficients() {
   
    /// All (inner, halo, boundary) points: mu and kappa
    #pragma acc parallel loop collapse(3) present(this, rho_field.vector[0:_ls_], T_field.vector[0:_ls_], mu_field.vector[0:_ls_], kappa_field.vector[0:_ls_])
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                mu_field[I1D(i,j,k)]    = transport_coefficients->calculateDynamicViscosity( P_thermo, T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
                kappa_field[I1D(i,j,k)] = transport_coefficients->calculateThermalConductivity( P_thermo, T_field[I1D(i,j,k)], rho_field[I1D(i,j,k)] );
            }
        }
    }

    /// Update halo values
    //mu_field.update();
    //kappa_field.update();

};

void FlowSolverRHEA::execute() {

    NVTX_PUSH("execute");
   
    /// Start wall-clock timer
    auto start_wall_clock_timer = chrono::steady_clock::now();

    /// Start timer: execute
    timers->start( "execute" );

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /// Set output (cout) precision
    cout.precision( cout_precision );

    /// Start RHEA simulation
    if( my_rank == 0 ) cout << "RHEA (v" << version_number << "): START SIMULATION" << endl;
   
    /// Initialize flow variables from restart file or by setting initial conditions
    if( use_restart ) {

        /// Initialize from restart file
        this->initializeFromRestart();

        if( artificial_compressibility_method ) {

            /// Calculate thermodynamic (bulk) pressure
            P_thermo = this->calculateVolumeAveragedPressure();
	    #pragma acc update device(P_thermo)

            /// Calculate alpha value of artificial compressibility method
            alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();
	    #pragma acc update device(alpha_acm) 
	    
	    /// Calculate artificially modified thermodynamics
            this->calculateArtificiallyModifiedThermodynamics();

	    /// Calculate artificially modified transport coefficients
            this->calculateArtificiallyModifiedTransportCoefficients();

	}

    } else {

        /// Set initial conditions
        this->setInitialConditions();

        /// Initialize thermodynamics
        this->initializeThermodynamics();

        if( artificial_compressibility_method ) {

            /// Calculate thermodynamic (bulk) pressure
            P_thermo = this->calculateVolumeAveragedPressure();
	    #pragma acc update device(P_thermo)

            /// Calculate alpha value of artificial compressibility method
            alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();
	    #pragma acc update device(alpha_acm) 

	    /// Calculate artificially modified thermodynamics
            this->calculateArtificiallyModifiedThermodynamics();	    

	    /// Calculate artificially modified transport coefficients
            this->calculateArtificiallyModifiedTransportCoefficients();

	} else {

            /// Calculate transport coefficients
            this->calculateTransportCoefficients();

	}

    }

    if( activate_immersed_boundary_method ) {

        /// Start timer: immersed_boundary_method
        NVTX_PUSH("immersed_boundary_method");
        timers->start( "immersed_boundary_method" );

        /// Tag immersed boundary method
        this->tagImmersedBoundaryMethod();

        /// Stop timer: immersed_boundary_method
        timers->stop( "immersed_boundary_method" );
        NVTX_POP(); // immersed_boundary_method

    }

    /// Calculate conserved variables from primitive variables
    this->primitiveToConservedVariables();
 
    /// Update previous state of conserved variables
    this->updatePreviousStateConservedVariables();    

    /// Start particles from restart / random distribution
    if( use_restart_particles ) {

        point_particles->read_from_file( restart_data_file_particles );
        this->updateLagrangianEulerianMeshIndexes0( my_rank );

    } else {

        point_particles->generate_prts_random( buffer_ratio_particles );
	this->updateLagrangianEulerianMeshIndexes0( my_rank );
        this->setInitialParticlesPositionsVelocities();
	this->updateLagrangianEulerianMeshIndexes0( my_rank );	

    }
    point_particles->copyToDeviceParticles();

    /// Start timer: time_iteration_loop
    NVTX_PUSH("time_iteration_loop");
    timers->start( "time_iteration_loop" );

    // Some copyin's to the GPU, need to figure out where to put them. UNDER DEVELOPMENT
    #pragma acc enter data copyin(x_field.vector[0:_ls_],y_field.vector[0:_ls_],z_field.vector[0:_ls_])

    /// Iterate flow solver RHEA in time
    for(int time_iter = current_time_iter; time_iter < final_time_iter; time_iter++) {

        /// Start timer: calculate_time_step
        NVTX_PUSH("calculate_time_step");
        timers->start( "calculate_time_step" );
	   
	/// Calculate time step
        this->calculateTimeStep();
	if( ( current_time + delta_t ) > final_time ) delta_t = final_time - current_time;
        #pragma acc update device(delta_t)

        /// Stop timer: calculate_time_step
        timers->stop( "calculate_time_step" );
        NVTX_POP(); // calculate_time_step

        /// Stop timer: execute
        timers->stop( "execute" );

        /// Start timer: output_solver_state
        NVTX_PUSH("output_solver_state");
        timers->start( "output_solver_state" );

        /// Print time iteration information (if criterion satisfied)
        if( ( current_time_iter%print_frequency_iter == 0 ) and ( my_rank == 0 ) ) {
	    auto diff_wall_clock_timer = chrono::steady_clock::now() - start_wall_clock_timer;
            cout << "Time iteration " << current_time_iter << ": " 
                 << "time = " << scientific << current_time << " [s], "
                 << "time-step = " << scientific << delta_t << " [s], "
                 << "wall-clock time = " << scientific << chrono::duration< double, std::ratio<3600> >( diff_wall_clock_timer ).count() << " [h]" << endl;
        }

	/// Output current state data to file (if criterion satisfied)
	updated_cpu = false;	/// Reset CPU data update tracker
        if( current_time_iter%output_frequency_iter == 0 ) this->outputCurrentStateData();
        
	/// Output current 2d slices state data to file (if criterion satisfied)
        this->output2dSlicesCurrentStateData();
        
	/// Output temporal point probes data to files (if criterion satisfied)
	this->outputTemporalPointProbesData();

	/// Output current state particles data to file (if criterion satisfied)
        if( current_time_iter%output_frequency_iter_particles == 0 ) point_particles->write_to_file( current_time, current_time_iter );
	
        /// Stop timer: output_solver_state
        timers->stop( "output_solver_state" );
        NVTX_POP(); // output_solver_state

        /// Start timer: execute
        timers->start( "execute" );

        /// Start timer: time_advance_point_particles
        NVTX_PUSH("time_advance_point_particles");
        timers->start( "time_advance_point_particles" );

        /// Update position & velocity of point particles
        this->timeAdvancePointParticles( my_rank );
            
        /// Stop timer: time_advance_point_particles
        timers->stop( "time_advance_point_particles" );
        NVTX_POP(); // time_advance_point_particles

        /// Start timer: rk_iteration_loop
	NVTX_PUSH("rk_iteration_loop");
	timers->start( "rk_iteration_loop" );

        /// Runge-Kutta time-integration steps
        for(rk_time_stage = 1; rk_time_stage <= rk_number_stages; rk_time_stage++) {

            /// Start timer: calculate_thermophysical_properties
            NVTX_PUSH("calculate_thermophysical_properties");
            timers->start( "calculate_thermophysical_properties" );
	
	    if( artificial_compressibility_method ) {

	        /// Calculate artificially modified transport coefficients
                this->calculateArtificiallyModifiedTransportCoefficients();

	    } else {

                /// Calculate transport coefficients
                this->calculateTransportCoefficients();

	    }

	    /// Stop timer: calculate_thermophysical_properties
            timers->stop( "calculate_thermophysical_properties" );
            NVTX_POP(); // calculate_thermophysical_properties

            /// Start timer: calculate_inviscid_fluxes
            NVTX_PUSH("calculate_inviscid_fluxes");
            timers->start( "calculate_inviscid_fluxes" );

            /// Calculate inviscid fluxes
            this->calculateInviscidFluxes();
	    
	    /// Stop timer: calculate_inviscid_fluxes
            timers->stop( "calculate_inviscid_fluxes" );
            NVTX_POP(); // calculate_inviscid_fluxes

            /// Start timer: calculate_viscous_fluxes
            NVTX_PUSH("calculate_viscous_fluxes");
            timers->start( "calculate_viscous_fluxes" );

            /// Calculate viscous fluxes
            this->calculateViscousFluxes();

            /// Stop timer: calculate_viscous_fluxes
            timers->stop( "calculate_viscous_fluxes" );
            NVTX_POP(); // calculate_viscous_fluxes

            /// Start timer: calculate_source_terms
            NVTX_PUSH("calculate_source_terms");
            timers->start( "calculate_source_terms" );

            /// Calculate source terms
            this->calculateSourceTerms();

            /// Stop timer: calculate_source_terms
            timers->stop( "calculate_source_terms" );
            NVTX_POP(); // calculate_source_terms

            /// Start timer: immersed_boundary_method
            NVTX_PUSH("immersed_boundary_method");
            timers->start( "immersed_boundary_method" );

            /// Impose immersed boundary method
            if( activate_immersed_boundary_method ) this->imposeImmersedBoundaryMethod();

            /// Stop timer: immersed_boundary_method
            timers->stop( "immersed_boundary_method" );
            NVTX_POP(); // immersed_boundary_method

            /// Start timer: time_advance_conserved_variables
            NVTX_PUSH("time_advance_conserved_variables");
            timers->start( "time_advance_conserved_variables" );

            /// Advance conserved variables in time
            this->timeAdvanceConservedVariables();
            
	    /// Stop timer: time_advance_conserved_variables
            timers->stop( "time_advance_conserved_variables" );
            NVTX_POP(); // time_advance_conserved_variables

            /// Start timer: conserved_to_primitive_variables
            NVTX_PUSH("conserved_to_primitive_variables");
            timers->start( "conserved_to_primitive_variables" );

            /// Calculate primitive variables from conserved variables
            this->conservedToPrimitiveVariables();

            /// Stop timer: conserved_to_primitive_variables
            timers->stop( "conserved_to_primitive_variables" );
            NVTX_POP(); // conserved_to_primitive_variables

            /// Start timer: calculate_thermodynamics_from_primitive_variables
            NVTX_PUSH("calculate_thermodynamics_from_primitive_variables");
            timers->start( "calculate_thermodynamics_from_primitive_variables" );

            /// Calculate thermodynamics from primitive variables
            this->calculateThermodynamicsFromPrimitiveVariables();

            if( artificial_compressibility_method ) {

                /// Calculate thermodynamic (bulk) pressure
                P_thermo = this->calculateVolumeAveragedPressure();
	        #pragma acc update device(P_thermo)

                /// Calculate alpha value of artificial compressibility method
                alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();
	        #pragma acc update device(alpha_acm) 

                /// Calculate artificially modified thermodynamics
                this->calculateArtificiallyModifiedThermodynamics();	    

	    }

            /// Stop timer: calculate_thermodynamics_from_primitive_variables
            timers->stop( "calculate_thermodynamics_from_primitive_variables" );
            NVTX_POP(); // calculate_thermodynamics_from_primitive_variables

            /// Start timer: update_boundaries
            NVTX_PUSH("update_boundaries");
            timers->start( "update_boundaries" );

            /// Update boundary values
            this->updateBoundaries();
            
            /// Stop timer: update_boundaries
            timers->stop( "update_boundaries" );
            NVTX_POP(); // update_boundaries

        }

        /// Stop timer: rk_iteration_loop
        timers->stop( "rk_iteration_loop" );
        NVTX_POP(); // rk_iteration_loop

        /// Start timer: temporal_hook_function
        NVTX_PUSH("temporal_hook_function");
        timers->start( "temporal_hook_function" );

        /// Temporal hook function
        this->temporalHookFunction();

        /// Stop timer: temporal_hook_function
        timers->stop( "temporal_hook_function" );
        NVTX_POP(); // temporal_hook_function

        /// Start timer: update_time_averaged_quantities
        NVTX_PUSH("update_time_averaged_quantities");
        timers->start( "update_time_averaged_quantities" );
	
        /// Update time-averaged quantities
        if( time_averaging_active ) this->updateTimeAveragedQuantities();
	
        /// Stop timer: update_time_averaged_quantities
        timers->stop( "update_time_averaged_quantities" );
        NVTX_POP(); // update_time_averaged_quantities

        /// Start timer: update_previous_state_conserved_variables
        NVTX_PUSH("update_previous_state_conserved_variables");
        timers->start( "update_previous_state_conserved_variables" );
        
        /// Update previous state of conserved variables
        this->updatePreviousStateConservedVariables();
     
        /// Update time and time iteration
        current_time += delta_t;
        current_time_iter += 1;

        /// Stop timer: update_previous_state_conserved_variables
        timers->stop( "update_previous_state_conserved_variables" );
        NVTX_POP(); // update_previous_state_conserved_variables

        /// Check if simulation is completed: current_time > final_time
        if( current_time >= final_time ) break;

    }
    
    // ... deleting methods/fields from GPU
    #pragma acc exit data delete(x_field.vector[0:_ls_],y_field.vector[0:_ls_],z_field.vector[0:_ls_])

    /// Stop timer: time_iteration_loop
    timers->stop( "time_iteration_loop" );
    NVTX_POP(); // time_iteration_loop

    /// Print timers information
    if( print_timers ) timers->printTimers( timers_information_file );

    /// Print time advancement information
    if( my_rank == 0 ) {
        cout << "Time advancement completed -> " 
             << "iteration = " << current_time_iter << ", "
             << "time = " << scientific << current_time << " [s]" << endl;
    }

    /// Output current state data to file
    updated_cpu = false;	/// Reset CPU data update tracker
    this->outputCurrentStateData();
    
    /// Output current 2d slices state data to file (if criterion satisfied)
    this->output2dSlicesCurrentStateData();
        
    /// Output temporal point probes data to files (if criterion satisfied)
    this->outputTemporalPointProbesData();

    /// Output current state particles data to file
    point_particles->write_to_file( current_time, current_time_iter );

    /// End RHEA simulation
    if( my_rank == 0 ) cout << "RHEA (v" << version_number << "): END SIMULATION" << endl;
    
    /// Stop timer: execute
    timers->stop( "execute" );

    /// Start wall-clock timer
    //auto end_wall_clock_timer = chrono::steady_clock::now();

    NVTX_POP(); // execute

};



////////// BaseRiemannSolver CLASS //////////

BaseRiemannSolver::BaseRiemannSolver() {};
        
BaseRiemannSolver::~BaseRiemannSolver() {};
	
void BaseRiemannSolver::calculateWavesSpeed(double &S_L, double &S_R, const double &rho_L, const double &rho_R, const double &u_L, const double &u_R, const double &P_L, const double &P_R, const double &a_L, const double &a_R) {

#if _PRESSURE_BASED_WAVE_SPEED_ESTIMATES_
    /// Pressure-based wave speed estimates: ... recommended by E. F. Toro, but not sure if applicable to non-ideal gas themodynamics
    /// E. F. Toro, M. Spruce, W. Speares.
    /// Restoration of the contact surface in the HLL-Riemann solver.
    /// Shock Waves, 4, 25-34, 1994.

    double P_bar   = 0.5*( P_L + P_R );
    double rho_bar = 0.5*( rho_L + rho_R );
    double gamma   = thermodynamics->calculateHeatCapacitiesRatio( P_bar, rho_bar );
    double a_bar   = 0.5*( a_L + a_R );
    double P_pvrs  = 0.5*( P_L + P_R ) - 0.5*( u_R - u_L )*rho_bar*a_bar;
    double P_star  = max( 0.0, P_pvrs );
    double q_L     = 1.0;
    if(P_star > P_L) q_L = sqrt( 1.0 + ( ( gamma + 1.0 )/( 2.0*gamma ) )*( ( P_star/P_L ) - 1.0 ) );
    double q_R     = 1.0;
    if(P_star > P_R) q_R = sqrt( 1.0 + ( ( gamma + 1.0 )/( 2.0*gamma ) )*( ( P_star/P_R ) - 1.0 ) );
    S_L = u_L - a_L*q_L;
    S_R = u_R + a_R*q_R;
#else
    /// Direct wave speed estimates:
    /// B. Einfeldt.
    /// On Godunov-type methods for gas dynamics.
    /// SIAM Journal on Numerical Analysis, 25, 294-318, 1988.

    double hat_u = ( u_L*sqrt( rho_L ) + u_R*sqrt( rho_R ) )/( sqrt( rho_L ) + sqrt( rho_R ) );
    double hat_a = sqrt( ( ( a_L*a_L*sqrt( rho_L ) + a_R*a_R*sqrt( rho_R ) )/( sqrt( rho_L ) + sqrt( rho_R ) ) ) + 0.5*( ( sqrt( rho_L )*sqrt( rho_R ) )/( ( sqrt( rho_L ) + sqrt( rho_R ) )*( sqrt( rho_L ) + sqrt( rho_R ) ) ) )*( u_R - u_L )*( u_R - u_L ) );

    S_L = min( u_L - a_L, hat_u - hat_a );
    S_R = max( u_R + a_R, hat_u + hat_a );
#endif

};


////////// DivergenceFluxApproximateRiemannSolver CLASS //////////

DivergenceFluxApproximateRiemannSolver::DivergenceFluxApproximateRiemannSolver() : BaseRiemannSolver() {};

DivergenceFluxApproximateRiemannSolver::~DivergenceFluxApproximateRiemannSolver() {};

double DivergenceFluxApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Divergence scheme obtained from a central differencing of the first derivative of the flux term:

    double F_L = rho_L*u_L;
    double F_R = rho_R*u_R;
    if( var_type == 0 ) {
        F_L *= 1.0;
        F_R *= 1.0;
    } else if ( var_type == 1 ) {
        F_L *= u_L; F_L += P_L;
        F_R *= u_R; F_R += P_R;
    } else if ( var_type == 2 ) {
        F_L *= v_L;
        F_R *= v_R;
    } else if ( var_type == 3 ) {
        F_L *= w_L;
        F_R *= w_R;
    } else if ( var_type == 4 ) {
        F_L *= E_L; F_L += u_L*P_L;
        F_R *= E_R; F_R += u_R*P_R;
    }
    double F = 0.5*( F_L + F_R );

    return( F );

};


////////// MurmanRoeFluxApproximateRiemannSolver CLASS //////////

MurmanRoeFluxApproximateRiemannSolver::MurmanRoeFluxApproximateRiemannSolver() : BaseRiemannSolver() {};

MurmanRoeFluxApproximateRiemannSolver::~MurmanRoeFluxApproximateRiemannSolver() {};

double MurmanRoeFluxApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Murman-Roe Riemman solver:
    /// P. L. Roe.
    /// Approximate Riemann solvers, parameter vectors and difference schemes.
    /// Journal of Computational Physics, 43, 357-372, 1981.

    double F_L = rho_L*u_L;
    double F_R = rho_R*u_R;
    double U_L = rho_L;
    double U_R = rho_R;
    if( var_type == 0 ) {
        F_L *= 1.0;
        F_R *= 1.0;
        U_L *= 1.0;
        U_R *= 1.0;
    } else if ( var_type == 1 ) {
        F_L *= u_L; F_L += P_L;
        F_R *= u_R; F_R += P_R;
        U_L *= u_L;
        U_R *= u_R;
    } else if ( var_type == 2 ) {
        F_L *= v_L;
        F_R *= v_R;
        U_L *= v_L;
        U_R *= v_R;
    } else if ( var_type == 3 ) {
        F_L *= w_L;
        F_R *= w_R;
        U_L *= w_L;
        U_R *= w_R;
    } else if ( var_type == 4 ) {
        F_L *= E_L; F_L += u_L*P_L;
        F_R *= E_R; F_R += u_R*P_R;
        U_L *= E_L;
        U_R *= E_R;
    }

    /// Wave speed
    //double S = abs( ( F_L - F_R )/( U_L - U_R + epsilon ) );
    double S = abs( ( F_L - F_R )/( U_L - U_R + 1.0e-10 ) );	// ... modified for OpenACC

    /// Conservative + dissipative flux form
    double F = 0.5*( F_L + F_R ) - 0.5*S*( U_R - U_L );

    return( F );

};


////////// KgpFluxApproximateRiemannSolver CLASS //////////

KgpFluxApproximateRiemannSolver::KgpFluxApproximateRiemannSolver() : BaseRiemannSolver() {};

KgpFluxApproximateRiemannSolver::~KgpFluxApproximateRiemannSolver() {};

double KgpFluxApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Kennedy, Gruber & Pirozzoli (KGP) scheme:
    /// G. Coppola, F. Capuano , S. Pirozzoli, L. de Luca.
    /// Numerically stable formulations of convective terms for turbulent compressible flows.
    /// Journal of Computational Physics, 382, 86-104, 2019.

    double F = ( 1.0/8.0 )*( rho_L + rho_R )*( u_L + u_R );
    if( var_type == 0 ) {
        F *= 1.0 + 1.0;
    } else if ( var_type == 1 ) {
        F *= u_L + u_R; F += ( 1.0/2.0 )*( P_L + P_R );
        //F *= u_L + u_R; F += ( 1.0/4.0 )*( rho_L + rho_R )*( P_L/rho_L + P_R/rho_R );
    } else if ( var_type == 2 ) {
        F *= v_L + v_R;
    } else if ( var_type == 3 ) {
        F *= w_L + w_R;
    } else if ( var_type == 4 ) {
        F *= E_L + P_L/rho_L + E_R + P_R/rho_R;
        //F *= E_L + E_R; F += ( 1.0/4.0 )*( u_L + u_R )*( P_L + P_R );
    }

    return( F );

};


////////// ShimaFluxApproximateRiemannSolver CLASS //////////

ShimaFluxApproximateRiemannSolver::ShimaFluxApproximateRiemannSolver() : BaseRiemannSolver() {};

ShimaFluxApproximateRiemannSolver::~ShimaFluxApproximateRiemannSolver() {};

double ShimaFluxApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Shima, Kuya, Tamaki & Kawai (SHIMA) scheme:
    /// N. Shima, Y. Kuya, Y. Tamaki, S. Kawai.
    /// Preventing spurious pressure oscillations in split convective form discretization for compressible flows
    /// Journal of Computational Physics, 427, 110060, 2021.
    
    double F = ( 1.0/8.0 )*( rho_L + rho_R )*( u_L + u_R );
    if( var_type == 0 ) {
        F *= 1.0 + 1.0;
    } else if ( var_type == 1 ) {
        F *= u_L + u_R; F += ( 1.0/2.0 )*( P_L + P_R );
    } else if ( var_type == 2 ) {
        F *= v_L + v_R;
    } else if ( var_type == 3 ) {
        F *= w_L + w_R;
    } else if ( var_type == 4 ) {
        double ke_L = ( 1.0/2.0 )*( u_L*u_L + v_L*v_L + w_L*w_L );
        double e_L  = E_L - ke_L;
        double ke_R = ( 1.0/2.0 )*( u_R*u_R + v_R*v_R + w_R*w_R );
        double e_R  = E_R - ke_R;
        F *= ke_L + ke_R;
        F += ( 1.0/4.0 )*( rho_L*e_L + rho_R*e_R )*( u_L + u_R );
        F += ( 1.0/2.0 )*( u_L*P_R + u_R*P_L );
    }

    return( F );

};


////////// HllApproximateRiemannSolver CLASS //////////

HllApproximateRiemannSolver::HllApproximateRiemannSolver() : BaseRiemannSolver() {};

HllApproximateRiemannSolver::~HllApproximateRiemannSolver() {};

double HllApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Harten-Lax-van Leer (HLL) Riemman solver:
    /// A. Harten, P. D. Lax, B. van Leer.
    /// On upstream differencing and Godunov-type schemes for hyperbolic conservation laws.
    /// SIAM Review, 25, 35-61, 1983.

    double F_L = rho_L*u_L;
    double F_R = rho_R*u_R;
    double U_L = rho_L;
    double U_R = rho_R;
    if( var_type == 0 ) {
        F_L *= 1.0;
        F_R *= 1.0;
        U_L *= 1.0;
        U_R *= 1.0;
    } else if ( var_type == 1 ) {
        F_L *= u_L; F_L += P_L;
        F_R *= u_R; F_R += P_R;
        U_L *= u_L;
        U_R *= u_R;
    } else if ( var_type == 2 ) {
        F_L *= v_L;
        F_R *= v_R;
        U_L *= v_L;
        U_R *= v_R;
    } else if ( var_type == 3 ) {
        F_L *= w_L;
        F_R *= w_R;
        U_L *= w_L;
        U_R *= w_R;
    } else if ( var_type == 4 ) {
        F_L *= E_L; F_L += u_L*P_L;
        F_R *= E_R; F_R += u_R*P_R;
        U_L *= E_L;
        U_R *= E_R;
    }

    double S_L, S_R;
    this->calculateWavesSpeed( S_L, S_R, rho_L, rho_R, u_L, u_R, P_L, P_R, a_L, a_R );

    double F = 0.0;
    if( 0.0 <= S_L ) {
        F = F_L;
    } else if( 0.0 >= S_R ) {
        F = F_R;
    } else {
        F = ( S_R*F_L - S_L*F_R + S_L*S_R*( U_R - U_L ) )/( S_R - S_L );
    }

    return( F );

};


////////// HllcApproximateRiemannSolver CLASS //////////

HllcApproximateRiemannSolver::HllcApproximateRiemannSolver() : BaseRiemannSolver() {};

HllcApproximateRiemannSolver::~HllcApproximateRiemannSolver() {};

double HllcApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Harten-Lax-van Leer-Contact (HLLC) Riemman solver:
    /// E. F. Toro, M. Spruce, W. Speares.
    /// Restoration of the contact surface in the HLL-Riemann solver.
    /// Shock Waves, 4, 25-34, 1994.

    double F_L = rho_L*u_L;
    double F_R = rho_R*u_R;
    double U_L = rho_L;
    double U_R = rho_R;
    if( var_type == 0 ) {
        F_L *= 1.0;
        F_R *= 1.0;
        U_L *= 1.0;
        U_R *= 1.0;
    } else if ( var_type == 1 ) {
        F_L *= u_L; F_L += P_L;
        F_R *= u_R; F_R += P_R;
        U_L *= u_L;
        U_R *= u_R;
    } else if ( var_type == 2 ) {
        F_L *= v_L;
        F_R *= v_R;
        U_L *= v_L;
        U_R *= v_R;
    } else if ( var_type == 3 ) {
        F_L *= w_L;
        F_R *= w_R;
        U_L *= w_L;
        U_R *= w_R;
    } else if ( var_type == 4 ) {
        F_L *= E_L; F_L += u_L*P_L;
        F_R *= E_R; F_R += u_R*P_R;
        U_L *= E_L;
        U_R *= E_R;
    }

    double S_L, S_R;
    this->calculateWavesSpeed( S_L, S_R, rho_L, rho_R, u_L, u_R, P_L, P_R, a_L, a_R );

    double S_star   = ( P_R - P_L + rho_L*u_L*( S_L - u_L ) - rho_R*u_R*( S_R - u_R ) )/( rho_L*( S_L - u_L ) - rho_R*( S_R - u_R ) );
    double U_star_L = rho_L*( ( S_L - u_L )/( S_L - S_star ) );
    double U_star_R = rho_R*( ( S_R - u_R )/( S_R - S_star ) );
    if( var_type == 0 ) {
        U_star_L *= 1.0;
        U_star_R *= 1.0;       
    } else if( var_type == 1 ) {
        U_star_L *= S_star;
        U_star_R *= S_star;
    } else if( var_type == 2 ) {
        U_star_L *= v_L;
        U_star_R *= v_R;
    } else if( var_type == 3 ) {
        U_star_L *= w_L;
        U_star_R *= w_R;
    } else if( var_type == 4 ) {
        U_star_L *= ( E_L + ( S_star - u_L )*( S_star + P_L/( rho_L*( S_L - u_L ) ) ) );
        U_star_R *= ( E_R + ( S_star - u_R )*( S_star + P_R/( rho_R*( S_R - u_R ) ) ) );
    }

    double F = 0.0;
    if( 0.0 <= S_L ) {
        F = F_L;
    } else if( ( S_L <= 0.0 ) && ( 0.0 <= S_star ) ) {
        F = F_L + S_L*( U_star_L - U_L );
    } else if( ( S_star <= 0.0 ) && ( 0.0 <= S_R ) ) {
        F = F_R + S_R*( U_star_R - U_R );
    } else if( 0.0 >= S_R ) {
        F = F_R;
    }

    return( F );

};


////////// HllcPlusApproximateRiemannSolver CLASS //////////

HllcPlusApproximateRiemannSolver::HllcPlusApproximateRiemannSolver() : BaseRiemannSolver() {};

HllcPlusApproximateRiemannSolver::~HllcPlusApproximateRiemannSolver() {};

double HllcPlusApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// HLLC-type Riemann solver for all-speed flows:
    /// S. Chen, B. Lin, Y. Li, C. Yan.
    /// HLLC+: low-Mach shock-stable HLLC-type Riemann solver for all-speed flows.
    /// SIAM Journal of Scientific Computing, 4, B921-B950, 2020.

    double F_L = rho_L*u_L;
    double F_R = rho_R*u_R;
    double U_L = rho_L;
    double U_R = rho_R;
    if( var_type == 0 ) {
        F_L *= 1.0;
        F_R *= 1.0;
        U_L *= 1.0;
        U_R *= 1.0;
    } else if ( var_type == 1 ) {
        F_L *= u_L; F_L += P_L;
        F_R *= u_R; F_R += P_R;
        U_L *= u_L;
        U_R *= u_R;
    } else if ( var_type == 2 ) {
        F_L *= v_L;
        F_R *= v_R;
        U_L *= v_L;
        U_R *= v_R;
    } else if ( var_type == 3 ) {
        F_L *= w_L;
        F_R *= w_R;
        U_L *= w_L;
        U_R *= w_R;
    } else if ( var_type == 4 ) {
        F_L *= E_L; F_L += u_L*P_L;
        F_R *= E_R; F_R += u_R*P_R;
        U_L *= E_L;
        U_R *= E_R;
    }

    double S_L, S_R;
    this->calculateWavesSpeed( S_L, S_R, rho_L, rho_R, u_L, u_R, P_L, P_R, a_L, a_R );

    double phi_L = rho_L*( S_L - u_L );
    double phi_R = rho_R*( S_R - u_R );
    double S_star   = ( P_R - P_L + phi_L*u_L - phi_R*u_R )/( phi_L - phi_R );
    double U_star_L = rho_L*( ( S_L - u_L )/( S_L - S_star ) );
    double U_star_R = rho_R*( ( S_R - u_R )/( S_R - S_star ) );
    double M        = min( 1.0, max( ( 1.0/a_L )*sqrt( u_L*u_L + v_L*v_L + w_L*w_L ), ( 1.0/a_R )*sqrt( u_R*u_R + v_R*v_R + w_R*w_R ) ) );
    double f_M      = M*sqrt( 4.0 + pow( 1.0 - M*M, 2.0 ) )/( 1.0 + M*M );
    double h        = min( P_L/P_R, P_R/P_L );
    double g        = 1.0 - pow( h, M );
    double A_p_L    = ( phi_L*phi_R )/( phi_R - phi_L );
    double A_p_R    = ( phi_L*phi_R )/( phi_R - phi_L );
    if( var_type == 0 ) {
        U_star_L *= 1.0;
        U_star_R *= 1.0;       
        A_p_L    *= 0.0;
        A_p_R    *= 0.0;
    } else if( var_type == 1 ) {
        U_star_L *= S_star;
        U_star_R *= S_star;
        A_p_L    *= ( f_M - 1.0 )*( u_R - u_L );
        A_p_R    *= ( f_M - 1.0 )*( u_R - u_L );
    } else if( var_type == 2 ) {
        U_star_L *= v_L;
        U_star_R *= v_R;
        A_p_L    *= ( S_L/( S_L - S_star ) )*g*( v_R - v_L );
        A_p_R    *= ( S_R/( S_R - S_star ) )*g*( v_R - v_L );
    } else if( var_type == 3 ) {
        U_star_L *= w_L;
        U_star_R *= w_R;
        A_p_L    *= ( S_L/( S_L - S_star ) )*g*( w_R - w_L );
        A_p_R    *= ( S_R/( S_R - S_star ) )*g*( w_R - w_L );
    } else if( var_type == 4 ) {
        U_star_L *= ( E_L + ( S_star - u_L )*( S_star + P_L/( rho_L*( S_L - u_L ) ) ) );
        U_star_R *= ( E_R + ( S_star - u_R )*( S_star + P_R/( rho_R*( S_R - u_R ) ) ) );
        A_p_L    *= ( f_M - 1.0 )*( u_R - u_L )*S_star;
        A_p_R    *= ( f_M - 1.0 )*( u_R - u_L )*S_star;
    }
    double F_star_L = F_L + S_L*( U_star_L - U_L ) + A_p_L;
    double F_star_R = F_R + S_R*( U_star_R - U_R ) + A_p_R;

    double F = 0.0;
    if( 0.0 <= S_L ) {
        F = F_L;
    } else if( ( S_L <= 0.0 ) && ( 0.0 <= S_star ) ) {
        F = F_star_L;
    } else if( ( S_star <= 0.0 ) && ( 0.0 <= S_R ) ) {
        F = F_star_R;
    } else if( 0.0 >= S_R ) {
        F = F_R;
    }

    return( F );

};


////////// EckepFluxApproximateRiemannSolver CLASS //////////

EckepFluxApproximateRiemannSolver::EckepFluxApproximateRiemannSolver() : BaseRiemannSolver() {};

EckepFluxApproximateRiemannSolver::~EckepFluxApproximateRiemannSolver() {};

double EckepFluxApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Entropy conservative and kinetic energy preserving (ECKEP) scheme:
    /// K. Bahuguna, R. Kolluru, S.V. R. Rao
    /// Structure-preserving schemes conserving entropy and kinetic energy.
    /// arXiv:2505.13374, 2025.

    double F = ( 1.0/8.0 )*( rho_L + rho_R )*( u_L + u_R );
    if( var_type == 0 ) {
        F *= 1.0 + 1.0;
    } else if ( var_type == 1 ) {
        F *= u_L + u_R; F += ( 1.0/2.0 )*( P_L + P_R );
    } else if ( var_type == 2 ) {
        F *= v_L + v_R;
    } else if ( var_type == 3 ) {
        F *= w_L + w_R;
    } else if ( var_type == 4 ) {
        double bar_F_1  = ( 1.0/4.0 )*( rho_L + rho_R )*( u_L + u_R );
        double bar_F_2u = ( 1.0/2.0 )*( bar_F_1*( u_L + u_R ) + ( P_L + P_R ) );
        double bar_F_2v = ( 1.0/2.0 )*bar_F_1*( v_L + v_R );
        double bar_F_2w = ( 1.0/2.0 )*bar_F_1*( w_L + w_R );
        double bar_F_3  = ( 1.0/2.0 )*bar_F_1*( E_L + P_L/rho_L + E_R + P_R/rho_R );
        double ds_drho_L  = ( u_L*u_L + v_L*v_L + w_L*w_L - E_L - P_L/rho_L )/( rho_L*T_L );
        double ds_drho_R  = ( u_R*u_R + v_R*v_R + w_R*w_R - E_R - P_R/rho_R )/( rho_R*T_R );
        double ds_drhou_L = ( -1.0 )*u_L/( rho_L*T_L );
        double ds_drhou_R = ( -1.0 )*u_R/( rho_R*T_R );
        double ds_drhov_L = ( -1.0 )*v_L/( rho_L*T_L );
        double ds_drhov_R = ( -1.0 )*v_R/( rho_R*T_R );
        double ds_drhow_L = ( -1.0 )*w_L/( rho_L*T_L );
        double ds_drhow_R = ( -1.0 )*w_R/( rho_R*T_R );
        double ds_drhoE_L = 1.0/( rho_L*T_L );
        double ds_drhoE_R = 1.0/( rho_R*T_R );	    
        double V_1_L  = ( -1.0 )*( s_L + rho_L*ds_drho_L );
        double V_1_R  = ( -1.0 )*( s_R + rho_R*ds_drho_R );
        double V_2u_L = ( -1.0 )*rho_L*ds_drhou_L;
        double V_2u_R = ( -1.0 )*rho_R*ds_drhou_R;
        double V_2v_L = ( -1.0 )*rho_L*ds_drhov_L;
        double V_2v_R = ( -1.0 )*rho_R*ds_drhov_R;
        double V_2w_L = ( -1.0 )*rho_L*ds_drhow_L;
        double V_2w_R = ( -1.0 )*rho_R*ds_drhow_R;
	double V_3_L  = ( -1.0 )*rho_L*ds_drhoE_L;
        double V_3_R  = ( -1.0 )*rho_R*ds_drhoE_R;
        double deltaV_1 = V_1_R - V_1_L;
        double deltaV_2u = V_2u_R - V_2u_L;
        double deltaV_2v = V_2v_R - V_2v_L;
        double deltaV_2w = V_2w_R - V_2w_L;
        double deltaV_3 = V_3_R - V_3_L;
        double psi_L = u_L*P_L/T_L;
        double psi_R = u_R*P_R/T_R;
        double deltaPsi = psi_R - psi_L;
        //double alpha_3  = 2.0*( bar_F_1*deltaV_1 + bar_F_2u*deltaV_2u + bar_F_2v*deltaV_2v + bar_F_2w*deltaV_2w + bar_F_3*deltaV_3 - deltaPsi )/( deltaV_3*deltaV_3 + epsilon );
        double alpha_3  = 2.0*( bar_F_1*deltaV_1 + bar_F_2u*deltaV_2u + bar_F_2v*deltaV_2v + bar_F_2w*deltaV_2w + bar_F_3*deltaV_3 - deltaPsi )/( deltaV_3*deltaV_3 + 1.0e-10 );	// ... modified for OpenACC
        F = bar_F_3 - ( 1.0/2.0 )*alpha_3*deltaV_3;
    }

    return( F );

};


////////// HesFluxApproximateRiemannSolver CLASS //////////

HesFluxApproximateRiemannSolver::HesFluxApproximateRiemannSolver() : BaseRiemannSolver() {};

HesFluxApproximateRiemannSolver::~HesFluxApproximateRiemannSolver() {};

double HesFluxApproximateRiemannSolver::calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) {

    /// Method of Optimal Viscosity for Enhanced Resolution of Shocks (MOVERS) scheme:
    /// S. Jaisankar, S.V. Raghurama Rao.
    /// A central Rankine-Hugoniot solver for hyperbolic conservation laws.
    /// Journal of Computational Physics, 228, 770-798, 2009.    

    /// Calculate eigenvalues
    double u = ( 1.0/2.0 )*( u_L + u_R );
    double a = ( 1.0/2.0 )*( a_L + a_R );
    double lambda_1 = abs( u - a );
    double lambda_2 = abs( u );
    double lambda_3 = abs( u + a );
    double lambda_min = min( lambda_1, min( lambda_2, lambda_3 ) );
    double lambda_max = max( lambda_1, max( lambda_2, lambda_3 ) );

    /// Conserved variable increment
    double deltaU_1  = rho_R - rho_L;          
    double deltaU_2u = rho_R*u_R - rho_L*u_L;
    double deltaU_2v = rho_R*v_R - rho_L*v_L;
    double deltaU_2w = rho_R*w_R - rho_L*w_L;
    double deltaU_3  = rho_R*E_R - rho_L*E_L;

    /// Flux increment 
    double deltaF_1  = rho_R*u_R - rho_L*u_L;
    double deltaF_2u = rho_R*u_R*u_R + P_R - rho_L*u_L*u_L - P_L;
    double deltaF_3  = rho_R*u_R*E_R + P_R*u_R - rho_L*u_L*E_L - P_L*u_L;
   
    /// Wave speed (limited)
    //double S_1 = deltaF_1 / ( deltaU_1 + epsilon );
    double S_1 = deltaF_1 / ( deltaU_1 + 1.0e-10 );					// ... modified for OpenACC
    if( abs(S_1) > lambda_max ) S_1 = copysign( lambda_max, S_1 );
    if( abs(S_1) < lambda_min ) S_1 = copysign( lambda_min, S_1 );
    //double S_2 = deltaF_2u / ( deltaU_2u + epsilon );
    double S_2 = deltaF_2u / ( deltaU_2u + 1.0e-10 );					// ... modified for OpenACC
    if( abs(S_2) > lambda_max ) S_2 = copysign( lambda_max, S_2 );
    if( abs(S_2) < lambda_min ) S_2 = copysign( lambda_min, S_2 );
    //double S_3 = deltaF_3 / ( deltaU_3 + epsilon );
    double S_3 = deltaF_3 / ( deltaU_3 + 1.0e-10 );					// ... modified for OpenACC
    if( abs(S_3) > lambda_max ) S_3 = copysign( lambda_max, S_3 );
    if( abs(S_3) < lambda_min ) S_3 = copysign( lambda_min, S_3 );
    double alpha_S = min( abs(S_1), min( abs(S_2), abs(S_3) ) );

    /// Entropic variables: derivatives
    double ds_drho_L   = ( u_L*u_L + v_L*v_L + w_L*w_L - E_L - P_L/rho_L )/( rho_L*T_L );
    double ds_drho_LL  = ( u_LL*u_LL + v_LL*v_LL + w_LL*w_LL - E_LL - P_LL/rho_LL )/( rho_LL*T_LL );
    double ds_drho_LLL = ( u_LLL*u_LLL + v_LLL*v_LLL + w_LLL*w_LLL - E_LLL - P_LLL/rho_LLL )/( rho_LLL*T_LLL );
    double ds_drho_R   = ( u_R*u_R + v_R*v_R + w_R*w_R - E_R - P_R/rho_R )/( rho_R*T_R );
    double ds_drho_RR  = ( u_RR*u_RR + v_RR*v_RR + w_RR*w_RR - E_RR - P_RR/rho_RR )/( rho_RR*T_RR );
    double ds_drho_RRR = ( u_RRR*u_RRR + v_RRR*v_RRR + w_RRR*w_RRR - E_RRR - P_RRR/rho_RRR )/( rho_RRR*T_RRR );
    double ds_drhou_L   = ( -1.0 )*u_L/( rho_L*T_L );
    double ds_drhou_LL  = ( -1.0 )*u_LL/( rho_LL*T_LL );
    double ds_drhou_LLL = ( -1.0 )*u_LLL/( rho_LLL*T_LLL );
    double ds_drhou_R   = ( -1.0 )*u_R/( rho_R*T_R );
    double ds_drhou_RR  = ( -1.0 )*u_RR/( rho_RR*T_RR );
    double ds_drhou_RRR = ( -1.0 )*u_RRR/( rho_RRR*T_RRR );
    double ds_drhoE_L   = 1.0/( rho_L*T_L );
    double ds_drhoE_LL  = 1.0/( rho_LL*T_LL );
    double ds_drhoE_LLL = 1.0/( rho_LLL*T_LLL );
    double ds_drhoE_R   = 1.0/( rho_R*T_R );
    double ds_drhoE_RR  = 1.0/( rho_RR*T_RR );
    double ds_drhoE_RRR = 1.0/( rho_RRR*T_RRR );

    /// Entropic variables: calculations
    double V_1_L   = ( -1.0 )*( s_L + rho_L*ds_drho_L );
    double V_1_LL  = ( -1.0 )*( s_LL + rho_LL*ds_drho_LL );
    double V_1_LLL = ( -1.0 )*( s_LLL + rho_LLL*ds_drho_LLL );
    double V_1_R   = ( -1.0 )*( s_R + rho_R*ds_drho_R );
    double V_1_RR  = ( -1.0 )*( s_RR + rho_RR*ds_drho_RR );
    double V_1_RRR = ( -1.0 )*( s_RRR + rho_RRR*ds_drho_RRR ); 
    double V_2_L   = ( -1.0 )*rho_L*ds_drhou_L;
    double V_2_LL  = ( -1.0 )*rho_LL*ds_drhou_LL;
    double V_2_LLL = ( -1.0 )*rho_LLL*ds_drhou_LLL;
    double V_2_R   = ( -1.0 )*rho_R*ds_drhou_R;
    double V_2_RR  = ( -1.0 )*rho_RR*ds_drhou_RR;
    double V_2_RRR = ( -1.0 )*rho_RRR*ds_drhou_RRR;
    double V_3_L   = ( -1.0 )*rho_L*ds_drhoE_L;
    double V_3_LL  = ( -1.0 )*rho_LL*ds_drhoE_LL;
    double V_3_LLL = ( -1.0 )*rho_LLL*ds_drhoE_LLL;
    double V_3_R   = ( -1.0 )*rho_R*ds_drhoE_R;
    double V_3_RR  = ( -1.0 )*rho_RR*ds_drhoE_RR;
    double V_3_RRR = ( -1.0 )*rho_RRR*ds_drhoE_RRR;

    /// SOBOLEV-LIKE SENSOR: approximated derivatives
    //double eps = 1e-12*pow( max( abs(V_1_L), max( abs(V_2_L), max( abs(V_3_L), max( abs(V_1_R), max( abs(V_2_R), abs(V_3_R) ) ) ) ) ), 2.0 ) + epsilon;
    double eps = 1e-12*pow( max( abs(V_1_L), max( abs(V_2_L), max( abs(V_3_L), max( abs(V_1_R), max( abs(V_2_R), abs(V_3_R) ) ) ) ) ), 2.0 ) + 1.0e-10;	
    double DV1_L = ( -V_1_RR + 8.0*V_1_R - 8.0*V_1_LL + V_1_LLL )/( 12.0*delta );
    double DV2_L = ( -V_2_RR + 8.0*V_2_R - 8.0*V_2_LL + V_2_LLL )/( 12.0*delta );
    double DV3_L = ( -V_3_RR + 8.0*V_3_R - 8.0*V_3_LL + V_3_LLL )/( 12.0*delta );
    double D2V1_L = ( -V_1_RR + 16.0*V_1_R - 30.0*V_1_L + 16.0*V_1_LL - V_1_LLL )/( 12.0*delta*delta );
    double D2V2_L = ( -V_2_RR + 16.0*V_2_R - 30.0*V_2_L + 16.0*V_2_LL - V_2_LLL )/( 12.0*delta*delta );
    double D2V3_L = ( -V_3_RR + 16.0*V_3_R - 30.0*V_3_L + 16.0*V_3_LL - V_3_LLL )/( 12.0*delta*delta );
    double D3V1_L = ( -V_1_LLL - V_1_LL + 10.0*V_1_L - 14.0*V_1_R + 7.0*V_1_RR - V_1_RRR )/( 4.0*delta*delta*delta );
    double D3V2_L = ( -V_2_LLL - V_2_LL + 10.0*V_2_L - 14.0*V_2_R + 7.0*V_2_RR - V_2_RRR )/( 4.0*delta*delta*delta );
    double D3V3_L = ( -V_3_LLL - V_3_LL + 10.0*V_3_L - 14.0*V_3_R + 7.0*V_3_RR - V_3_RRR )/( 4.0*delta*delta*delta );
    double DV1_R  = ( -V_1_RRR + 8.0*V_1_RR - 8.0*V_1_L  + V_1_LL )/( 12.0*delta );
    double DV2_R  = ( -V_2_RRR + 8.0*V_2_RR - 8.0*V_2_L  + V_2_LL )/( 12.0*delta );
    double DV3_R  = ( -V_3_RRR + 8.0*V_3_RR - 8.0*V_3_L  + V_3_LL )/( 12.0*delta );
    double D2V1_R = ( -V_1_RRR + 16.0*V_1_RR - 30.0*V_1_R + 16.0*V_1_L - V_1_LL )/( 12.0*delta*delta );
    double D2V2_R = ( -V_2_RRR + 16.0*V_2_RR - 30.0*V_2_R + 16.0*V_2_L - V_2_LL )/( 12.0*delta*delta );
    double D2V3_R = ( -V_3_RRR + 16.0*V_3_RR - 30.0*V_3_R + 16.0*V_3_L - V_3_LL )/( 12.0*delta*delta );
    double D3V1_R = ( V_1_LLL - 7.0*V_1_LL + 14.0*V_1_L - 10.0*V_1_R + V_1_RR + V_1_RRR)/( 4.0*delta*delta*delta );
    double D3V2_R = ( V_2_LLL - 7.0*V_2_LL + 14.0*V_2_L - 10.0*V_2_R + V_2_RR + V_2_RRR)/( 4.0*delta*delta*delta );
    double D3V3_R = ( V_3_LLL - 7.0*V_3_LL + 14.0*V_3_L - 10.0*V_3_R + V_3_RR + V_3_RRR)/( 4.0*delta*delta*delta );

    /// SOBOLEV-LIKE SENSOR: norms
    double normV_L_sq   = V_1_L*V_1_L + V_2_L*V_2_L + V_3_L*V_3_L;     
    double normDV_L_sq  = DV1_L*DV1_L + DV2_L*DV2_L + DV3_L*DV3_L;
    double normD2V_L_sq = D2V1_L*D2V1_L + D2V2_L*D2V2_L + D2V3_L*D2V3_L;
    double normD3V_L_sq = D3V1_L*D3V1_L + D3V2_L*D3V2_L + D3V3_L*D3V3_L;
    double normV_R_sq   = V_1_R*V_1_R + V_2_R*V_2_R + V_3_R*V_3_R;     
    double normDV_R_sq  = DV1_R*DV1_R + DV2_R*DV2_R + DV3_R*DV3_R;
    double normD2V_R_sq = D2V1_R*D2V1_R + D2V2_R*D2V2_R + D2V3_R*D2V3_R;
    double normD3V_R_sq = D3V1_R*D3V1_R + D3V2_R*D3V2_R + D3V3_R*D3V3_R;

    /// SOBOLEV-LIKE SENSOR: result
    double sensor_L = ( delta*delta*normD3V_L_sq )/( normV_L_sq + normDV_L_sq + normD2V_L_sq + eps );
    double sensor_R = ( delta*delta*normD3V_R_sq )/( normV_R_sq + normDV_R_sq + normD2V_R_sq + eps );
    double Phi_sensor = max( sensor_L, sensor_R );    

    /// SOBOLEV-LIKE SENSOR: normalization
    //double s = 80.0;
    double s = 25.0;
    //double phi_epsilon = 0.05;
    double phi_epsilon = 0.04;
    Phi_sensor = 0.5*( 1.0 + tanh( s*( Phi_sensor - phi_epsilon ) ) );
    Phi_sensor = max( 0.0, min( 1.0, Phi_sensor ) );
    alpha_S *= Phi_sensor; 

    ///// ---------------------------------///
    ///// START: SHOCK SENSOR MODIFICATION ///
    ///// ---------------------------------///
    
    ///// Pressure sensor: shocks have sharp pressure jump
    //double P = ( 1.0/2.0 )*( P_L + P_R );
    ////double sensor_P = min( 1.0, 50.0*abs( P_R - P_L )/( P + epsilon ) );     // Factor: 10, 50, 100
    //double sensor_P = min( 1.0, 50.0*abs( P_R - P_L )/( P + 1.0e-10 ) );     // Factor: 10, 50, 100		// ... modified for OpenACC

    /////// Compression switch: compressions (shocks) will be negative; expansions (rarefactions) will be positive
    ////double delta_u = u_R - u_L;
    /////// Apply sensor: if delta_u > 0 (rarefaction/expansion)
    ////if( delta_u > 0.0 ) alpha_S = 0.0;
    
    //alpha_S *= sensor_P;

    ///// -------------------------------///
    ///// END: SHOCK SENSOR MODIFICATION ///
    ///// -------------------------------///

    double F = ( 1.0/8.0 )*( rho_L + rho_R )*( u_L + u_R );
    if( var_type == 0 ) {
        F *= 1.0 + 1.0;
        F -= ( 1.0/2.0 )*alpha_S*deltaU_1;
    } else if ( var_type == 1 ) {
        F *= u_L + u_R; F += ( 1.0/2.0 )*( P_L + P_R );
        F -= ( 1.0/2.0 )*alpha_S*deltaU_2u;
    } else if ( var_type == 2 ) {
        F *= v_L + v_R;
        F -= ( 1.0/2.0 )*alpha_S*deltaU_2v;
    } else if ( var_type == 3 ) {
        F *= w_L + w_R;
        F -= ( 1.0/2.0 )*alpha_S*deltaU_2w;
    } else if ( var_type == 4 ) {
        double bar_F_1  = ( 1.0/4.0 )*( rho_L + rho_R )*( u_L + u_R );
        double bar_F_2u = ( 1.0/2.0 )*( bar_F_1*( u_L + u_R ) + ( P_L + P_R ) );
        double bar_F_2v = ( 1.0/2.0 )*bar_F_1*( v_L + v_R );
        double bar_F_2w = ( 1.0/2.0 )*bar_F_1*( w_L + w_R );
        double bar_F_3  = ( 1.0/2.0 )*bar_F_1*( E_L + P_L/rho_L + E_R + P_R/rho_R );
        double ds_drho_L  = ( u_L*u_L + v_L*v_L + w_L*w_L - E_L - P_L/rho_L )/( rho_L*T_L );
        double ds_drho_R  = ( u_R*u_R + v_R*v_R + w_R*w_R - E_R - P_R/rho_R )/( rho_R*T_R );
        double ds_drhou_L = ( -1.0 )*u_L/( rho_L*T_L );
        double ds_drhou_R = ( -1.0 )*u_R/( rho_R*T_R );
        double ds_drhov_L = ( -1.0 )*v_L/( rho_L*T_L );
        double ds_drhov_R = ( -1.0 )*v_R/( rho_R*T_R );
        double ds_drhow_L = ( -1.0 )*w_L/( rho_L*T_L );
        double ds_drhow_R = ( -1.0 )*w_R/( rho_R*T_R );
        double ds_drhoE_L = 1.0/( rho_L*T_L );
        double ds_drhoE_R = 1.0/( rho_R*T_R );	    
        double V_1_L  = ( -1.0 )*( s_L + rho_L*ds_drho_L );
        double V_1_R  = ( -1.0 )*( s_R + rho_R*ds_drho_R );
        double V_2u_L = ( -1.0 )*rho_L*ds_drhou_L;
        double V_2u_R = ( -1.0 )*rho_R*ds_drhou_R;
        double V_2v_L = ( -1.0 )*rho_L*ds_drhov_L;
        double V_2v_R = ( -1.0 )*rho_R*ds_drhov_R;
        double V_2w_L = ( -1.0 )*rho_L*ds_drhow_L;
        double V_2w_R = ( -1.0 )*rho_R*ds_drhow_R;
	double V_3_L  = ( -1.0 )*rho_L*ds_drhoE_L;
        double V_3_R  = ( -1.0 )*rho_R*ds_drhoE_R;
        double deltaV_1 = V_1_R - V_1_L;
        double deltaV_2u = V_2u_R - V_2u_L;
        double deltaV_2v = V_2v_R - V_2v_L;
        double deltaV_2w = V_2w_R - V_2w_L;
        double deltaV_3 = V_3_R - V_3_L;
        double psi_L = u_L*P_L/T_L;
        double psi_R = u_R*P_R/T_R;
        double deltaPsi = psi_R - psi_L;
        //double alpha_3  = 2.0*( bar_F_1*deltaV_1 + bar_F_2u*deltaV_2u + bar_F_2v*deltaV_2v + bar_F_2w*deltaV_2w + bar_F_3*deltaV_3 - deltaPsi )/( deltaV_3*deltaV_3 + epsilon );
        double alpha_3  = 2.0*( bar_F_1*deltaV_1 + bar_F_2u*deltaV_2u + bar_F_2v*deltaV_2v + bar_F_2w*deltaV_2w + bar_F_3*deltaV_3 - deltaPsi )/( deltaV_3*deltaV_3 + 1.0e-10 );	// ... modified for OpenACC
        F = bar_F_3 - ( 1.0/2.0 )*alpha_3*deltaV_3;
        F -= ( 1.0/2.0 )*alpha_S*deltaU_3;
    }    

    return( F );

};


////////// BaseExplicitRungeKuttaMethod CLASS //////////

BaseExplicitRungeKuttaMethod::BaseExplicitRungeKuttaMethod() {};
        
BaseExplicitRungeKuttaMethod::~BaseExplicitRungeKuttaMethod() {};


////////// RungeKutta1Method CLASS //////////

RungeKutta1Method::RungeKutta1Method() : BaseExplicitRungeKuttaMethod() {};

RungeKutta1Method::~RungeKutta1Method() {};

void RungeKutta1Method::setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage) {

    /// Explicit first-order Runge-Kutta (RK1) method:
    /// S. Gottlieb, C.-W. Shu & E. Tadmor.
    /// Strong stability-preserving high-order time discretization methods.
    /// SIAM Review 43, 89-112, 2001.

    /// First Runge-Kutta stage
    rk_a = 1.0; rk_b = 0.0; rk_c = 1.0;

};


////////// StrongStabilityPreservingRungeKutta2Method CLASS //////////

StrongStabilityPreservingRungeKutta2Method::StrongStabilityPreservingRungeKutta2Method() : BaseExplicitRungeKuttaMethod() {};

StrongStabilityPreservingRungeKutta2Method::~StrongStabilityPreservingRungeKutta2Method() {};

void StrongStabilityPreservingRungeKutta2Method::setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage) {

    /// Explicit second-order strong-stability-preserving Runge-Kutta (SSP-RK2) method:
    /// S. Gottlieb, C.-W. Shu & E. Tadmor.
    /// Strong stability-preserving high-order time discretization methods.
    /// SIAM Review 43, 89-112, 2001.

    if(rk_time_stage == 1) {
        /// First Runge-Kutta stage
        rk_a = 1.0; rk_b = 0.0; rk_c = 1.0;
    } else if(rk_time_stage == 2) {
        /// Second Runge-Kutta stage
        rk_a = 1.0/2.0; rk_b = 1.0/2.0; rk_c = 1.0/2.0;
    }

};


////////// StrongStabilityPreservingRungeKutta3Method CLASS //////////

StrongStabilityPreservingRungeKutta3Method::StrongStabilityPreservingRungeKutta3Method() : BaseExplicitRungeKuttaMethod() {};

StrongStabilityPreservingRungeKutta3Method::~StrongStabilityPreservingRungeKutta3Method() {};

void StrongStabilityPreservingRungeKutta3Method::setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage) {

    /// Explicit third-order strong-stability-preserving Runge-Kutta (SSP-RK3) method:
    /// S. Gottlieb, C.-W. Shu & E. Tadmor.
    /// Strong stability-preserving high-order time discretization methods.
    /// SIAM Review 43, 89-112, 2001.

    if(rk_time_stage == 1) {
        /// First Runge-Kutta stage
        rk_a = 1.0; rk_b = 0.0; rk_c = 1.0;
    } else if(rk_time_stage == 2) {
        /// Second Runge-Kutta stage
        rk_a = 3.0/4.0; rk_b = 1.0/4.0; rk_c = 1.0/4.0;
    } else if(rk_time_stage == 3) {
        /// Third Runge-Kutta stage
        rk_a = 1.0/3.0; rk_b = 2.0/3.0; rk_c = 2.0/3.0;
    }

};
