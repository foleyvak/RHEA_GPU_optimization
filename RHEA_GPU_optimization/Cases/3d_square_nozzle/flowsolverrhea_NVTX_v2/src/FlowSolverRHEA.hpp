#ifndef _RHEA_FLOW_SOLVER_
#define _RHEA_FLOW_SOLVER_

////////// INCLUDES //////////
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string.h>
#include <sstream>
#include <limits>
#include <iomanip>
#include <chrono>
#include <mpi.h>
#include <functional>
#include "yaml-cpp/yaml.h"
#include "MacroParameters.hpp"
#include "ThermodynamicModel.hpp"
#include "TransportCoefficients.hpp"
#include "ComputationalDomain.hpp"
#include "ParallelTopology.hpp"
#include "DistributedArray.hpp"
#include "InputOutputManager.hpp"
#include "ParallelTimer.hpp"
#include "ImmersedBoundaryMethod.hpp"
#include "LagrangianPointParticles.hpp"
#include "stl_reader.hpp"

////////// CLASS DECLARATION //////////
class FlowSolverRHEA;					/// Flow solver RHEA

class BaseRiemannSolver;				/// Base Riemann solver
class DivergenceFluxApproximateRiemannSolver;		/// Divergence scheme approximate Riemann solver
class MurmanRoeFluxApproximateRiemannSolver;		/// Murman-Roe scheme approximate Riemann solver
class KgpFluxApproximateRiemannSolver;			/// KGP scheme approximate Riemann solver
class ShimaFluxApproximateRiemannSolver;		/// SHIMA scheme approximate Riemann solver
class HllApproximateRiemannSolver;			/// HLL approximate Riemann solver
class HllcApproximateRiemannSolver;			/// HLLC approximate Riemann solver
class HllcPlusApproximateRiemannSolver;			/// HLLC+ approximate Riemann solver
class EckepFluxApproximateRiemannSolver;		/// ECKEP scheme approximate Riemann solver
class HesFluxApproximateRiemannSolver;			/// Hybrid Entropy Stable (HES) scheme approximate Riemann solver

class BaseExplicitRungeKuttaMethod;			/// Base explicit Runge-Kutta method
class RungeKutta1Method;				/// Runge-Kutta 1 (RK1) method
class StrongStabilityPreservingRungeKutta2Method;	/// Strong stability preserving Runge-Kutta 2 (SSP-RK2) method
class StrongStabilityPreservingRungeKutta3Method;	/// Strong stability preserving Runge-Kutta 3 (SSP-RK3) method

////////// FUNCTION DECLARATION //////////


////////// FlowSolverRHEA CLASS //////////
class FlowSolverRHEA {
   
    ////////// VARIABLES & PARAMETERS DESCRIPTION //////////

    /// Primitive variables:
    ///   - Density: rho
    ///   - Velocities: u, v, w
    ///   - Specific total energy: E = e + ke
    ///     ... sum of specific internal energy e, and specific kinetic energy ke = (u*u + v*v + w*w)/2

    /// Conserved variables:
    ///   - Mass: rho
    ///   - Momentum: rho*u, rho*v, rho*w
    ///   - Total energy: rho*E

    /// Thermodynamic state:
    ///   - Pressure: P
    ///   - Temperature: T
    ///   - Speed of sound: sos
    ///   - Specific gas constant: R_specific
    ///   - Ratio of heat capacities: gamma

    /// Transport coefficients:
    ///   - Dynamic viscosity: mu
    ///   - Thermal conductivity: kappa
 
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        FlowSolverRHEA();					/// Default constructor
        //FlowSolverRHEA(const std::string configuration_file);	/// Parametrized constructor
        FlowSolverRHEA(const std::string name_configuration_file, std::function<void(FlowSolverRHEA*)> custom_mesh_filler = nullptr);	/// Parametrized constructor
        virtual ~FlowSolverRHEA();				/// Destructor

	////////// GET FUNCTIONS //////////
        inline double getCurrentTime() { return( current_time ); };
        inline int getCurrentTimeIteration() { return( current_time_iter ); };

	////////// SET FUNCTIONS //////////
        inline void setCurrentTime(double current_time_) { current_time = current_time_; };
        inline void setCurrentTimeIteration(int current_time_iter_) { current_time_iter = current_time_iter_; };

	////////// SOLVER METHODS //////////
        
	/// Execute (aggregated method) RHEA
        virtual void execute();

        /// Read configuration (input) file written in YAML language
        virtual void readConfigurationFile();

        /// Fill mesh x, y, z, delta_x, delta_y, delta_z fields
        virtual void fillMeshCoordinatesSizesFields();

        /// Set initial conditions: u, v, w, P and T ... needs to be modified/overwritten according to the problem under consideration
        virtual void setInitialConditions();

        /// Initialize from restart file: x, y, z, rho, u, v, w, E, P, T, sos, mu and kappa
        virtual void initializeFromRestart();

        /// Initialize thermodynamic state: rho, E and sos
        virtual void initializeThermodynamics();

        /// Calculate conserved variables from primitive variables: variable -> rho*variable
        virtual void primitiveToConservedVariables();

        /// Calculate primitive variables from conserved variables: rho*variable -> variable
        virtual void conservedToPrimitiveVariables();

        /// Calculate thermodynamics from primitive variables
        virtual void calculateThermodynamicsFromPrimitiveVariables();

        /// Update boundary values: rho, rhou, rhov, rhow and rhoE
        virtual void updateBoundaries();

        /// Advance point particles in time: position and velocity
        virtual void timeAdvancePointParticles(const int &my_rank);

        /// Advance point particles in time: velocity ... needs to be modified/overwritten according to the problem under consideration
        virtual void timeAdvanceVelocityPointParticles();

        /// Update previous state of conserved variables: rho*variable -> rho_0*variable_0
        virtual void updatePreviousStateConservedVariables();

        /// Calculate time step satisfying CFL constraint
        virtual void calculateTimeStep();

        /// Calculate transport coefficients
        virtual void calculateTransportCoefficients();

        /// Calculate rhou, rhov, rhow and rhoE source terms ... needs to be modified/overwritten according to the problem under consideration
        virtual void calculateSourceTerms();

        /// Impose immersed boundary method
        virtual void imposeImmersedBoundaryMethod();

        /// Tag immersed boundary method ... needs to be modified/overwritten according to the problem under consideration
        virtual void tagImmersedBoundaryMethod();

        /// Reconstruct immersed boundary method
        virtual void reconstructImmersedBoundaryMethod();

        /// Force immersed boundary method
        virtual void forceImmersedBoundaryMethod();

        /// Calculate inviscid fluxes in x-, y- and z-direction
        virtual void calculateInviscidFluxes();

        /// Calculate viscous fluxes
        virtual void calculateViscousFluxes();

        /// Advance conserved variables in time
        virtual void timeAdvanceConservedVariables();

        /// Advance pressure in time
        virtual void timeAdvancePressure();

        /// Update CPU data
        virtual void updateHost();

        /// Output current solver state data
        virtual void outputCurrentStateData();

        /// Output current 2d slices solver state data
        virtual void output2dSlicesCurrentStateData();

        /// Output temporal point probes data
        virtual void outputTemporalPointProbesData();

        /// Update time-averaged quantities
        virtual void updateTimeAveragedQuantities();

        /// Update time mean quantity
        //virtual double updateTimeMeanQuantity(const double &quantity, const double &mean_quantity, const double &delta_t, const double &averaging_time);
        #pragma acc routine
	static double updateTimeMeanQuantity(const double &quantity, const double &mean_quantity, const double &delta_t, const double &averaging_time);

        /// Update time rmsf quantity
        //virtual double updateTimeRmsfQuantity(const double &quantity, const double &mean_quantity, const double &rmsf_quantity, const double &delta_t, const double &averaging_time);
	#pragma acc routine
	static double updateTimeRmsfQuantity(const double &quantity, const double &mean_quantity, const double &rmsf_quantity, const double &delta_t, const double &averaging_time);

        ///// Update time Reynolds-averaged quantity
        ////virtual double updateTimeReynoldsAveragedQuantity(const double &quantity_1, const double &mean_quantity_1, const double &quantity_2, const double &mean_quantity_2, const double &reynolds_averaged_quantity, const double &delta_t, const double &averaging_time);
        //static double updateTimeReynoldsAveragedQuantity(const double &quantity_1, const double &mean_quantity_1, const double &quantity_2, const double &mean_quantity_2, const double &reynolds_averaged_quantity, const double &delta_t, const double &averaging_time);

        /// Update time Favre-averaged quantity
        //virtual double updateTimeFavreAveragedQuantity(const double &quantity_1, const double &mean_rho_quantity_1, const double &quantity_2, const double &mean_rho_quantity_2, const double &rho, const double &mean_rho, const double &favre_averaged_quantity, const double &delta_t, const double &averaging_time);
	#pragma acc routine
	static double updateTimeFavreAveragedQuantity(const double &quantity_1, const double &mean_rho_quantity_1, const double &quantity_2, const double &mean_rho_quantity_2, const double &rho, const double &mean_rho, const double &favre_averaged_quantity, const double &delta_t, const double &averaging_time);
       
        /// Temporal hook function ... needs to be modified/overwritten according to the problem under consideration
        virtual void temporalHookFunction();

	/// Calculate volume-averaged pressure
        virtual double calculateVolumeAveragedPressure();

	/// Calculate alpha value of artificial compressibility method 
        virtual double calculateAlphaArtificialCompressibilityMethod();

	/// Calculate artificially modified thermodynamics
        virtual void calculateArtificiallyModifiedThermodynamics();

	/// Calculate artificially modified transport coefficients
        virtual void calculateArtificiallyModifiedTransportCoefficients();

        /// Set initial conditions of particles: x, y, z, u, v, and w ... needs to be modified/overwritten according to the problem under consideration
        virtual void setInitialParticlesPositionsVelocities();

	/// Update Lagrangian-Eulerian Mesh indexes 0
	void updateLagrangianEulerianMeshIndexes0(const int &my_rank);

	/// Obtain Lagrangian-Eulerian velocity and dynamic viscosity values
        //virtual void obtainLagrangianEulerianVelocityDynamicViscosityValues(double &u_velocity_fluid_point_particle, double &v_velocity_fluid_point_particle, double &w_velocity_fluid_point_particle, double &dynamic_viscosity_fluid, const int &p);
	//#pragma acc routine
	void obtainLagrangianEulerianVelocityDynamicViscosityValues(double &u_velocity_fluid_point_particle, double &v_velocity_fluid_point_particle, double &w_velocity_fluid_point_particle, double &dynamic_viscosity_fluid, const int &index_particle);

	/// Find if point particle is within fluid cell
        virtual bool pointParticleIsContainedWithinFluidCell(const double &x_position_particle, const double &y_position_particle, const double &z_position_particle, const int &local_index_i, const int &local_index_j, const int &local_index_k);

	/// Trilinear interpolation of scalar field at particular position
        /// Target point: x, y, z
        /// Grid coordinates: x0, x1, y0, y1, z0, z1
        /// Scalar values at grid coordinates: f000, f100, f010, f110, f001, f101, f011, f111
        //virtual double trilinearInterpolation(const double &x, const double &y, const double &z, const double &xm, const double &xc, const double &xp, const double &ym, const double &yc, const double &yp, const double &zm, const double &zc, const double &zp, const double &fmmm, const double &fmmc, const double &fmmp, const double &fmcm, const double &fmcc, const double &fmcp, const double &fmpm, const double &fmpc, const double &fmpp, const double &fcmm, const double &fcmc, const double &fcmp, const double &fccm, const double &fccc, const double &fccp, const double &fcpm, const double &fcpc, const double &fcpp, const double &fpmm, const double &fpmc, const double &fpmp, const double &fpcm, const double &fpcc, const double &fpcp, const double &fppm, const double &fppc, const double &fppp);
        //#pragma acc routine
        double trilinearInterpolation(const double &x, const double &y, const double &z, const double &xm, const double &xc, const double &xp, const double &ym, const double &yc, const double &yp, const double &zm, const double &zc, const double &zp, const double &fmmm, const double &fmmc, const double &fmmp, const double &fmcm, const double &fmcc, const double &fmcp, const double &fmpm, const double &fmpc, const double &fmpp, const double &fcmm, const double &fcmc, const double &fcmp, const double &fccm, const double &fccc, const double &fccp, const double &fcpm, const double &fcpc, const double &fcpp, const double &fpmm, const double &fpmc, const double &fpmp, const double &fpcm, const double &fpcc, const double &fpcp, const double &fppm, const double &fppc, const double &fppp);

    //protected:
    public:

        ////////// SOLVER PARAMETERS //////////
	
        /// Fluid & flow properties 
	std::string thermodynamic_model;			/// Thermodynamic model
	std::string transport_coefficients_model;		/// Transport coefficients model

        /// Problem parameters
        double x_0;           	        	 	 	/// Domain origin in x-direction [m]
        double y_0;             	        		/// Domain origin in y-direction [m]
        double z_0;                     			/// Domain origin in z-direction [m]
        double L_x;   						/// Domain size in x-direction [m]
        double L_y;      					/// Domain size in y-direction [m]
        double L_z;						/// Domain size in z-direction [m]
        double current_time;   					/// Current time [s]
        double final_time;		      			/// Final time [s]
        double averaging_time;		      			/// Averaging time [s]
	std::string configuration_file;				/// Configuration file name (YAML language)	

        /// Computational parameters
        int num_grid_x;						/// Number of inner grid points in x-direction
        int num_grid_y;						/// Number of inner grid points in y-direction
        int num_grid_z;						/// Number of inner grid points in z-direction
        double A_x;						/// Stretching factor in x-direction
        double A_y;						/// Stretching factor in y-direction
        double A_z;						/// Stretching factor in z-direction
        bool external_mesh;					/// Activate external mesh
	std::string external_mesh_file;				/// External mesh file
        double CFL;						/// CFL coefficient
        double delta_t = 0.0;	      				/// Time step [s]
        int current_time_iter;					/// Current time iteration
        int final_time_iter;					/// Final time iteration
        int rk_time_stage;					/// Stage of Runge-Kutta time discretization method
        int rk_number_stages;					/// Number of stages of Runge-Kutta time discretization method
	std::string riemann_solver_scheme;			/// Riemann solver scheme
	std::string runge_kutta_time_scheme;			/// Runge-Kutta time scheme
        bool transport_pressure_scheme;				/// Activate transport P instead of rhoE
        bool artificial_compressibility_method;			/// Artificially decrease velocity of acoustic waves
        double epsilon_acm;	      				/// Relative error of artificial compressibility method [-]
        double alpha_acm = 1.0;	      				/// Speedup factor of artificial compressibility method [-]
        double P_thermo = 0.0;	      				/// Thermodynamic (bulk) pressure for artificial compressibility method [Pa]

        /// Local mesh values for I1D macro
        int _lNx_;
        int _lNy_;
        int _lNz_;
	int _ls_;

	/// Updated CPU data tracker 				/// Indicates if CPU memory is updated ... added for OpenACC
	bool updated_cpu = false;

        /// Boundary conditions
        int bocos_type[6];					/// Array of boundary conditions type
        double bocos_u[6];					/// Array of boundary conditions u
        double bocos_v[6];					/// Array of boundary conditions v
        double bocos_w[6];					/// Array of boundary conditions w
        double bocos_P[6];					/// Array of boundary conditions P
        double bocos_T[6];					/// Array of boundary conditions T

        /// Immersed boundary method
        bool activate_immersed_boundary_method;			/// Activate immersed boundary method

        /// Print/Write/Read file parameters
        int print_frequency_iter;				/// Print information iteration frequency
	std::string output_data_file_name;			/// Output data file name (HDF5 format)	
        int output_frequency_iter;				/// Data output iteration frequency
        bool generate_xdmf_file;				/// Generate xdmf file reader
        bool use_restart;					/// Use restart file for initialization
	std::string restart_data_file;				/// Restart data file
        bool time_averaging_active;				/// Activate time averaging
        bool reset_time_averaging;				/// Reset time averaging
      
	/// 2D data output slices
        int number_two_dimensional_data_output_slices = 0;	/// Number of 2D data output slices (default is zero)	
	std::vector<std::string> dos_normal_directions;		/// Normal directions of 2D data output slices
	std::vector<double> dos_x_positions;			/// Positions in x-direction of 2D data output slices [m]			
	std::vector<double> dos_y_positions;			/// Positions in y-direction of 2D data output slices [m]			
	std::vector<double> dos_z_positions;			/// Positions in z-direction of 2D data output slices [m]			
	std::vector<int> dos_output_frequency_iters;		/// Output frequency iterations of 2D data output slices
	std::vector<bool> dos_generate_xdmf_files;		/// Generate xdmf files of 2D data output slices
	std::vector<std::string> dos_output_file_names;		/// Output file names of 2D data output slices
 
	/// Temporal point probes
	std::vector<TemporalPointProbe> temporal_point_probes;	/// Temporal point probes
        int number_temporal_point_probes = 0;			/// Number of temporal point probes (default is zero)	
	std::vector<double> tpp_x_positions;			/// Positions in x-direction of temporal point probes [m]			
	std::vector<double> tpp_y_positions;			/// Positions in y-direction of temporal point probes [m]			
	std::vector<double> tpp_z_positions;			/// Positions in z-direction of temporal point probes [m]			
	std::vector<int> tpp_output_frequency_iters;		/// Output frequency iterations of temporal point probes
	std::vector<std::string> tpp_output_file_names;		/// Output file names of temporal point probes
 
        /// Lagrangian point particles
        bool activate_pure_tracer_particles;			/// Activate pure tracer particles
        bool activate_two_way_coupling_particles;		/// Activate two-way coupling particles
	double buffer_ratio_particles;				/// Buffer ratio of allocated particles per MPI task; e.g., 2, 3, 5
        int output_frequency_iter_particles;			/// Particles data output iteration frequency
        bool use_restart_particles;				/// Particles use restart file for initialization
	std::string restart_data_file_particles;		/// Particles restart data file
	int number_particles_local_in_use = 0;			/// Number of particles in use for each task

        /// Timers information
        bool print_timers;					/// Print timers information
	std::string timers_information_file;			/// Timers information file

        /// Parallelization scheme
        int np_x;						/// Number of processes in x-direction
        int np_y;						/// Number of processes in y-direction
        int np_z;						/// Number of processes in z-direction

	////////// SOLVER (PARALLEL) VARIABLES //////////
	
        /// Mesh coordinates (input/output data)
        DistributedArray x_field;				/// 3-D field of x-coordinate
        DistributedArray y_field;				/// 3-D field of y-coordinate
        DistributedArray z_field;				/// 3-D field of z-coordinate

        /// Primitive, conserved, thermodynamic and thermophysical variables
        DistributedArray rho_field;				/// 3-D field of rho
        DistributedArray u_field;				/// 3-D field of u
        DistributedArray v_field;				/// 3-D field of v
        DistributedArray w_field;				/// 3-D field of w
        DistributedArray E_field;				/// 3-D field of E
        DistributedArray s_field;				/// 3-D field of s
        DistributedArray rhou_field;				/// 3-D field of rhou
        DistributedArray rhov_field;				/// 3-D field of rhov
        DistributedArray rhow_field;				/// 3-D field of rhow
        DistributedArray rhoE_field;				/// 3-D field of rhoE
        DistributedArray P_field;				/// 3-D field of P
        DistributedArray T_field;				/// 3-D field of T
        DistributedArray sos_field;				/// 3-D field of sos
        DistributedArray mu_field;				/// 3-D field of mu
        DistributedArray kappa_field;				/// 3-D field of kappa
        DistributedArray c_v_field;				/// 3-D field of c_v
        DistributedArray c_p_field;				/// 3-D field of c_p

        /// Time-integration variables
        DistributedArray rho_0_field;				/// 3-D previous field of rho
        DistributedArray rhou_0_field;				/// 3-D previous field of rhou
        DistributedArray rhov_0_field;				/// 3-D previous field of rhov
        DistributedArray rhow_0_field;				/// 3-D previous field of rhow
        DistributedArray rhoE_0_field;				/// 3-D previous field of rhoE
        DistributedArray P_0_field;				/// 3-D previous field of P

        /// Inviscid fluxes
        DistributedArray rho_inv_flux;				/// 3-D inviscid fluxes of rho
        DistributedArray rhou_inv_flux;				/// 3-D inviscid fluxes of rhou
        DistributedArray rhov_inv_flux;				/// 3-D inviscid fluxes of rhov
        DistributedArray rhow_inv_flux;				/// 3-D inviscid fluxes of rhow
        DistributedArray rhoE_inv_flux;				/// 3-D inviscid fluxes of rhoE
        DistributedArray P_inv_flux;				/// 3-D inviscid fluxes of P

        /// Viscous fluxes
        DistributedArray rhou_vis_flux;				/// 3-D viscous fluxes of rhou
        DistributedArray rhov_vis_flux;				/// 3-D viscous fluxes of rhov
        DistributedArray rhow_vis_flux;				/// 3-D viscous fluxes of rhow
        DistributedArray rhoE_vis_flux;				/// 3-D viscous fluxes of rhoE
        DistributedArray work_vis_rhoe_flux;			/// 3-D viscous fluxes of work rhoe
        DistributedArray P_vis_flux;				/// 3-D viscous fluxes of P

        /// Source terms
        DistributedArray f_rhou_field;				/// 3-D field of f_rhou
        DistributedArray f_rhov_field;				/// 3-D field of f_rhov
        DistributedArray f_rhow_field;				/// 3-D field of f_rhow
        DistributedArray f_rhoE_field;				/// 3-D field of f_rhoE

        /// Time averaging
        DistributedArray avg_rho_field;				/// 3-D field of time-averaged rho
        DistributedArray avg_rhou_field;			/// 3-D field of time-averaged rhou
        DistributedArray avg_rhov_field;			/// 3-D field of time-averaged rhov
        DistributedArray avg_rhow_field;			/// 3-D field of time-averaged rhow
        DistributedArray avg_rhoE_field;			/// 3-D field of time-averaged rhoE
        DistributedArray avg_rhoP_field;			/// 3-D field of time-averaged rhoP
        DistributedArray avg_rhoT_field;			/// 3-D field of time-averaged rhoT
        DistributedArray avg_u_field;				/// 3-D field of time-averaged u
        DistributedArray avg_v_field;				/// 3-D field of time-averaged v
        DistributedArray avg_w_field;				/// 3-D field of time-averaged w
        DistributedArray avg_E_field;				/// 3-D field of time-averaged E
        DistributedArray avg_s_field;				/// 3-D field of time-averaged s
        DistributedArray avg_P_field;				/// 3-D field of time-averaged P
        DistributedArray avg_T_field;				/// 3-D field of time-averaged T
        DistributedArray avg_sos_field;				/// 3-D field of time-averaged sos
        DistributedArray avg_mu_field;				/// 3-D field of time-averaged mu
        DistributedArray avg_kappa_field;			/// 3-D field of time-averaged kappa
        DistributedArray avg_c_v_field;				/// 3-D field of time-averaged c_v
        DistributedArray avg_c_p_field;				/// 3-D field of time-averaged c_p
        DistributedArray rmsf_rho_field;			/// 3-D field of root-mean-square-fluctuation rho
        DistributedArray rmsf_rhou_field;			/// 3-D field of root-mean-square-fluctuation rhou
        DistributedArray rmsf_rhov_field;			/// 3-D field of root-mean-square-fluctuation rhov
        DistributedArray rmsf_rhow_field;			/// 3-D field of root-mean-square-fluctuation rhow
        DistributedArray rmsf_rhoE_field;			/// 3-D field of root-mean-square-fluctuation rhoE
        DistributedArray rmsf_u_field;				/// 3-D field of root-mean-square-fluctuation u
        DistributedArray rmsf_v_field;				/// 3-D field of root-mean-square-fluctuation v
        DistributedArray rmsf_w_field;				/// 3-D field of root-mean-square-fluctuation w
        DistributedArray rmsf_E_field;				/// 3-D field of root-mean-square-fluctuation E
        DistributedArray rmsf_s_field;				/// 3-D field of root-mean-square-fluctuation s
        DistributedArray rmsf_P_field;				/// 3-D field of root-mean-square-fluctuation P
        DistributedArray rmsf_T_field;				/// 3-D field of root-mean-square-fluctuation T
        DistributedArray rmsf_sos_field;			/// 3-D field of root-mean-square-fluctuation sos
        DistributedArray rmsf_mu_field;				/// 3-D field of root-mean-square-fluctuation mu
        DistributedArray rmsf_kappa_field;			/// 3-D field of root-mean-square-fluctuation kappa
        DistributedArray rmsf_c_v_field;			/// 3-D field of root-mean-square-fluctuation c_v
        DistributedArray rmsf_c_p_field;			/// 3-D field of root-mean-square-fluctuation c_p
        DistributedArray favre_uffuff_field;			/// 3-D field of Favre-averaged widetilde{u''u''} = overline{rhou''u''}/bar{rho}
        DistributedArray favre_uffvff_field;			/// 3-D field of Favre-averaged widetilde{u''v''} = overline{rhou''v''}/bar{rho}
        DistributedArray favre_uffwff_field;			/// 3-D field of Favre-averaged widetilde{u''w''} = overline{rhou''w''}/bar{rho}
        DistributedArray favre_vffvff_field;			/// 3-D field of Favre-averaged widetilde{v''v''} = overline{rhov''v''}/bar{rho}
        DistributedArray favre_vffwff_field;			/// 3-D field of Favre-averaged widetilde{v''w''} = overline{rhov''w''}/bar{rho}
        DistributedArray favre_wffwff_field;			/// 3-D field of Favre-averaged widetilde{w''w''} = overline{rhow''w''}/bar{rho}
        DistributedArray favre_uffEff_field;			/// 3-D field of Favre-averaged widetilde{u''E''} = overline{rhou''E''}/bar{rho}
        DistributedArray favre_vffEff_field;			/// 3-D field of Favre-averaged widetilde{v''E''} = overline{rhov''E''}/bar{rho}
        DistributedArray favre_wffEff_field;			/// 3-D field of Favre-averaged widetilde{w''E''} = overline{rhow''E''}/bar{rho}
        
        /// Immersed boundary method (IBM)
        DistributedArray tag_IBM_field;				/// 3-D field of IBM tag
        DistributedArray u_IBM_field;				/// 3-D field of IBM u-velocity
        DistributedArray v_IBM_field;				/// 3-D field of IBM v-velocity
        DistributedArray w_IBM_field;				/// 3-D field of IBM w-velocity
        DistributedArray T_IBM_field;				/// 3-D field of IBM temperature

	// Added for cylindrical coordinates
	DistributedArray det_Jacobian_field;              /// 3-D field of the determinant of Jacobian

	////////// THERMODYNAMIC MODEL, TRANSPORT COEFFICIENTS, RIEMANN SOLVER, EXPLICIT RUNGE-KUTTA METHOD //////////
	////////// IMMERSED BOUNDARY METHOD, COMPUTATIONAL DOMAIN, PARALLEL TOPOLOGY, WRITER/READER, PARALLEL TIMER //////////
        //BaseThermodynamicModel *thermodynamics;		/// Thermodynamic model
        _ThermodynamicModel_ *thermodynamics;			/// Thermodynamic model ... modified for OpenACC
        //BaseTransportCoefficients *transport_coefficients;	/// Transport coefficients
        _TransportCoefficientsModel_ *transport_coefficients;	/// Transport coefficients ... modified for OpenACC
        //BaseRiemannSolver *riemann_solver;			/// Riemann solver
        _RiemannSolverScheme_ *riemann_solver;			/// Riemann solver ... modified for OpenACC
        //BaseExplicitRungeKuttaMethod *runge_kutta_method;	/// Runge-Kutta method
        _ExplicitRungeKuttaMethod_ *runge_kutta_method;		/// Runge-Kutta method ... modified for OpenACC
        //BaseImmersedBoundaryMethod *immersed_boundary_method;	/// Immersed boundary method
        _ImmersedBoundaryMethod_ *immersed_boundary_method;	/// Immersed boundary method ... modified for OpenACC
        ComputationalDomain *mesh;				/// Computational domain
        ParallelTopology *topo;					/// Parallel topology
        DistributedPointParticles *point_particles;		/// Distributed point particles
        WriteReadHDF5 *writer_reader;				/// HDF5 data writer/reader
        ParallelTimer *timers;					/// Parallel timer

	
	////////// VERSION STUFF //////////
        /// Version numbers consist of three numbers separated by dots; i.e., 1.2.3.
        /// The leftmost number (1) is the major version.
        /// The middle number (2) is the minor version.
        /// The rightmost number (3) is the revision, but it may also refer to a "point release" or "subminor version".
	
        /// Version number (updated 10/09/2026)
	std::string version_number = "9.0.0";			/// Version number	

    private:

};

////////// BaseRiemannSolver CLASS //////////
class BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        BaseRiemannSolver();					/// Default constructor
        virtual ~BaseRiemannSolver();				/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Calculate waves speed
        //virtual void calculateWavesSpeed(double &S_L, double &S_R, const double &rho_L, const double &rho_R, const double &u_L, const double &u_R, const double &P_L, const double &P_R, const double &a_L, const double &a_R);
	#pragma acc routine
        void calculateWavesSpeed(double &S_L, double &S_R, const double &rho_L, const double &rho_R, const double &u_L, const double &u_R, const double &P_L, const double &P_R, const double &a_L, const double &a_R);	// ... modified for OpenACC

        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
        //virtual double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) = 0;
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type ) { return 0.0; };	// ... modified for OpenACC

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// DivergenceFluxApproximateRiemannSolver CLASS //////////
class DivergenceFluxApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        DivergenceFluxApproximateRiemannSolver();						/// Default constructor
        virtual ~DivergenceFluxApproximateRiemannSolver();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// MurmanRoeFluxApproximateRiemannSolver CLASS //////////
class MurmanRoeFluxApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        MurmanRoeFluxApproximateRiemannSolver();					/// Default constructor
        virtual ~MurmanRoeFluxApproximateRiemannSolver();				/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// KgpFluxApproximateRiemannSolver CLASS //////////
class KgpFluxApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        KgpFluxApproximateRiemannSolver();						/// Default constructor
        virtual ~KgpFluxApproximateRiemannSolver();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
     
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// ShimaFluxApproximateRiemannSolver CLASS //////////
class ShimaFluxApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        ShimaFluxApproximateRiemannSolver();						/// Default constructor
        virtual ~ShimaFluxApproximateRiemannSolver();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// HllApproximateRiemannSolver CLASS //////////
class HllApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        HllApproximateRiemannSolver();							/// Default constructor
        virtual ~HllApproximateRiemannSolver();						/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
      
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// HllcApproximateRiemannSolver CLASS //////////
class HllcApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        HllcApproximateRiemannSolver();							/// Default constructor
        virtual ~HllcApproximateRiemannSolver();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// HllcPlusApproximateRiemannSolver CLASS //////////
class HllcPlusApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        HllcPlusApproximateRiemannSolver();					/// Default constructor
        virtual ~HllcPlusApproximateRiemannSolver();				/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// EckepFluxApproximateRiemannSolver CLASS //////////
class EckepFluxApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        EckepFluxApproximateRiemannSolver();						/// Default constructor
        virtual ~EckepFluxApproximateRiemannSolver();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// HesFluxApproximateRiemannSolver CLASS //////////
class HesFluxApproximateRiemannSolver : public BaseRiemannSolver {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        HesFluxApproximateRiemannSolver();						/// Default constructor
        virtual ~HesFluxApproximateRiemannSolver();					/// Destructor

	////////// GET FUNCTIONS //////////

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Calculate intercell flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
	#pragma acc routine
        double calculateIntercellFlux( const double &rho_LLL, const double &u_LLL, const double &v_LLL, const double &w_LLL, const double &E_LLL, const double &s_LLL, const double &P_LLL, const double &T_LLL, const double &a_LLL, const double &rho_LL, const double &u_LL, const double &v_LL, const double &w_LL, const double &E_LL, const double &s_LL, const double &P_LL, const double &T_LL, const double &a_LL, const double &rho_L, const double &u_L, const double &v_L, const double &w_L, const double &E_L, const double &s_L, const double &P_L, const double &T_L, const double &a_L, const double &rho_R, const double &u_R, const double &v_R, const double &w_R, const double &E_R, const double &s_R, const double &P_R, const double &T_R, const double &a_R, const double &rho_RR, const double &u_RR, const double &v_RR, const double &w_RR, const double &E_RR, const double &s_RR, const double &P_RR, const double &T_RR, const double &a_RR, const double &rho_RRR, const double &u_RRR, const double &v_RRR, const double &w_RRR, const double &E_RRR, const double &s_RRR, const double &P_RRR, const double &T_RRR, const double &a_RRR, const double &delta, const int &var_type );

    protected:

        ////////// PARAMETERS //////////

    private:

};

////////// BaseExplicitRungeKuttaMethod CLASS //////////
class BaseExplicitRungeKuttaMethod {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        BaseExplicitRungeKuttaMethod();				/// Default constructor
        virtual ~BaseExplicitRungeKuttaMethod();		/// Destructor

	////////// GET FUNCTIONS //////////
	//virtual inline int getNumberRungeKuttaStages() { return( number_runge_kutta_stages ); };
	inline int getNumberRungeKuttaStages() { return( number_runge_kutta_stages ); };	// ... modified for OpenACC

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
       
        /// Set stage coefficients: y_rk+1 = rk_a*y_n + rk_b*y_rk + rk_c*delta_t*f( y_rk )
	#pragma acc routine
        //virtual void setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage);
        void setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage) {};	// ... modified for OpenACC

    protected:

        ////////// PARAMETERS //////////
        int number_runge_kutta_stages = 0;

    private:

};

////////// RungeKutta1Method CLASS //////////
class RungeKutta1Method : public BaseExplicitRungeKuttaMethod {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        RungeKutta1Method();				/// Default constructor
        virtual ~RungeKutta1Method();			/// Destructor

	////////// GET FUNCTIONS //////////
	inline int getNumberRungeKuttaStages() { return( number_runge_kutta_stages ); };

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Set stage coefficients: y_rk+1 = rk_a*y_n + rk_b*y_rk + rk_c*delta_t*f( y_rk )
	#pragma acc routine
        void setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage);

    protected:

        ////////// PARAMETERS //////////
        int number_runge_kutta_stages = 1;

    private:

};

////////// StrongStabilityPreservingRungeKutta2Method CLASS //////////
class StrongStabilityPreservingRungeKutta2Method : public BaseExplicitRungeKuttaMethod {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        StrongStabilityPreservingRungeKutta2Method();				/// Default constructor
        virtual ~StrongStabilityPreservingRungeKutta2Method();			/// Destructor

	////////// GET FUNCTIONS //////////
	inline int getNumberRungeKuttaStages() { return( number_runge_kutta_stages ); };

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Set stage coefficients: y_rk+1 = rk_a*y_n + rk_b*y_rk + rk_c*delta_t*f( y_rk )
	#pragma acc routine
        void setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage);

    protected:

        ////////// PARAMETERS //////////
        int number_runge_kutta_stages = 2;

    private:

};

////////// StrongStabilityPreservingRungeKutta3Method CLASS //////////
class StrongStabilityPreservingRungeKutta3Method : public BaseExplicitRungeKuttaMethod {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        StrongStabilityPreservingRungeKutta3Method();				/// Default constructor
        virtual ~StrongStabilityPreservingRungeKutta3Method();			/// Destructor

	////////// GET FUNCTIONS //////////
	inline int getNumberRungeKuttaStages() { return( number_runge_kutta_stages ); };

	////////// SET FUNCTIONS //////////

	////////// METHODS //////////
        
        /// Set stage coefficients: y_rk+1 = rk_a*y_n + rk_b*y_rk + rk_c*delta_t*f( y_rk )
	#pragma acc routine
        void setStageCoefficients(double &rk_a, double &rk_b, double &rk_c, const int &rk_time_stage);

    protected:

        ////////// PARAMETERS //////////
        int number_runge_kutta_stages = 3;

    private:

};

#endif /*_RHEA_FLOW_SOLVER_*/
