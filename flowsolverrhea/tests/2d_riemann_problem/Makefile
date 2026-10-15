EXECUTABLE   = RHEA.exe
MAIN         = myRHEA.cpp
CONFIG_FILE  = configuration_file.yaml
PROJECT_PATH = $(RHEA_PATH)
SRC_DIR      = $(PROJECT_PATH)/src
# COOLPROP PATHS & EXTERNALS
CP_DIR       = $(PROJECT_PATH)/externals/CoolProp
CP_EXT_DIR   = $(CP_DIR)/externals
CP_LIB_FILE  = $(CP_DIR)/libcoolprop.a
CP_SOURCES   = $(wildcard $(CP_DIR)/src/*.cpp)
CP_INC       = -I$(CP_DIR) \
               -I$(CP_DIR)/include \
               -I$(CP_DIR)/src \
               -I$(CP_EXT_DIR)/fmtlib/include \
               -I$(CP_EXT_DIR)/Eigen \
               -I$(CP_EXT_DIR)/rapidjson/include \
               -I$(CP_EXT_DIR)/nlohmann-json \
               -I$(CP_EXT_DIR)/msgpack-c/include \
               -I$(CP_EXT_DIR)/IF97
# COMPILER
CXX          = mpicxx.openmpi
# CPU FLAGS
CXXFLAGS     = -Ofast -Wall -std=c++17 -Wno-unknown-pragmas -Wno-unused-variable -I$(PROJECT_PATH) $(CP_INC) -DFMT_HEADER_ONLY -DCOOLPROP_LIB -include $(PROJECT_PATH)/externals/boost_fix.h
## CPU-GPU FLAGS
#CXXFLAGS     = -fast -acc -gpu=cc75,mem:separate:pinnedalloc -Minfo=accel -O3 -Wall -std=c++17 -I$(PROJECT_PATH) $(CP_INC) -DFMT_HEADER_ONLY -DCOOLPROP_LIB -include $(PROJECT_PATH)/externals/boost_fix.h
# CUDA_AWARE MPI FLAG
CXXFLAGS += -D_GPU_AWARE_MPI_DEACTIVATED_=1
# UBUNTU - LINUX
INC_LIB_YAML =
INC_DIR_YAML =
INC_LIB_HDF5 = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 
INC_DIR_HDF5 = -I/usr/include/hdf5/openmpi
## MAC - OS X
#INC_LIB_YAML = -L/usr/local/lib
#INC_DIR_YAML = -I/usr/local/include
#INC_LIB_HDF5 =
#INC_DIR_HDF5 =
LDFLAGS      = -lyaml-cpp -lhdf5 -lpthread



# !! THE LINES BELOW SHOULD NOT BE MODIFIED !! #

OBJS = $(wildcard $(SRC_DIR)/*.cpp)
INC_LIB = $(INC_LIB_YAML) $(INC_LIB_HDF5)
INC_DIR = $(INC_DIR_YAML) $(INC_DIR_HDF5)

all: preprocess $(CP_LIB_FILE) build postprocess

# ! OPENACC STUFF TO AVOID PROBLEMS WITH POINTERS ... CAN BE REMOVED ONCE OPENACC ACCEPTS POINTERS WITH DERIVED CLASSES ! #
EXTRACTED_THERMODYNAMIC_MODEL := $(shell grep "thermodynamic_model" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")
THERMODYNAMIC_MODEL = dummy
ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), IDEAL_GAS)
	THERMODYNAMIC_MODEL = IdealGasThermodynamicModel
else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), STIFFENED_GAS)
	THERMODYNAMIC_MODEL = StiffenedGasThermodynamicModel
else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), PENG_ROBINSON)
	THERMODYNAMIC_MODEL = PengRobinsonThermodynamicModel
else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), COOLPROP)
	THERMODYNAMIC_MODEL = CoolPropThermodynamicModel
else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), SVD_SURROGATE)
	THERMODYNAMIC_MODEL = SvdSurrogateThermodynamicModel	
endif
EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL := $(shell grep "transport_coefficients_model" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")
TRANSPORT_COEFFICIENTS_MODEL = dummy
ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), CONSTANT)
        TRANSPORT_COEFFICIENTS_MODEL = ConstantTransportCoefficients
else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), LOW_PRESSURE_GAS)
        TRANSPORT_COEFFICIENTS_MODEL = LowPressureGasTransportCoefficients
else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), HIGH_PRESSURE)
        TRANSPORT_COEFFICIENTS_MODEL = HighPressureTransportCoefficients
else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), COOLPROP)
        TRANSPORT_COEFFICIENTS_MODEL = CoolPropTransportCoefficients
else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), SVD_SURROGATE)
        TRANSPORT_COEFFICIENTS_MODEL = SvdSurrogateTransportCoefficients	
endif
EXTRACTED_RIEMANN_SOLVER_SCHEME := $(shell grep "riemann_solver_scheme" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")
RIEMANN_SOLVER_SCHEME = dummy
ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), DIVERGENCE)
        RIEMANN_SOLVER_SCHEME = DivergenceFluxApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), MURMAN-ROE)
        RIEMANN_SOLVER_SCHEME = MurmanRoeFluxApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), KGP)
        RIEMANN_SOLVER_SCHEME = KgpFluxApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), SHIMA)
        RIEMANN_SOLVER_SCHEME = ShimaFluxApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), HLL)
        RIEMANN_SOLVER_SCHEME = HllApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), HLLC)
        RIEMANN_SOLVER_SCHEME = HllcApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), HLLC+)
        RIEMANN_SOLVER_SCHEME = HllcPlusApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), ECKEP)
        RIEMANN_SOLVER_SCHEME = EckepFluxApproximateRiemannSolver
else ifeq ($(EXTRACTED_RIEMANN_SOLVER_SCHEME), HES)
        RIEMANN_SOLVER_SCHEME = HesFluxApproximateRiemannSolver
endif
EXTRACTED_RUNGE_KUTTA_METHOD := $(shell grep "runge_kutta_time_scheme" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")
EXPLICIT_RUNGE_KUTTA_METHOD = dummy
ifeq ($(EXTRACTED_RUNGE_KUTTA_METHOD), RK1)
	EXPLICIT_RUNGE_KUTTA_METHOD = RungeKutta1Method
else ifeq ($(EXTRACTED_RUNGE_KUTTA_METHOD), SSP-RK2)
	EXPLICIT_RUNGE_KUTTA_METHOD = StrongStabilityPreservingRungeKutta2Method
else ifeq ($(EXTRACTED_RUNGE_KUTTA_METHOD), SSP-RK3)
	EXPLICIT_RUNGE_KUTTA_METHOD = StrongStabilityPreservingRungeKutta3Method
endif
#EXTRACTED_IMMERSED_BOUNDARY_METHOD := $(shell grep "immersed_boundary_method" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")
#IMMERSED_BOUNDARY_METHOD = dummy
IMMERSED_BOUNDARY_METHOD = BaseImmersedBoundaryMethod

# ! OPENACC STUFF TO AVOID PROBLEMS WITH POINTERS ... CAN BE REMOVED ONCE OPENACC ACCEPTS POINTERS WITH DERIVED CLASSES ! #
preprocess:
	@cp $(SRC_DIR)/FlowSolverRHEA.hpp $(SRC_DIR)/FlowSolverRHEA.hpp.original
	@cp $(SRC_DIR)/FlowSolverRHEA.cpp $(SRC_DIR)/FlowSolverRHEA.cpp.original
	@for file in $(SRC_DIR)/FlowSolverRHEA.*pp; do \
		sed -i 's/\b_ThermodynamicModel_\b/$(THERMODYNAMIC_MODEL)/g' $$file; \
		sed -i 's/\b_TransportCoefficientsModel_\b/$(TRANSPORT_COEFFICIENTS_MODEL)/g' $$file; \
		sed -i 's/\b_RiemannSolverScheme_\b/$(RIEMANN_SOLVER_SCHEME)/g' $$file; \
		sed -i 's/\b_ExplicitRungeKuttaMethod_\b/$(EXPLICIT_RUNGE_KUTTA_METHOD)/g' $$file; \
		sed -i 's/\b_ImmersedBoundaryMethod_\b/$(IMMERSED_BOUNDARY_METHOD)/g' $$file; \
	done

# ! BUILD THE COOLPROP STATIC LIBRARY AUTOMATICALLY ! #
$(CP_LIB_FILE):
	@echo "Applying brute-force path fixes to CoolProp..."
	@mkdir -p $(CP_DIR)/build
	# Physically copy Backends into include if not already there 
	# (This fixes the 'Backends/Helmholtz/...' not found error)
	@cp -rn $(CP_DIR)/src/Backends $(CP_DIR)/include/ 2>/dev/null || true
	# Run the standard CMake build
	@cd $(CP_DIR)/build && cmake .. \
		-DCOOLPROP_STATIC_LIBRARY=ON \
		-DCMAKE_BUILD_TYPE=Release
	@cd $(CP_DIR)/build && cmake --build . --config Release -- -j$$(( $$(nproc) > 4 ? 4 : $$(nproc) ))
	# Move the resulting library to the expected filename if CMake named it libCoolProp.a
	@cp $(CP_DIR)/build/libCoolProp.a $(CP_LIB_FILE) 2>/dev/null || true
	@echo "CoolProp build completed successfully."

# ! BUILD RHEA ! #
build: $(CP_LIB_FILE)
	@echo "Building RHEA executable..."
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(MAIN) $(OBJS) $(INC_LIB) $(INC_DIR) $(CP_LIB_FILE) $(LDFLAGS)

# ! OPENACC STUFF TO AVOID PROBLEMS WITH POINTERS ... CAN BE REMOVED ONCE OPENACC ACCEPTS POINTERS WITH DERIVED CLASSES ! #
postprocess:
	@mv $(SRC_DIR)/FlowSolverRHEA.hpp.original $(SRC_DIR)/FlowSolverRHEA.hpp
	@mv $(SRC_DIR)/FlowSolverRHEA.cpp.original $(SRC_DIR)/FlowSolverRHEA.cpp

# ! STANDARD CLEAN: REMOVES THE RHEA EXECUTABLE & OBJECT FILES ! #
.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) *.o

# ! DEEP CLEAN: REMOVES EVERYTHING, INCLUDING THE COMPILED COOLPROP LIBRARY
.PHONY: distclean
distclean: clean
	$(RM) -rf $(CP_DIR)/build
	$(RM) $(CP_LIB_FILE)
