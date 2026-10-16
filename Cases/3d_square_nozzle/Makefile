EXECUTABLE   = RHEA.exe
MAIN         = myRHEA.cpp
CONFIG_FILE  = configuration_file.yaml

# ============================================================
# PROJECT PATHS
# ============================================================

#PROJECT_PATH = $(RHEA_PATH_OPTI)
#SRC_DIR      = $(PROJECT_PATH)/src

PROJECT_PATH = /home/eckert/RHEA_GPU_optimization/RHEA/flowsolverrhea_NVTX_v2
SRC_DIR      = $(PROJECT_PATH)/src


# ============================================================
# COMPILATION OPTIONS
# ============================================================

# COOLPROP COMPILATION (for cpu)
COOLPROP_ACTIVATED_ = 1

# NVTX PROFILING
NVTX ?= 1

# CUDA-AWARE MPI FLAG
CXXFLAGS += -D_GPU_AWARE_MPI_DEACTIVATED_=0


# ============================================================
# COOLPROP PATHS & EXTERNALS
# ============================================================

CP_DIR       = $(PROJECT_PATH)/externals/CoolProp
CP_EXT_DIR   = $(CP_DIR)/externals
CP_LIB_FILE  = $(CP_DIR)/libcoolprop.a
CP_SOURCES   = $(wildcard $(CP_DIR)/src/*.cpp)

CP_INC       = -I$(CP_DIR) \
               -I$(CP_DIR)/include \
               -I$(CP_DIR)/src \
               -I$(CP_DIR)/src/Backends/Helmholtz \
               -I$(CP_EXT_DIR)/fmtlib/include \
               -I$(CP_EXT_DIR)/Eigen \
               -I$(CP_EXT_DIR)/rapidjson/include \
               -I$(CP_EXT_DIR)/nlohmann-json \
               -I$(CP_EXT_DIR)/msgpack-c/include \
               -I$(CP_EXT_DIR)/IF97


# ============================================================
# COMPILER
# ============================================================

CXX          = mpicxx.openmpi
#CXX          = mpicxx

# ============================================================
# COMPILATION FLAGS
# ============================================================

CXXFLAGS     = -Ofast -Wall -std=c++17 -Wno-unknown-pragmas -Wno-unused-variable -I$(PROJECT_PATH)

#CXXFLAGS     = -Ofast -Wall -std=c++17 -Wno-unused-variable -I$(PROJECT_PATH)

#CXXFLAGS     = -fast -acc -gpu=cc90,maxregcount:128 -Minfo=accel -O3 -Wall -std=c++17 -I$(PROJECT_PATH)



# ============================================================
# COOLPROP COMPILATION FLAGS
# ============================================================

CXXFLAGS += -DCOOLPROP_ACTIVATED_=$(COOLPROP_ACTIVATED_)

ifeq ($(COOLPROP_ACTIVATED_),1)

    # CoolProp include directories
    CXXFLAGS += $(CP_INC)

    # CoolProp compatibility flags
    CXXFLAGS += -DFMT_HEADER_ONLY
    CXXFLAGS += -DCOOLPROP_LIB

    # Boost compatibility header
    CXXFLAGS += -include $(PROJECT_PATH)/externals/boost_fix.h

    # CoolProp static library
    COOLPROP_LIB = $(CP_LIB_FILE)

else

    COOLPROP_LIB =

endif


# ============================================================
# NVTX COMPILATION FLAGS
# ============================================================

ifeq ($(NVTX),1)

    # NVIDIA HPC SDK 26.5 - MareNostrum 5

    NVTX_INC = /apps/ACC/NVIDIA-HPC-SDK/26.5/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/target-linux-x64/nvtx/include

    CXXFLAGS += -DUSE_NVTX -I$(NVTX_INC)

endif


# ============================================================
# UBUNTU - LINUX
# ============================================================

INC_LIB_YAML =
INC_DIR_YAML =

INC_LIB_HDF5 = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi
INC_DIR_HDF5 = -I/usr/include/hdf5/openmpi


# ============================================================
# MAC - OS X
# ============================================================

#INC_LIB_YAML = -L/usr/local/lib
#INC_DIR_YAML = -I/usr/local/include
#INC_LIB_HDF5 =
#INC_DIR_HDF5 =


# ============================================================
# LINKER FLAGS
# ============================================================

LDFLAGS = -lyaml-cpp -lhdf5 -lpthread

ifeq ($(NVTX),1)
    LDFLAGS += -ldl
endif


# ============================================================
# SOURCE FILES AND DIRECTORIES
# ============================================================

OBJS = $(wildcard $(SRC_DIR)/*.cpp)

INC_LIB = $(INC_LIB_YAML) $(INC_LIB_HDF5)
INC_DIR = $(INC_DIR_YAML) $(INC_DIR_HDF5)



# ============================================================
# MODEL SELECTION FROM CONFIGURATION FILE
# ============================================================

# ! OPENACC STUFF TO AVOID PROBLEMS WITH POINTERS ...
# CAN BE REMOVED ONCE OPENACC ACCEPTS POINTERS WITH DERIVED CLASSES ! #

# ============================================================
# THERMODYNAMIC MODEL
# ============================================================

EXTRACTED_THERMODYNAMIC_MODEL := $(shell grep "thermodynamic_model" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")

THERMODYNAMIC_MODEL = dummy

ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), IDEAL_GAS)
    THERMODYNAMIC_MODEL = IdealGasThermodynamicModel

else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), STIFFENED_GAS)
    THERMODYNAMIC_MODEL = StiffenedGasThermodynamicModel

else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), PENG_ROBINSON)
    THERMODYNAMIC_MODEL = PengRobinsonThermodynamicModel

else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), COOLPROP)

    ifeq ($(COOLPROP_ACTIVATED_),1)
        THERMODYNAMIC_MODEL = CoolPropThermodynamicModel
    else
        $(error CoolProp thermodynamic model selected but COOLPROP_ACTIVATED_=0)
    endif

else ifeq ($(EXTRACTED_THERMODYNAMIC_MODEL), SVD_SURROGATE)
    THERMODYNAMIC_MODEL = SvdSurrogateThermodynamicModel

endif


# ============================================================
# TRANSPORT COEFFICIENTS MODEL
# ============================================================

EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL := $(shell grep "transport_coefficients_model" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")

TRANSPORT_COEFFICIENTS_MODEL = dummy

ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), CONSTANT)
    TRANSPORT_COEFFICIENTS_MODEL = ConstantTransportCoefficients

else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), LOW_PRESSURE_GAS)
    TRANSPORT_COEFFICIENTS_MODEL = LowPressureGasTransportCoefficients

else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), HIGH_PRESSURE)
    TRANSPORT_COEFFICIENTS_MODEL = HighPressureTransportCoefficients

else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), COOLPROP)

    ifeq ($(COOLPROP_ACTIVATED_),1)
        TRANSPORT_COEFFICIENTS_MODEL = CoolPropTransportCoefficients
    else
        $(error CoolProp transport coefficients model selected but COOLPROP_ACTIVATED_=0)
    endif

else ifeq ($(EXTRACTED_TRANSPORT_COEFFICIENTS_MODEL), SVD_SURROGATE)
    TRANSPORT_COEFFICIENTS_MODEL = SvdSurrogateTransportCoefficients

endif


# ============================================================
# RIEMANN SOLVER SCHEME
# ============================================================

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


# ============================================================
# RUNGE-KUTTA METHOD
# ============================================================

EXTRACTED_RUNGE_KUTTA_METHOD := $(shell grep "runge_kutta_time_scheme" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")

EXPLICIT_RUNGE_KUTTA_METHOD = dummy

ifeq ($(EXTRACTED_RUNGE_KUTTA_METHOD), RK1)
    EXPLICIT_RUNGE_KUTTA_METHOD = RungeKutta1Method

else ifeq ($(EXTRACTED_RUNGE_KUTTA_METHOD), SSP-RK2)
    EXPLICIT_RUNGE_KUTTA_METHOD = StrongStabilityPreservingRungeKutta2Method

else ifeq ($(EXTRACTED_RUNGE_KUTTA_METHOD), SSP-RK3)
    EXPLICIT_RUNGE_KUTTA_METHOD = StrongStabilityPreservingRungeKutta3Method

endif


# ============================================================
# IMMERSED BOUNDARY METHOD
# ============================================================

#EXTRACTED_IMMERSED_BOUNDARY_METHOD := $(shell grep "immersed_boundary_method" $(CONFIG_FILE) | sed -n "s/.*'\([^']*\)'.*/\1/p")
#IMMERSED_BOUNDARY_METHOD = dummy

IMMERSED_BOUNDARY_METHOD = BaseImmersedBoundaryMethod


# ============================================================
# BUILD THE COOLPROP STATIC LIBRARY
# ============================================================

# This target is only required when COOLPROP_ACTIVATED_=1.

ifeq ($(COOLPROP_ACTIVATED_),1)

$(CP_LIB_FILE):
	@echo "Building CoolProp static library..."
	@mkdir -p $(CP_DIR)/build
	@cp -rn $(CP_DIR)/src/Backends $(CP_DIR)/include/ 2>/dev/null || true
	@cd $(CP_DIR)/build && cmake .. \
		-DCOOLPROP_STATIC_LIBRARY=ON \
		-DCMAKE_BUILD_TYPE=Release
	@cd $(CP_DIR)/build && cmake --build . --config Release -- -j$$(( $$(nproc) > 4 ? 4 : $$(nproc) ))
	@cp $(CP_DIR)/build/libCoolProp.a $(CP_LIB_FILE) 2>/dev/null || \
	  test -f $(CP_LIB_FILE)
	@test -f $(CP_LIB_FILE)
	@echo "CoolProp build completed successfully."

endif


# ============================================================
# BUILD RHEA
# ============================================================

# Preprocess, compile, and restore sources in a single recipe.
# This guarantees their order even with parallel Make.
# The shell trap restores the original files if compilation fails.

.PHONY: all build

all: build

build: $(COOLPROP_LIB)
	@echo "Building RHEA executable..."
	@set -e; \
	cp $(SRC_DIR)/FlowSolverRHEA.hpp $(SRC_DIR)/FlowSolverRHEA.hpp.original; \
	cp $(SRC_DIR)/FlowSolverRHEA.cpp $(SRC_DIR)/FlowSolverRHEA.cpp.original; \
	trap 'mv -f $(SRC_DIR)/FlowSolverRHEA.hpp.original $(SRC_DIR)/FlowSolverRHEA.hpp; \
	      mv -f $(SRC_DIR)/FlowSolverRHEA.cpp.original $(SRC_DIR)/FlowSolverRHEA.cpp' EXIT HUP INT TERM; \
	for file in $(SRC_DIR)/FlowSolverRHEA.hpp $(SRC_DIR)/FlowSolverRHEA.cpp; do \
		sed -i 's/\b_ThermodynamicModel_\b/$(THERMODYNAMIC_MODEL)/g' $$file; \
		sed -i 's/\b_TransportCoefficientsModel_\b/$(TRANSPORT_COEFFICIENTS_MODEL)/g' $$file; \
		sed -i 's/\b_RiemannSolverScheme_\b/$(RIEMANN_SOLVER_SCHEME)/g' $$file; \
		sed -i 's/\b_ExplicitRungeKuttaMethod_\b/$(EXPLICIT_RUNGE_KUTTA_METHOD)/g' $$file; \
		sed -i 's/\b_ImmersedBoundaryMethod_\b/$(IMMERSED_BOUNDARY_METHOD)/g' $$file; \
	done; \
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(MAIN) $(OBJS) \
	    $(INC_LIB) $(INC_DIR) $(COOLPROP_LIB) $(LDFLAGS)


# ============================================================
# CLEAN
# ============================================================

.PHONY: clean

clean:
	$(RM) $(EXECUTABLE) *.o


# ============================================================
# DEEP CLEAN
# ============================================================

.PHONY: distclean

distclean: clean
	$(RM) -rf $(CP_DIR)/build
	$(RM) $(CP_LIB_FILE)
