#/usr/bin/env python3
"""
svd_surrogate_modeling_thermophysical_properties.py — SVD thermophysical surrogate model for fluids
============================================================================

Generates a compact SVD-based surrogate model of a fluid's thermophysical
properties and writes it to a single human-readable .yaml file. For a given
substance, the script builds a uniform NxN grid and computes one SVD
per property, covering both the forward and inverse directions in a single run:

Forward direction (T, P) grid:
    rho, e, c_p, c_v, sos, s, volume_expansivity (alpha),
    isothermal_compressibility (kappa_T), isentropic_compressibility (kappa_S),
    conductivity (k), viscosity (mu)

Inverse direction (rho, e) grid:
    T, P

Mixed (rho, P) grid:
    T

Mixed (rho, T) grid:
    P

For an NxN grid, the maximum mathematically possible SVD rank for any of
these matrices is min(N, N) = N. By default, the script computes the full
SVD (rank N, exact reconstruction of the grid) for every property and
records the achieved rank in substance_SVD_rank. Truncation to a lower
rank is available via --rank.

T range and P range are provided directly on the command line via
--T_min/--T_max/--P_min/--P_max (required); FLUID_DATABASE holds the
substance name and CoolProp identifier.

Pipeline, applied uniformly to every property matrix:
    1. Generation of the property points (raw matrices from CoolProp)
    2. SVD via np.linalg.svd, directly on the raw matrix, optionally
       truncated to a given rank

Usage:
    python3 svd_surrogate_modeling_thermophysical_properties.py --substance CO2 --n 100 --T_min 315.0 --T_max 500.0 --P_min 50.00e5 --P_max 350e5 --output SVD_Surrogate_CO2.yaml
    python3 svd_surrogate_modeling_thermophysical_properties.py --substance N2 --rank 6 --T_min 100.0 --T_max 200.0 --P_min 30e5 --P_max 50e5 --output SVD_Surrogate.yaml
"""

import argparse
from dataclasses import dataclass
import numpy as np
import yaml
import CoolProp.CoolProp as CP


# ── Self-contained fluid registry ────────────────────────────────────────
@dataclass
class FluidMetadata:
    name: str
    coolprop_name: str


FLUID_DATABASE = {
    "CO2": FluidMetadata(
        name="CarbonDioxide",
        coolprop_name="CarbonDioxide",
    ),
    "N2": FluidMetadata(
        name="Nitrogen",
        coolprop_name="Nitrogen",
    ),
    "CH4": FluidMetadata(
        name="Methane",
        coolprop_name="Methane",
    ),
    "O2": FluidMetadata(
        name="Oxygen",
        coolprop_name="Oxygen",
    ),
}

# ── Forward properties: CoolProp full-name keys, in generation order ────────
PROPS = {
    'viscosity':          'viscosity',
    'conductivity':       'conductivity',
    'e':                  'Umass',
    's':                  'Smass',
    'rho':                'Dmass',
    'c_p':                'Cpmass',
    'c_v':                'Cvmass',
    'sos':                'speed_of_sound',
    'isobaric_expansion': 'isobaric_expansion_coefficient',
    'isothermal_compressibility': 'isothermal_compressibility',
}

# Mapping from our short property name -> the tag used in the YAML matrix
# key (rho, e, mu, kappa are named explicitly there; the rest follow the 
# same "T_P_to_<prop>" pattern). 
YAML_TAG = {
    'viscosity': 'mu', 'conductivity': 'kappa', 'e': 'e',
    's': 's', 'rho': 'rho', 'c_p': 'c_p', 'c_v': 'c_v', 'sos': 'sos',
    'volume_expansivity': 'alpha',
    'isothermal_compressibility': 'kappa_T', 'isentropic_compressibility': 'kappa_S',
}


# ── Step 1: generation helpers ───────────────────────────────────────────

def build_grid(low, high, n):
    return np.linspace(low, high, n)


def matrix_from_two_inputs(output_key, in1_name, in1_grid, in2_name, in2_grid, name):
    """M[i, j] = property(in1_name=in1_grid[i], in2_name=in2_grid[j]), NaN if the
    point falls outside CoolProp's valid single-phase table for this backend
    (filled in afterwards by mean of valid points so the SVD has no NaNs)."""
    M = np.full((len(in1_grid), len(in2_grid)), np.nan)
    for i, v1 in enumerate(in1_grid):
        for j, v2 in enumerate(in2_grid):
            try:
                M[i, j] = CP.PropsSI(output_key, in1_name, v1, in2_name, v2, name)
            except Exception:
                pass
    if np.isnan(M).any():
        # Fill any failed (two-phase / out-of-range) points with the mean
        # of their valid neighbours so the small demo grid stays dense.
        fill = np.nanmean(M)
        M[np.isnan(M)] = fill
    return M


# ── Step 2: SVD (optionally truncated) ───────────────────────────────────

def truncated_svd(M, rank=None):
    """SVD of M -> U (m x r), S (r,), V (n x r) such that M ~= U @ diag(S) @ V.T.
    If `rank` is None, all singular values are kept (full rank = min(M.shape)).
    Otherwise the result is truncated to `rank` (capped to min(M.shape))."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    full_rank = len(S)
    r = full_rank if rank is None else max(1, min(rank, full_rank))
    return U[:, :r], S[:r], Vt[:, :r]


def to_flat(a):
    return [float(x) for x in np.ravel(a)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--substance', type=str, default='CO2', choices=list(FLUID_DATABASE.keys()))
    ap.add_argument('--n', type=int, default=10, help='grid points per axis')
    ap.add_argument('--rank', type=int, default=None,
                     help='SVD rank to keep (default: None -> full rank = n, i.e. min(n,n) -- '
                          'exact reconstruction of the grid). Must be <= n; if you ask '
                          'for more than n it is silently capped to n, since a n x n '
                          'matrix cannot have more than n singular values.')
    ap.add_argument('--output', type=str, default='SVD_Surrogate.yaml')
    ap.add_argument('--T_min', type=float, required=True, help='minimum temperature [K]')
    ap.add_argument('--T_max', type=float, required=True, help='maximum temperature [K]')
    ap.add_argument('--P_min', type=float, required=True, help='minimum pressure [Pa]')
    ap.add_argument('--P_max', type=float, required=True, help='maximum pressure [Pa]')
    args = ap.parse_args()

    meta = FLUID_DATABASE[args.substance]
    name = meta.coolprop_name
    n = args.n

    # For the metadata section of the YAML file
    molecular_weight = CP.PropsSI('M', name)  # [kg/mol]

    T_min, T_max = args.T_min, args.T_max
    P_min, P_max = args.P_min, args.P_max

    if T_min >= T_max:
        raise ValueError(f"T_min ({T_min}) must be < T_max ({T_max})")
    if P_min >= P_max:
        raise ValueError(f"P_min ({P_min}) must be < P_max ({P_max})")

    rank = n if args.rank is None else max(1, min(args.rank, n))
    if args.rank is not None and args.rank > n:
        print(f"NOTE: --rank {args.rank} requested but the grid is only {n}x{n}, "
              f"so a matrix here cannot have more than {n} singular values. "
              f"Capping rank to {n}.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Generation of the property points
    # All raw matrices (forward, inverse, and the two mixed-input ones)
    # are collected together into a single dict, keyed by the same name
    # that will be used to build the YAML matrix keys later on.
    # ══════════════════════════════════════════════════════════════════
    T_grid = build_grid(T_min, T_max, n)
    P_grid = build_grid(P_min, P_max, n)

    print(f"[1/3] Generating property points: forward grid {n}x{n} = {n*n} points, "
          f"T in [{T_min}, {T_max}] K, P in [{P_min}, {P_max}] Pa")

    raw_matrices = {}

    for prop, key in PROPS.items():
        print(f"      computing T_P_to_{YAML_TAG.get(prop, prop)} ...")
        raw_matrices[f'T_P_to_{prop}'] = matrix_from_two_inputs(key, 'T', T_grid, 'P', P_grid, name)

    # ── Derived forward property (no direct CoolProp key -- computed from
    # matrices already fetched above)
    print("      computing T_P_to_isentropic_compressibility (kappa_T / (c_p/c_v)) ...")
    _gamma = raw_matrices['T_P_to_c_p'] / raw_matrices['T_P_to_c_v']
    raw_matrices['T_P_to_isentropic_compressibility'] = (
        raw_matrices['T_P_to_isothermal_compressibility'] / _gamma
    )

    # ── Build the (e, rho) grid for the inverse direction from the actual
    # rho/e values realised on the forward (T,P) grid ────────
    rho_vals = raw_matrices['T_P_to_rho']
    e_vals = raw_matrices['T_P_to_e']
    rho_min, rho_max = float(rho_vals.min()), float(rho_vals.max())
    e_min, e_max = float(e_vals.min()), float(e_vals.max())
    rho_grid = build_grid(rho_min, rho_max, n)
    e_grid = build_grid(e_min, e_max, n)

    print(f"      inverse grid: {n}x{n} = {n*n} points, "
          f"rho in [{rho_grid[0]:.3f}, {rho_grid[-1]:.3f}] kg/m3, "
          f"e in [{e_grid[0]:.3g}, {e_grid[-1]:.3g}] J/kg")

    raw_matrices['e_rho_to_T'] = matrix_from_two_inputs('T', 'Umass', e_grid, 'Dmass', rho_grid, name) 
    raw_matrices['e_rho_to_P'] = matrix_from_two_inputs('P', 'Umass', e_grid, 'Dmass', rho_grid, name)

    # ── Additional (rho, P) -> T and (rho, T) -> P matrices. These reuse
    # the rho_grid and the forward T_grid/P_grid built above for the
    # inverse (e, rho) direction, so all three grids stay consistent. ────
    print("      computing rho_P_to_T and rho_T_to_P ...")
    raw_matrices['rho_P_to_T'] = matrix_from_two_inputs('T', 'Dmass', rho_grid, 'P', P_grid, name)
    raw_matrices['rho_T_to_P'] = matrix_from_two_inputs('P', 'Dmass', rho_grid, 'T', T_grid,name)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — SVD (optionally truncated), applied uniformly to every
    # entry of raw_matrices, in the exact order that determines the
    # YAML layout below.
    # ══════════════════════════════════════════════════════════════════
    print(f"[2/3] Running SVD (rank {rank}) on every property matrix ...")

    svd_factors = {}
    for mat_key, M in raw_matrices.items():
        svd_factors[mat_key] = truncated_svd(M, rank)

    achieved_rank = rank

    # ── Assemble the single "substance_SVD_data" key that holds every
    # matrix's U, S, VT together (grids first, then rho_P/rho_T, then the
    # forward properties in the requested order, then the inverse ones). ─
    svd_data = {}
    svd_data['substance_P_vector'] = to_flat(P_grid)
    svd_data['substance_T_vector'] = to_flat(T_grid)
    svd_data['substance_rho_vector'] = to_flat(rho_grid)
    svd_data['substance_e_vector'] = to_flat(e_grid)

    def emit(yaml_name, mat_key):
        U, S, V = svd_factors[mat_key]
        svd_data[f'substance_{yaml_name}_matrix_U'] = to_flat(U)
        svd_data[f'substance_{yaml_name}_matrix_S'] = to_flat(S)
        svd_data[f'substance_{yaml_name}_matrix_VT'] = to_flat(V)

    # T(rho, P) and P(rho, T) matrices, written immediately after the
    # T/P/e/rho vectors and before the rest of the forward/inverse matrices.
    emit('rho_P_to_T', 'rho_P_to_T')
    emit('rho_T_to_P', 'rho_T_to_P')

    # Forward + inverse matrices, in this exact order: rho, e, s, c_p, c_v,
    # sos, volume_expansivity (alpha), isothermal_compressibility (kappa_T),
    # isentropic_compressibility (kappa_S), P (inverse), T (inverse),
    # viscosity (mu), conductivity (kappa).
    ordered_props = ['rho', 'e', 's', 'c_p', 'c_v', 'sos',
                      'volume_expansivity', 'isothermal_compressibility',
                      'isentropic_compressibility',
                      'P', 'T',
                      'viscosity', 'conductivity']

    # 'volume_expansivity' (output name / YAML tag) is generated under the
    # CoolProp-facing key 'isobaric_expansion' in PROPS.
    RAW_KEY_ALIAS = {'volume_expansivity': 'isobaric_expansion'}

    for prop in ordered_props:
        if prop in ('P', 'T'):
            emit(f'e_rho_to_{prop}', f'e_rho_to_{prop}')
        else:
            tag = YAML_TAG[prop]
            raw_prop = RAW_KEY_ALIAS.get(prop, prop)
            emit(f'T_P_to_{tag}', f'T_P_to_{raw_prop}')
    print(f"[3/3] Writing {args.output} ...")

    header = "##### SUBSTANCE METADATA #####"

    metadata = {
        'substance_metadata': {
            'substance_name': name,
            'substance_molecular_weight': float(molecular_weight),
            'substance_P_min': float(P_min),
            'substance_P_max': float(P_max),
            'substance_T_min': float(T_min),
            'substance_T_max': float(T_max),
            'substance_rho_min': float(rho_min),
            'substance_rho_max': float(rho_max),
            'substance_e_min': float(e_min),
            'substance_e_max': float(e_max),
            'substance_SVD_rank': achieved_rank,
        }
    }

    with open(args.output, 'w') as f:
        f.write(header + "\n")
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)
        f.write("\n\n##### SUBSTANCE SVD DATA #####\n")
        yaml.safe_dump({'substance_SVD_data': svd_data}, f, default_flow_style=None, sort_keys=False, width=1_000_000)

    print("Done.")
    print(f"T range: [{T_min}, {T_max}] K | P range: [{P_min}, {P_max}] Pa | "
          f"grid: {n}x{n} | SVD rank: {achieved_rank}/{n}")


if __name__ == '__main__':
    main()
