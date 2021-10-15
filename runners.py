import numpy as np

import refl_processing as rp
import hambuilding as hb
import mc

def run(triplets, var_size, mc_iters, mc_runs=1, friedel=True, symmetric=True, \
    hamiltonian=None):

    if hamiltonian is None:
        Jmat_orig, hvec_orig, ic_orig, refl_stats, int_to_refl_orig, \
            refl_to_int_orig = hb.structure_ham(var_size, triplets, \
            friedel=friedel, symmetric=symmetric, verbose=True)
        hamiltonian = (Jmat_orig, hvec_orig, ic_orig, refl_stats, \
            int_to_refl_orig, refl_to_int_orig)
    else:
        Jmat_orig, hvec_orig, ic_orig, refl_stats, int_to_refl_orig, \
            refl_to_int_orig = hamiltonian

    Jmat, hvec, ic, int_to_refl, refl_to_int, toprefl, toprefl_vec = \
        hb.fix_variable(Jmat_orig, hvec_orig, ic_orig, refl_stats, \
        refl_to_int_orig)

    total_qubits = len(hvec)
    total_vars = total_qubits // var_size

    runs = mc_runs

    verbose = False
    if runs == 1: verbose= True

    total_qubits = len(hvec)
    states, costss = [], []
    for j in range(runs):
        if runs > 1: print(f'Run {j+1} of {runs}   ', end='\r')
        input_state = mc.generate_state(total_qubits)
        state, costs = mc.mc(Jmat, hvec, ic, (var_size,)*total_vars, \
            input_state, mc_iters, 3.0, cost_freq=1, verbose=verbose)
        states.append(state)
        costss.append(costs)

    idx = np.argmin([c[-1] for c in costss])
    costs, state = costss[idx], states[idx]
    toprefl_start = refl_to_int_orig[toprefl]*var_size
    state = rp.reduced_state_to_original(state, toprefl_vec, toprefl_start)
    input_state = rp.reduced_state_to_original(input_state, toprefl_vec, \
        toprefl_start)

    cost_check = mc.cost(state, Jmat_orig, hvec_orig, ic_orig)
    assert np.abs(costs[-1] - cost_check) < 1e-8
    cost_init = mc.cost(input_state, Jmat_orig, hvec_orig, ic_orig)

    print(f'cost={costs[-1]}, cost_check={cost_check}, cost_init={cost_init}   ')

    num_refls = len(refl_stats)

    tmp = rp.decode_binary(state, (var_size,)*num_refls)
    tmp_init = rp.decode_binary(input_state, (var_size,)*num_refls)

    angles = rp.ints_to_angles(tmp, (var_size,)*num_refls, symmetric=symmetric)
    angles_init = rp.ints_to_angles(tmp_init, (var_size,)*num_refls, \
        symmetric=symmetric)

    ang_sums = rp.compute_angle_sums(angles, triplets, refl_to_int_orig, \
        friedel=friedel)
    ang_sums_init = rp.compute_angle_sums(angles_init, triplets, \
        refl_to_int_orig, friedel=friedel)

    unmapped_cost = np.sum(ang_sums**2)
    print(f'unmapped_cost={unmapped_cost}  ')

    return costs, angles, angles_init, ang_sums, ang_sums_init, hamiltonian
