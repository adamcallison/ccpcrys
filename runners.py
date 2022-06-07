import numpy as np

import refl_processing as rp
import hambuilding as hb
import mc
import anneal

def run(triplets, var_size, mc_iters, mc_runs=1, friedel=True, symmetric=True, \
    hamiltonian=None, weights=None, stop_cost=None, fix=True):

    if hamiltonian is None:
        Jmat_orig, hvec_orig, ic_orig, refl_stats, int_to_refl_orig, \
            refl_to_int_orig = hb.structure_ham(var_size, triplets, \
            friedel=friedel, symmetric=symmetric, weights=weights, verbose=True)
        hamiltonian = (Jmat_orig, hvec_orig, ic_orig, refl_stats, \
            int_to_refl_orig, refl_to_int_orig)
    else:
        Jmat_orig, hvec_orig, ic_orig, refl_stats, int_to_refl_orig, \
            refl_to_int_orig = hamiltonian

    if fix:
        refls_to_fix = hb.find_fixable_refls(refl_stats)
    else:
        refls_to_fix = []
    #Jmat, hvec, ic, int_to_refl, refl_to_int, fixed_reflvecs = hb.fix_variables(Jmat_orig, hvec_orig, ic_orig, refl_stats, refl_to_int_orig, refls_to_fix)


    #total_qubits = len(hvec)
    #total_vars = total_qubits // var_size

    runs = mc_runs

    verbose = False
    if runs == 1: verbose= True

    #states, states_init, costss = [], [], []
    best_cost = float('inf')
    for j in range(runs):
        if runs > 1: print(f'Run {j+1} of {runs}. (Current best cost {best_cost})   ', end='\r')

        if fix:
            fixres = hb.fix_variables(Jmat_orig, hvec_orig, ic_orig, refl_stats, refl_to_int_orig, refls_to_fix)
        else:
            fixres = Jmat_orig, hvec_orig, ic_orig, int_to_refl_orig, refl_to_int_orig, []
        Jmat, hvec, ic = fixres[0], fixres[1], fixres[2]
        total_qubits = len(hvec)
        total_vars = total_qubits // var_size

        input_state = mc.generate_state(total_qubits)
        state, costs = mc.mc(Jmat, hvec, ic, (var_size,)*total_vars, \
            input_state, mc_iters, 3.0, cost_freq=1, verbose=verbose)
        #states.append(state)
        #states_init.append(input_state)
        #costss.append(costs)
        cost = costs[-1]
        if cost < best_cost:
            best_cost = cost
            best_state = state
            best_state_init = input_state
            best_costs = costs
            best_fixres = fixres
        if (not (stop_cost is None)) and (costs[-1] <= stop_cost):
            print(f"Breaking loop after {j+1} runs.                 ")
            break
        if j == runs-1:
            print(f"Loop after {j+1} runs.                 ") 

    Jmat, hvec, ic, int_to_refl, refl_to_int, fixed_reflvecs = best_fixres
    #idx = np.argmin([c[-1] for c in costss])
    #costs, state, input_state = costss[idx], states[idx], states_init[idx]
    costs, state, input_state = best_costs, best_state, best_state_init
    rtf_starts = []
    for j, rtf in enumerate(refls_to_fix):
        rtf_starts.append(refl_to_int_orig[rtf]*var_size)
    rtf_starts = np.array(rtf_starts)
    if fix:
        state = rp.reduced_state_to_original(state, fixed_reflvecs, rtf_starts)
        input_state = rp.reduced_state_to_original(input_state, fixed_reflvecs, rtf_starts)

    cost_check = mc.cost(state, Jmat_orig, hvec_orig, ic_orig)
    assert np.abs(costs[-1] - cost_check) < 1e-8
    cost_init = mc.cost(input_state, Jmat_orig, hvec_orig, ic_orig)

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

    unmapped_cost_old = np.sum(ang_sums**2)
    if weights is None:
        unmapped_cost = np.sum((ang_sums-(2*np.pi))**2)
        unmapped_cost_init = np.sum((ang_sums_init-(2*np.pi))**2)
    else:
        unmapped_cost = np.dot(weights,(ang_sums-(2*np.pi))**2)
        unmapped_cost_init = np.dot(weights, (ang_sums_init-(2*np.pi))**2)

    print(f'unmapped_cost={unmapped_cost}  ')
    print(f'unmapped_cost_init={unmapped_cost_init}  ')

    return costs, angles, angles_init, ang_sums, ang_sums_init, refls_to_fix, fixed_reflvecs, hamiltonian
