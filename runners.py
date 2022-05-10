import numpy as np

import refl_processing as rp
import hambuilding as hb
import mc
import anneal

def run(triplets, var_size, mc_iters, mc_runs=1, friedel=True, symmetric=True, \
    hamiltonian=None, weights=None):

    if hamiltonian is None:
        Jmat_orig, hvec_orig, ic_orig, refl_stats, int_to_refl_orig, \
            refl_to_int_orig = hb.structure_ham(var_size, triplets, \
            friedel=friedel, symmetric=symmetric, weights=weights, verbose=True)
        hamiltonian = (Jmat_orig, hvec_orig, ic_orig, refl_stats, \
            int_to_refl_orig, refl_to_int_orig)
    else:
        Jmat_orig, hvec_orig, ic_orig, refl_stats, int_to_refl_orig, \
            refl_to_int_orig = hamiltonian

    #Jmat, hvec, ic, int_to_refl, refl_to_int, toprefl, toprefl_vec = \
    #    hb.fix_variable(Jmat_orig, hvec_orig, ic_orig, refl_stats, \
    #    refl_to_int_orig)
    #refls_to_fix = hb.find_top_refls(refl_stats, 3)
    refls_to_fix = hb.find_fixable_refls(refl_stats)
    print(refls_to_fix)
    Jmat, hvec, ic, int_to_refl, refl_to_int, fixed_reflvecs = hb.fix_variables(Jmat_orig, hvec_orig, ic_orig, refl_stats, refl_to_int_orig, refls_to_fix)

    print(hvec.shape)

    total_qubits = len(hvec)
    total_vars = total_qubits // var_size

    runs = mc_runs

    verbose = False
    if runs == 1: verbose= True

    total_qubits = len(hvec)
    states, states_init, costss = [], [], []
    for j in range(runs):
        if runs > 1: print(f'Run {j+1} of {runs}   ', end='\r')
        input_state = mc.generate_state(total_qubits)
        state, costs = mc.mc(Jmat, hvec, ic, (var_size,)*total_vars, \
            input_state, mc_iters, 3.0, cost_freq=1, verbose=verbose)
        states.append(state)
        states_init.append(input_state)
        costss.append(costs)

    idx = np.argmin([c[-1] for c in costss])
    costs, state, input_state = costss[idx], states[idx], states_init[idx]
    rtf_starts = []
    for j, rtf in enumerate(refls_to_fix):
        rtf_starts.append(refl_to_int_orig[rtf]*var_size)
    rtf_starts = np.array(rtf_starts)
    state = rp.reduced_state_to_original(state, fixed_reflvecs, rtf_starts)
    input_state = rp.reduced_state_to_original(input_state, fixed_reflvecs, rtf_starts)

        #rtf, rtfvec = refl_to_fix, fixed_reflvecs[j]
        #rtf_start = refl_to_int_orig[rtf]*var_size
        #state = rp.reduced_state_to_original(state, rtfvec, rtf_start)
        #input_state = rp.reduced_state_to_original(input_state, rtfvec, \
        #    rtf_start)

    cost_check = mc.cost(state, Jmat_orig, hvec_orig, ic_orig)
    print(costs[-1], cost_check)
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

def run_simanneal(triplets, var_size, mc_minutes, mc_iters, mc_runs=1, friedel=True, \
    symmetric=True, hamiltonian=None):

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

    var_sizes = (var_size,)*total_vars

    class Solver(anneal.Annealer):
        def move(self):
            output = mc.update(var_sizes, self.state)
            self.state = output
        def energy(self):
            return mc.cost(self.state, Jmat, hvec, ic)
        if mc_runs > 1:
            def update(self, *args, **kwargs):
                pass

    input_state = mc.generate_state(total_qubits)
    solver = Solver(input_state)
    schedule = solver.auto(minutes=mc_minutes, steps=mc_iters)

    runs = mc_runs

    verbose = False
    if runs == 1: verbose = True

    total_qubits = len(hvec)
    states, costss = [], []
    for j in range(runs):
        if runs > 1: print(f'Run {j+1} of {runs}   ', end='\r')
        input_state = mc.generate_state(total_qubits)
        solver = Solver(input_state)
        solver.set_schedule(schedule)
        state, costs = solver.anneal()
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
