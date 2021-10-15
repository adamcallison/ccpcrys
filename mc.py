import numpy as np

def generate_state(total_qubits):
    state = np.random.default_rng().choice([-1, 1], size=total_qubits, replace=True)
    return state

def update(var_sizes, input_state):
    total_qubits = input_state.shape[0]
    assert sum(var_sizes) == total_qubits
    num_vars = len(var_sizes)
    choose_var = np.random.default_rng().integers(0, num_vars)

    substate_idxmin, substate_idxmax = np.sum(var_sizes[:choose_var], dtype=int), np.sum(var_sizes[:choose_var+1])
    substate = input_state[substate_idxmin:substate_idxmax]
    substate_bin = ''.join(['0' if x == 1.0 else '1' for x in substate][::-1])
    substate_int = int(substate_bin, 2)
    if substate_int == 0:
        new_substate_int = 1
    elif substate_int == (2**var_sizes[choose_var]) - 1:
        new_substate_int = (2**var_sizes[choose_var]) - 2
    else:
        change = np.random.default_rng().choice([-1, 1])
        new_substate_int = substate_int + change
    new_substate_bin = bin(new_substate_int)[2:]
    new_substate_bin = ('0'*(var_sizes[choose_var] - len(new_substate_bin))) + new_substate_bin
    new_substate = np.array([(1-(2*int(x))) for x in new_substate_bin[::-1]])

    output_state = np.array(input_state)
    output_state[substate_idxmin:substate_idxmax] = new_substate

    return output_state

def cost(state, Jmat, hvec, ic):
    return (state.dot(Jmat.dot(state))) + (hvec.dot(state)) + ic

def acceptreject(e_current, e_candidate):
    return e_candidate < e_current

def acceptrejectprob(e_current, e_candidate, probfunc):
    if e_candidate < e_current:
        return True
    rnd = np.random.default_rng().uniform()
    return rnd <= probfunc(e_current, e_candidate)

def boltzprob(e_current, e_candidate, T):
    if T == 0:
        return 0.0
    else:
        return np.exp(-(e_candidate-e_current)/T)

def mc(Jmat, hvec, ic, var_sizes, initial_state, iterations, T_start, cost_freq=1, verbose=False):
    e_current = cost(initial_state, Jmat, hvec, ic)
    state = initial_state
    costs = [e_current]
    for j in range(1, iterations+1):
        if verbose: print(f'Iteration {j} of {iterations}   ', end='\r')
        new_state = update(var_sizes, state)
        e_candidate = cost(new_state, Jmat, hvec, ic)
        T = T_start*(1-(j/iterations))
        pf = lambda ecur, ecand: boltzprob(ecur, ecand, T)
        if acceptrejectprob(e_current, e_candidate, pf):
            state = new_state
            e_current = e_candidate
        if (j == iterations) or (j%cost_freq == 0): costs.append(e_current)
    return state, costs
