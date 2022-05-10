import numpy as np
from copy import deepcopy as dc

import m2swork.hams as hams

def friedel_standard_old(r):
    rl = [int(x) for x in r.strip('[').strip(']').split(',')]
    if rl[0] < 0:
        return False
    elif rl[0] == 0:
        if rl[1] < 0:
            return False
        elif rl[1] == 0:
            if rl[2] < 0:
                return False

    return True

def friedel_standard(r):
    rl = [int(x) for x in r.strip('[').strip(']').split(',')]
    if (rl[0] == 0) and (rl[2] == 0):
        return rl[1] >= 0
    elif (rl[2] == 0):
        return rl[0] >= 0
    else:
        return rl[2] >= 0
    raise Exception("Shouldn't have reached here")

def friedel_standardise(r):
    if not friedel_standard(r):
        rl = [int(x) for x in r.strip('[').strip(']').split(',')]
        rl = [-1*x for x in rl]
        r = ''.join(str(rl).split())
    return r

def reflection_stats(triplets, friedel=False):
    refls = {}
    for t in triplets:
        for r in t:
            if friedel: r = friedel_standardise(r)
            try:
                refls[r] += 1
            except KeyError:
                refls[r] = 1
    return refls

#def reduced_state_to_original(reduced_state, toprefl_vec, toprefl_start):
#    state = reduced_state
#    state = np.array(list(state[:toprefl_start]) + list(toprefl_vec) + \
#        list(state[toprefl_start:]))
#    return state

def reduced_state_to_original(reduced_state, fixedrefl_vecs, fixedrefl_starts):
    idx = np.argsort(fixedrefl_starts)
    fixedrefl_starts = np.array(fixedrefl_starts)[idx]
    fixedrefl_vecs = [fixedrefl_vecs[i] for i in idx]
    state = reduced_state
    while len(fixedrefl_starts)>0:
        fs, fv = fixedrefl_starts[0], fixedrefl_vecs[0]
        state = np.array(list(state[:fs]) + list(fv) + \
                list(state[fs:]))
        fixedrefl_starts = fixedrefl_starts[1:]
        fixedrefl_vecs = fixedrefl_vecs[1:]
        #veclen = len(fv)
        #fixedrefl_starts += veclen
    return state

def decode_binary(state, var_sizes):
    var_start = 0
    decoded = []
    for var_size in var_sizes:
        var_end = var_start + var_size
        var_state = state[var_start:var_end]
        var_val = np.sum([(2**j)*x for j, x in enumerate(((1 - var_state) / 2))])
        decoded.append(var_val)
        var_start = var_end
    return np.array(decoded)

def ints_to_angles(ints, var_sizes, symmetric=True):
    var_sizes = np.array(var_sizes, dtype=int)
    if symmetric:
        ms = ints + 1
        angles = ( 2*np.pi*ms/((2**var_sizes) + 2) ) - np.pi
    else:
        ms = ints
        angles = ( 2*np.pi*ms/((2**var_sizes)) ) - np.pi
    return angles

def compute_angle_sums(angles, triplets, refl_to_int, friedel=False):
    ang_sums = []
    for t in triplets:
        angs, signs = [], []
        for r in t:
            if friedel:
                r, sign = friedel_standardise(r), (1.0 if friedel_standard(r) else -1.0)
            else:
                r, sign = r, 1.0
            angs.append(angles[refl_to_int[r]])
            signs.append(sign)
        ang_sums.append(np.dot(signs, angs))

    ang_sums = np.array(ang_sums)
    return ang_sums
