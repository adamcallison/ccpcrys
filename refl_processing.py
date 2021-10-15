import numpy as np
from copy import deepcopy as dc

import m2swork.hams as hams

def friedel_standard(r):
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
