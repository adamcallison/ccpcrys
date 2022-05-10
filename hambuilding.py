import numpy as np
from copy import deepcopy as dc

import m2swork.hams as hams

import refl_processing as rp

def get_zeros(total_qubits):
    Jmat = np.zeros((total_qubits, total_qubits), dtype=np.float64)
    hvec = np.zeros(total_qubits, dtype=np.float64)
    ic = 0.0
    return Jmat, hvec, ic

def prod_binary_old(total_qubits, var1_start, var1_end, var2_start, var2_end, symmetric=True):
    # will be removed
    Jmat, hvec, ic = get_zeros(total_qubits)
    var1_size, var2_size = var1_end - var1_start, var2_end - var2_start

    if not symmetric:
        ic += 1
        for ind1, qubit_ind1 in enumerate(range(var1_start, var1_end)):
            hvec[qubit_ind1] += 2**ind1
        for ind2, qubit_ind2 in enumerate(range(var2_start, var2_end)):
            hvec[qubit_ind2] += 2**ind2

    for ind1, qubit_ind1 in enumerate(range(var1_start, var1_end)):
        for ind2, qubit_ind2 in enumerate(range(var2_start, var2_end)):
            if qubit_ind1 == qubit_ind2:
                ic += 2**(ind1+ind2)
            else:
                Jmat[qubit_ind1, qubit_ind2] += 2**(ind1+ind2)

    M1 = 2**var1_size
    M2 = 2**var2_size

    if symmetric:
        scale = (np.pi**2)/( (M1+1)*(M2+1) )
    else:
        scale = (np.pi**2)/( (M1)*(M2) )

    return scale*Jmat, scale*hvec, scale*ic

def prod_binary(total_qubits, var1_start, var1_end, var2_start, var2_end, symmetric=True):
    Jmat, hvec, ic = get_zeros(total_qubits)
    var1_size, var2_size = var1_end - var1_start, var2_end - var2_start

    symval = int(symmetric)

    for ind1, qubit_ind1 in enumerate(range(var1_start, var1_end)):
        hvec[qubit_ind1] += (1-symval) * (2**ind1)
    for ind2, qubit_ind2 in enumerate(range(var2_start, var2_end)):
        hvec[qubit_ind2] += (1-symval) * (2**ind2)

    for ind1, qubit_ind1 in enumerate(range(var1_start, var1_end)):
        for ind2, qubit_ind2 in enumerate(range(var2_start, var2_end)):
            if qubit_ind1 == qubit_ind2:
                ic += 2**(ind1+ind2)
            else:
                Jmat[qubit_ind1, qubit_ind2] += 2**(ind1+ind2)

    ic += (symval-1)**2

    M1 = 2**var1_size
    M2 = 2**var2_size

    scale = (np.pi**2)/( (M1)*(M2) )

    return scale*Jmat, scale*hvec, scale*ic

def triplet_binary(total_qubits, var1_start, var1_end, var2_start, var2_end, var3_start, var3_end, var1_sign=1.0, var2_sign=1.0, var3_sign=1.0, symmetric=True):
    var1_size, var2_size, var3_size = var1_end - var1_start, var2_end - var2_start, var3_end - var3_start

    Jmat, hvec, ic = get_zeros(total_qubits)

    Jmatterm, hvecterm, icterm = prod_binary(total_qubits, var1_start, var1_end, var1_start, var1_end, symmetric=symmetric)
    Jmat, hvec, ic = Jmat + Jmatterm, hvec + hvecterm, ic + icterm

    Jmatterm, hvecterm, icterm = prod_binary(total_qubits, var2_start, var2_end, var2_start, var2_end, symmetric=symmetric)
    Jmat, hvec, ic = Jmat + Jmatterm, hvec + hvecterm, ic + icterm

    Jmatterm, hvecterm, icterm = prod_binary(total_qubits, var3_start, var3_end, var3_start, var3_end, symmetric=symmetric)
    Jmat, hvec, ic = Jmat + Jmatterm, hvec + hvecterm, ic + icterm

    Jmatterm, hvecterm, icterm = prod_binary(total_qubits, var1_start, var1_end, var2_start, var2_end, symmetric=symmetric)
    sign = var1_sign*var2_sign
    Jmat, hvec, ic = Jmat + (sign*2*Jmatterm), hvec + (sign*2*hvecterm), ic + (sign*2*icterm)

    Jmatterm, hvecterm, icterm = prod_binary(total_qubits, var1_start, var1_end, var3_start, var3_end, symmetric=symmetric)
    sign = var1_sign*var3_sign
    Jmat, hvec, ic = Jmat + (sign*2*Jmatterm), hvec + (sign*2*hvecterm), ic + (sign*2*icterm)

    Jmatterm, hvecterm, icterm = prod_binary(total_qubits, var2_start, var2_end, var3_start, var3_end, symmetric=symmetric)
    sign = var2_sign*var3_sign
    Jmat, hvec, ic = Jmat + (sign*2*Jmatterm), hvec + (sign*2*hvecterm), ic + (sign*2*icterm)

    return Jmat, hvec, ic

def structure_ham(var_size, triplets, friedel=False, symmetric=True, weights=None, verbose=False):

    num_triplets = len(triplets)
    refl_stats = rp.reflection_stats(triplets, friedel=friedel)
    refls = sorted(list(refl_stats.keys()))
    num_refls = len(refls)
    refl_to_int = dict(zip(refls, range(num_refls)))
    int_to_refl = dict(zip(range(num_refls), refls))
    total_qubits = (var_size*num_refls)
    Jmat, hvec, ic = get_zeros(total_qubits)

    if weights is None:
        weights = np.ones(num_triplets)

    for i, t in enumerate(triplets):
        if verbose:
            pline = f'processing triplet {i+1} of {num_triplets}...  '
            print(pline, end='\r')

        if friedel:
            t_idx1 = refl_to_int[rp.friedel_standardise(t[0])]
            t_idx2 = refl_to_int[rp.friedel_standardise(t[1])]
            t_idx3 = refl_to_int[rp.friedel_standardise(t[2])]
            t_idx = np.array([t_idx1, t_idx2, t_idx3])

            r1_sign = 1.0 if rp.friedel_standard(t[0]) else -1.0
            r2_sign = 1.0 if rp.friedel_standard(t[1]) else -1.0
            r3_sign = 1.0 if rp.friedel_standard(t[2]) else -1.0
        else:
            t_idx = np.array([refl_to_int[t[0]], refl_to_int[t[1]], refl_to_int[t[2]]], dtype=int)
            r1_sign, r2_sign, r3_sign = 1.0, 1.0, 1.0


        Jmattmp, hvectmp, ictmp = triplet_binary(total_qubits, \
            var_size*t_idx[0], var_size*(t_idx[0]+1), var_size*t_idx[1], \
            var_size*(t_idx[1]+1), var_size*t_idx[2], var_size*(t_idx[2]+1), \
            var1_sign=r1_sign, var2_sign=r2_sign, var3_sign=r3_sign, symmetric=symmetric)
        Jmat += weights[i]*Jmattmp
        hvec += weights[i]*hvectmp
        ic += weights[i]*ictmp

    if verbose:
        print(' '*len(pline), end='\r')
        print('Processed all triplets.')

    return Jmat, hvec, ic, refl_stats, int_to_refl, refl_to_int

def sort_refls(refl_stats):
    idx = np.argsort(list(refl_stats.values()))
    all_refls = list(refl_stats.keys())
    refls = []
    for j in range(len(all_refls)):
        refls.append(all_refls[idx[-j]])
    return refls

def find_top_refls(refl_stats, number):
    sorted_refls = sort_refls(refl_stats)
    refls = []
    for j in range(number):
        refls.append(sorted_refls[j])
    return refls

def find_fixable_refls(refl_stats):
    # only fixes 3
    sorted_refls = sort_refls(refl_stats)
    first_refl_str = sorted_refls[0]
    first_refl = np.array([int(y.strip()) for y in (first_refl_str[1:-1].split(","))])
    first_refln = first_refl / np.sqrt(np.dot(first_refl, first_refl))

    num = 1
    while True:
        second_refl_str = sorted_refls[num]
        second_refl = np.array([int(y.strip()) for y in (second_refl_str[1:-1].split(","))])
        second_refln = second_refl / np.sqrt(np.dot(second_refl, second_refl))
        overlap = np.dot(first_refln, second_refln)
        orth_comp2 = second_refln - (overlap*first_refln)
        #print(num, np.abs(orth_comp2)**2)
        if np.dot(orth_comp2, orth_comp2) > 1e-4:
            break
        else:
            num += 1

    orth_comp2n = orth_comp2 / np.sqrt(np.dot(orth_comp2, orth_comp2))
    #print('done')

    num += 1
    while True:
        third_refl_str = sorted_refls[num]
        third_refl = np.array([int(y.strip()) for y in (third_refl_str[1:-1].split(","))])
        third_refln = third_refl / np.sqrt(np.dot(third_refl, third_refl))
        overlap1 = np.dot(first_refln, third_refln)
        overlap2 = np.dot(orth_comp2n, third_refln)
        orth_comp3 = third_refln - ( (overlap1*first_refln) + (overlap2*orth_comp2n) )
        #print(num, np.abs(overlap1)**2, np.abs(overlap2)**2, np.dot(orth_comp3, orth_comp3))
        if np.dot(orth_comp3, orth_comp3) > 1e-4:
            break
        else:
            num += 1

    return [first_refl_str, second_refl_str, third_refl_str]



def fix_variables(Jmat, hvec, ic, refl_stats, refl_to_int, refls_to_fix):
    Jmat_mod, hvec_mod, ic_mod =  Jmat, hvec, ic
    refl_to_int_mod = refl_to_int
    refl_stats_mod = refl_stats

    fixedrefl_vecs = []
    for refl_to_fix in refls_to_fix:
        Jmat_mod, hvec_mod, ic_mod, int_to_refl_mod, refl_to_int_mod, fixedrefl_vec = fix_variable(Jmat_mod, hvec_mod, ic_mod, refl_stats_mod, refl_to_int_mod, refl_to_fix)
        fixedrefl_vecs.append(fixedrefl_vec)
        refl_stats_mod = dict(refl_stats_mod)
        del refl_stats_mod[refl_to_fix]
    return Jmat_mod, hvec_mod, ic_mod, int_to_refl_mod, refl_to_int_mod, fixedrefl_vecs


#def fix_variable(Jmat, hvec, ic, refl_stats, refl_to_int):
def fix_variable(Jmat, hvec, ic, refl_stats, refl_to_int, refl_to_fix):
    num_refls = len(refl_stats)
    total_qubits = hvec.shape[0]
    var_size = total_qubits//num_refls

    #toprefl = list(refl_stats.keys())[np.argmax(list(refl_stats.values()))]
    toprefl = refl_to_fix # due to old code
    toprefl_int = refl_to_int[toprefl]
    toprefl_start, toprefl_end = toprefl_int*var_size, (toprefl_int+1)*var_size
    toprefl_vec = np.array([1]*var_size, dtype=np.float64)
    Jmat_mod, Jmat_rs, Jmat_cs, Jmat_rcs = np.delete(np.delete(Jmat, range(toprefl_start, toprefl_end), axis=0), range(toprefl_start, toprefl_end), axis=1), \
        Jmat[toprefl_start:toprefl_end,:], \
        Jmat[:,toprefl_start:toprefl_end], \
        Jmat[toprefl_start:toprefl_end,toprefl_start:toprefl_end]
    Jmat_rs, Jmat_cs = np.delete(Jmat_rs, range(toprefl_start, toprefl_end), axis=1), np.delete(Jmat_cs, range(toprefl_start, toprefl_end), axis=0)

    hvec_mod, hvec_rs = np.delete(hvec, range(toprefl_start, toprefl_end), axis=0), hvec[toprefl_start:toprefl_end]

    ic_mod = ic + (toprefl_vec.dot(Jmat_rcs.dot(toprefl_vec))) + (hvec_rs.dot(toprefl_vec))

    for j, el in enumerate(toprefl_vec):
        hvec_mod = hvec_mod + (Jmat_rs[j,:]*el) + (Jmat_cs[:,j]*el)

    refls_mod = sorted(list(refl_stats.keys()))
    refls_mod = [refl for refl in refls_mod if not refl == toprefl]
    num_refls_mod = len(refls_mod)
    refl_to_int_mod = dict(zip(refls_mod, range(num_refls_mod)))
    int_to_refl_mod = dict(zip(range(num_refls_mod), refls_mod))

    #return Jmat_mod, hvec_mod, ic_mod, int_to_refl_mod, refl_to_int_mod, toprefl, toprefl_vec
    return Jmat_mod, hvec_mod, ic_mod, int_to_refl_mod, refl_to_int_mod, toprefl_vec
