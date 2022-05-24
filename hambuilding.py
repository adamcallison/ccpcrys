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

def prod_binary(total_qubits, var1_start, var1_end, var2_start, var2_end, symmetric=True):
    Jmat, hvec, ic = get_zeros(total_qubits)
    var1_size, var2_size = var1_end - var1_start, var2_end - var2_start

    symval = int(symmetric)

    M1 = 2**var1_size
    M2 = 2**var2_size

    for ind1, qubit_ind1 in enumerate(range(var1_start, var1_end)):
        hvec[qubit_ind1] += (1-M2-symval) * (2**ind1)
    for ind2, qubit_ind2 in enumerate(range(var2_start, var2_end)):
        hvec[qubit_ind2] += (1-M1-symval) * (2**ind2)

    for ind1, qubit_ind1 in enumerate(range(var1_start, var1_end)):
        for ind2, qubit_ind2 in enumerate(range(var2_start, var2_end)):
            if qubit_ind1 == qubit_ind2:
                ic += 2**(ind1+ind2)
            else:
                Jmat[qubit_ind1, qubit_ind2] += 2**(ind1+ind2)

    ic += (symval-1)**2
    ic += (M1+M2)*(symval-1)
    ic += M1*M2

    scale = (np.pi**2)/( (M1)*(M2) )

    return scale*Jmat, scale*hvec, scale*ic

def binary(total_qubits, var_start, var_end, symmetric=True):
    Jmat, hvec, ic = get_zeros(total_qubits)
    var_size = var_end - var_start

    symval = int(symmetric)

    M = 2**var_size

    for ind, qubit_ind in enumerate(range(var_start, var_end)):
        hvec[qubit_ind] -= (2**ind)

    ic += (M+(-1)+symval)

    scale = np.pi/M

    return scale*Jmat, scale*hvec, scale*ic


def triplet_binary_old(total_qubits, var1_start, var1_end, var2_start, var2_end, var3_start, var3_end, var1_sign=1.0, var2_sign=1.0, var3_sign=1.0, symmetric=True):
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

def triplet_binary(total_qubits, var1_start, var1_end, var2_start, var2_end, var3_start, var3_end, var1_comp=False, var2_comp=False, var3_comp=False, symmetric=True):
    var1_size, var2_size, var3_size = var1_end - var1_start, var2_end - var2_start, var3_end - var3_start

    Jmat, hvec, ic = get_zeros(total_qubits)

    # single variables
    J1, h1, ic1 = binary(total_qubits, var1_start, var1_end, symmetric=symmetric)
    J2, h2, ic2 = binary(total_qubits, var2_start, var2_end, symmetric=symmetric)
    J3, h3, ic3 = binary(total_qubits, var3_start, var3_end, symmetric=symmetric)

    # squares
    J11, h11, ic11 = prod_binary(total_qubits, var1_start, var1_end, var1_start, var1_end, symmetric=symmetric)
    J22, h22, ic22 = prod_binary(total_qubits, var2_start, var2_end, var2_start, var2_end, symmetric=symmetric)
    J33, h33, ic33 = prod_binary(total_qubits, var3_start, var3_end, var3_start, var3_end, symmetric=symmetric)

    # products
    J12, h12, ic12 = prod_binary(total_qubits, var1_start, var1_end, var2_start, var2_end, symmetric=symmetric)
    J13, h13, ic13 = prod_binary(total_qubits, var1_start, var1_end, var3_start, var3_end, symmetric=symmetric)
    J23, h23, ic23 = prod_binary(total_qubits, var2_start, var2_end, var3_start, var3_end, symmetric=symmetric)

    for j in range(1,4):
        if j == 1:
            Jaa, haa, icaa, Ja, ha, ica = J11, h11, ic11, J1, h1, ic1
            vara_comp = var1_comp
        if j == 2:
            Jaa, haa, icaa, Ja, ha, ica = J22, h22, ic22, J2, h2, ic2
            vara_comp = var2_comp
        if j == 3:
            Jaa, haa, icaa, Ja, ha, ica = J33, h33, ic33, J3, h3, ic3
            vara_comp = var3_comp

        if not vara_comp:
            Jterm, hterm, icterm = Jaa, haa, icaa
        if vara_comp:
            Jterm = Jaa + (-4*np.pi*Ja)
            hterm = haa + (-4*np.pi*ha)
            icterm = icaa + (-4*np.pi*ica) + (4*(np.pi**2))
        Jmat += Jterm
        hvec += hterm
        ic += icterm

    for j in range(1, 3):
        for k in range(j+1, 4):
            if (j==1) and (k==2):
                Jab, hab, icab, Ja, ha, ica, Jb, hb, icb = J12, h12, ic12, J1, h1, ic1, J2, h2, ic2
                vara_comp, varb_comp = var1_comp, var2_comp
            if (j==1) and (k==3):
                Jab, hab, icab, Ja, ha, ica, Jb, hb, icb = J13, h13, ic13, J1, h1, ic1, J3, h3, ic3
                vara_comp, varb_comp = var1_comp, var3_comp
            if (j==2) and (k==3):
                Jab, hab, icab, Ja, ha, ica, Jb, hb, icb = J23, h23, ic23, J2, h2, ic2, J3, h3, ic3
                vara_comp, varb_comp = var2_comp, var3_comp

            if (not vara_comp) and (not varb_comp):
                Jterm, hterm, icterm = 2*Jab, 2*hab, 2*icab
            if (not vara_comp) and (varb_comp):
                Jterm = (-2*Jab) + 4*np.pi*Ja
                hterm = (-2*hab) + 4*np.pi*ha
                icterm = (-2*icab) + 4*np.pi*ica
            if (vara_comp) and (not varb_comp):
                Jterm = (-2*Jab) + 4*np.pi*Jb
                hterm = (-2*hab) + 4*np.pi*hb
                icterm = (-2*icab) + 4*np.pi*icb
            if (vara_comp) and (varb_comp):
                Jterm = (2*Jab) + (-4*np.pi*(Ja+Jb))
                hterm = (2*hab) + (-4*np.pi*(ha+hb))
                icterm = (2*icab) + (-4*np.pi*(ica+icb)) + (8*(np.pi**2))

            Jmat += Jterm
            hvec += hterm
            ic += icterm

    for j in range(1,4):
        if j == 1:
            Ja, ha, ica = J1, h1, ic1
            vara_comp = var1_comp
        if j == 2:
            Ja, ha, ica = J2, h2, ic2
            vara_comp = var2_comp
        if j == 3:
            Ja, ha, ica = J3, h3, ic3
            vara_comp = var3_comp

        if not vara_comp:
            Jterm, hterm, icterm = -4*np.pi*Ja, -4*np.pi*ha, -4*np.pi*ica
        if vara_comp:
            Jterm = 4*np.pi*Ja
            hterm = 4*np.pi*ha
            icterm = (4*np.pi*ica) + (-8*(np.pi**2))
        Jmat += Jterm
        hvec += hterm
        ic += icterm

    icterm = 4*(np.pi**2)
    ic = ic + icterm

    return Jmat, hvec, ic

def structure_ham_old(var_size, triplets, friedel=False, symmetric=True, weights=None, verbose=False):

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

            r1_comp = False if rp.friedel_standard(t[0]) else True
            r2_comp = False if rp.friedel_standard(t[1]) else True
            r3_comp = False if rp.friedel_standard(t[2]) else True
        else:
            t_idx = np.array([refl_to_int[t[0]], refl_to_int[t[1]], refl_to_int[t[2]]], dtype=int)
            r1_comp, r2_comp, r3_comp = False, False, False


        Jmattmp, hvectmp, ictmp = triplet_binary(total_qubits, \
            var_size*t_idx[0], var_size*(t_idx[0]+1), var_size*t_idx[1], \
            var_size*(t_idx[1]+1), var_size*t_idx[2], var_size*(t_idx[2]+1), \
            var1_comp=r1_comp, var2_comp=r2_comp, var3_comp=r3_comp, symmetric=symmetric)
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
        if np.dot(orth_comp2, orth_comp2) > 1e-4:
            break
        else:
            num += 1

    orth_comp2n = orth_comp2 / np.sqrt(np.dot(orth_comp2, orth_comp2))


    num += 1
    while True:
        third_refl_str = sorted_refls[num]
        third_refl = np.array([int(y.strip()) for y in (third_refl_str[1:-1].split(","))])
        third_refln = third_refl / np.sqrt(np.dot(third_refl, third_refl))
        overlap1 = np.dot(first_refln, third_refln)
        overlap2 = np.dot(orth_comp2n, third_refln)
        orth_comp3 = third_refln - ( (overlap1*first_refln) + (overlap2*orth_comp2n) )
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

def fix_variable(Jmat, hvec, ic, refl_stats, refl_to_int, refl_to_fix):
    num_refls = len(refl_stats)
    total_qubits = hvec.shape[0]
    var_size = total_qubits//num_refls

    toprefl = refl_to_fix # due to old code
    toprefl_int = refl_to_int[toprefl]
    toprefl_start, toprefl_end = toprefl_int*var_size, (toprefl_int+1)*var_size
    toprefl_vec = sign = np.random.default_rng().choice([-1, 1], size=var_size)
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

    return Jmat_mod, hvec_mod, ic_mod, int_to_refl_mod, refl_to_int_mod, toprefl_vec
