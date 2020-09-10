import os

import numpy as np
from numpy import genfromtxt

Env = 'Sim'
K = 5
p = 200
Nk = 150
loss_function = 'DWDnc'
l = '0.06'
al = '0.1'
q = '1'
stat = np.zeros(10, dtype=np.int)
for i in range(100):
    pat = os.path.join(os.getcwd(), "results", "env=" + str(Env), "K=" + str(K), "p=" + str(p),
                       "Nk=" + str(Nk), "loss_function=" + str(loss_function), "Lambda=" + str(l),
                       "Alpha=" + str(al), "q=" + str(q), str(i), "B.csv")
    d = np.abs(genfromtxt(pat, delimiter=','))
    d_t = d[:2, :]
    group_d = np.mean(d, axis=1)
    group_d_t = group_d[:2]
    print(d_t.shape)
    total = p * K

    # element
    total = p * K
    correct_included = np.count_nonzero(d_t > 1e-4)
    correct_not_included = 2 * K - correct_included
    not_correct_included = np.count_nonzero(d > 1e-4) - correct_included
    not_correct_not_included = total - correct_included - correct_not_included - not_correct_included

    # group
    gtotal = p
    gcorrect_included = np.count_nonzero(group_d_t > 1e-4)
    gcorrect_not_included = 2 - np.count_nonzero(group_d_t > 1e-4)
    gnot_correct_included = np.count_nonzero(group_d > 1e-4) - gcorrect_included
    gnot_correct_not_included = gtotal - gcorrect_included - gcorrect_not_included - gnot_correct_included

    stat += np.array(
        [total, correct_included, correct_not_included, not_correct_included, not_correct_not_included, gtotal,
         gcorrect_included, gcorrect_not_included, gnot_correct_included, gnot_correct_not_included])

total, correct_included, correct_not_included, not_correct_included, not_correct_not_included, gtotal, gcorrect_included, gcorrect_not_included, gnot_correct_included, gnot_correct_not_included = stat
print("Among 100 runs:")
print("Elementwise:")
print("Total: %d" % total)
print("Useful variables included: %d" % correct_included)
print("Useful variables excluded: %d" % correct_not_included)
print("Useless variables included: %d" % not_correct_included)
print("Useless variables excluded: %d" % not_correct_not_included)

print("Groupwise:")
print("Total: %d" % gtotal)
print("Useful variable-groups included: %d" % gcorrect_included)
print("Useful variable-groups excluded: %d" % gcorrect_not_included)
print("Useless variable-groups included: %d" % gnot_correct_included)
print("Useless variable-groups excluded: %d" % gnot_correct_not_included)
