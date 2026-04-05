"""Lyapunov-based reward evaluation for NOMA scheduling actions.

This module exposes the `bisection` function used by both training and
exhaustive-search scripts. Given channel state, battery state, and an action
vector, it computes the optimization objective and the next system state.
"""

import random

import numpy as np


def bisection(h, g, BEnergy, AoI, M, V):
    """Evaluate one scheduling action and return the resulting system state.

    Args:
        h: Uplink channel gain vector.
        g: Downlink channel gain vector used for energy harvesting.
        BEnergy: Current battery energy vector (Joules).
        AoI: Current age-of-information vector.
        M: Binary action vector where 1 means transmission and 0 means harvest.
        V: Lyapunov control parameter.

    Returns:
        A tuple `(objective, avg_aoi, next_battery, next_aoi)`.
        For infeasible transmission (battery drops below zero), this function
        returns `(-100000000, wrong_indices)` to preserve original behavior.
    """
    tau = 274
    nu = 0.29
    P_max = 0.004927
    P0 = 0.000064
    Pin = 5
    v = 1 / V

    AoI_k = [x for x in AoI]
    Amax = 1000
    Bmax = 0.0003
    sigma = 3.162277660168375 * 10 ** (-9)
    S = 1.5
    AverSumAoI = 0
    eta = 0.5

    EnergyHarvest = [0 for _ in range(len(M))]
    BEnergy_k = [x for x in BEnergy]

    FangSuo = 0
    aa = 0
    Trans_nodes = []
    Trans_nodes_h = []
    wrongA = []
    flatT = 0

    for i in range(len(M)):
        if M[i] == 1:
            flatT = 1
            break

    if flatT == 0:
        for j in range(len(M)):
            EnergyHarvest[j] = eta * Pin * g[j]
            B_next = BEnergy_k[j] + EnergyHarvest[j]
            if B_next >= Bmax:
                BEnergy_k[j] = Bmax
            else:
                BEnergy_k[j] += EnergyHarvest[j]

        for j in range(len(M)):
            if AoI[j] < Amax:
                AoI_k[j] = AoI[j] + 1
            else:
                AoI_k[j] = Amax

        for j in range(len(M)):
            AverSumAoI += AoI_k[j]
        AverSumAoI /= len(M)
    else:
        Energy_Trans = []
        for i in range(len(M)):
            EnergyHarvest[i] = eta * Pin * g[i]
            B_next = BEnergy_k[i] + EnergyHarvest[i]
            if B_next >= Bmax:
                EnergyHarvest[i] = Bmax - BEnergy_k[i]

            aa += v * (Bmax - BEnergy[i]) * EnergyHarvest[i]

            if M[i] == 1:
                AoI_k[i] = 1
                Trans_nodes.append(i)
                Trans_nodes_h.append(h[i])
            else:
                AoI_k[i] += 1

        list_in = np.argsort(Trans_nodes_h)
        for i in range(len(list_in)):
            Energy_noise = 0
            for k in range(len(Energy_Trans)):
                Energy_noise += Energy_Trans[k] * Trans_nodes_h[list_in[k]]

            Energy_Trans_Node = (sigma + Energy_noise) / Trans_nodes_h[list_in[i]] * (2 ** S - 1)
            aa += v * ((Bmax - BEnergy[Trans_nodes[i]]) * Energy_Trans_Node) - (AoI[Trans_nodes[i]] / len(M))
            Energy_Trans.append(Energy_Trans_Node)

        for i in range(len(list_in)):
            BEnergy_k[Trans_nodes[list_in[i]]] -= Energy_Trans[i]

        for i in range(len(M)):
            if BEnergy_k[i] < 0:
                wrongA.append(i)
                return -100000000, wrongA

        for i in range(len(M)):
            AverSumAoI += AoI_k[i]
        AverSumAoI /= len(M)

    for i in range(len(M)):
        FangSuo += (Bmax - BEnergy[i]) * (BEnergy[i] - BEnergy_k[i])

    LyapnovDrift = -4 * 10 ** -7 * AverSumAoI - FangSuo
    _ = LyapnovDrift

    return -aa, AverSumAoI, BEnergy_k, AoI_k


if __name__ == "__main__":
    for _ in range(3):
        dlt = []
        for _ in range(5):
            dlt.append(random.randint(1, 35))
        for _ in range(2):
            dlt.append(random.randint(1, 12))
        print(dlt)
