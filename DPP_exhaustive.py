"""Exhaustive action search baseline for AoI optimization.

This script evaluates all binary actions for a small-user setting and uses the
best objective value at each time frame as a baseline against learning methods.
"""

import itertools
import time

import numpy as np
import scipy.io as sio

from bisection import bisection


def save_to_txt(AoI_his, file_path):
    """Save a 1-D metric list to a plain text file.

    Args:
        AoI_his: Sequence of scalar values.
        file_path: Output file path.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for AoI in AoI_his:
            f.write(f"{AoI} \n")


if __name__ == "__main__":
    N = 10
    n = 3000
    print("#user = %d, #channel=%d" % (N, n))

    data = sio.loadmat("./data/data_10.mat")
    channel_h = data["input_h"]
    channel_g = data["input_g"]

    # NOMA exhaustive binary actions.
    all_action = np.array(list(map(list, itertools.product([0, 1], repeat=N))))
    print(all_action)

    repeat_times = 1
    total_time_his = []

    for times in range(repeat_times):
        AoI = [1 for _ in range(N)]
        B = [0.0003 for _ in range(N)]
        LTAAoI = 0
        AoI_his = []
        start_time = time.time()

        for i in range(n):
            h = channel_h[i, :]
            g = channel_g[i, :]
            results = []
            for action in all_action:
                results.append(bisection(h, g, B, AoI, action, 1e-7))

            best_idx = np.argmax([r[0] for r in results])
            avg_sum = results[best_idx][1]
            new_B = results[best_idx][2]
            new_AoI = results[best_idx][3]

            B = new_B.copy()
            AoI = new_AoI.copy()
            LTAAoI = (LTAAoI * i + avg_sum) / (i + 1)
            AoI_his.append(LTAAoI)

        total_time = time.time() - start_time
        total_time_his.append(total_time)
        print("Count:", times + 1)
        print("Total time consumed:%s" % total_time)
        print(LTAAoI)

    # save_to_txt(total_time_his, "time_total_his.txt")
    # save_to_txt(AoI_his, "AoI_his.txt")
    print(LTAAoI)
