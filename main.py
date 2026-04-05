"""Main training and evaluation script for DPP-DRL AoI optimization.

This script trains `MemoryDNN` to geneAoI candidate scheduling actions,
selects the best action by calling `bisection`, and evaluates final AoI/Yi
metrics over multiple runs and V settings.
"""

import os
import time

import numpy as np
import scipy.io as sio

from bisection import bisection
from memory import MemoryDNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_AoI(AoI_his, rolling_intv=50):
    """Plot average AoI curve.

    Args:
        AoI_his: AoI metric history.
        rolling_intv: Reserved parameter for compatibility.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _ = rolling_intv
    print("Plotting AoI...")
    AoI_array = np.asarray(AoI_his)
    mpl.style.use("seaborn")
    plt.plot(np.arange(len(AoI_array)) + 1, AoI_his)
    plt.ylabel("Average Sum AoI")
    plt.xlabel("Time Frames")
    plt.savefig("plot_AoI.png")
    plt.show()


def plot_Yi(AoI_his, rolling_intv=50):
    """Plot average Yi curve.

    Args:
        AoI_his: Yi metric history.
        rolling_intv: Reserved parameter for compatibility.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _ = rolling_intv
    print("Plotting Yi...")
    AoI_array = np.asarray(AoI_his)
    mpl.style.use("seaborn")
    plt.plot(np.arange(len(AoI_array)) + 1, AoI_his)
    plt.ylabel("Network average Yi")
    plt.xlabel("Time Frames")
    plt.savefig("plot_Yi.png")
    plt.show()


def plot_V(AoI_his, rolling_intv=50):
    """Plot metric curve indexed by V scale.

    Args:
        AoI_his: Metric history.
        rolling_intv: Reserved parameter for compatibility.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _ = rolling_intv
    AoI_array = np.asarray(AoI_his)
    mpl.style.use("seaborn")
    plt.plot(np.arange(len(AoI_array)) / 100, AoI_his)
    plt.ylabel("Average Sum AoI")
    plt.xlabel("Time Frames")
    plt.savefig("plot_V.png")
    plt.show()


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
    n = 30000
    K = N
    decoder_mode = "OP"
    Memory = 1024
    Delta = 32
    count = 1

    VArrays = [1e-7]
    # VArrays = [1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 8e-5, 1e-4]
    Bmax = 0.3  # mJ

    print(
        "#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d"
        % (N, n, K, decoder_mode, Memory, Delta)
    )

    data = sio.loadmat("./data/data_10.mat")
    channel_h = data["input_h"]
    channel_g = data["input_g"]

    # Scale channel gains for easier neural-network training.
    channel_h = channel_h * 10000
    channel_g = channel_g * 10000
    channel = [x for x in channel_h]

    split_idx = int(0.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(0.8 * n))

    for V in VArrays:
        print("V=", V)
        total_time_his = []

        avetrainingAoI = 0
        avetrainingYi = 0
        AvetestAoI = 0
        AvetestYi = 0
        final_aoi_records = []
        final_Yi_records = []

        for times in range(count):
            mem = MemoryDNN(
                net=[4 * N, 120, 80, N],
                learning_rate=0.001,
                training_interval=10,
                batch_size=64,
                memory_size=Memory,
            )

            k_idx_his = []
            K_his = []
            Energy_train = [Bmax for _ in range(N)]  # mJ
            AoI_text = []
            AoI = [1 for _ in range(N)]

            trainingAoI = 0
            trainingYi = 0
            tr_AoI = []
            tr_Yi = []

            start_time = time.time()

            for i in range(n):
                if i > 0 and i % Delta == 0:
                    if Delta > 1:
                        max_k = max(k_idx_his[-Delta:-1])
                    else:
                        max_k = k_idx_his[-1]
                    K = min(max_k + 1, N)

                if i < n - num_test:
                    i_idx = i % split_idx
                else:
                    i_idx = i - n + num_test + split_idx

                h = channel_h[i_idx, :]
                g = channel_g[i_idx, :]

                m_list = mem.decode(h, g, Energy_train, AoI, K, decoder_mode)
                r_list = []
                Energy_now = [x / 1000 for x in Energy_train]  # J

                for m in m_list:
                    r_list.append(bisection(h / 10000, g / 10000, Energy_now, AoI, m, V)[0])

                number = 0
                for r in r_list:
                    if r == -100000000:
                        number += 1
                _ = number

                best_action = m_list[np.argmax(r_list)]
                mem.encode(h, g, Energy_train, AoI, best_action)

                trainingAoI_k = bisection(
                    h / 10000,
                    g / 10000,
                    Energy_now,
                    AoI,
                    best_action,
                    V,
                )[1]
                Energy_bb = [
                    x
                    for x in bisection(
                        h / 10000,
                        g / 10000,
                        Energy_now,
                        AoI,
                        best_action,
                        V,
                    )[2]
                ]
                Energy_train = [x * 1000 for x in Energy_bb]

                AverSumE = 0
                for j in range(len(Energy_train)):
                    AverSumE += Energy_train[j]
                AverSumE /= len(Energy_train)
                trainingYi_k = Bmax - AverSumE

                AoI = [
                    x
                    for x in bisection(
                        h / 10000,
                        g / 10000,
                        Energy_now,
                        AoI,
                        best_action,
                        V,
                    )[3]
                ]

                trainingAoI = (trainingAoI * i + trainingAoI_k) / (i + 1)
                trainingYi = (trainingYi * i + trainingYi_k) / (i + 1)
                tr_AoI.append(trainingAoI)
                tr_Yi.append(trainingYi)

                k_idx_his.append(np.argmax(r_list))
                K_his.append(K)

            total_time = time.time() - start_time
            print("trainingAoI:", trainingAoI)
            print("trainingYi:", trainingYi)
            total_time_his.append(total_time)

            avetrainingAoI += trainingAoI
            avetrainingYi += trainingYi

            AoI_t = [1 for _ in range(N)]
            BEnergy_t = [Bmax / 1000 for _ in range(N)]  # J
            pl_AoI = []

            FinalAoI = 0
            FinalYi = 0

            Energy_change = []
            Bat = []

            for i in range(3000):
                AverSumAoI = 0
                AverSumEnergy = 0
                h_t = channel_h[i, :]
                g_t = channel_g[i, :]

                B_test = [x * 1000 for x in BEnergy_t]
                m_list1 = mem.decode(h_t, g_t, B_test, AoI_t, K, decoder_mode)
                r_list1 = []

                for m in m_list1:
                    r_list1.append(bisection(h_t / 10000, g_t / 10000, BEnergy_t, AoI_t, m, V)[0])

                LyaDrift, AverSumAoI, B_tt, AoI_t = bisection(
                    h_t / 10000,
                    g_t / 10000,
                    BEnergy_t,
                    AoI_t,
                    m_list1[np.argmax(r_list1)],
                    V,
                )
                _ = LyaDrift
                BEnergy_t = [x for x in B_tt]

                Bat.append(BEnergy_t)

                BEnergy_tt = [x * 1000 for x in BEnergy_t]
                for j in range(len(BEnergy_tt)):
                    AverSumEnergy += BEnergy_tt[j]
                AverSumEnergy /= len(BEnergy_tt)
                testYi_k = Bmax - AverSumEnergy

                AoI_text.append([x for x in AoI_t])

                FinalAoI = (FinalAoI * i + AverSumAoI) / (i + 1)
                pl_AoI.append(FinalAoI)

                Energy_change.append([x for x in B_tt])

                FinalYi = (FinalYi * i + testYi_k) / (i + 1)
                pl_AoI.append(FinalYi)

            _ = (AoI_text, Energy_change, Bat, tr_AoI, tr_Yi, K_his)

            final_aoi_records.append(FinalAoI)
            final_Yi_records.append(FinalYi)

            AvetestAoI += FinalAoI
            print("test AoI:", FinalAoI)
            AvetestYi += FinalYi
            print("test Yi:", FinalYi)

        # target_file = "L_AoI_P=0.5W_B=10mJ.txt"
        # with open(target_file, "a", encoding="utf-8") as f:
        #     f.write(f"=== Results for V = {V} ===\\n")
        #     for record in final_aoi_records:
        #         f.write(f"{record}\\n")

        # target_file = "L_Yi_P=0.5W_B=10mJ.txt"
        # with open(target_file, "a", encoding="utf-8") as f:
        #     f.write(f"=== Results for V = {V} ===\\n")
        #     for record in final_Yi_records:
        #         f.write(f"{record}\\n")

        print("Average training AoI:", avetrainingAoI / count)
        print("Average test AoI:", AvetestAoI / count)
        print("Average training Yi:", avetrainingYi / count)
        print("Average test Yi:", AvetestYi / count)

    _ = total_time_his
