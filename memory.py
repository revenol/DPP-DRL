from __future__ import print_function

"""Deep memory network utilities for action generation in DPP-DRL.

This module defines `MemoryDNN`, a replay-memory-based binary-action predictor
used by the main training script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print(torch.__version__)


class MemoryDNN:
    """DNN-backed memory for action encoding and decoding.

    Args:
        net: Layer-width list, e.g. `[input_dim, hidden1, hidden2, output_dim]`.
        learning_rate: Optimizer learning rate.
        training_interval: Number of encode calls between training steps.
        batch_size: Mini-batch size used for each training step.
        memory_size: Maximum replay-memory size.
        output_graph: Reserved flag for compatibility.
    """

    def __init__(
        self,
        net,
        learning_rate=0.001,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False,
    ):
        self.net = net
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.output_graph = output_graph

        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []

        # Memory layout: [state_features, action_bits]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        self._build_net()

    def _build_net(self):
        """Build the feed-forward predictor network."""
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),
            nn.ReLU(),
            nn.Linear(self.net[1], self.net[2]),
            nn.ReLU(),
            nn.Linear(self.net[2], self.net[3]),
            nn.BatchNorm1d(self.net[3]),
            nn.Sigmoid(),
        )

    def remember(self, h, g, BEnergy, AoI, m):
        """Append one `(state, action)` pair into replay memory."""
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, g, BEnergy, AoI, m))
        self.memory_counter += 1

    def encode(self, h, g, BEnergy, AoI, m):
        """Store one sample and trigger training periodically."""
        self.remember(h, g, BEnergy, AoI, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        """Sample from replay memory and run one BCE training step."""
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0 : self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0] :])

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        self.cost_his.append(self.cost)

    def decode(self, h, g, BEnergy, AoI, k=1, decoder_mode="OP"):
        """Decode candidate actions from the current state.

        Args:
            h: Uplink channel gains.
            g: Downlink channel gains.
            BEnergy: Current battery levels.
            AoI: Current age-of-information vector.
            k: Number of top candidates to generate.
            decoder_mode: One of `"OP"`, `"KNN"`, or `"TDMA"`.

        Returns:
            A list/array of candidate binary actions, based on the selected mode.
        """
        self.model.eval()
        temp = torch.Tensor([np.hstack((h, g, BEnergy, AoI))])

        m_pred = self.model(temp)
        m_pred = m_pred.detach().numpy()

        if decoder_mode == "OP":
            return self.knm(h, g, BEnergy, AoI, m_pred[0], k) + self.knm(
                h,
                g,
                BEnergy,
                AoI,
                m_pred[0] + np.random.normal(0, 1, len(m_pred[0])),
                k,
            )
        if decoder_mode == "KNN":
            return self.knn(h, g, BEnergy, AoI, m_pred[0], k)
        if decoder_mode == "TDMA":
            return self.knm_tdma(h, g, BEnergy, AoI, m_pred[0], k)

        print("The action selection must be 'OP' or 'KNN'")
        return []

    def opn(self, m, k=1):
        """Legacy helper retained for compatibility."""
        return self.knm(m, k) + self.knm(m + np.random.normal(0, 1, len(m)), k)

    def knm(self, h, g, BEnergy, AoI, m, k=1):
        """Generate up to `k` order-preserving binary actions from logits."""
        m_list = []
        m_list.append(1 * (m > 0.5))

        if k > 1:
            m_abs = abs(m)
            idx_list = np.argsort(m_abs)[: k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))

        m_list.append([0 for _ in range(len(m))])
        return m_list

    def knm_tdma(self, h, g, BEnergy, AoI, m, k=1):
        """Generate TDMA-style one-hot actions using sorted prediction values."""
        m_list = []
        m_list.append([0 for _ in range(len(m))])

        sorted_indices = np.argsort(-m)
        for i in range(min(k, len(m))):
            decision = [0] * len(m)
            decision[sorted_indices[i]] = 1
            m_list.append(decision)

        return m_list

    def tdma(self, h, g, BEnergy, AoI, m, k=1):
        """Generate all one-hot TDMA actions."""
        m_list = []
        for i in range(len(m)):
            m_list.append([1 if x == i else 0 for x in range(len(m))])
        return m_list

    def knn(self, h, g, BEnergy, AoI, m, k=1):
        """Enumerate all binary actions and rank by Euclidean distance."""
        if len(self.enumerate_actions) == 0:
            import itertools

            self.enumerate_actions = np.array(
                list(map(list, itertools.product([0, 1], repeat=self.net[3])))
            )

        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:]]

    def plot_cost(self):
        """Plot and save training loss over time."""
        import matplotlib.pyplot as plt

        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel("Training Loss")
        plt.xlabel("Time Frames")
        plt.savefig("plot_cost.png")
        plt.show()
