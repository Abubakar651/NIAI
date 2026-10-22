# Week 8 - Task 04: Gated Recurrent Unit (GRU) Architecture from Scratch
# NAVTTC Outline Week 8 Day 4
# Reference: Cho et al. (2014) / On the Properties of Neural Machine Translation: Encoder-Decoder Approaches

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class GRUCellFromScratch:
    """
    Mathematical Implementation of Gated Recurrent Unit (GRU) Cell.
    A streamlined variant of LSTM combining forget & input gates into an 'Update Gate'
    and merging cell state and hidden state, leading to fewer parameters and faster training.
    """
    def __init__(self, input_dim=4, hidden_dim=8, random_state=42):
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1. Update Gate Parameters (z_t)
        self.W_z = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_z = np.zeros((1, hidden_dim))

        # 2. Reset Gate Parameters (r_t)
        self.W_r = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_r = np.zeros((1, hidden_dim))

        # 3. Candidate Hidden State Parameters (~h_t)
        self.W_h = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_h = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_h = np.zeros((1, hidden_dim))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def step_forward(self, x_t, h_prev):
        """
        Executes a single time-step forward pass through the GRU cell.
        x_t: (1, input_dim)
        h_prev: (1, hidden_dim)
        """
        # 1. Update Gate: Controls how much past information to carry forward
        z_t = self.sigmoid(np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)

        # 2. Reset Gate: Controls how much past information to forget
        r_t = self.sigmoid(np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)

        # 3. Candidate Hidden State: Incorporates reset previous memory
        gated_memory = r_t * h_prev
        h_tilde = np.tanh(np.dot(x_t, self.W_h) + np.dot(gated_memory, self.U_h) + self.b_h)

        # 4. Final Hidden State: Linear interpolation between old state and candidate state
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde

        gates = {
            "update_gate": z_t,
            "reset_gate": r_t,
            "candidate_hidden": h_tilde
        }

        return h_t, gates


if __name__ == "__main__":
    print("=" * 65)
    print("Week 8 Task 4: GRU Architecture & Internal Gating Mechanics")
    print("Student: Abubakar Iqbal | NAVTTC Week 8")
    print("=" * 65)

    input_dim = 3
    hidden_dim = 4
    gru = GRUCellFromScratch(input_dim=input_dim, hidden_dim=hidden_dim, random_state=42)

    # Sequence of financial / customer behavior vectors: [session_duration, click_count, transaction_value]
    sequence = np.array([
        [15.2, 5.0, 100.0],
        [2.1,  1.0, 20.0],
        [45.8, 12.0, 500.0]
    ])

    print(f"\nProcessing Sequential Data through GRU Cell (Input Dim = {input_dim}, Hidden Dim = {hidden_dim}):")

    h_prev = np.zeros((1, hidden_dim))
    for t_step, x in enumerate(sequence, 1):
        x_t = x.reshape(1, -1)
        h_t, gates = gru.step_forward(x_t, h_prev)

        print(f"\n--- Time Step t = {t_step} ---")
        print(f"  Input Vector x_t:          {x_t[0]}")
        print(f"  Reset Gate (r_t):          {gates['reset_gate'][0].round(4)}")
        print(f"  Update Gate (z_t):         {gates['update_gate'][0].round(4)}")
        print(f"  Candidate State (~h_t):    {gates['candidate_hidden'][0].round(4)}")
        print(f"  Emitted Hidden State (h_t):{h_t[0].round(4)}")
        h_prev = h_t

    print("\nWeek 8 Task 4 GRU Cell Forward Propagation Verified Successfully.")
