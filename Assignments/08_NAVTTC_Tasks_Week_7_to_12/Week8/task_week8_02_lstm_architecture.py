# Week 8 - Task 02: Long Short-Term Memory (LSTM) Architecture from Scratch
# NAVTTC Outline Week 8 Day 2
# Reference: Hochreiter & Schmidhuber (1997) / Understanding LSTM Networks

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class LSTMCellFromScratch:
    """
    Mathematical Implementation of an LSTM Cell from ground up.
    Solves the Vanishing/Exploding Gradient problem of vanilla RNNs
    via Constant Error Carousel (CEC) and non-linear gating mechanisms.
    """
    def __init__(self, input_dim=4, hidden_dim=8, random_state=42):
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Combined weight matrices [x_t, h_{t-1}] -> dim = (input_dim + hidden_dim)
        concat_dim = input_dim + hidden_dim

        # Gate weights: Forget (f), Input (i), Candidate (c), Output (o)
        self.W_f = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_f = np.ones((1, hidden_dim)) # Initialize forget gate bias to 1 to prevent early forgetting

        self.W_i = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_i = np.zeros((1, hidden_dim))

        self.W_c = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_c = np.zeros((1, hidden_dim))

        self.W_o = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_o = np.zeros((1, hidden_dim))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def step_forward(self, x_t, h_prev, c_prev):
        """
        Executes a single time-step forward pass through the LSTM cell.
        x_t: (1, input_dim)
        h_prev: (1, hidden_dim)
        c_prev: (1, hidden_dim)
        """
        # Concatenate input and prior hidden state: [x_t, h_{t-1}]
        concat = np.hstack((x_t, h_prev))

        # 1. Forget Gate: What old information to discard from cell state
        f_t = self.sigmoid(np.dot(concat, self.W_f) + self.b_f)

        # 2. Input Gate: Which values will be updated
        i_t = self.sigmoid(np.dot(concat, self.W_i) + self.b_i)

        # 3. Candidate Cell State: New candidate values that could be added
        c_tilde = np.tanh(np.dot(concat, self.W_c) + self.b_c)

        # 4. Cell State Update: Combine old state and new candidate
        c_t = f_t * c_prev + i_t * c_tilde

        # 5. Output Gate: Decide what parts of the cell state to output
        o_t = self.sigmoid(np.dot(concat, self.W_o) + self.b_o)

        # 6. Hidden State: Filtered cell state
        h_t = o_t * np.tanh(c_t)

        gate_activations = {
            "forget_gate": f_t,
            "input_gate": i_t,
            "candidate_cell": c_tilde,
            "output_gate": o_t
        }

        return h_t, c_t, gate_activations

    def forward_sequence(self, sequence_inputs):
        """Processes a sequence of length T through the LSTM cell."""
        T = len(sequence_inputs)
        h_states = []
        c_states = []

        h_t = np.zeros((1, self.hidden_dim))
        c_t = np.zeros((1, self.hidden_dim))

        for t in range(T):
            x_t = sequence_inputs[t].reshape(1, -1)
            h_t, c_t, gates = self.step_forward(x_t, h_t, c_t)
            h_states.append(h_t)
            c_states.append(c_t)

        return np.array(h_states), np.array(c_states)


if __name__ == "__main__":
    print("=" * 65)
    print("Week 8 Task 2: LSTM Cell Architecture & Gating Mechanics")
    print("Student: Abubakar Iqbal | NAVTTC Week 8")
    print("=" * 65)

    input_dim = 3
    hidden_dim = 4
    lstm = LSTMCellFromScratch(input_dim=input_dim, hidden_dim=hidden_dim, random_state=42)

    # Simulate sequential financial transactions: [amount, velocity, risk_index]
    sample_seq = np.array([
        [120.0, 1.2, 0.1],
        [450.0, 3.5, 0.4],
        [890.0, 7.8, 0.9],
        [25.0,  0.5, 0.0]
    ])

    print(f"\nProcessing Sequence of {len(sample_seq)} Financial Time-Steps:")
    print("Input Vector Dimension:", input_dim)
    print("LSTM Hidden Memory State Dimension:", hidden_dim)

    h_prev = np.zeros((1, hidden_dim))
    c_prev = np.zeros((1, hidden_dim))

    for step, x in enumerate(sample_seq, 1):
        x_t = x.reshape(1, -1)
        h_t, c_t, gates = lstm.step_forward(x_t, h_prev, c_prev)
        print(f"\n--- Time-Step t = {step} ---")
        print(f"  Input x_t:               {x_t[0]}")
        print(f"  Forget Gate (f_t):       {gates['forget_gate'][0].round(3)} (Retain Ratio)")
        print(f"  Input Gate (i_t):        {gates['input_gate'][0].round(3)} (Update Ratio)")
        print(f"  Candidate State (~C_t):  {gates['candidate_cell'][0].round(3)}")
        print(f"  Updated Cell State (C_t):{c_t[0].round(3)}")
        print(f"  Output Gate (o_t):       {gates['output_gate'][0].round(3)}")
        print(f"  Emitted Hidden (h_t):    {h_t[0].round(3)}")
        h_prev, c_prev = h_t, c_t

    print("\nWeek 8 Task 2 LSTM Gating Implementation Verified.")
