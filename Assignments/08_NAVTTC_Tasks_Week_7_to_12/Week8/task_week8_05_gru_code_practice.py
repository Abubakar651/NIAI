# Week 8 - Task 05: GRU Code Practice & Comparative Architecture Benchmark
# NAVTTC Outline Week 8 Day 5
# Reference: Comparative Benchmark of SimpleRNN vs LSTM vs GRU

import time
import numpy as np
import pandas as pd

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class SequentialArchitectureBenchmark:
    """
    Empirical and theoretical comparison across the 3 dominant sequential architectures:
      1. SimpleRNN: 1 recurrent weight matrix (Fast, but suffers from vanishing gradients)
      2. LSTM: 4 distinct gates / states (f, i, c, o) (Robust long-term memory, heavier parameter overhead)
      3. GRU: 2 gated mechanisms (r, z) (Optimal trade-off: 25% fewer parameters than LSTM, faster convergence)
    """

    @staticmethod
    def calculate_parameter_count(input_dim, hidden_dim, output_dim):
        """Calculates exact theoretical parameters across architectures."""
        # SimpleRNN: W_xh (in * h), W_hh (h * h), b_h (h) + output head (h * out + out)
        rnn_params = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim + output_dim)

        # LSTM: 4 sets of gates (W_xi, W_hi, b for each of 4 gates) + output head
        lstm_gate_params = 4 * ((input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim)
        lstm_params = lstm_gate_params + (hidden_dim * output_dim + output_dim)

        # GRU: 3 sets of matrices (z, r, h_tilde) + output head
        gru_gate_params = 3 * ((input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim)
        gru_params = gru_gate_params + (hidden_dim * output_dim + output_dim)

        return rnn_params, lstm_params, gru_params

    @staticmethod
    def benchmark_latency(num_steps=1000, input_dim=8, hidden_dim=16):
        """Simulates latency and execution runtime across 1,000 temporal forward transitions."""
        np.random.seed(42)
        X = np.random.randn(num_steps, input_dim)

        # 1. SimpleRNN Latency
        W_rnn = np.random.randn(input_dim + hidden_dim, hidden_dim)
        h_rnn = np.zeros((1, hidden_dim))
        t0 = time.time()
        for x in X:
            concat = np.hstack((x.reshape(1, -1), h_rnn))
            h_rnn = np.tanh(np.dot(concat, W_rnn))
        rnn_time = (time.time() - t0) * 1000

        # 2. LSTM Latency (4 gates)
        W_lstm = np.random.randn(input_dim + hidden_dim, 4 * hidden_dim)
        h_lstm = np.zeros((1, hidden_dim))
        c_lstm = np.zeros((1, hidden_dim))
        t0 = time.time()
        for x in X:
            concat = np.hstack((x.reshape(1, -1), h_lstm))
            gates = np.dot(concat, W_lstm)
            f = 1.0 / (1.0 + np.exp(-gates[:, :hidden_dim]))
            i = 1.0 / (1.0 + np.exp(-gates[:, hidden_dim:2*hidden_dim]))
            c_bar = np.tanh(gates[:, 2*hidden_dim:3*hidden_dim])
            o = 1.0 / (1.0 + np.exp(-gates[:, 3*hidden_dim:]))
            c_lstm = f * c_lstm + i * c_bar
            h_lstm = o * np.tanh(c_lstm)
        lstm_time = (time.time() - t0) * 1000

        # 3. GRU Latency (3 gates)
        W_gru = np.random.randn(input_dim + hidden_dim, 2 * hidden_dim)
        W_gru_cand = np.random.randn(input_dim + hidden_dim, hidden_dim)
        h_gru = np.zeros((1, hidden_dim))
        t0 = time.time()
        for x in X:
            concat = np.hstack((x.reshape(1, -1), h_gru))
            gates = 1.0 / (1.0 + np.exp(-np.dot(concat, W_gru)))
            z, r = gates[:, :hidden_dim], gates[:, hidden_dim:]
            concat_cand = np.hstack((x.reshape(1, -1), r * h_gru))
            h_cand = np.tanh(np.dot(concat_cand, W_gru_cand))
            h_gru = (1.0 - z) * h_gru + z * h_cand
        gru_time = (time.time() - t0) * 1000

        return rnn_time, lstm_time, gru_time


if __name__ == "__main__":
    print("=" * 65)
    print("Week 8 Task 5: GRU Code Practice & Architectural Comparison")
    print("Student: Abubakar Iqbal | NAVTTC Week 8")
    print("=" * 65)

    bench = SequentialArchitectureBenchmark()

    in_d, hid_d, out_d = 10, 32, 2
    rnn_p, lstm_p, gru_p = bench.calculate_parameter_count(in_d, hid_d, out_d)
    rnn_t, lstm_t, gru_t = bench.benchmark_latency(num_steps=1000, input_dim=in_d, hidden_dim=hid_d)

    summary = [
        {
            "Architecture": "SimpleRNN",
            "Internal Gates": "None (1 Recurrent Layer)",
            "Total Parameters": rnn_p,
            "1k Step Latency (ms)": f"{rnn_t:.2f}",
            "Primary Suitability": "Short temporal sequences (<10 steps)"
        },
        {
            "Architecture": "LSTM",
            "Internal Gates": "4 (Forget, Input, Candidate, Output)",
            "Total Parameters": lstm_p,
            "1k Step Latency (ms)": f"{lstm_t:.2f}",
            "Primary Suitability": "Complex long-range dependencies (>100 steps)"
        },
        {
            "Architecture": "GRU",
            "Internal Gates": "2 (Reset, Update)",
            "Total Parameters": gru_p,
            "1k Step Latency (ms)": f"{gru_t:.2f}",
            "Primary Suitability": "Fast convergence, low-latency production"
        }
    ]

    df_summary = pd.DataFrame(summary)
    print("\n--- Empirical Benchmark Matrix ---")
    print(df_summary.to_string(index=False))

    param_savings = ((lstm_p - gru_p) / lstm_p) * 100
    print(f"\nArchitectural Insight: GRU achieves a {param_savings:.1f}% reduction in parameter count compared to LSTM, reducing VRAM footprint while mitigating the vanishing gradient challenge.")
    print("Week 8 Sequential Architectures Module Completed.")
