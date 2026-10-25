# Abubakar Iqbal
# NAVTTC Week 8 - Task 05: RNN vs LSTM vs GRU Benchmark

import time
import numpy as np
import pandas as pd

class SequenceBenchmark:
    @staticmethod
    def count_parameters(in_dim, h_dim, out_dim):
        rnn_p = (in_dim * h_dim) + (h_dim * h_dim) + h_dim + (h_dim * out_dim + out_dim)
        lstm_p = 4 * ((in_dim * h_dim) + (h_dim * h_dim) + h_dim) + (h_dim * out_dim + out_dim)
        gru_p = 3 * ((in_dim * h_dim) + (h_dim * h_dim) + h_dim) + (h_dim * out_dim + out_dim)
        return rnn_p, lstm_p, gru_p

    @staticmethod
    def time_forward_passes(n_steps=1000, in_dim=8, h_dim=16):
        np.random.seed(42)
        X = np.random.randn(n_steps, in_dim)

        W_rnn = np.random.randn(in_dim + h_dim, h_dim)
        h_rnn = np.zeros((1, h_dim))
        t0 = time.time()
        for x in X:
            concat = np.hstack((x.reshape(1, -1), h_rnn))
            h_rnn = np.tanh(np.dot(concat, W_rnn))
        rnn_time = (time.time() - t0) * 1000

        W_lstm = np.random.randn(in_dim + h_dim, 4 * h_dim)
        h_lstm = np.zeros((1, h_dim))
        c_lstm = np.zeros((1, h_dim))
        t0 = time.time()
        for x in X:
            concat = np.hstack((x.reshape(1, -1), h_lstm))
            gates = np.dot(concat, W_lstm)
            f = 1.0 / (1.0 + np.exp(-gates[:, :h_dim]))
            i = 1.0 / (1.0 + np.exp(-gates[:, h_dim:2*h_dim]))
            c_bar = np.tanh(gates[:, 2*h_dim:3*h_dim])
            c_lstm = f * c_lstm + i * c_bar
            o = 1.0 / (1.0 + np.exp(-gates[:, 3*h_dim:]))
            h_lstm = o * np.tanh(c_lstm)
        lstm_time = (time.time() - t0) * 1000

        W_gru = np.random.randn(in_dim + h_dim, 3 * h_dim)
        h_gru = np.zeros((1, h_dim))
        t0 = time.time()
        for x in X:
            concat = np.hstack((x.reshape(1, -1), h_gru))
            gates = np.dot(concat, W_gru)
            z = 1.0 / (1.0 + np.exp(-gates[:, :h_dim]))
            r = 1.0 / (1.0 + np.exp(-gates[:, h_dim:2*h_dim]))
            h_cand = np.tanh(gates[:, 2*h_dim:])
            h_gru = (1.0 - z) * h_gru + z * h_cand
        gru_time = (time.time() - t0) * 1000

        return rnn_time, lstm_time, gru_time

if __name__ == "__main__":
    print("NAVTTC Week 8 Task 05 - Abubakar Iqbal")

    in_dim, h_dim, out_dim = 10, 32, 2
    rnn_p, lstm_p, gru_p = SequenceBenchmark.count_parameters(in_dim, h_dim, out_dim)
    rnn_t, lstm_t, gru_t = SequenceBenchmark.time_forward_passes(n_steps=1000, in_dim=in_dim, h_dim=h_dim)

    abubakar_results_df = pd.DataFrame([
        {"Architecture": "SimpleRNN", "Parameters": rnn_p, "Latency (ms)": round(rnn_t, 2)},
        {"Architecture": "LSTM", "Parameters": lstm_p, "Latency (ms)": round(lstm_t, 2)},
        {"Architecture": "GRU", "Parameters": gru_p, "Latency (ms)": round(gru_t, 2)}
    ])

    print("\nBenchmark Summary (1000 Forward Steps):")
    print(abubakar_results_df.to_string(index=False))

    param_saving = (1.0 - gru_p / lstm_p) * 100
    print(f"\nAbubakar's Analysis: GRU reduces parameter count by {param_saving:.1f}% vs LSTM.")
