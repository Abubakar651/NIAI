# Abubakar Iqbal
# NAVTTC Week 7 - Task 63: 2D CNN (Image) & 1D CNN (Text) Mechanics

import numpy as np

class CNNLayers:
    @staticmethod
    def conv2d(image, kernel, stride=1, padding=0):
        if padding > 0:
            image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
        H, W = image.shape
        kH, kW = kernel.shape
        out_H = (H - kH) // stride + 1
        out_W = (W - kW) // stride + 1

        feature_map = np.zeros((out_H, out_W))
        for i in range(out_H):
            for j in range(out_W):
                patch = image[i*stride : i*stride + kH, j*stride : j*stride + kW]
                feature_map[i, j] = np.sum(patch * kernel)
        return feature_map

    @staticmethod
    def max_pool2d(matrix, pool_size=2, stride=2):
        H, W = matrix.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1
        pooled = np.zeros((out_H, out_W))
        for i in range(out_H):
            for j in range(out_W):
                patch = matrix[i*stride : i*stride + pool_size, j*stride : j*stride + pool_size]
                pooled[i, j] = np.max(patch)
        return pooled

    @staticmethod
    def conv1d(sequence, kernel, stride=1):
        seq_len, in_dim = sequence.shape
        k_len, _ = kernel.shape
        out_len = (seq_len - k_len) // stride + 1
        out = np.zeros(out_len)
        for i in range(out_len):
            patch = sequence[i*stride : i*stride + k_len, :]
            out[i] = np.sum(patch * kernel)
        return out

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

if __name__ == "__main__":
    print("NAVTTC Week 7 Task 63 - Abubakar Iqbal")

    abubakar_img = np.array([
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80]
    ], dtype=float)

    sobel_filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    conv2d_out = CNNLayers.conv2d(abubakar_img, sobel_filter, stride=1, padding=0)
    relu_out = CNNLayers.relu(conv2d_out)
    pool_out = CNNLayers.max_pool2d(relu_out, pool_size=2, stride=2)
    flat = pool_out.flatten()

    np.random.seed(42)
    W_dense = np.random.randn(len(flat), 2) * 0.1
    b_dense = np.array([0.05, -0.05])
    probs_2d = CNNLayers.softmax(np.dot(flat, W_dense) + b_dense)

    print("2D CNN Image Output:")
    print(f"Feature Map Shape: {conv2d_out.shape} -> Pooled Shape: {pool_out.shape}")
    print(f"Class Probabilities: Class 0: {probs_2d[0]:.4f}, Class 1: {probs_2d[1]:.4f}")

    print("\n1D CNN Text Document Classification:")
    abubakar_doc_embeddings = np.array([
        [0.8, 0.1, 0.4, 0.9],
        [0.2, 0.7, 0.3, 0.1],
        [0.9, 0.8, 0.6, 0.8],
        [0.1, 0.2, 0.9, 0.3],
        [0.5, 0.6, 0.7, 0.4]
    ], dtype=float)

    kernel_1d = np.array([
        [0.3, 0.2, 0.1, 0.5],
        [0.4, 0.1, 0.3, 0.2]
    ], dtype=float)

    conv1d_out = CNNLayers.conv1d(abubakar_doc_embeddings, kernel_1d, stride=1)
    act1d_out = CNNLayers.relu(conv1d_out)
    global_max = np.max(act1d_out)

    W1d = np.array([[0.6, -0.4]])
    b1d = np.array([0.1, -0.1])
    probs_1d = CNNLayers.softmax(np.dot(np.array([[global_max]]), W1d) + b1d)[0]

    print(f"1D Conv Output Vector: {conv1d_out.round(3)}")
    print(f"Global Max-Pooled Value: {global_max:.3f}")
    print(f"Text Document Class Probabilities: {probs_1d.round(4)}")
