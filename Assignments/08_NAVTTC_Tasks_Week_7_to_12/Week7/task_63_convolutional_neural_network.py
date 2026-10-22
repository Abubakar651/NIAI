# Task 63 - Demonstrate Convolutional Neural Network (CNN) Architecture
# NAVTTC Outline Week 7 / Task 63
# Reference: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neuralnetworks-the-eli5-way-3bd2b1164a53

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class CNNFromScratch:
    """
    Step-by-step mathematical demonstration of Convolutional Neural Network (CNN) layers:
      1. 2D Convolution Operation (Filter / Kernel sliding across input matrix)
      2. Non-linear Activation (ReLU)
      3. Spatial Downsampling (2x2 Max-Pooling)
      4. Flattening & Fully Connected (Dense) Classification Layer
    """

    @staticmethod
    def conv2d(image, kernel, stride=1, padding=0):
        """Applies a 2D convolution with specified kernel, stride, and padding."""
        if padding > 0:
            image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")

        H, W = image.shape
        kH, kW = kernel.shape
        out_H = (H - kH) // stride + 1
        out_W = (W - kW) // stride + 1

        feature_map = np.zeros((out_H, out_W))

        for i in range(0, out_H):
            for j in range(0, out_W):
                h_start = i * stride
                h_end = h_start + kH
                w_start = j * stride
                w_end = w_start + kW
                patch = image[h_start:h_end, w_start:w_end]
                feature_map[i, j] = np.sum(patch * kernel)

        return feature_map

    @staticmethod
    def relu(matrix):
        return np.maximum(0, matrix)

    @staticmethod
    def max_pooling2d(matrix, pool_size=2, stride=2):
        """Downsamples feature map by taking maximum in each pool_size x pool_size window."""
        H, W = matrix.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1

        pooled = np.zeros((out_H, out_W))
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride
                patch = matrix[h_start:h_start + pool_size, w_start:w_start + pool_size]
                pooled[i, j] = np.max(patch)

        return pooled

    @staticmethod
    def flatten(matrix):
        return matrix.flatten()

    @staticmethod
    def dense(flat_vector, weights, bias):
        z = np.dot(flat_vector, weights) + bias
        # Softmax probability distribution
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)


if __name__ == "__main__":
    print("=" * 65)
    print("Task 63: Convolutional Neural Network (CNN) Mathematical Mechanics")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    cnn = CNNFromScratch()

    # Create a synthetic 8x8 grayscale image patch with a sharp vertical edge
    synthetic_image = np.array([
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80],
        [10, 10, 10, 10, 80, 80, 80, 80]
    ], dtype=float)

    print("\n[Layer 0] Raw Input Image Matrix (8x8):")
    print(synthetic_image)

    # Define Sobel Vertical Edge Detection Filter (3x3)
    sobel_vertical = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    print("\n[Layer 1 - Kernel] 3x3 Sobel Vertical Edge Filter:")
    print(sobel_vertical)

    # 1. Convolution (Stride=1, Padding=0) -> Output size = (8-3)/1 + 1 = 6x6
    conv_output = cnn.conv2d(synthetic_image, sobel_vertical, stride=1, padding=0)
    print(f"\n[Layer 1 - Conv2D] Feature Map (6x6):")
    print(conv_output.round(1))

    # 2. Activation Function (ReLU)
    relu_output = cnn.relu(conv_output)
    print(f"\n[Layer 2 - Activation] Post-ReLU Feature Map (Negative values zeroed):")
    print(relu_output.round(1))

    # 3. Max Pooling (2x2, Stride=2) -> Output size = (6-2)/2 + 1 = 3x3
    pooled_output = cnn.max_pooling2d(relu_output, pool_size=2, stride=2)
    print(f"\n[Layer 3 - MaxPool2D] Downsampled 3x3 Feature Map:")
    print(pooled_output.round(1))

    # 4. Flattening Layer -> 3x3 = 9-dimensional vector
    flat_vector = cnn.flatten(pooled_output)
    print(f"\n[Layer 4 - Flatten] 1D Feature Representation (Length {len(flat_vector)}):")
    print(flat_vector.round(1))

    # 5. Fully Connected (Dense) Layer with 2 Output Classes (Edge Present vs Edge Absent)
    np.random.seed(42)
    dense_weights = np.random.randn(len(flat_vector), 2) * 0.1
    dense_bias = np.array([0.1, -0.1])
    class_probs = cnn.dense(flat_vector, dense_weights, dense_bias)

    print("\n[Layer 5 - Dense/Softmax] Predicted Class Probabilities:")
    print(f"  Class 0 (No Edge): {class_probs[0]:.4f}")
    print(f"  Class 1 (Edge Detected): {class_probs[1]:.4f}")
    print(f"  Predicted Class Label: {np.argmax(class_probs)}")

    print("\nTask 63 CNN Architecture Pipeline Executed Successfully.")
