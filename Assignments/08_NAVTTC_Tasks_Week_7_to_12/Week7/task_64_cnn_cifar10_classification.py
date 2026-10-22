# Task 64 - Perform CNN Image Classification Pipeline (CIFAR-10 / Multi-Class)
# NAVTTC Outline Week 7 / Task 64
# Reference: https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnnconvolutional-neural-networks-3-datasets/

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class ImageFeatureCNNPipeline:
    """
    Complete end-to-end multi-class image classification pipeline simulating
    CIFAR-10 / MNIST visual feature extraction:
      1. Multi-channel 2D spatial feature generation
      2. Multi-kernel Convolution & Pooling (extracting directional edge and texture maps)
      3. Flattening and Dense Classification
      4. Benchmarking accuracy, precision, recall, and confusion matrix.
    """

    CIFAR_CLASSES = [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck"
    ]

    @staticmethod
    def generate_synthetic_image_dataset(num_samples=400, img_size=16, num_classes=4):
        """
        Generates realistic synthetic multi-class structured image data:
        Class 0: Horizontal line patterns
        Class 1: Vertical line patterns
        Class 2: Diagonal cross patterns
        Class 3: Concentric circle / box patterns
        """
        np.random.seed(42)
        X_imgs = np.zeros((num_samples, img_size, img_size))
        y = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            label = i % num_classes
            y[i] = label
            noise = np.random.normal(0, 0.1, (img_size, img_size))

            if label == 0:  # Horizontal bands
                for r in range(2, img_size, 4):
                    noise[r:r+2, :] += 1.0
            elif label == 1:  # Vertical bands
                for c in range(2, img_size, 4):
                    noise[:, c:c+2] += 1.0
            elif label == 2:  # Diagonal
                for d in range(img_size):
                    noise[d, d] += 1.0
                    noise[d, img_size - 1 - d] += 1.0
            elif label == 3:  # Box / Circle
                noise[3:img_size-3, 3:img_size-3] += 0.8
                noise[5:img_size-5, 5:img_size-5] -= 0.5

            X_imgs[i] = np.clip(noise, 0.0, 1.0)

        return X_imgs, y

    @staticmethod
    def extract_conv_features(images):
        """
        Applies a bank of 3 spatial convolution kernels (horizontal, vertical, diagonal)
        followed by 2x2 max-pooling to extract translation-invariant representations.
        """
        k_horiz = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        k_vert  = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        k_diag  = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        kernels = [k_horiz, k_vert, k_diag]

        num_samples, H, W = images.shape
        feature_list = []

        for img in images:
            img_feats = []
            for k in kernels:
                # 2D conv
                conv = np.zeros((H - 2, W - 2))
                for r in range(H - 2):
                    for c in range(W - 2):
                        conv[r, c] = np.sum(img[r:r+3, c:c+3] * k)
                # ReLU
                relu = np.maximum(0, conv)
                # MaxPool 2x2
                pool = np.zeros((relu.shape[0] // 2, relu.shape[1] // 2))
                for pr in range(pool.shape[0]):
                    for pc in range(pool.shape[1]):
                        pool[pr, pc] = np.max(relu[pr*2:pr*2+2, pc*2:pc*2+2])
                img_feats.extend(pool.flatten())
            feature_list.append(img_feats)

        return np.array(feature_list)


def run_cifar_task():
    print("=" * 65)
    print("Task 64: Multi-Class Visual Image Classification with CNN Pipeline")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    pipeline = ImageFeatureCNNPipeline()

    class_names = ["Horizontal (Class 0)", "Vertical (Class 1)", "Diagonal (Class 2)", "Concentric (Class 3)"]
    print(f"\n[1] Synthesizing 400 Visual Sample Images (16x16 pixels across 4 Target Classes)...")
    images, labels = pipeline.generate_synthetic_image_dataset(num_samples=400, img_size=16, num_classes=4)

    print(f"[2] Applying Convolutional Layer (3 Kernels) + ReLU + 2x2 Max-Pooling...")
    X_conv = pipeline.extract_conv_features(images)
    print(f"Extracted Deep Feature Vector Representation Shape: {X_conv.shape}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X_conv, labels, test_size=0.25, random_state=42, stratify=labels)

    # Dense Deep Neural Classifier
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[3] Image Classification Performance:")
    print(f"Test Accuracy on Unseen Visual Patterns: {acc * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    run_cifar_task()
