# Abubakar Iqbal
# NAVTTC Week 7 - Task 64: CNN Feature Extraction on Multi-Class Image Data

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_image_data(n_samples=320, size=16, n_classes=4):
    np.random.seed(42)
    X = np.zeros((n_samples, size, size))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        cls = i % n_classes
        y[i] = cls
        base = np.random.normal(0, 0.08, (size, size))
        if cls == 0:
            for r in range(2, size, 4):
                base[r:r+2, :] += 1.0
        elif cls == 1:
            for c in range(2, size, 4):
                base[:, c:c+2] += 1.0
        elif cls == 2:
            for d in range(size):
                base[d, d] += 1.0
                base[d, size - 1 - d] += 1.0
        elif cls == 3:
            base[3:size-3, 3:size-3] += 0.8
        X[i] = np.clip(base, 0.0, 1.0)
    return X, y

def extract_features(images):
    h_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    v_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    features = []
    for img in images:
        H, W = img.shape
        conv_h = np.zeros((H - 2, W - 2))
        conv_v = np.zeros((H - 2, W - 2))
        for r in range(H - 2):
            for c in range(W - 2):
                patch = img[r:r+3, c:c+3]
                conv_h[r, c] = np.sum(patch * h_filter)
                conv_v[r, c] = np.sum(patch * v_filter)

        feat_vector = np.array([
            np.max(conv_h), np.mean(conv_h), np.std(conv_h),
            np.max(conv_v), np.mean(conv_v), np.std(conv_v),
            np.mean(img)
        ])
        features.append(feat_vector)
    return np.array(features)

def main():
    print("NAVTTC Week 7 Task 64 - Abubakar Iqbal")

    abubakar_imgs, abubakar_labels = create_image_data(n_samples=320, size=16, n_classes=4)
    print(f"Generated Image Dataset: {abubakar_imgs.shape} across 4 classes")

    abubakar_feats = extract_features(abubakar_imgs)
    X_train, X_test, y_train, y_test = train_test_split(
        abubakar_feats, abubakar_labels, test_size=0.25, random_state=42, stratify=abubakar_labels
    )

    abubakar_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
    abubakar_mlp.fit(X_train, y_train)

    abubakar_preds = abubakar_mlp.predict(X_test)
    acc = accuracy_score(y_test, abubakar_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, abubakar_preds, zero_division=0))

if __name__ == "__main__":
    main()
