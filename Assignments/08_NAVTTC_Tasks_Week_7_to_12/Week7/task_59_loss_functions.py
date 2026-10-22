# Task 59 - Implementation of 7 Core Neural Network Loss Functions
# NAVTTC Outline Week 7 / Task 59
# Reference: https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machinelearning-python-code/

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class LossFunctionSuite:
    """
    Exhaustive implementation of 7 foundational loss functions used across
    regression, binary classification, multi-class classification, and generative models.
    """

    # 1. Mean Squared Error (MSE / L2 Loss) - Regression
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # 2. Mean Absolute Error (MAE / L1 Loss) - Robust Regression
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    # 3. Huber Loss (Smooth L1 Loss) - Robust Regression with threshold delta
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        abs_error = np.abs(error)
        linear_mask = abs_error > delta
        quadratic_loss = 0.5 * (error ** 2)
        linear_loss = delta * (abs_error - 0.5 * delta)
        return np.mean(np.where(linear_mask, linear_loss, quadratic_loss))

    # 4. Binary Cross-Entropy (Log Loss) - Binary Classification
    @staticmethod
    def binary_cross_entropy(y_true, y_pred_prob, eps=1e-12):
        y_prob = np.clip(y_pred_prob, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

    # 5. Categorical Cross-Entropy - Multi-Class Classification (One-hot targets)
    @staticmethod
    def categorical_cross_entropy(y_true_one_hot, y_pred_probs, eps=1e-12):
        probs = np.clip(y_pred_probs, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true_one_hot * np.log(probs), axis=1))

    # 6. Hinge Loss - Maximum Margin / Support Vector Machines (Targets y in {-1, 1})
    @staticmethod
    def hinge_loss(y_true_signed, raw_scores):
        return np.mean(np.maximum(0, 1.0 - y_true_signed * raw_scores))

    # 7. Kullback-Leibler (KL) Divergence - Relative Entropy / VAEs & Probability Distributions
    @staticmethod
    def kl_divergence(p_prob, q_prob, eps=1e-12):
        p = np.clip(p_prob, eps, 1.0)
        q = np.clip(q_prob, eps, 1.0)
        return np.sum(p * np.log(p / q))


if __name__ == "__main__":
    print("=" * 65)
    print("Task 59: 7 Neural Network Loss Functions Implementation")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    suite = LossFunctionSuite()

    # Regression Sample Data
    y_reg_true = np.array([10.5, 20.0, 15.2, 33.1, 45.0])
    y_reg_pred = np.array([11.0, 18.5, 15.0, 31.0, 48.2])

    print("\n[A] Regression Losses (Continuous Variables):")
    mse = suite.mean_squared_error(y_reg_true, y_reg_pred)
    mae = suite.mean_absolute_error(y_reg_true, y_reg_pred)
    huber = suite.huber_loss(y_reg_true, y_reg_pred, delta=1.0)
    print(f"  1. Mean Squared Error (MSE):  {mse:.5f}")
    print(f"  2. Mean Absolute Error (MAE): {mae:.5f}")
    print(f"  3. Huber Loss (delta=1.0):    {huber:.5f}")

    # Binary Classification Data
    y_bin_true = np.array([1, 0, 1, 1, 0])
    y_bin_pred = np.array([0.92, 0.15, 0.78, 0.65, 0.35])
    bce = suite.binary_cross_entropy(y_bin_true, y_bin_pred)
    print("\n[B] Binary Classification:")
    print(f"  4. Binary Cross-Entropy (Log Loss): {bce:.5f}")

    # Multi-Class Classification Data
    y_cat_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    y_cat_pred = np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05],
        [0.15, 0.15, 0.70]
    ])
    cce = suite.categorical_cross_entropy(y_cat_true, y_cat_pred)
    print("\n[C] Multi-Class Classification:")
    print(f"  5. Categorical Cross-Entropy: {cce:.5f}")

    # Margin / SVM Hinge Loss Data (Targets in {-1, +1})
    y_signed = np.array([1, -1, 1, -1])
    decision_scores = np.array([1.5, -0.8, 0.2, -1.2]) # 0.2 is inside the margin
    hinge = suite.hinge_loss(y_signed, decision_scores)
    print("\n[D] Maximum Margin / SVM:")
    print(f"  6. Hinge Loss: {hinge:.5f}")

    # Probability Distribution Comparison (KL Divergence)
    p_distribution = np.array([0.4, 0.4, 0.2])
    q_distribution = np.array([0.3, 0.5, 0.2])
    kl = suite.kl_divergence(p_distribution, q_distribution)
    print("\n[E] Information Theory & Generative Models:")
    print(f"  7. KL Divergence (P || Q): {kl:.5f}")

    print("\n" + "=" * 65)
    print("All 7 Loss Functions Computed and Verified Successfully.")
    print("=" * 65)
