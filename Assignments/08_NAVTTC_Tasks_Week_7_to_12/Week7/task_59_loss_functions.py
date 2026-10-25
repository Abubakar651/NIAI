# Abubakar Iqbal
# NAVTTC Week 7 - Task 59: Loss Functions

import numpy as np

class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def huber(y_true, y_pred, delta=1.0):
        err = y_true - y_pred
        abs_err = np.abs(err)
        return np.mean(np.where(abs_err <= delta, 0.5 * (err ** 2), delta * (abs_err - 0.5 * delta)))

    @staticmethod
    def binary_crossentropy(y_true, y_pred, eps=1e-12):
        p = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    @staticmethod
    def categorical_crossentropy(y_true, y_pred, eps=1e-12):
        p = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(p), axis=1))

    @staticmethod
    def hinge(y_true, y_scores):
        return np.mean(np.maximum(0, 1.0 - y_true * y_scores))

    @staticmethod
    def kl_divergence(p, q, eps=1e-12):
        p_c = np.clip(p, eps, 1.0)
        q_c = np.clip(q, eps, 1.0)
        return np.sum(p_c * np.log(p_c / q_c))

if __name__ == "__main__":
    print("NAVTTC Week 7 Task 59 - Abubakar Iqbal")

    abubakar_y_reg = np.array([12.0, 15.5, 20.0, 24.2])
    abubakar_pred_reg = np.array([11.8, 16.0, 19.1, 25.0])

    print("Regression Losses:")
    print(f"MSE:   {LossFunctions.mse(abubakar_y_reg, abubakar_pred_reg):.4f}")
    print(f"MAE:   {LossFunctions.mae(abubakar_y_reg, abubakar_pred_reg):.4f}")
    print(f"Huber: {LossFunctions.huber(abubakar_y_reg, abubakar_pred_reg, delta=1.0):.4f}")

    abubakar_bin_true = np.array([1, 0, 1, 1])
    abubakar_bin_pred = np.array([0.92, 0.15, 0.88, 0.76])
    print("\nClassification Losses:")
    print(f"Binary Cross-Entropy:      {LossFunctions.binary_crossentropy(abubakar_bin_true, abubakar_bin_pred):.4f}")

    abubakar_cat_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    abubakar_cat_pred = np.array([[0.85, 0.10, 0.05], [0.15, 0.75, 0.10], [0.05, 0.20, 0.75]])
    print(f"Categorical Cross-Entropy: {LossFunctions.categorical_crossentropy(abubakar_cat_true, abubakar_cat_pred):.4f}")

    abubakar_hinge_y = np.array([1, -1, 1, -1])
    abubakar_hinge_scores = np.array([1.5, -0.8, 0.4, -1.2])
    print(f"Hinge Loss:                {LossFunctions.hinge(abubakar_hinge_y, abubakar_hinge_scores):.4f}")

    abubakar_p = np.array([0.4, 0.4, 0.2])
    abubakar_q = np.array([0.33, 0.33, 0.34])
    print(f"KL Divergence:             {LossFunctions.kl_divergence(abubakar_p, abubakar_q):.4f}")
