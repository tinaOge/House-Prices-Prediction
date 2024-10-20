import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(best_model_tuned, X_test_selected, y_test):
    # Step 1: Predict on the test set (keep predictions in log scale)
    y_pred_log = best_model_tuned.predict(X_test_selected)

    # Calculate RMSE on the log scale
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)

    print(f"RMSE on log scale: {rmse_log:.4f}")
    print(f"R² Score on log scale: {r2_log:.4f}")
    print(f"Mean Absolute Error (MAE) on log scale: {mae_log:.4f}")

    # Inverse log transform predictions
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test)  # Inverse transform the log-scaled y_test

    # Step 3: Calculate RMSE and other metrics on the original scale
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    print(f"RMSE on original scale: {rmse_original:.4f}")
    print(f"Mean Absolute Error (MAE) on original scale: {mae:.4f}")
    print(f"R² Score on original scale: {r2:.4f}")
