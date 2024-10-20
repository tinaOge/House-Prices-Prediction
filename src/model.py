import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV


def train_and_select_model(X_train_processed, y_train, X_test_processed, df_test_processed, numerical_features,
                           categorical_features, preprocessor):
    """Train the model and perform feature selection using RFECV."""
    model = LinearRegression()

    # Perform RFECV
    selector = RFECV(estimator=model, step=1, cv=5, scoring='neg_mean_squared_error')
    selector.fit(X_train_processed, y_train)

    # Access feature names
    onehot_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        categorical_features)

    # The best features are in selector.support_
    selected_features = np.array(numerical_features + list(onehot_feature_names))[selector.support_]

    # Print the selected features
    print(f"Selected Features: {selected_features.tolist()}")

    # Prepare the selected features
    X_train_selected = X_train_processed[:, selector.support_]
    X_test_selected = X_test_processed[:, selector.support_]
    df_test_selected = df_test_processed[:, selector.support_]

    # Cross-validated score of the best model
    initial_rmse = np.sqrt(-selector.cv_results_['mean_test_score'].min())
    print(f"Initial RMSE: {initial_rmse:.4f} ")

    # Define models for training
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(),
    }

    model_rmse_scores = {}
    for model_name, model_instance in models.items():
        cv_neg_rmse = cross_val_score(model_instance, X_train_selected, y_train, scoring='neg_mean_squared_error', cv=5)
        model_rmse_scores[model_name] = np.sqrt(-cv_neg_rmse.mean())
        print(f"{model_name} Cross-Validated RMSE: {model_rmse_scores[model_name]:.4f}")

    # Identify the best model based on RMSE
    best_model_name = min(model_rmse_scores, key=model_rmse_scores.get)
    best_model_rmse = model_rmse_scores[best_model_name]
    print(f"Best Model: {best_model_name} with Cross-Validated RMSE: {best_model_rmse:.4f}")

    # Fit the best model on the selected features
    best_model = models[best_model_name].fit(X_train_selected, y_train)
    return best_model, X_train_selected, y_train, X_test_selected, df_test_selected


def hyperparameter_tuning(X_train_selected, y_train):
    """Hyperparameter tuning for the Gradient Boosting model."""
    best_params = {
        'n_estimators': [200],
        'learning_rate': [0.1],
        'max_depth': [3],
        'min_samples_leaf': [4],
        'min_samples_split': [2],
        'subsample': [0.8]
    }

    # Initialize GridSearchCV for Gradient Boosting
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        best_params,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    # Fit the grid search
    grid_search.fit(X_train_selected, y_train)

    # Best parameters and score
    best_model_tuned = grid_search.best_estimator_
    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"Best Gradient Boosting RMSE: {best_rmse:.4f} with params: {grid_search.best_params_}")

    return best_model_tuned
