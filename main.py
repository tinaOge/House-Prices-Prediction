import numpy as np
import pandas as pd
from src.eda import perform_eda
from src.visualization import create_visualizations
from src.data_preprocessing import preprocess_data
from src.model import train_and_select_model
from src.model_evaluation import evaluate_model
from src.model import hyperparameter_tuning
from src.submission import create_submission
from sklearn.model_selection import train_test_split


def main():
    # Load Data
    train_data_path = 'data/train.csv'  # Update with your training data path
    test_data_path = 'data/test.csv'  # Update with your test data path

    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    # Combine the train and test DataFrames for EDA
    combined_df = pd.concat([df_train, df_test], keys=['train', 'test'])

    # Exploratory Data Analysis (EDA)
    perform_eda()

    # Data Visualization
    create_visualizations(df_train)

    # Separate features and target variable
    X = df_train.drop('SalePrice', axis=1)
    y = df_train['SalePrice']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log transform the target variable
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # Preprocess the data
    X_train_processed, X_test_processed, df_test_processed, y_train, numerical_features, categorical_features, preprocessor = preprocess_data(
        X_train, X_test, df_test, y_train)

    # Train and select the model
    best_model, X_train_selected, y_train, X_test_selected, df_test_selected = train_and_select_model(
        X_train_processed, y_train, X_test_processed, df_test_processed, numerical_features, categorical_features,
        preprocessor
    )

    # Hyperparameter Tuning
    best_model_tuned = hyperparameter_tuning(X_train_selected, y_train)

    # Evaluate Model
    evaluate_model(best_model_tuned, X_test_selected, y_test)

    # Prepare Submission
    create_submission(best_model_tuned, df_test_selected, 'data/sample_submission.csv', 'data/submission_file.csv')


if __name__ == "__main__":
    main()