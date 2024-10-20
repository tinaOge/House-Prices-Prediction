import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def preprocess_data(X_train, X_test, df_test, y_train):
    """Preprocess the data, handling multicollinearity and zero correlation."""
    numeric_features = X_train.select_dtypes(include=np.number).columns

    # Calculate correlation matrix using only numeric features
    correlation_matrix = X_train[numeric_features].corr()

    # Drop features with high multicollinearity (> 0.8)
    high_corr_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)

    # Drop the highly correlated features from both training and testing sets
    X_train = X_train.drop(columns=high_corr_features)
    X_test = X_test.drop(columns=high_corr_features)  # Ensure to drop the same features from the test set

    # Handle Zero Correlation
    # Select only numeric features for correlation calculation
    numeric_features = X_train.select_dtypes(include=np.number).columns
    correlation_with_target = X_train[numeric_features].corrwith(y_train)

    zero_corr_features = correlation_with_target[correlation_with_target.abs() < 0.04].index.tolist()  # Adjust the threshold as needed

    # Drop the features from both training and testing sets
    X_train = X_train.drop(columns=zero_corr_features)
    X_test = X_test.drop(columns=zero_corr_features)  # Drop the same features from the test set

    # Drop the same features from the unseen dataset df_test
    df_test = df_test.drop(columns=high_corr_features.union(zero_corr_features), errors='ignore')  # Use union to combine both sets

    # Identify categorical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing pipelines
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Apply preprocessing to the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    df_test_processed = preprocessor.transform(df_test)

    return X_train_processed, X_test_processed, df_test_processed, y_train, numerical_features, categorical_features, preprocessor
