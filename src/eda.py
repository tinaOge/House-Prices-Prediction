import pandas as pd

def perform_eda():
    # Load training and test datasets
    df_train = pd.read_csv('data/train.csv', index_col='Id')
    df_test = pd.read_csv('data/test.csv', index_col='Id')

    # Add a placeholder for the target variable in the test set
    df_test['SalePrice'] = -1

    # Combine train and test datasets
    combined_df = pd.concat([df_train, df_test], axis=0)

    # Display initial EDA metrics
    print(f"Combined DataFrame Shape: {combined_df.shape}")
    print("Head of Combined DataFrame:")
    print(combined_df.head())
    print(f"Number of Duplicates: {combined_df.duplicated().sum()}")

    # Drop duplicates
    combined_df.drop_duplicates(inplace=True)

    # Check for missing values
    missing_values = combined_df.isna().sum()
    print("Missing Values Count:")
    print(missing_values)

    # Drop columns with more than 50% missing values
    combined_df = combined_df.loc[:, combined_df.isna().mean() < 0.5]

    # More insights
    print(f"Updated DataFrame Shape after dropping missing values: {combined_df.shape}")
    print("DataFrame Info:")
    combined_df.info()

    print("Descriptive Statistics (numerical):")
    print(combined_df.describe())

    print("Descriptive Statistics (categorical):")
    print(combined_df.describe(include='object'))

    # Split combined_df back into train and test
    df_train = combined_df[combined_df['SalePrice'] != -1]
    df_test = combined_df[combined_df['SalePrice'] == -1].drop(columns=['SalePrice'])

    return df_train, df_test

