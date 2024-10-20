import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(df_train):
    # Create a histogram to visualize target column
    plt.figure(figsize=(10, 6))
    sns.histplot(df_train['SalePrice'], kde=True)
    plt.title('Distribution of SalePrice')
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.show()

    # Create the correlation matrix for numerical columns
    numerical_columns = df_train.select_dtypes(include=['number']).columns
    corr_matrix = df_train[numerical_columns].corr()

    # Generate a heatmap
    plt.figure(figsize=(22, 14))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # Scatter plot of SalePrice vs. OverallQual
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_train['OverallQual'], y=df_train['SalePrice'],
                    hue=df_train['OverallQual'], palette='viridis', s=100)
    plt.title('Scatter Plot of SalePrice vs. OverallQual')
    plt.xlabel('Overall Quality')
    plt.ylabel('Sale Price')
    plt.show()

    # Scatter plot of SalePrice vs. GrLivArea
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_train['GrLivArea'], y=df_train['SalePrice'],
                    hue=df_train['GrLivArea'], palette='viridis', s=100)
    plt.title('Scatter Plot of SalePrice vs. GrLivArea')
    plt.xlabel('Ground Living Area')
    plt.ylabel('Sale Price')
    plt.show()

    # Scatter plot of SalePrice vs. Year built after 1950
    built = df_train[df_train['YearBuilt'] > 1950]
    built[['SalePrice', 'YearBuilt']].plot(
        kind='scatter', x='YearBuilt', y='SalePrice', figsize=(10, 6))
    plt.title('Scatter Plot of SalePrice vs. Year Built (after 1950)')
    plt.xlabel('Year Built')
    plt.ylabel('Sale Price')
    plt.show()

    # Scatter plot of SalePrice vs. Year remodeled after 1950
    remodel = df_train[df_train['YearRemodAdd'] > 1950]
    remodel[['SalePrice', 'YearRemodAdd']].plot(
        kind='scatter', x='YearRemodAdd', y='SalePrice', figsize=(10, 6))
    plt.title('Scatter Plot of SalePrice vs. Year Remodeled (after 1950)')
    plt.xlabel('Year Remodeled')
    plt.ylabel('Sale Price')
    plt.show()

    # Visualize 'Neighborhood' in a dataframe and plot it
    pd.DataFrame(df_train['Neighborhood'].value_counts()
                 ).plot(kind='bar', figsize=(10, 5))
    plt.title('Neighborhood Counts')
    plt.xlabel('Neighborhood')
    plt.ylabel('Count')
    plt.show()

    # Box plot of SalePrice by Neighborhood
    plt.figure(figsize=(14, 8))
    sns.boxplot(x=df_train['Neighborhood'], y=df_train['SalePrice'])
    plt.title('Box Plot of SalePrice by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Sale Price')
    plt.xticks(rotation=90)
    plt.show()

    # Box plot of SalePrice by CentralAir
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_train['CentralAir'], y=df_train['SalePrice'])
    plt.title('Box Plot of SalePrice by CentralAir')
    plt.xlabel('Central Air (Y/N)')
    plt.ylabel('Sale Price')
    plt.show()
