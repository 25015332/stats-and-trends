"""
This script analyzes nutritional data for breakfast cereals to explore
health trends. It generates visualizations and calculates statistical
moments for consumer ratings.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates a relational scatter plot to examine the relationship
    between sugar content and cereal ratings.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='sugars', y='rating')
    plt.title('Impact of Sugar Content on Consumer Ratings')
    plt.xlabel('Sugars (g)')
    plt.ylabel('Consumer Rating')
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Creates a bar plot showing the average calorie count for the
    top cereal manufacturers.
    """
    plt.figure(figsize=(12, 6))
    # Map manufacturer codes to names for better readability
    mfr_map = {'A': 'Am.H.', 'G': 'Gen.M.', 'K': 'Kellogg',
               'N': 'Nabisco', 'P': 'Post', 'Q': 'Quaker', 'R': 'Ralston'}
    df['manufacturer_name'] = df['mfr'].map(mfr_map)

    mfr_cal = df.groupby('manufacturer_name')['calories'].mean().sort_values()
    sns.barplot(
        x=mfr_cal.index,
        y=mfr_cal.values,
        hue=mfr_cal.index,
        palette='terrain',
        legend=False
    )
    plt.title('Average Calories per Serving by Manufacturer')
    plt.xlabel('Manufacturer')
    plt.ylabel('Average Calories')
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Creates a box plot to visualize the distribution of fiber
    across the main manufacturers in the dataset.
    """
    plt.figure(figsize=(10, 6))
    top_mfrs = df['mfr'].value_counts().nlargest(5).index
    df_subset = df[df['mfr'].isin(top_mfrs)]

    sns.boxplot(data=df_subset, x='mfr', y='fiber')
    plt.title('Statistical Distribution of Fiber Content by Manufacturer')
    plt.xlabel('Manufacturer Code')
    plt.ylabel('Fiber (g)')
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculates Mean, Standard Deviation, Skewness, and Excess Kurtosis
    for the specified column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Cleans the data by normalizing column names to lowercase and
    dropping entries with missing nutritional info.
    """
    # Fix for KeyError: Convert all columns to lowercase
    df.columns = [col.lower() for col in df.columns]

    print("--- Dataset Overview ---")
    print(df.head())
    print("\n--- Correlation Analysis ---")
    print(df.select_dtypes(include=['number']).corr())

    # Drop cereals with missing nutritional data (using lowercase names)
    df = df.dropna(subset=['rating', 'sugars', 'calories', 'fiber'])

    return df


def writing(moments, col):
    """
    Interprets the statistical moments for the cereal ratings.
    """
    mean, stddev, skew, excess_kurtosis = moments

    print(f'\nFor the attribute {col}:')
    print(f'Mean = {mean:.2f}, '
          f'Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, and '
          f'Excess Kurtosis = {excess_kurtosis:.2f}.')

    # Distribution interpretation logic
    skew_type = "not skewed"
    if skew > 0.5:
        skew_type = "right skewed"
    elif skew < -0.5:
        skew_type = "left skewed"

    kurt_type = "mesokurtic"
    if excess_kurtosis > 1:
        kurt_type = "leptokurtic"
    elif excess_kurtosis < -1:
        kurt_type = "platykurtic"

    print(f'The data was {skew_type} and {kurt_type}.')
    return


def main():
    """
    Main execution pipeline for breakfast cereal analysis.
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'rating'

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
