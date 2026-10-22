import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Data Loading and Initial Analysis
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)

    print("--- First 5 Rows ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())

    # 2. Data Transformations and Aggregations
    # Impute missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop Cabin due to high missing values
    df.drop('Cabin', axis=1, inplace=True)
    
    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. Visualizations
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Visualization 1: Survival by Gender
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Survived', hue='Sex', palette='pastel')
    plt.title('Survival Count by Gender')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Gender')
    plt.savefig('survival_by_gender.png')
    plt.close()

    # Visualization 2: Survival by Passenger Class
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Pclass', y='Survived', palette='viridis', errorbar=None)
    plt.title('Survival Rate by Passenger Class')
    plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
    plt.ylabel('Survival Probability')
    plt.savefig('survival_by_class.png')
    plt.close()

    # Visualization 3: Age Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Survived'] == 1]['Age'], label='Survived', fill=True, color='green')
    sns.kdeplot(data=df[df['Survived'] == 0]['Age'], label='Did Not Survive', fill=True, color='red')
    plt.title('Age Distribution: Survivors vs. Non-Survivors')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('age_distribution.png')
    plt.close()

    # Visualization 4: Correlation Matrix
    plt.figure(figsize=(10, 8))
    numerical_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']]
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    print("\nEDA complete. Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
