import pandas as pd
df=pd.read_csv("dataset/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
print(df.head())

print("\n Datatset shape;")
print(df.shape)

print("\n Column names:")
print(df.columns.tolist())

print("\n Data types:")
print(df.dtypes)

print("\n Missing values:")
print(df.isnull().sum())

print("\n Important Categorical Columns:")

""" for column in df.columns:
    print(f"\n{column}:") 
    print(df[column].unique()[:20])  # Displaying only the first 20 unique values for brevity
"""
print("\nGender:")
print(df["gender"].unique())

print("\nAcademic work impact:")
print(df["academic_work_impact"].unique())

print("\nAddiction level:")
print(df["addiction_level"].unique())

print("\nTarget label:" )
print(df["addicted_label"].unique())

print("\nTarget distribution:")
print(df["addicted_label"].value_counts())

print(pd.crosstab(df["addiction_level"], df["addicted_label"], dropna=False))

#removing useless columns
df_model=df.drop(columns=["transaction_id","user_id","addiction_level"])

print("\n Columns after dropping useless columns:")
print(df_model.columns.tolist())

df_model.to_csv("dataset/model_cleaned_dataset.csv", index=False)

print("\nDuplicate rows:")
print(df_model.duplicated().sum())

print("\nNumerical columns:")
print(df_model.select_dtypes(include="number").columns.tolist())

print("\nCategorical columns:")
print(df_model.select_dtypes(include=["object", "string"]).columns.tolist())