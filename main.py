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

# removing useless columns
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

#seaparate features and target
X = df_model.drop("addicted_label", axis=1)
y = df_model["addicted_label"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining data:",X_train.shape)
print("Testing data:",X_test.shape)

"""using sklearn's ColumnTransformer to preprocess categorical features using OneHotEncoder. 
The remainder of the columns will be passed through without any changes."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
categorical_columns=["gender","stress_level","academic_work_impact"]
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_columns)
    ],
    remainder=StandardScaler()
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)

# Training Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
print("\nPredictions:")
print(y_pred)