import pandas as pd

# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv(
    "dataset/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
)

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# =========================
# 2. Explore Categorical Data
# =========================

print("\nGender:")
print(df["gender"].unique())

print("\nAcademic work impact:")
print(df["academic_work_impact"].unique())

print("\nAddiction level:")
print(df["addiction_level"].unique())

print("\nTarget label:")
print(df["addicted_label"].unique())

print("\nTarget distribution:")
print(df["addicted_label"].value_counts())

print("\nAddiction level vs Target:")
print(
    pd.crosstab(
        df["addiction_level"],
        df["addicted_label"],
        dropna=False
    )
)


# =========================
# 3. Remove Unnecessary Columns
# =========================

df_model = df.drop(
    columns=["transaction_id", "user_id", "addiction_level"]
)

print("\nColumns after dropping unnecessary columns:")
print(df_model.columns.tolist())

# Save cleaned dataset
df_model.to_csv(
    "dataset/model_cleaned_dataset.csv",
    index=False
)


# =========================
# 4. Check Duplicates
# =========================

print("\nDuplicate rows:")
print(df_model.duplicated().sum())


# =========================
# 5. Identify Column Types
# =========================

print("\nNumerical columns:")
print(
    df_model.select_dtypes(include="number").columns.tolist()
)

print("\nCategorical columns:")
print(
    df_model.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
)


# =========================
# 6. Separate Features and Target
# =========================

X = df_model.drop("addicted_label", axis=1)
y = df_model["addicted_label"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)


# =========================
# 7. Train-Test Split
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================
# 8. Preprocessing
# =========================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_columns = [
    "gender",
    "stress_level",
    "academic_work_impact"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder=StandardScaler()
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    "\nProcessed training data shape:",
    X_train_processed.shape
)

print(
    "Processed testing data shape:",
    X_test_processed.shape
)


# =========================
# 9. Train Logistic Regression Model
# =========================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)


# =========================
# 10. Make Predictions
# =========================

y_pred = model.predict(X_test_processed)

print("\nPredictions:")
print(y_pred)


# =========================
# 11. Calculate Accuracy
# =========================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
