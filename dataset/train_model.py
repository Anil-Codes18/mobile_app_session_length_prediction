import os
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MOBILE APP SESSION LENGTH PREDICTION
# ============================================================

print("=" * 60)
print("MOBILE APP SESSION LENGTH PREDICTION")
print("=" * 60)


# ------------------------------------------------------------
# 1. DATASET PATH
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# If train_model.py is inside dataset folder
if (BASE_DIR / "mobile_app_session_cleaned.csv").exists():
    DATA_PATH = BASE_DIR / "mobile_app_session_cleaned.csv"
else:
    DATA_PATH = BASE_DIR / "dataset" / "mobile_app_session_cleaned.csv"

print("\nLoading dataset...")
print("Dataset path:", DATA_PATH)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", df.shape)

TARGET = "session_duration_sec"

print("Target:", TARGET)


# ------------------------------------------------------------
# 3. REMOVE ID / UNNECESSARY COLUMNS
# ------------------------------------------------------------

columns_to_remove = [
    "timestamp",
    "user_id",
    "session_id",
    "ip_address",
    "phone_number",
    "is_subscribed",
    "push_enabled"
]

columns_to_remove = [
    col for col in columns_to_remove
    if col in df.columns
]

df = df.drop(columns=columns_to_remove)

print("\nRemoved columns:")
print(columns_to_remove)


# ------------------------------------------------------------
# 4. REMOVE ROWS WITH MISSING TARGET
# ------------------------------------------------------------

df = df.dropna(subset=[TARGET])


# ------------------------------------------------------------
# 5. FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ------------------------------------------------------------
# 6. IDENTIFY FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ------------------------------------------------------------
# 7. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ------------------------------------------------------------
# 8. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ------------------------------------------------------------
# 9. CREATE MODELS
# ------------------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=80,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )
}


# ------------------------------------------------------------
# 10. CREATE MODELS FOLDER
# ------------------------------------------------------------

models_dir = BASE_DIR / "models"

models_dir.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 11. TRAIN MODELS
# ------------------------------------------------------------

results = []


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))


    # Save model
    file_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
        + ".pkl"
    )

    model_path = models_dir / file_name

    joblib.dump(
        pipeline,
        model_path
    )

    print("Saved:", model_path)


    # Store results
    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ------------------------------------------------------------
# 12. MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 13. SAVE RESULTS
# ------------------------------------------------------------

results_path = models_dir / "model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)


print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)