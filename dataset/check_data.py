import pandas as pd

# Load dataset
df = pd.read_csv("dataset/mobile_app_session_cleaned.csv")

print("===================================")
print("MOBILE APP SESSION DATASET")
print("===================================")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Statistics:")
print(df["session_duration_sec"].describe())