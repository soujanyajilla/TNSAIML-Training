import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("dataset/retail_sales_forecast_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

features = [
    "Year", "Month", "Product_Category", "Region",
    "Weather_Condition", "Season", "Inventory_Level",
    "Unit_Price", "Discount_Percent", "Promotion",
    "Competitor_Price"
]

target = "Units_Sold"

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),

    ("num", SimpleImputer(strategy="median"), numerical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100, random_state=42
    ))
])

model.fit(X_train, y_train)
print("\nModel Training Completed!")

pred = model.predict(X_test)

print("\nModel Evaluation")
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2 Score:", r2_score(y_test, pred))

joblib.dump(model, "model.joblib")
print("\nModel saved successfully!")
