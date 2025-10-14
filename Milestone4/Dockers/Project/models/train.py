import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ---------------------------------------------------------------------
# 🔧 Step 1: Safe paths (works no matter where script is run)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train_database.csv")  # corrected file name
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lgb_damage_model.pkl")

# ---------------------------------------------------------------------
# 📂 Step 2: Check data file existence
# ---------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Data file not found at {DATA_PATH}\n"
        f"Please make sure 'train_database.csv' exists in the Project/data/ folder."
    )

# ---------------------------------------------------------------------
# 📊 Step 3: Load Data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------
# ⚙️ Step 4: Clean and preprocess
# ---------------------------------------------------------------------
# Drop rows with missing target
df = df.dropna(subset=["Damage_Potential"])

# Convert categorical/object columns safely
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category")

# ---------------------------------------------------------------------
# 🧩 Step 5: Define features (adjust as per your dataset)
# ---------------------------------------------------------------------
features = [
    "Latitude",
    "Longitude",
    "Depth",
    "Magnitude",
    "Root Mean Square",
    "Magnitude Type_MD",
    "Magnitude Type_MH",
    "Magnitude Type_ML",
    "Magnitude Type_MS",
    "Magnitude Type_MW",
    "Magnitude Type_MWB",
    "Magnitude Type_MWC",
    "Magnitude Type_MWR",
    "Magnitude Type_MWW",
    "Status_Reviewed"
]

# Check that all features exist in the dataset
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"❌ Missing columns in dataset: {missing_features}")

# ---------------------------------------------------------------------
# 🔢 Step 6: Split data
# ---------------------------------------------------------------------
X = df[features]
y = df["Damage_Potential"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 🤖 Step 7: Train LightGBM Regressor
# ---------------------------------------------------------------------
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 📈 Step 8: Evaluate
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Training Complete")
print(f"📊 RMSE: {rmse:.3f}")
print(f"📈 R²: {r2:.3f}")

# ---------------------------------------------------------------------
# 💾 Step 9: Save trained model
# ---------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    MODEL_PATH
)
print(f"✅ Saved model and feature list to {MODEL_PATH}")
