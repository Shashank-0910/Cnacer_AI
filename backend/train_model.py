# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ------------------------------------------------------------
# 1️⃣  Locate the dataset automatically (NO hard-coded drive)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "cancer_patient_training_dataset.xlsx")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset NOT FOUND at: {file_path}")

print(f"📁 Loading dataset from: {file_path}")

df = pd.read_excel(file_path)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------------------------------------
# 2️⃣ Separate features and target
# ------------------------------------------------------------

target_col = "stage"
X = df.drop(target_col, axis=1)
y = df[target_col]

# ------------------------------------------------------------
# 3️⃣ Detect categorical & numerical columns
# ------------------------------------------------------------

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# ------------------------------------------------------------
# 4️⃣ Preprocessing
# ------------------------------------------------------------

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# ------------------------------------------------------------
# 5️⃣ Define the model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

# ------------------------------------------------------------
# 6️⃣ Create ML pipeline
# ------------------------------------------------------------

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# ------------------------------------------------------------
# 7️⃣ Split data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 8️⃣ Train model
# ------------------------------------------------------------

clf.fit(X_train, y_train)

# ------------------------------------------------------------
# 9️⃣ Model Evaluation
# ------------------------------------------------------------

y_pred = clf.predict(X_test)
print("\n🎯 Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 🔟 Save Model
# ------------------------------------------------------------

model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "cancer_model.pkl")
joblib.dump(clf, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")

