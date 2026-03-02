# ===============================
# Loan Approval ML Model
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Dataset
data = pd.read_csv("loan_data.csv")

# 2. Separate Input and Output
X = data[["Income", "LoanAmount", "Credit_History"]]
y = data["Loan_Status"]

# 3. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Test Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# 6. Save Model
joblib.dump(model, "loan_model.pkl")

print("Model saved successfully!")