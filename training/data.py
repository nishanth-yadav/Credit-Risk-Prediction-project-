import pandas as pd
import numpy as np
import random

# For reproducibility
np.random.seed(42)

rows = 10000   # ✅ Generate 10,000 Records

data = []

for _ in range(rows):

    age = np.random.randint(21, 65)

    income = np.random.randint(120000, 2500000)  # Annual income

    employment_type = random.choice(["Salaried", "Self-Employed", "Business"])

    years_employed = np.random.randint(0, age - 20)

    loan_amount = np.random.randint(50000, 2000000)

    loan_term = random.choice([12, 24, 36, 60, 120, 180])

    credit_history = random.choice(["Good", "Average", "Bad"])

    existing_debt = np.random.randint(0, 800000)

    debt_to_income = existing_debt / (income + 1)

    previous_default = random.choice([0, 1])

    savings = np.random.randint(0, 1500000)

    dependents = np.random.randint(0, 6)

    property_owned = random.choice([0, 1])

    loan_purpose = random.choice(["Home", "Car", "Business", "Education", "Personal"])

    # -------------------------
    # Risk Logic (Bank-like Rules)
    # -------------------------
    risk_score = 0

    if income < 400000:
        risk_score += 2

    if debt_to_income > 0.45:
        risk_score += 2

    if credit_history == "Bad":
        risk_score += 3
    elif credit_history == "Average":
        risk_score += 1

    if previous_default == 1:
        risk_score += 3

    if loan_amount > income * 0.7:
        risk_score += 2

    if years_employed < 2:
        risk_score += 1

    if savings < 50000:
        risk_score += 1

    # Final Label
    risk_label = 1 if risk_score >= 6 else 0   # 1 = High Risk, 0 = Low Risk

    data.append([
        age, income, employment_type, years_employed,
        loan_amount, loan_term, credit_history,
        existing_debt, debt_to_income, previous_default,
        savings, dependents, property_owned, loan_purpose,
        risk_label
    ])

columns = [
    "Age", "Income", "Employment_Type", "Years_Employed",
    "Loan_Amount", "Loan_Term", "Credit_History",
    "Existing_Debt", "Debt_to_Income", "Previous_Default",
    "Savings", "Dependents", "Property_Owned", "Loan_Purpose",
    "Risk_Label"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset
output_path = "../dataset/synthetic_credit_risk_10000.csv"
df.to_csv(output_path, index=False)

print("✅ Dataset Generated Successfully!")
print("Saved at:", output_path)
print("Shape:", df.shape)