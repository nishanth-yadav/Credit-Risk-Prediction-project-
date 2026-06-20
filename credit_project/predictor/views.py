import os
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from django.conf import settings
from .model_trainer import train_all_models
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.contrib import messages

MODEL_PATH = os.path.join(settings.BASE_DIR, "saved_model", "credit_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "saved_model", "scaler.pkl")
COLUMNS_PATH = os.path.join(settings.BASE_DIR, "saved_model", "model_columns.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_columns = joblib.load(COLUMNS_PATH)


latest_result = {}


def landing(request):
    return render(request, "landing.html")



def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("register")

        User.objects.create_user(username=username, password=password)
        messages.success(request, "Account created successfully")
        return redirect("login")

    return render(request, "register.html")



def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid login")

    return render(request, "login.html")



def user_logout(request):
    logout(request)
    return redirect("landing")


def home(request):
    return render(request, "home.html")


def input_page(request):
    return render(request, "input.html")


def predict(request):
    global latest_result

    if request.method == "POST":
        try:
            # Collect user inputs
            input_data = {
                "Age": float(request.POST["Age"]),
                "Income": float(request.POST["Income"]),
                "Employment_Type": request.POST["Employment_Type"],
                "Years_Employed": float(request.POST["Years_Employed"]),
                "Loan_Amount": float(request.POST["Loan_Amount"]),
                "Loan_Term": float(request.POST["Loan_Term"]),
                "Credit_History": request.POST["Credit_History"],
                "Existing_Debt": float(request.POST["Existing_Debt"]),
                "Debt_to_Income": float(request.POST["Debt_to_Income"]),
                "Previous_Default": int(request.POST["Previous_Default"]),
                "Savings": float(request.POST["Savings"]),
                "Dependents": int(request.POST["Dependents"]),
                "Property_Owned": int(request.POST["Property_Owned"]),
                "Loan_Purpose": request.POST["Loan_Purpose"],
            }

            # Convert to DataFrame
            df = pd.DataFrame([input_data])

            # One-Hot Encode (same as training)
            df = pd.get_dummies(df)

            # Align with training columns
            df = df.reindex(columns=model_columns, fill_value=0)

            # Scale features
            scaled = scaler.transform(df)

            # Prediction
            prediction = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][prediction]

            latest_result = {
                "prediction": "High Risk" if prediction == 1 else "Low Risk",
                "probability": round(probability * 100, 2),
                "input_data": input_data
            }

            return render(request, "prediction.html", latest_result)

        except Exception as e:
            return render(request, "prediction.html", {"error": str(e)})

    return render(request, "input.html")


def explanation(request):
    global latest_result

    if not latest_result:
        return render(request, "explanation.html", {"error": "No prediction available."})

    data = latest_result["input_data"]
    reasons = []

    if data["Income"] < 400000:
        reasons.append("Income is below safe lending threshold.")

    if data["Debt_to_Income"] > 0.45:
        reasons.append("Debt-to-Income ratio is too high.")

    if data["Credit_History"] == "Bad":
        reasons.append("Poor credit history detected.")

    if data["Previous_Default"] == 1:
        reasons.append("Applicant has previous loan default.")

    if data["Years_Employed"] < 2:
        reasons.append("Employment stability is low.")

    if not reasons:
        reasons.append("Financial profile is stable and meets approval criteria.")

    latest_result["reasons"] = reasons

    return render(request, "explanation.html", latest_result)


def about_model(request):
    details = {
        "dataset": "Credit Risk Dataset",
        "features": "14 Financial Attributes",
        "model": "Random Forest Classifier",
        "training": "Supervised Learning",
        "explainability": "Rule-Based LLM Simulation",
        "framework": "Django",
        "language": "Python"
    }

    return render(request, "about.html", details)


def train_page(request):
    return render(request, "train.html")


def start_training(request):
    dataset_path = os.path.join(settings.BASE_DIR, "dataset", "credit_risk.csv")

    # Folder to store confusion matrix images
    static_dir = os.path.join(settings.BASE_DIR, "static")

    # Create static folder if not exists
    os.makedirs(static_dir, exist_ok=True)

    # Train models and generate images
    results = train_all_models(dataset_path, static_dir)

    return render(request, "training_results.html", {"results": results})