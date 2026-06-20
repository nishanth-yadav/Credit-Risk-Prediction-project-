import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def save_confusion_matrix(cm, name, static_dir):
    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()

    labels = ["Low Risk", "High Risk"]
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    image_path = os.path.join(static_dir, f"{name}_cm.png")
    plt.savefig(image_path)
    plt.close()

    return f"{name}_cm.png"


def train_all_models(dataset_path, static_dir):

    df = pd.read_csv(dataset_path)

    X = df.drop("Risk_Label", axis=1)
    y = df["Risk_Label"]

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "DecisionTree": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=2000)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        image_name = save_confusion_matrix(cm, name, static_dir)

        results.append({
            "name": name,
            "accuracy": round(acc*100,2),
            "precision": round(prec*100,2),
            "recall": round(rec*100,2),
            "f1": round(f1*100,2),
            "image": image_name
        })

    return results