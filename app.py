from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load BOTH models
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")


# ===============================
# HOME PAGE
# ===============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===============================
# CSV TRANSACTION PREDICTION
# ===============================
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    try:
        df = pd.read_csv(file.file)
        original_df = df.copy()

        y_true = None
        if "Class" in df.columns:
            y_true = df["Class"]
            df = df.drop("Class", axis=1)

        # ===============================
        # Batch Prediction
        # ===============================
        batch_size = 10000
        rf_predictions = []
        lr_predictions = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            rf_pred = rf_model.predict(batch)
            lr_pred = lr_model.predict(batch)

            rf_predictions.extend(rf_pred)
            lr_predictions.extend(lr_pred)

        rf_predictions = np.array(rf_predictions)
        lr_predictions = np.array(lr_predictions)

        # Use Random Forest as primary dashboard model
        predictions = rf_predictions

        # ===============================
        # Basic Statistics
        # ===============================
        total = len(predictions)
        fraud_count = int(predictions.sum())
        legit_count = total - fraud_count
        fraud_percent = round((fraud_count / total) * 100, 2)

        # ===============================
        # Fraud Trend
        # ===============================
        original_df["Prediction"] = predictions

        if "Time" in original_df.columns:
            original_df["Hour"] = (original_df["Time"] // 3600).astype(int)
            fraud_trend = (
                original_df[original_df["Prediction"] == 1]
                .groupby("Hour")
                .size()
                .to_dict()
            )
        else:
            fraud_trend = {}

        # ===============================
        # Feature Importance (RF only)
        # ===============================
        feature_importance = {}
        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            feature_importance = dict(
                sorted(
                    zip(df.columns, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )

        # ===============================
        # Metrics + ROC
        # ===============================
        accuracy = precision = recall = f1 = None
        conf_matrix = None
        rf_auc = lr_auc = None
        rf_fpr = rf_tpr = lr_fpr = lr_tpr = []

        if y_true is not None:

            # Metrics using RF (primary)
            accuracy = round(accuracy_score(y_true, predictions), 4)
            precision = round(precision_score(y_true, predictions), 4)
            recall = round(recall_score(y_true, predictions), 4)
            f1 = round(f1_score(y_true, predictions), 4)
            conf_matrix = confusion_matrix(y_true, predictions).tolist()

            # Probabilities
            rf_probs = rf_model.predict_proba(df)[:, 1]
            lr_probs = lr_model.predict_proba(df)[:, 1]

            # AUC
            rf_auc = round(roc_auc_score(y_true, rf_probs), 4)
            lr_auc = round(roc_auc_score(y_true, lr_probs), 4)

            # ROC Curves
            rf_fpr, rf_tpr, _ = roc_curve(y_true, rf_probs)
            lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_probs)

        # ===============================
        # Save Predictions
        # ===============================
        original_df.to_csv("static/predictions.csv", index=False)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "total": total,
                "fraud_count": fraud_count,
                "legit_count": legit_count,
                "fraud_percent": fraud_percent,
                "fraud_trend": fraud_trend,
                "feature_importance": feature_importance,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": conf_matrix,
                "rf_auc": rf_auc,
                "lr_auc": lr_auc,
                "rf_fpr": rf_fpr.tolist() if y_true is not None else [],
                "rf_tpr": rf_tpr.tolist() if y_true is not None else [],
                "lr_fpr": lr_fpr.tolist() if y_true is not None else [],
                "lr_tpr": lr_tpr.tolist() if y_true is not None else [],
                "error": None,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": str(e)},
        )


# ===============================
# JSON API ENDPOINT
# ===============================
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict_api")
def predict_api(transaction: Transaction):

    data = [[
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount
    ]]

    prediction = rf_model.predict(data)[0]
    probability = rf_model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
