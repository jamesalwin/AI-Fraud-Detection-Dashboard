from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load trained model
model = joblib.load("model.pkl")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===============================
# SINGLE TRANSACTION PREDICTION
# ===============================
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    try:
        df = pd.read_csv(file.file)

        original_df = df.copy()

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        batch_size = 10000
        predictions = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_pred = model.predict(batch)
            predictions.extend(batch_pred)

        predictions = np.array(predictions)

        total = len(predictions)
        fraud_count = int(predictions.sum())
        legit_count = total - fraud_count
        fraud_percent = round((fraud_count / total) * 100, 2)

        # ---- Fraud Trend ----
        original_df["Prediction"] = predictions
        original_df["Hour"] = (original_df["Time"] // 3600).astype(int)

        fraud_trend = (
            original_df[original_df["Prediction"] == 1]
            .groupby("Hour")
            .size()
            .to_dict()
        )

        # ---- Feature Importance ----
        importances = model.feature_importances_
        feature_names = df.columns

        feature_importance = dict(
            sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )

        # ---- Save Prediction File ----
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
                "error": None
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": str(e)
            },
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
