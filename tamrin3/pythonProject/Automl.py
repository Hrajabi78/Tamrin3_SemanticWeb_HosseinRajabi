import requests
import pandas as pd
from flask import Flask, request, render_template
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from datetime import datetime

# -----------------------------
# تنظیمات
# -----------------------------
APP_PORT = 5000
SEED = 42
MAX_RUNTIME_SECS = 60
MAX_MODELS = 10

app = Flask(__name__)
aml = None

TARGET = "magnitude"
FEATURES = ["longitude", "latitude", "depth", "time"]

# -----------------------------
# init H2O
# -----------------------------
def init_h2o():
    if not h2o.connection():
        h2o.init()
        h2o.remove_all()

# -----------------------------
# بارگذاری داده زلزله از USGS و آموزش
# -----------------------------
def load_and_train():
    global aml
    init_h2o()

    # دریافت داده زلزله از USGS (۳۰ روز گذشته، بزرگی > 4)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": "2025-08-01",
        "endtime": "2025-09-01",
        "minmagnitude": 4,
        "minlatitude": -1,   # نزدیک استوا
        "maxlatitude": 80,   # تا شمال آسیا
        "minlongitude": 25,  # از خاورمیانه
        "maxlongitude": 180  # تا شرق آسیا و اقیانوس آرام
    }
    res = requests.get(url, params=params)
    data = res.json()

    # استخراج ویژگی‌ها
    records = []
    for feature in data["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]  # [lon, lat, depth]
        records.append({
            "time": pd.to_datetime(props["time"], unit="ms").timestamp(),  # تبدیل به timestamp
            "magnitude": props["mag"],
            "longitude": coords[0],
            "latitude": coords[1],
            "depth": coords[2]
        })

    df = pd.DataFrame(records).dropna()

    # تقسیم داده
    train, _ = train_test_split(df, test_size=0.2, random_state=SEED)
    train_h2o = h2o.H2OFrame(train)

    # آموزش AutoML
    aml = H2OAutoML(
        max_models=MAX_MODELS,
        max_runtime_secs=MAX_RUNTIME_SECS,
        seed=SEED,
        sort_metric="RMSE",
        verbosity="info"
    )
    aml.train(x=FEATURES, y=TARGET, training_frame=train_h2o)

load_and_train()

# -----------------------------
# فانکشن پیش‌بینی
# -----------------------------
def predict_values(data_dict):
    df = pd.DataFrame([data_dict])
    h2o_df = h2o.H2OFrame(df)
    preds = aml.leader.predict(h2o_df).as_data_frame()
    return float(preds.iloc[0, 0])

# -----------------------------
# صفحات Flask
# -----------------------------
@app.get("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.post("/predict")
def predict_form():
    try:
        # خواندن طول، عرض، عمق
        longitude = float(request.form.get("longitude"))
        latitude = float(request.form.get("latitude"))
        depth = float(request.form.get("depth"))

        # تبدیل رشته زمان به timestamp
        time_str = request.form.get("time_input")  # مثال: "2025-09-06 12:30:00"
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        timestamp = dt.timestamp()  # عدد ثانیه از 1970-01-01

        # داده ورودی به مدل
        data_dict = {
            "longitude": longitude,
            "latitude": latitude,
            "depth": depth,
            "timestamp": timestamp
        }

        yhat = predict_values(data_dict)  # پیش‌بینی ریشتر
        return render_template("result.html", yhat=f"{yhat:.2f}")
    except Exception as e:
        return render_template("error.html", error=str(e))
@app.get("/leaderboard")
def leaderboard():
    lb = aml.leaderboard.as_data_frame()
    wanted_cols = ["model_id", "algo", "rmse", "mae", "r2"]
    cols = [c for c in wanted_cols if c in lb.columns]
    lb = lb[cols]
    rows = lb.values.tolist()
    return render_template("leaderboard.html", columns=lb.columns, rows=rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)

