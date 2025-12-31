from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/delivery_time_model.pkl")
model_features = joblib.load("model/model_features.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Distance_km": float(request.form["Distance_km"]),
        "Preparation_Time_min": float(request.form["Preparation_Time_min"]),
        "Courier_Experience_yrs": float(request.form["Courier_Experience_yrs"]),
        "Weather": request.form["Weather"],
        "Traffic_Level": request.form["Traffic_Level"],
        "Time_of_Day": request.form["Time_of_Day"],
        "Vehicle_Type": request.form["Vehicle_Type"]
    }


    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(input_df)[0]

 
    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        data=data
    )

if __name__ == "__main__":
    app.run(debug=True)
