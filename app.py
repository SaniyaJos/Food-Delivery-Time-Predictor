from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/delivery_time_model.pkl")
model_features = joblib.load("model/model_features.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Step 1: Read form data
    data = {
        "Distance_km": float(request.form["Distance_km"]),
        "Preparation_Time_min": float(request.form["Preparation_Time_min"]),
        "Courier_Experience_yrs": float(request.form["Courier_Experience_yrs"]),
        "Weather": request.form["Weather"],
        "Traffic_Level": request.form["Traffic_Level"],
        "Time_of_Day": request.form["Time_of_Day"],
        "Vehicle_Type": request.form["Vehicle_Type"]
    }

    # Step 2: Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Step 3: Encode categoricals
    input_df = pd.get_dummies(input_df)

    # Step 4: Align with training features
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Step 5: Predict
    prediction = model.predict(input_df)[0]

    # Step 6: Show result page
    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        data=data
    )

if __name__ == "__main__":
    app.run(debug=True)
