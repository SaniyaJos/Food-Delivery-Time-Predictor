import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import joblib


df = pd.read_csv("Food_Delivery_Times.csv")
df = df.drop(columns=["Order_ID"], errors="ignore")


df = df.dropna()

df_encoded = pd.get_dummies(
    df,
    columns=["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"],
    drop_first=True
)


X = df_encoded.drop("Delivery_Time_min", axis=1)
y = df_encoded["Delivery_Time_min"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=70
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(r2_score(y_test, y_pred)*100)
print("MAE:", mae)
print("MSE:", mse)

joblib.dump(model,"delivery_time_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
