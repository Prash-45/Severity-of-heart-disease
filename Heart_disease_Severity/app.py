from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

num_cols = ['age','trestbps','chol','thalch','oldpeak']

model_columns = [
    'age',
    'trestbps',
    'chol',
    'thalch',
    'oldpeak',
    'sex_Male',
    'dataset_Hungary',
    'dataset_Switzerland',
    'dataset_VA Long Beach',
    'cp_atypical angina',
    'cp_non-anginal',
    'cp_typical angina',
    'fbs_True',
    'restecg_normal',
    'restecg_st-t abnormality',
    'exang_True',
    'slope_flat',
    'slope_upsloping',
    'thal_normal',
    'thal_reversable defect'
]

labels = {
    0:"No Heart Disease",
    1:"Stage 1 Heart Disease",
    2:"Stage 2 Heart Disease",
    3:"Stage 3 Heart Disease",
    4:"Stage 4 Heart Disease"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "age":[float(request.form["age"])],
        "sex":[request.form["sex"]],
        "dataset":[request.form["dataset"]],
        "cp":[request.form["cp"]],
        "trestbps":[float(request.form["trestbps"])],
        "chol":[float(request.form["chol"])],
        "fbs":[request.form["fbs"]],
        "restecg":[request.form["restecg"]],
        "thalch":[float(request.form["thalch"])],
        "exang":[request.form["exang"]],
        "oldpeak":[float(request.form["oldpeak"])],
        "slope":[request.form["slope"]],
        "thal":[request.form["thal"]]
    }

    df = pd.DataFrame(data)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Correct column order
    df = df[model_columns]

    # Scale numerical columns
    df[num_cols] = scaler.transform(df[num_cols])

    prediction = model.predict(df)[0]

    return render_template(
        "index.html",
        prediction_text=labels[prediction]
    )


if __name__ == "__main__":
    app.run(debug=True)