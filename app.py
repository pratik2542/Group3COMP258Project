from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow import keras
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = keras.models.load_model("assets/model.h5")

scaler = joblib.load("assets/scaler.pkl")


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == "POST":
        data = []

        fgpa = float(request.form.get("fgpa"))
        sgpa = float(request.form.get("sgpa"))
        fln = float(request.form.get("fln"))
        fn = float(request.form.get("fn"))
        ftn = float(request.form.get("ftn"))
        cn = float(request.form.get("cn"))
        rn = float(request.form.get("rn"))
        gn = float(request.form.get("gn"))
        pen = float(request.form.get("pen"))
        agn = float(request.form.get("agn"))
        egn = float(request.form.get("egn"))
        

        data = {
            "fgpa": fgpa,
            "sgpa": sgpa,
            "fln": fln,
            "fn": fn,
            "ftn": ftn,
            "cn": cn,
            "rn": rn,
            "gn": gn,
            "pen": pen,
            "agn": agn,
            "egn": egn
        }
        # print(data)

        df = pd.DataFrame(data, index=[0])
        df.columns = ['First_Term_Gpa', 'Second_Term_Gpa', 'First_Language_numeric', 'Funding_numeric', 'FastTrack_numeric',
                      'Coop_numeric', 'Residency_numeric', 'Gender_numeric', 'Previous_Education_numeric', 'Age_Group_numeric', 'English_Grade_numeric']

        df = scaler.transform(df)
        df = df.reshape(1,1,11)
        res = model.predict(df)
        if (res > 0.5):
            res = 1
        else:
            res = 0
        res = str(res)
        res = "Result is : "+res
        return render_template("main.html", result=res)
    else:
        return render_template("main.html", result="Make prediction")