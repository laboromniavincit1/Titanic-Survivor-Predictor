import pickle

import numpy as np
from flask import Flask, render_template, request, url_for

app = Flask(__name__)


@app.route("/")
def tst():
    return "hello"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")
    else:
        Pclass = float(request.form["Pclass"])
        Sex = request.form["Sex"]
        Age = float(request.form["Age"])
        SibSp = float(request.form["SibSp"])
        Parch = float(request.form["Parch"])
        Fare = float(request.form["Fare"])
        Embarked = request.form["Embarked"]
        data = np.array(
            [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked], dtype=object
        ).reshape(1, 7)

        model = pickle.load(open("model.pk1", "rb"))
        prediction = model.predict(data)
        prediction = prediction[0]
        if prediction == 1:
            return render_template(
                "index.html",
                final_result="Congratulations! You survived the Titanic journey!",
            )
        else:
            return render_template(
                "index.html",
                final_result="Our condolences, you were not able to survive the Titanic disaster.",
            )


if __name__ == "__main__":
    print("Server Started for TSP")
    app.run(debug=True)
