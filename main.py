import flask
from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string
import re
import string
import training as tr


LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

def manual_test(news, alg):
    testing_news = {'text':[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(tr.wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = tr.vectorization.fit_transform(new_x_test)
    if alg == 'LR':
        pred = LR.predict(new_xv_test)
    elif alg == 'DT':
        pred = DT.predict(new_xv_test)
    elif alg == 'GBC':
        pred = GBC.predict(new_xv_test)
    elif alg == 'RFC':
        pred = RFC.predict(new_xv_test)
    else:
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)
        pred = (pred_LR + pred_DT + pred_GBC + pred_RFC)/4
        if pred < 0.5:
            pred = 0
        else:
            pred = 1
    return output_label(pred)


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return "The API is running"


@app.route("/clean", methods=['GET'])
def clean():
    return "clean"

@app.route("/train", methods=['GET'])
def train():
    LR, DT, GBC, RFC = tr.trainAll()
    return flask.Response(response="Successfully Trained", status=200)

@app.route("/evaluate", methods=['POST'])
def evaluate():
    newsText = flask.request.form['text']
    algorithm = flask.request.form['alg']
    prediction = manual_test(newsText, algorithm)
    return flask.Response(response=prediction, status=200)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
