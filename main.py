import flask
from flask import Flask
import pandas as pd
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import string
import time


vectorization = TfidfVectorizer()
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()
data = {"isTrained" : False}

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

def manual_test(news, alg):
    if data['isTrained']:
        testing_news = {'text':[news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        if alg == 'LR':
            pred = LR.predict(new_xv_test)
        elif alg == "DT":
            pred = DT.predict(new_xv_test)
        elif alg == "GBC":
            pred = GBC.predict(new_xv_test)
        elif alg == "RFC":
            pred = RFC.predict(new_xv_test)
        return output_label(pred)
    else:
        return False

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def trainAll():
    print("Pulling From Dataset.....")
    df_fake = pd.read_csv("input/fake-news-detection/Fake.csv")
    df_true = pd.read_csv("input/fake-news-detection/True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1
    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480, 23470, -1):
        df_fake.drop([i], axis=0, inplace=True)
    df_true_manual_testing = df_true.tail(10)
    for i in range(21416, 21406, -1):
        df_true.drop([i], axis=0, inplace=True)
    df_fake_manual_testing["class"] = 0
    df_true_manual_testing["class"] = 1
    df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
    df_manual_testing.to_csv("manual-testing.csv")
    df_merge = pd.concat([df_fake, df_true], axis=0)
    df_merge.columns
    df = df_merge.drop(["title", "subject", "date"], axis=1)
    df.isnull().sum()
    df = df.sample(frac=1)
    df.reset_index(inplace=True)
    df.drop(["index"], axis=1, inplace=True)
    df.head()
    print("Formatting Data for Training.....")
    df["text"] = df["text"].apply(wordopt)
    x = df["text"]
    y = df["class"]
    print("Splitting Training and Testing Data....")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    xv_train = vectorization.fit_transform(x_train)
    start = time.time()
    print("Training in Logistic Regression....")
    LR.fit(xv_train, y_train)
    end = time.time()
    print(f"{round(end - start, 2)}")
    start = time.time()
    print("Training in Decision Trees....")
    DT.fit(xv_train, y_train)
    end = time.time()
    print(f"{round(end - start, 2)}")
    start = time.time()
    print("Training in Gradient Boosting Classifier....")
    GBC.fit(xv_train, y_train)
    end = time.time()
    print(f"{round(end - start, 2)}")
    start = time.time()
    print("Training in Random Forest Classifier....")
    RFC.fit(xv_train, y_train)
    end = time.time()
    print(f"{round(end - start, 2)}")
    print("Training Complete")
    data['isTrained'] = True


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return "The API is running"


@app.route("/clean", methods=['POST'])
def clean():
    newsText = flask.request.form['text']
    cleanText = wordopt(newsText)
    return flask.Response(response=cleanText, status=200)

@app.route("/train", methods=['GET'])
def train():
    if not data['isTrained']:
        trainAll()
        return flask.Response(response="Successfully Trained", status=200)
    return flask.Response(response="Already Trained", status=200)

@app.route("/evaluate", methods=['POST'])
def evaluate():
    if not data['isTrained']:
        return flask.Response(response="Not Trained", status=405)
    newsText = flask.request.form['text']
    algorithm = flask.request.form['alg']
    prediction = manual_test(newsText, algorithm)
    return flask.Response(response=prediction, status=200)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
