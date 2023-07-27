from flask import Flask
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string

def manualTesting(df_fake, df_true):
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
    return df_manual_testing

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



app = Flask(__name__)
df_fake = pd.read_csv("input/fake-news-detection/Fake.csv")
df_true = pd.read_csv("input/fake-news-detection/True.csv")
df_fake["class"] = 0
df_true["class"] = 1
try:
    df_manual_testing = pd.read_csv("manual-testing.csv")
except:
    df_manual_testing = manualTesting(df_fake,df_true)
df_merge = pd.concat([df_fake,df_true], axis=0)
df_merge.columns
df = df_merge.drop(["title","subject","date"], axis = 1)
df.isnull().sum()
df = df.sample(frac = 1)
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.fit_transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

GBC = GradientBoostingClassifier()
GBC.fit(xv_train,y_train)

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

def manual_test(news, alg):
    testing_news = {'text':[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_vx_test = vectorization.fit_transform(new_x_test)
    if alg == 'LR':
        pred = LR.predict(new_vx_test)
    elif alg == 'DT':
        pred = DT.predict(new_vx_test)
    elif alg == 'GBC':
        pred = GBC.predict(new_vx_test)
    elif alg == 'RFC':
        pred = RFC.predict(new_vx_test)
    else:
        pred_LR = LR.predict(new_vx_test)
        pred_DT = DT.predict(new_vx_test)
        pred_GBC = GBC.predict(new_vx_test)
        pred_RFC = RFC.predict(new_vx_test)
        pred = (pred_LR + pred_DT + pred_GBC + pred_RFC)/4
        if pred < 0.5:
            pred = 0
        else:
            pred = 1
    return output_label(pred)


@app.route("/", methods=['GET'])
def home():
    return "The API is running"


@app.route("/clean", methods=['GET'])
def clean():
    return "clean"

@app.route("/train", methods=['GET'])
def train():
    return "train"

@app.route("/evaluate", methods=['POST'])
def evaluate():
    return "evaluate"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
