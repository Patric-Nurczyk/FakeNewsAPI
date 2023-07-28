import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string
import time

LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier(random_state=0)
vectorization = TfidfVectorizer()

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
    #GBC.fit(xv_train, y_train)
    end = time.time()
    print(f"{round(end - start, 2)}")
    start = time.time()
    print("Training in Random Forest Classifier....")
    #RFC.fit(xv_train, y_train)
    end = time.time()
    print(f"{round(end - start, 2)}")
    start = time.time()
    print("Training Complete")
    return LR, DT, GBC, RFC


