from flask import Flask, render_template, request,flash
import pandas as pd
from flask import Response
import csv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import re
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from flask import session
from werkzeug.utils import secure_filename
import sys
import os
import io
import base64
import warnings

warnings.filterwarnings('ignore')
import shutil
from random import randint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from sklearn.model_selection import train_test_split
import numpy as np

from DBConfig import DBConnection

from RFC import rfc_evaluation

from DTC import dt_evaluation

from KNN import knn_evaluation



from DBConfig import DBConnection

import xgboost as xgb

from VTC import vtc_evaluation

app = Flask(__name__)
app.secret_key = "abc"

dict={}
metrics = []
metrics.clear()

accuracy_list=[]
accuracy_list.clear()
precision_list=[]
precision_list.clear()
recall_list=[]
recall_list.clear()
f1score_list=[]
f1score_list.clear()

mcc_list=[]
mcc_list.clear()

kapa_list=[]
kapa_list.clear()



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/user")
def user():
    return render_template("user.html")


@app.route("/newuser")
def newuser():
    return render_template("register.html")


@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        uid = request.form.get('unm')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into register values(%s,%s,%s,%s,%s)"
            values = (name,uid, pwd,email,mno)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user.html", msg2="Registered Successfully..! Login Here.")


        else:
            return render_template("register.html", msg="User name already exists..!")



    except Exception as e:
        print(e)

    return ""

@app.route("/admin_home")
def admin_home():
    return render_template("admin_home.html")


@app.route("/perevaluations")
def perevaluations():
    accuracy_graph()
    precision_graph()
    recall_graph()
    f1score_graph()
    return render_template("metrics.html")


@app.route("/upload_dataset")
def upload_dataset():
    return render_template("upload_dataset.html")


@app.route("/cropyield_prediction")
def cropyield_prediction():
    return render_template("crop_yield_prediction.html")


@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")



        return ""



@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html")
        else:

            return render_template("user.html", msg="Invalid Credentials")

        return ""

@app.route("/preprocessing")
def preprocessing():
    return render_template("data_preprocessing.html")



@app.route("/data_preprocessing" ,methods =["GET", "POST"] )
def data_preprocessing():
    fname = request.form.get("file")
    df = pd.read_csv("../CropYield_Prediction/dataset/"+fname)
    y= df['label']
    del df['label']
    del df['Area']
    del df['Production']
    X=df
    dict['X'] = X
    dict['y'] = y

    return render_template("data_preprocessing.html",msg="Data Preprocessing Completed..!")

@app.route("/evaluations" )
def evaluations():

    rf_list=[]
    dt_list = []
    knn_list = []
    vtc_list = []

    cropdata = dict['X']

    labels = dict['y']

    # Split train test: 70 % - 30 %
    X_train, X_test, y_train, y_test = train_test_split(cropdata, labels, test_size=0.3, random_state=29)

    accuracy_rf, precision_rf, recall_rf, fscore_rf= rfc_evaluation(X_train, X_test, y_train, y_test)
    rf_list.append("RFC")
    rf_list.append(accuracy_rf)
    rf_list.append(precision_rf)
    rf_list.append(recall_rf)
    rf_list.append(fscore_rf)



    accuracy_dt, precision_dt, recall_dt, fscore_dt  = dt_evaluation(X_train, X_test, y_train, y_test)
    dt_list.append("DTC")
    dt_list.append(accuracy_dt)
    dt_list.append(precision_dt)
    dt_list.append(recall_dt)
    dt_list.append(fscore_dt)



    accuracy_knn, precision_knn, recall_knn, fscore_knn  = knn_evaluation(X_train, X_test, y_train, y_test)
    knn_list.append("KNN")
    knn_list.append(accuracy_knn)
    knn_list.append(precision_knn)
    knn_list.append(recall_knn)
    knn_list.append(fscore_knn)



    rfc_clf = RandomForestClassifier()
    dt_clf = DecisionTreeClassifier()
    knn_clf = KNeighborsClassifier()



    accuracy_vtc, precision_vtc, recall_vtc, fscore_vtc=vtc_evaluation(X_train, X_test, y_train, y_test, rfc_clf, dt_clf,knn_clf)
    vtc_list.append("VTC")
    vtc_list.append(accuracy_vtc)
    vtc_list.append(precision_vtc)
    vtc_list.append(recall_vtc)
    vtc_list.append(fscore_vtc)

    metrics.clear()

    metrics.append(rf_list)
    metrics.append(dt_list)
    metrics.append(knn_list)
    metrics.append(vtc_list)

    return render_template("evaluations.html", evaluations=metrics)





@app.route("/prediction", methods =["GET", "POST"])
def prediction():

    try:
        cropdata = pd.read_csv("../CropYield_Prediction/dataset/cropyield_dataset.csv")
        y_train = cropdata['label']
        del cropdata['label']
        del cropdata['Area']
        del cropdata['Production']

        x_train = cropdata
        # print(x_train)
        temperature = request.form.get("temp")
        humidity = request.form.get("humd")
        ph = request.form.get("ph")
        rainfall = request.form.get("rain")

        X_test = [[float(temperature), float(humidity), float(ph), float(rainfall)]]
        print(X_test)

        #Crop Prediction
        rfc_clf = RandomForestClassifier()
        dt_clf = DecisionTreeClassifier()
        knn_clf = KNeighborsClassifier()

        voting_clf = VotingClassifier(estimators=[('RF', rfc_clf), ('dt', dt_clf), ('knn', knn_clf)], voting='hard')

        voting_clf.fit(x_train, y_train)
        predicted = voting_clf.predict(np.array(X_test))
        result = predicted[0]
        print("res=", result)


        #Crop Yield Prediction

        cropdata2 = pd.read_csv("../CropYield_Prediction/dataset/cropyield_dataset.csv")
        x_train = cropdata2[['temperature', 'humidity', 'ph', 'rainfall']]
        y_train = cropdata2[['Area','Production']]

        dt = DecisionTreeRegressor()
        rf = RandomForestRegressor(n_estimators=10, random_state=1)
        knn = KNeighborsRegressor()

        vr= VotingRegressor([('dt', dt), ('rf', rf), ('knn', knn)])

        clf_yield = MultiOutputRegressor(vr)

        clf_yield.fit(x_train, y_train)

        print("res=", clf_yield.predict(X_test).tolist()[0])

        result2=clf_yield.predict(X_test).tolist()[0]

        area=round(result2[0], 2)

        production=round(result2[1],2)

        crop_yield =round((production/area),2)


        print(result)
        print(area)
        print(production)
        print(crop_yield)


    except Exception as e:
        print(e)


    return render_template("crop_yield_prediction.html",result=result,area=area,production=production,crop_yield=crop_yield)



def accuracy_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    accuracy_list.clear()

    cursor.execute("select accuracy from mlevaluations")
    acdata=cursor.fetchall()

    for record in acdata:
        accuracy_list.append(float(record[0]))

    height = accuracy_list
    print("height=",height)
    bars = ('RFC','DTC','KNN','VTC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['red', 'green', 'blue','orange'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Analysis on ML Accuracies')
    plt.savefig('static/accuracy.png')
    plt.clf()

    return ""



def precision_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select precesion from mlevaluations")
    pdata = cursor.fetchall()

    precision_list.clear()
    for record in pdata:
        precision_list.append(float(record[0]))

    height = precision_list
    print("pheight=",height)
    bars = ('RFC', 'DTC', 'KNN', 'VTC')
    y_pos = np.arange(len(bars))
    plt2.bar(y_pos, height, color=['green', 'brown', 'violet', 'blue'])
    plt2.xticks(y_pos, bars)
    plt2.xlabel('Algorithms')
    plt2.ylabel('Precision')
    plt2.title('Analysis on ML Precisions')
    plt2.savefig('static/precision.png')
    plt2.clf()
    return ""


def recall_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select recall from mlevaluations")
    recdata = cursor.fetchall()

    recall_list.clear()
    for record in recdata:
        recall_list.append(float(record[0]))

    height = recall_list
    #print("height=",height)
    bars = ('RFC', 'DTC', 'KNN', 'VTC')
    y_pos = np.arange(len(bars))
    plt3.bar(y_pos, height, color=['orange', 'cyan', 'gray', 'green'])
    plt3.xticks(y_pos, bars)
    plt3.xlabel('Algorithms')
    plt3.ylabel('Recall')
    plt3.title('Analysis on ML Recall')
    plt3.savefig('static/recall.png')
    plt3.clf()
    return ""



def f1score_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select f1score from mlevaluations")
    fsdata = cursor.fetchall()
    f1score_list.clear()
    for record in fsdata:
        f1score_list.append(float(record[0]))

    height = f1score_list
    print("fheight=",height)
    bars = ('RFC', 'DTC', 'KNN', 'VTC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['gray', 'green', 'orange', 'brown'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('F1-Score')
    plt.title('Analysis on ML F1-Score')
    plt4.savefig('static/f1score.png')
    plt4.clf()
    return ""



if __name__ == '__main__':
    app.run(host="localhost", port=2468, debug=True)
