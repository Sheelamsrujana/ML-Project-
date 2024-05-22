
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix

from DBConfig import DBConnection

def vtc_evaluation(X_train, X_test, y_train, y_test, rfc_clf, dt_clf,knn_clf):

    db = DBConnection.getConnection()
    cursor = db.cursor()

    voting_clf = VotingClassifier(estimators=[('RF', rfc_clf),('dt', dt_clf),('knn', knn_clf)], voting='hard')

    voting_clf.fit(X_train, y_train)

    predicted = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100

    values = ("VTC",str(accuracy),str(precision),str(recall),str(fscore))
    sql = "insert into mlevaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("VTC=",accuracy,precision,recall,fscore)

    return accuracy, precision, recall, fscore




