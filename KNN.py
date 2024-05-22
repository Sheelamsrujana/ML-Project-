
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
from DBConfig import DBConnection

def knn_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()

    knn_clf = KNeighborsClassifier()

    knn_clf.fit(X_train, y_train)

    predicted = knn_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100
    

    values = ("KNN",str(accuracy),str(precision),str(recall),str(fscore))
    sql = "insert into mlevaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("KNN=",accuracy,precision,recall,fscore)

    return accuracy, precision, recall, fscore




