import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def run_nb():
    df = pd.read_csv('EmotionWheelFinal (1).csv')

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    gnb = GaussianNB()

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))