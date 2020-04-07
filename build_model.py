import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def loadData(path):
    df = pandas.read_csv(path, names = ['Date','UUID','Diesel','E5','E10'])
    return df

def convertTimestamp(df):
    df['Date'] = pandas.to_datetime(df.Date)
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['Hour'] = df.Date.dt.hour
    df['Minute'] = df.Date.dt.minute

def calcDelta(df):
    df['dDiesel'] = df.groupby(['Year', 'Month', 'Day'])['Diesel'].diff().fillna(0)
    df['dE5'] = df.groupby(['Year', 'Month', 'Day'])['E5'].diff().fillna(0)
    df['dE10'] = df.groupby(['Year', 'Month', 'Day'])['E10'].diff().fillna(0)

def build(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                     test_size = 0.2,
                                      random_state = 123
                                    )
    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 200))
    hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__min_samples_leaf' : [2,4,8],
                        'randomforestregressor__min_samples_split' : [2,5,10], 'randomforestregressor__max_depth' : [None, 5,3,1]}

    clf = GridSearchCV(pipeline, hyperparameters, cv = 10)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    joblib.dump(clf, 'price_pred.pkl')

    print(r2_score(y_test, pred))
    print(mean_squared_error(y_test, pred))

data = pandas.DataFrame()
data = loadData("jet_prices.csv")
convertTimestamp(data)
data = data.drop(['Date','UUID'], axis = 1)
data = data.sort_values(by = ['Year', 'Month', 'Day'])
calcDelta(data)

y = data.dE10
print(y.head())
X = data.drop(['Diesel','E5','E10','dDiesel','dE5','dE10'], axis = 1)
build(X,y)
