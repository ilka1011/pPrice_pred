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

def loadData(path1, path2, pnames, onames):
    df1 = pandas.read_csv(path1, names = pnames)
    df2 = pandas.read_csv(path2, names = onames)
    return df1, df2

def convertTimestamp(df):
    df['Date'] = pandas.to_datetime(df.Date)
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['Hour'] = df.Date.dt.hour
    df['Minute'] = df.Date.dt.minute

def calcDelta(df):
    df = df.set_index('Date')
    df.sort_index(inplace = True)
    df['deltaDiesel'] = df.groupby(['Year', 'Month', 'Day'])['Diesel'].transform(lambda x: x - x[0])
    df['deltaE5'] = df.groupby(['Year', 'Month', 'Day'])['E5'].transform(lambda x: x - x[0])
    df['deltaE10'] = df.groupby(['Year', 'Month', 'Day'])['E10'].transform(lambda x: x - x[0])
    df.reset_index(drop = True)
    df = df.drop(['UUID','Year'], axis = 1)
    return df

def build(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                     test_size = 0.2,
                                      random_state = 0)

    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 120))
    hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__min_samples_leaf' : [2,4,8],
                        'randomforestregressor__min_samples_split' : [2,5,10], 'randomforestregressor__max_depth' : [None, 5,3,1]}

    clf = GridSearchCV(pipeline, hyperparameters, cv = 10)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    joblib.dump(clf, 'price_pred.pkl')

    print(r2_score(y_test, pred))
    print(mean_squared_error(y_test, pred))

data = pandas.DataFrame()
[data, data2] = loadData("prices.csv", "oil_prices.csv", ['Date','UUID','Diesel','E5','E10'], ['Date', 'Price'])
#convertTimestamp(data)
#data = calcDelta(data)
#y = data.deltaE10
#print(y.head())
#X = data.drop(['Diesel','E5','E10','deltaDiesel','deltaE5','deltaE10'], axis = 1)
#build(X,y)
print(data2.describe())