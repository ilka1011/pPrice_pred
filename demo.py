import joblib
import pandas

clf = joblib.load('price_pred.pkl')
test = {'Year' : [2020, 2020, 2020], 'Month' : [4, 4, 4], 'Day' : [2, 3, 4],
        'Hour' : [14, 14, 14], 'Minute' : [30, 30, 30], 'Oil' : [25.3, 28.4, 26]}

data = pandas.DataFrame(test, columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Oil'])
data.set_index(['Year','Month','Day'], inplace = True)

pred = clf.predict(data)
print(pred)
#print(data.head())