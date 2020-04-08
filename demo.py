import joblib
import pandas

clf = joblib.load('price_pred.pkl')
test = {'Year' : [2020, 2020, 2020], 'Month' : [3, 4, 4], 'Day' : [2, 5, 8],
        'Hour' : [10, 12, 17], 'Minute' : [30, 15, 00], 'Oil' : [35.3, 28.4, 26]}

data = pandas.DataFrame(test, columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Oil'])
data.set_index(['Year','Month','Day'], inplace = True)

pred = clf.predict(data)
print(pred/10)
#print(data.head())