import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_ml_data():
    engine=create_engine('sqlite:///retail_data.db')


    data=pd.read_sql_query(""" 
          SELECT t.*, c.Country,p.Description
          FROM clean_transactions t
          LEFT JOIN customer_demographics c ON t.CustomerID=c.CustomerID
          LEFT JOIN product_info p ON t.StockCode=p.StockCode                
     """, engine)
    
    data['DayofWeek']=data['InvoiceDate'].dt.dayofweek
    data['Month']=data['InvoiceDate'].dt.month
    data['Year']=data['InvoiceDate'].dt.year


    data=pd.get_dummies(data,columns=['Country', 'Description'])

    features=['Price', 'DayOfWeek', 'Month','Year']+[col for col in data.columns if col.startswith(('Country_', 'Description_'))]
    X=data[features]
    y=data['Quantity']

    return train_test_split(X,y, test_size=0.2,random_state=42)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models={
        'DecisionTree':DecisionTreeRegressor(random_state=42),
        'RandomForest':RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost':XGBRegressor(n_estimators=100, random_state=42)
    }


    results={}


    for name,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)


        mse=mean_squared_error(y_test,y_pred)
        mae=mean_absolute_error(y_test,y_pred)

        results[name]={'MSE': mse, 'MAE':mae}
        print(f"{name}-MSE:{mse:.2f}, MAE:{mae:.2f}")

    return results

X_train, X_test, y_train, y_test= prepare_ml_data()
results=train_and_evaluate_models(X_train, X_test,y_train, y_test)


from sklearn.model_selection import TimeSeriesSplit

def time_based_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train and evaluate your model here
        # ...

# Example usage
X, y = prepare_ml_data()
time_based_cv(X, y)