import pandas as pd
import numpy as np
from sqlalchemy import create_engine

transaction_1=pd.read_csv('Transactional_data_retail01.csv')
transaction_2=pd.read_csv('Transactional_data_retail02.csv')
customer_demographics=pd.read_csv('CustomerDemographics.csv')
product_info=pd.read_csv('ProductInfo.csv')

transactions=pd.concat([transaction_1,transaction_2],ignore_index=True)

engine=create_engine('sqlite:///retail_data.db')

transactions.to_sql('transactions', engine, if_exists='replace',index=False)
customer_demographics.to_sql('customer_demographics', engine,if_exists='replace', index=False)
product_info.to_sql('product_info', engine, if_exists='replace', index=False)

#performing data cleaning and preprocessing

transactions['InvoiceDate']=pd.to_datetime(transactions['InvoiceDate'])

transactions=transactions[(transactions['Quanity']>0) & (transactions['Price']>0)]

transactions['TotalAmount']=transactions['Quanity']* transactions['Price']

transactions.to_sql('clean_transactions', engine, if_exists='replace', index=False)


