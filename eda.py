import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
from sqlalchemy import create_engine

engine=create_engine('sqlite:///retail_data.db')

customer_summary=pd.read_sql_query(""" 
    SELECT Customer ID, COUNT(DISTINCT Invoice ) as TotalTransactions,
        SUM(Quantity) as TotalQuantity, SUM(TotalAmount) as TotalRevenue
    FROM clean_transactions
    GROUP BY Customer ID
    """, engine)


item_summary=pd.read_sql_query(""" 
    SELECT StockCode, Description, COUNT(DISTINCT Invoice) as TotalTransactions,
        SUM(Quantity) as TotalQuantity, SUM(TotalAmount) as TotalRevenue
    From clean_transactions
    LEFT JOIN product_info USING (StockCode)
    GROUP BY StockCode
""", engine)


transaction_summary=pd.read_sql_query(""" 
   SELECT Invoice, COUNT(DISTINCT StockCode) as UniqueItems,
        SUM(Quantity) as TotalQuantity, SUM(TotalAmount) as TotalAmount
    FROM clean_transactions
    GROUP BY Invoice
""", engine)


plt.figure(figsize=(12,6))
sns.histplot(customer_summary['TotalTransactions'], kde=True)
plt.title('Distribution of Transactions per Customer')
plt.xlabel('Number of Transactions')
plt.savefig('customer_transactions_dist.png')
plt.close()


plt.figure(figsize=(12,6))
sns.scatterplot(data=item_summary, x='TotalQuantity', y='TotalRevenue')
plt.title('Item Quantity vs Revenue')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Total Revenue')
plt.savefig('item_quantity_revenue.png')
plt.close()


plt.figure(figsize=(12,6))
plt.boxplot(data=transaction_summary,y='TotalAmount')
plt.title('Distribution of Transaction Amount')
plt.ylabel('Total Amount')
plt.savefig('transaction_amount_dist.png')
plt.close()