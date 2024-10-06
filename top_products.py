import pandas as pd
from sqlalchemy import create_engine

engine=create_engine('sqlite:///retail_data.db')


top_10_quantity=pd.read_sql_query(""" 
   SELECT StockCode, Description, SUM(Quantity) as TotalQuantity
   FROM clean_transactions
   LEFT JOIN product_info USING (StockCode)
   GROUP BY StockCode
   ORDER BY TotalQuantity DESC
   LIMIT 10
""", engine)

top_10_revenue=pd.read_sql_query("""
    SELECT StockCode, Description, SUM(TotalAmount) as TotalRevenue
    FROM clean_transactions
    LEFT JOIN product_info USING (StockCode)
    GROUP BY StockCode
    ORDER BY TotalRevenue DESC
    LIMIT 10
""", engine)


print("Top 10 product by Quantity Sold")
print(top_10_quantity)
print("\n Top 10 Product by Revenue:")
print(top_10_revenue)

top_10_quantity.to_csv('top_10_quantity.csv', index=False)
top_10_revenue.to_csv('top_10_revenue.csv', index=False)