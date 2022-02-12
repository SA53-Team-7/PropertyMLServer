# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('land_transaction_data.csv')
print(df.sample(3), "\n")

df.columns = ["index", "land_txn_date", "proj_name", "land_txn_price", "openning_date", "openning_price"]

print(df.sample(3), "\n")

x_train = df[["land_txn_price"]]
y_train = df["openning_price"]

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x_train, y_train)

linReg = LinearRegression()
linReg_model = linReg.fit(x_train, y_train)

print(linReg.intercept_)
print(linReg.coef_)

y_hat = linReg.predict(x_train)
print(r2_score(y_train, y_hat))
