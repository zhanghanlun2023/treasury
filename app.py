# Streamlit app using sklearn MLPRegressor placeholder
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

st.title("AI赋能司库：现金流预测（MLP版）")

dates = pd.date_range("2023-01-01", periods=300)
cash = np.sin(np.linspace(0,10,300))*10000 + np.random.randn(300)*3000
df = pd.DataFrame({"日期":dates, "净现金流":cash})

st.write("示例数据", df.head())

window = 30
X, y = [], []
vals = df["净现金流"].values
for i in range(len(vals)-window):
    X.append(vals[i:i+window])
    y.append(vals[i+window])
X, y = np.array(X), np.array(y)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_s = scaler_X.fit_transform(X)
y_s = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
model.fit(X_s, y_s)

st.success("模型训练完成（MLPRegressor）")

last = X_s[-1].copy()
steps = 30
preds = []
for _ in range(steps):
    p_s = model.predict(last.reshape(1,-1))[0]
    p = scaler_y.inverse_transform([[p_s]])[0,0]
    preds.append(p)
    last = np.append(last[1:], p_s)

future_dates = [df["日期"].iloc[-1] + pd.Timedelta(days=i+1) for i in range(steps)]
out = pd.DataFrame({"日期":future_dates, "预测净现金流":preds})
st.write("预测结果", out)
