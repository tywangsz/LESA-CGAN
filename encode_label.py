import pandas as pd
# import tensorview as tv
import numpy as np


def labels(url):
    stock = pd.read_csv(url)

    rolling_20_mean = stock.Close.rolling(window=20).mean()[50:]
    rolling_50_mean = stock.Close.rolling(window=50).mean()[50:]
    rolling_50_std = stock.Close.rolling(window=50).std()[50:]

    stock = stock.iloc[50:, :]

    cond_0 = np.where((stock["Close"] < rolling_20_mean) & (stock["Close"] < rolling_50_mean) & (
                rolling_50_std > rolling_50_std.median()), 1, 0)
    cond_1 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] < rolling_50_mean) & (
                rolling_50_std > rolling_50_std.median()), 2, 0)
    cond_2 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] > rolling_50_mean) & (
                rolling_50_std > rolling_50_std.median()), 3, 0)
    cond_3 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] > rolling_50_mean) & (
                rolling_50_std < rolling_50_std.median()), 4, 0)
    cond_4 = np.where((stock["Close"] < rolling_20_mean) & (stock["Close"] > rolling_50_mean) & (
                rolling_50_std < rolling_50_std.median()), 5, 0)
    cond_5 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] < rolling_50_mean) & (
                rolling_50_std < rolling_50_std.median()), 6, 0)

    cond_all = cond_0 + cond_1 + cond_2 + cond_3 + cond_4 + cond_5
    stock["Label"] = cond_all

    return stock


def end_cond(X_train):
    vals = X_train[:, 21, 4] / X_train[:, 0, 4] - 1

    comb1 = np.where(vals < -.1, 0, 0)
    comb2 = np.where((vals >= -.1) & (vals <= -.05), 1, 0)
    comb3 = np.where((vals >= -.05) & (vals <= -.0), 2, 0)
    comb4 = np.where((vals > 0) & (vals <= 0.05), 3, 0)
    comb5 = np.where((vals > 0.05) & (vals <= 0.1), 4, 0)
    comb6 = np.where(vals > 0.1, 5, 0)
    cond_all = comb1 + comb2 + comb3 + comb4 + comb5 + comb6

    print(np.unique(cond_all, return_counts=True))
    arr = np.repeat(cond_all, 22, axis=0).reshape(len(cond_all), 22)
    X_train = np.dstack((X_train, arr))
    return X_train