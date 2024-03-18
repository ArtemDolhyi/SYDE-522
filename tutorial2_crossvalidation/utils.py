# code to plot kfold indices from https://github.com/vaasha/Machine-leaning-in-examples/blob/master/sklearn/cross-validation/Cross%20Validation.ipynb

import pandas as pd
import numpy as np

def kfoldize(splits, N, shift=.1):
    train = pd.DataFrame()
    test = pd.DataFrame()
    i = 1

    indices = np.arange(N)

    for train_index, test_index in splits:
        train_df = pd.DataFrame(np.take(indices, np.array(train_index)), columns=["x"])
        train_df["val"] = i - shift
        train = pd.concat([train, train_df], ignore_index=True)

        test_df = pd.DataFrame(np.take(indices, np.array(test_index)), columns=["x"])
        test_df["val"] = i + shift
        test = pd.concat([test, test_df], ignore_index=True)
        i += 1
    return train, test

def plot_kfold(ax, splits, N, shift = .1):
    train, test = kfoldize(splits, N, shift=shift)
    ax.scatter(x="x",y="val",c="b",label="train",s=15,data=train)
    ax.scatter(x="x",y="val",c="r",label="test",s=15,data=test)
    ax.set_ylabel("fold")
    ax.set_xlabel("indices")
    return
