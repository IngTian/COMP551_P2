import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model.gd import LogisticRegression

diabetes_train = pd.read_csv("data_sets/data_A2/diabetes/diabetes_train.csv")


def process_dataset(data):
    data=data.to_numpy()
    x, y =data[:,:-1],data[:,-1:]
    y=np.squeeze(y)
    return(x,y)

diabetes_train=process_dataset(diabetes_train)
x=diabetes_train[0]
y=diabetes_train[1]

                              #generate synthetic data
model = LogisticRegression(verbose=True, )
yh = model.fit(x,y).predict(x)
plt.plot(x, y, '.', label='dataset')
plt.plot(x, yh, 'g', alpha=.5, label='predictions')
plt.xlabel('x')
plt.ylabel(r'$y$')
plt.legend()
plt.show()
