from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import pandas as pd
dataframe = pd.read_csv(r"/Users/raminmadani/Documents/jupyter Projects/Datas/EURUSDindicator.csv",delimiter="\t", encoding="UTF-16")
dataframe.columns=['A', 'B', 'C', 'D','E', 'F', 'G', 'H','I','J','target','target2']
target = dataframe['target']
dataframe1 = dataframe.drop(columns=['target','target2'])
dataframe2 = dataframe1.values
X = dataframe2
y = target.values

target_column = ['target', 'target2']
predictors = list(set(list(dataframe.columns))-set(target_column))
dataframe[predictors] = dataframe[predictors]/dataframe[predictors].max()
print(dataframe.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
count_classes = y_test.shape[1]
print(count_classes)

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-10 * 300 ** (epoch/30))
optimizer = keras.optimizers.Adam(lr=1e-6)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100)

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-10, 1e3, 0, 1])
# plt.show()