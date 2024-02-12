# pip install pandas, tensorflow, numpy, matplotlib

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

df = pd.read_csv(
    os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv')
)

df.head(10)
df.tail(10)
df.iloc[6]['comment_text']
df[df.columns[2:]].iloc[6]


x = df['comment_text']
y = df[df.columns[2:]].values
MAX_WORDS = 200000

vectorizer = TextVectorization(
    max_tokens=MAX_WORDS, 
    output_sequence_length=1800,
    output_mode='int'
    )

vectorizer.adapt(x.values)
vectorizer('Hello, AI')
vectorized_text = vectorizer(x.values)


dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

batch_x, batch_y = dataset.as_numpy_iterator().next()

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


model = Sequential()
model.add(Embedding(MAX_WORDS+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(6, activation='sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()

history = model.fit(train, epochs=1, validation_data=val)


history.history

input_text = vectorizer('I will love you and your family')

batch = test.as_numpy_iterator().next()
batch_x, batch_y = test.as_numpy_iterator().next()
(model.predict(batch_x) > 0.5).astype(int)
model.predict(np.expand_dims(input_text, 0))


pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    x_true, y_true = batch
    yhat = model.predict(x_true)

    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')


model.save('harm_text.h5')
model = tf.keras.models.load_model('harm_text.h5')
input_test = vectorizer('I freaken hate you! I am coming for you, I am gonna hurt you real bad')
res = model.predict(np.expand_dims(input_test, 0))

def score_comment(comment):
    vectorized_comment = vectorizer(comment)
    results = model.predict(np.expand_dims(vectorized_comment, 0))

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '\n{}: {}'.format(col, results[0][idx] > 0.5)

    print(comment, text)
