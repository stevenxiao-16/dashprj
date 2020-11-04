import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from data_clean_general import clean
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
import tensorflow as tf
from functools import partial

k = tf.keras
m = k.models
l = k.layers
checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

fn = 'sentiment_db.csv'
# fn = './train_data/sentiment_db.csv'
data = pd.read_csv(fn, encoding='UTF-8')
del data['message_id']
del data['symbol']

data['message'] = data['message'].apply(clean)
data['message_length'] = data['message'].apply(len)
data = data[(data['message_length'] > 3)]
data.sentiment = data.sentiment.apply(lambda x: 1 if x == 'Bullish' else 0)
data = data.iloc[:10000,:]

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
features = np.array(model.encode(data['message'].tolist()))
labels = data.sentiment.to_numpy()
train_feature, test_feature, train_label, test_label = train_test_split(features, labels)

# print(test_feature.shape)
# lr_clf = LogisticRegression()
# lr_clf.fit(train_feature, train_label)
# print(lr_clf.score(test_feature, test_label))

model = m.Sequential([
  l.Flatten(),
  # l.Dense(1024, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=k.initializers.GlorotNormal()),
  l.Dense(512, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=k.initializers.GlorotNormal()),
  l.GaussianDropout(0.1),
  l.Dense(256, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=k.initializers.GlorotNormal()),
  l.GaussianDropout(0.1),
  l.Dense(128, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=k.initializers.GlorotNormal()),
  l.GaussianDropout(0.1),
  l.Dense(64, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=k.initializers.GlorotNormal()),
  l.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(
      optimizer=tf.optimizers.Adam(learning_rate = 0.001, beta_1=0.93, amsgrad=False),
      loss='binary_crossentropy',
      metrics=['accuracy']
  )


class CallbackLimit(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('loss') < 0.25:
      print("\n\nReached under 0.25 of loss.\nWe stop the process here.\n\n")
      self.model.stop_training = True


class CallbackLimitAcc(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('val_accuracy') > 0.882:
      print("\n\nReached highest acc.\nWe stop the process here.\n\n")
      self.model.stop_training = True


callback_limit = CallbackLimit()
callback_limit_acc = CallbackLimitAcc()
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
  checkpoint_path,
  verbose=1,
  save_weights_only=True,
  period=1
)
model.fit(
  train_feature,
  train_label,
  batch_size=400,
  epochs=8,
  callbacks=[callback_limit_acc, callback_checkpoint],
  # callbacks = [callback_checkpoint, callback_limit],
  validation_data=(test_feature, test_label)
)
model.save_weights(checkpoint_path.format(epoch=0))
# load model
import os
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
prediction = np.round(model.predict(test_feature))


print("Test Accuract")
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(test_label, prediction, digits=3))

