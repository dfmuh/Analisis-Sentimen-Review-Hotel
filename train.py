import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from keras.models import load_model

df = pd.read_csv("data/train.csv")
df.text=df.text.astype(str)
df.head()

text_df = df[['text','labels']]
# print(text_df.shape)
# text_df.head(5)

text_df["labels"].value_counts()

sentiment_label = text_df.labels.factorize()
sentiment_label

text = text_df.text.values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(text)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)
# print(tokenizer.word_index)

# print(text[0])
# print(encoded_docs[0])

# print(padded_sequence[0])

model = load_model('lstm_model.h5')

def main() :
  embedding_vector_length = 32
  model = Sequential()
  model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
  model.add(SpatialDropout1D(0.025))
  model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())

  history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.1, epochs=10, batch_size=20)
  model.save('lstm_model.h5')

  plt.plot(history.history['accuracy'], label= 'acc')
  plt.plot(history.history['val_accuracy'], label= 'val_acc')
  plt.legend()
  plt.show()
  plt.savefig("Akurasi.jpg")

  plt.plot(history.history['loss'], label= 'loss')
  plt.plot(history.history['val_loss'], label= 'val_loss')
  plt.legend()
  plt.show()
  plt.savefig("Loss.jpg")

if __name__ == '__main__' :
  main()

def predict_sentiment(text):
  tw = tokenizer.texts_to_sequences([text])
  tw = pad_sequences(tw,maxlen=200)
  prediction = int(model.predict(tw).round().item())
  if sentiment_label[1][prediction] == 0:
    result = "Negatif"
  else :
    result = "Positif"
  print("Hasil Prediksi : ", result)