import pandas as pd
import numpy as np
from keras.models import load_model

# Memuat model LSTM yang sudah dilatih sebelumnya
model = load_model('lstm_model.h5')

# Memuat data testing dari file CSV
df_test = pd.read_csv('data/test.csv')

# Menentukan variabel dependen dan independen
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

# Menormalisasi data
max_value = np.max(X_test)
X_test = X_test / max_value

# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Mengembalikan data ke skala semula
y_pred = y_pred * max_value

# Menggabungkan hasil prediksi dengan data testing asli dan menyimpan ke file CSV
df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred[:, 0]})
df_result.to_csv('hasil_prediksi.csv', index=False)
