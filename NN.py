from extract_csv import load_carbon_csv_content
import tensorflow as tf
from sklearn.model_selection import train_test_split

classes = ['normal', 'alerta', 'anômalo']
num_classes = len(classes)

data, target = load_carbon_csv_content('dataset/dataset_carbon_filter.csv')

m = data.shape[0]
f = data.shape[1]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(f,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes))

model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['accuracy' ,'mean_absolute_error'])
