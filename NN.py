import numpy as np
from extract_csv import load_carbon_csv_content
import model_manager as mm
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Config classes
classes = ['normal', 'alert', 'anomaly']
num_classes = len(classes)

# Config modelo
model_types = ['dense', 'norm_dense']
model_type = model_types[1]
model_path = f'models/{model_type}_model.h2'

# Config parâmentros
num_iters = 20 # Default: 20
mini_batch_size = None # Default: 64
test_percent = 0.33 # Default: 0.33
val_percent = 0.1 # Default: 0.1
learn_rate = 0.01 # Default: 0.001

# Config arquivo de comparação de acurácia
acc_path = f'models/{model_type}_acc'
prev_acc = mm.load_model_acc(acc_path)
print(f'Acurácia anterior: {prev_acc}')

# Config 
dataset_path = 'dataset/dataset_carbon_filter.csv'
data, target = load_carbon_csv_content(dataset_path)

m = data.shape[0]
f = data.shape[1]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_percent, random_state=42)

def create_model(type: str):

    model = tf.keras.Sequential()

    match type:
        case 'dense':
            model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(f,)))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='softmax'))
            model.add(tf.keras.layers.Dense(num_classes))
        case 'norm_dense':
            normalizer = tf.keras.layers.Normalization()
            normalizer.adapt(X_train)
            model.add(normalizer)
            model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(f,)))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='softmax'))
            model.add(tf.keras.layers.Dense(num_classes))
        case _:
            print('Modelo inválido!')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), 
                  loss='mean_squared_error', 
                  metrics=['sparse_categorical_accuracy' ,'mean_absolute_error'])
    
    return model

model = create_model(model_type)

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
model.fit(X_train, y_train, epochs=num_iters, batch_size=mini_batch_size, validation_split=val_percent)

X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
loss, accuracy, mae = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Acurácia: {accuracy}, MAE: {mae}")

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print(f'previsto: {predicted_classes}')
print(f'real: {y_test}')

if mm.compare(prev_acc, accuracy, acc_path): model.save(model_path)
