from mlxtend.plotting import plot_decision_regions
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def loss_decay(history):

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Acurácia de treino')
    plt.plot(epochs, val_acc, 'b', label='Acurácia de validação')
    plt.title('Acurácia de treino e validação')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Loss de treino')
    plt.plot(epochs, val_loss, 'b', label='Loss de validação')
    plt.title('Loss de treino e validação')
    plt.legend()

    plt.show()

def plot_decision_boundary(X_test, y_test, model):

    plt.scatter(X_test[:,0], X_test[:,1])
    plt.title('Fronteira de Decisão')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    plot_decision_regions(X_test, y_test, clf=model, legend=2)
