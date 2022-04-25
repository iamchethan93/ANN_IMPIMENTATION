import tensorflow as tf

def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (X_train_full, Y_train_full),(X_test, Y_test) = mnist.load_data()
    X_val, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    Y_val, Y_train = Y_train_full[:validation_datasize], Y_train_full[validation_datasize:]
    X_test = X_test / 255.
    return(X_train, Y_train),(X_test, Y_test),(X_val, Y_val)