import mlflow
import mlflow.keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train[:1000] / 255.0
y_train = y_train[:1000]


mlflow.set_experiment("Task 3.1")


learning_rate = 0.001  # Try 0.001 on your second run!
epochs = 5
batch_size = 64

with mlflow.start_run():

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("dataset", "MNIST subset")


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)


    for epoch in range(epochs):
        mlflow.log_metric("epoch_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("epoch_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("epoch_val_loss", history.history['val_loss'][epoch], step=epoch)
        mlflow.log_metric("epoch_val_accuracy", history.history['val_accuracy'][epoch], step=epoch)


    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
    plt.title(f"Training and Validation Loss (LR={learning_rate})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_filename = "loss_curve.png"
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.close()

    mlflow.keras.log_model(model, "model")

    print("Run successfully completed, logged, and tracked!")