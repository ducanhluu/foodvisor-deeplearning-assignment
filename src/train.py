import tensorflow as tf
import logging
import coloredlogs
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

logger = tf.get_logger()
coloredlogs.install(
    logger=logger,
    level=tf.compat.v1.logging.INFO,
    fmt="%(asctime)s ModelTraining] [%(levelname)s] %(message)s"
)
logger.propagate = False


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to directory containing images")
    parser.add_argument("--display_images",
                        action="store_true", help="Display some images")
    parser.add_argument("--batch_size", default=32, help="Training batch_size")
    parser.add_argument("--model_input_size",
                        default=[224, 224], help="Model input size")
    parser.add_argument("--learning_rate", default=0.0001,
                        help="Base learning rate")
    parser.add_argument("--training_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--finetuning_epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--val_split_size", default=0.2,
                        help="Proportion of validation split")
    parser.add_argument("--da_random_rotation", default=0.1, help="Random rotation factor used in data augmentation")
    parser.add_argument("--da_random_zoom", default=0.1, help="Random zooming factor used in data augmentation")
    parser.add_argument("--export_dir", default="resources", help="Path to folder to save trained model")

    return vars(parser.parse_args())


def visualize_training(train_history, finetuning_history, epochs, export_dir):
    acc = train_history.history['accuracy'] + finetuning_history.history['accuracy']
    val_acc = train_history.history['val_accuracy'] + finetuning_history.history['val_accuracy']

    loss = train_history.history['loss'] + finetuning_history.history['loss']
    val_loss = train_history.history['val_loss'] + finetuning_history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, 1 - np.array(acc), label='Training Error Rate')
    plt.plot(epochs_range, 1 - np.array(val_acc), label='Validation Error Rate')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Error Rate')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(export_dir, "learning_curves.png"))
    plt.show()


def visualize_images(dataset_batch, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset_batch:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy().astype("uint8")[0]])
            plt.axis("off")
    plt.show()


def train():
    args = parse_args()

    # Load dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        args["dataset_dir"],
        label_mode="binary",
        validation_split=args["val_split_size"],
        subset="training",
        seed=789,
        image_size=args["model_input_size"],
        batch_size=args["batch_size"])

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        args["dataset_dir"],
        label_mode="binary",
        validation_split=args["val_split_size"],
        subset="validation",
        seed=789,
        image_size=args["model_input_size"],
        batch_size=args["batch_size"])

    # Optimize dataset loading
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(
        1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Image preprocessing
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip(
                input_shape=(*args["model_input_size"], 3)),
            layers.experimental.preprocessing.RandomRotation(args["da_random_rotation"]),
            layers.experimental.preprocessing.RandomZoom(args["da_random_zoom"]),
        ]
    )

    base_model = tf.keras.applications.EfficientNetB1(input_shape=(*args["model_input_size"], 3),
                                                      include_top=False,
                                                      weights='imagenet')
    # Firstly, freeze the base model
    base_model.trainable = False

    # Use the same preprocessing operations as original model
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    # Add global pooling to reduce number of parameters and add a dense layer for the prediction
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    # Build model
    inputs = tf.keras.Input(shape=(*args["model_input_size"], 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args["learning_rate"]),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # Train new layer
    logger.info("Training model ...")
    training_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args["training_epochs"]
    )

    total_epochs = args["training_epochs"] + args["finetuning_epochs"]

    # Now train the whole model with lower learning rate
    logger.info("Finetuning model ...")
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args["learning_rate"] / 10),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    finetuning_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,
        initial_epoch=args["training_epochs"]
    )

    logger.info(
        "Model training completed, trained model and learning curves are exported to {}".format(args["export_dir"]))
    model.save(os.path.join(args["export_dir"], "trained_model"))
    visualize_training(training_history, finetuning_history, total_epochs, args["export_dir"])


if __name__ == "__main__":
    train()
