#!/usr/bin/env python3
"""Train EfficientNetB0 on rice nutrient dataset and save model."""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "rice_plant_lacks_nutrients")
MODEL_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_efficientnetb0.h5")


def train(batch_size: int = 32, target_size=(224, 224), epochs: int = 30):

    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {BASE_DIR}")

    # Ensure channels_last to match ImageNet weights
    K.set_image_data_format('channels_last')

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        BASE_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_gen.num_classes

    # Build with an explicit 3-channel input tensor to avoid grayscale (1-channel) misconfiguration
    input_tensor = tf.keras.Input(shape=(target_size[0], target_size[1], 3))
    # Use random init to avoid weight shape mismatch issues
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=input_tensor)
    # Fine-tuning: unfreeze only the last 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    # Lower LR for fine-tuning
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    cbs = [
        callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cbs, verbose=1)

    print(f"Saved best model to: {MODEL_OUT}")


if __name__ == "__main__":
    train()


