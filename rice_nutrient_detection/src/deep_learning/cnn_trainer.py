#!/usr/bin/env python3
"""Deep Learning CNN Implementation for Rice Nutrient Deficiency Detection."""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')
from src.utils.data_preprocessor import DataPreprocessor

class CNNModelTrainer:
    def __init__(self, base_dir, target_size=(224, 224), batch_size=32):
        self.base_dir = base_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = ['Nitrogen', 'Phosphorus', 'Potassium']
        self.num_classes = len(self.classes)
        
    def create_data_generators(self):
        """Create data generators with augmentation for training and validation."""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.base_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.val_generator = val_datagen.flow_from_directory(
            self.base_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
        
    def create_efficientnet_model(self):
        """Create EfficientNetB0 based model."""
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.target_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet_model(self):
        """Create ResNet50 based model."""
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.target_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_custom_cnn(self):
        """Create a custom CNN from scratch."""
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.target_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, model_name):
        """Compile the model with appropriate optimizer and loss."""
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{model_name} Model Summary:")
        print("-" * 50)
        model.summary()
        
        return model
    
    def train_model(self, model, model_name, epochs=50):
        """Train the CNN model."""
        
        # Create callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                f'models/best_{model_name.lower()}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\nTraining {model_name}...")
        print("=" * 50)
        
        # Train the model
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate the trained model."""
        
        # Get test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.base_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Predictions
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{model_name} Evaluation Results:")
        print("=" * 50)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.classes))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_true,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, history, model_name):
        """Plot training history."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'models/{model_name.lower()}_training_history.png')
        plt.show()

def train_all_cnn_models():
    """Train all CNN models and compare their performance."""
    
    print("🌾 DEEP LEARNING CNN MODELS TRAINING")
    print("=" * 60)
    
    BASE_DIR = 'data/rice_plant_lacks_nutrients'
    
    # Create trainer
    trainer = CNNModelTrainer(BASE_DIR)
    
    # Create data generators
    trainer.create_data_generators()
    
    # Define models to train
    models_to_train = {
        'EfficientNetB0': trainer.create_efficientnet_model(),
        'ResNet50': trainer.create_resnet_model(),
        'CustomCNN': trainer.create_custom_cnn()
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Compile model
        model = trainer.compile_model(model, model_name)
        
        # Train model
        history = trainer.train_model(model, model_name, epochs=30)
        
        # Evaluate model
        result = trainer.evaluate_model(model, model_name)
        results[model_name] = result
        
        # Plot training history
        trainer.plot_training_history(history, model_name)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("CNN MODELS COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    
    return results

if __name__ == "__main__":
    train_all_cnn_models()
