import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Set CUDA_VISIBLE_DEVICES to control GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras
from keras.utils import plot_model

# Import custom utility functions and model architecture
from video_utils import *
from ResNetConv2Plus1D_Architecture import *
# from Conv3D_Architecture import *

# Define save path for results with current timestamp
save_path = "./Results/" + datetime.now().strftime('%y%m%d_%H%M%S')
data_dir = Path('./Supination_dataset/')

# Define constants
n_frames = 16
batch_size = 8

# Define output signature for dataset
output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

# Load and preprocess training, validation, and test datasets
train_ds = tf.data.Dataset.from_generator(FrameGenerator(data_dir / 'train', n_frames, training=True),
                                          output_signature=output_signature)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(data_dir / 'val', n_frames),
                                        output_signature=output_signature)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(data_dir / 'test', n_frames),
                                         output_signature=output_signature)

train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

# Build the model architecture
model = keras.Model(input, x)

frames, label = next(iter(train_ds))
model.build(frames)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Define custom callback for evaluation metrics and visualization
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, labels, save_path):
        super().__init__()
        self.test_data = test_data
        self.labels = labels
        self.model = None
        self.save_path = save_path
        self.file_path = os.path.join(self.save_path, "metrics_summary.txt")
        os.makedirs(self.save_path, exist_ok=True)

    # Method to get actual and predicted labels from dataset
    def get_actual_predicted_labels(self, dataset):
        actual = [labels for _, labels in dataset.unbatch()]
        predicted = self.model.predict(dataset)

        actual = tf.stack(actual, axis=0)
        predicted = tf.concat(predicted, axis=0)
        predicted = tf.argmax(predicted, axis=1)

        return actual, predicted

    # Method to calculate classification metrics
    def calculate_classification_metrics(self, y_actual, y_pred):
        cm = tf.math.confusion_matrix(y_actual, y_pred)
        tp = np.diag(cm)
        precision = dict()
        recall = dict()
        accuracy = dict()
        f1_score = dict()

        for i in range(len(self.labels)):
            col = cm[:, i]
            fp = np.sum(col) - tp[i]

            row = cm[i, :]
            fn = np.sum(row) - tp[i]

            precision[self.labels[i]] = round(tp[i] / (tp[i] + fp) if (tp[i] + fp) > 0 else 0, 4)
            recall[self.labels[i]] = round(tp[i] / (tp[i] + fn) if (tp[i] + fn) > 0 else 0, 4)
            accuracy[self.labels[i]] = round(tp[i] / np.sum(row) if np.sum(row) > 0 else 0, 4)

            f1_score[self.labels[i]] = round(2 * (precision[self.labels[i]] * recall[self.labels[i]]) / 
                    (precision[self.labels[i]] + recall[self.labels[i]]) if (precision[self.labels[i]] + recall[self.labels[i]]) > 0 else 0, 4)

        return precision, recall, accuracy, f1_score

    # Callback method called at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # Check if the current epoch is a multiple of 5
            actual_val, predicted_val = self.get_actual_predicted_labels(val_ds)
            plot_confusion_matrix(actual_val, predicted_val, self.labels, 'validation', self.save_path)
    
        actual, predicted = self.get_actual_predicted_labels(self.test_data)
        precision, recall, accuracy, f1_score = self.calculate_classification_metrics(actual, predicted)
        
        avg_precision = sum(precision.values()) / len(precision)
        avg_recall = sum(recall.values()) / len(recall)
        avg_accuracy = sum(accuracy.values()) / len(accuracy)
        avg_f1_score = sum(f1_score.values()) / len(f1_score)
        loss = logs['loss']

        with open(os.path.join(self.save_path, "metrics_summary.txt"), 'a') as f:
            f.write("Epoch {} Metrics: ".format(epoch + 1))
            f.write("Average Accuracy: {:.4f}, ".format(avg_accuracy))
            f.write("Loss: {:.4f}, ".format(loss))
            f.write("Precision: {:.4f}, ".format(avg_precision))
            f.write("Recall: {:.4f}, ".format(avg_recall))
            f.write("F1-score: {:.4f}\n".format(avg_f1_score))

# Create FrameGenerator to extract labels
fg = FrameGenerator(data_dir / 'train', n_frames, training=True)
labels = list(fg.class_ids_for_name.keys())

# Instantiate MetricsCallback for evaluation during training
metrics_callback = MetricsCallback(test_ds, labels, save_path)

# Create directory for model checkpoints
checkpoint_path = os.path.join(save_path, "checkpoints/")
os.makedirs(checkpoint_path, exist_ok=True)

# Plot the model architecture and save to file
plot_model(model, show_shapes=True, to_file=os.path.join(save_path, 'model_architecture.png'))
with open(os.path.join(save_path, "metrics_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write('\n')

# Define ModelCheckpoint callback to save model weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, "model_checkpoint"),
                                                save_weights_only=True,
                                                save_freq='epoch')

# Train the model for a fixed number of epochs
history = model.fit(x=train_ds,
                    epochs=40,
                    validation_data=val_ds,
                    callbacks=[checkpoint, metrics_callback])

# Plot training history
plot_history(history, save_path)

# Evaluate model on test dataset
model.evaluate(test_ds, return_dict=True)

# Generate confusion matrix and save plot
actual, predicted = metrics_callback.get_actual_predicted_labels(train_ds)
plot_confusion_matrix(actual, predicted, labels, 'train', save_path)

actual, predicted = metrics_callback.get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, labels, 'test', save_path)

# Calculate and print classification metrics
precision, recall, accuracy, f1_score = metrics_callback.calculate_classification_metrics(actual, predicted)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1-score:", f1_score)

# Save the entire model to a HDF5 file
model.save(os.path.join(save_path, "my_model.h5"))