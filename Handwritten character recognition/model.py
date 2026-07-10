import tensorflow as tf
from tensorflow.keras import layers, Model
from utils import logger

def build_cnn_model(num_classes, input_shape=(28, 28, 1)):
    """
    Builds a Convolutional Neural Network (CNN) for handwritten character classification.
    Design contains:
    - Block 1: Conv2D(32) + BatchNorm + ReLU + Conv2D(32) + BatchNorm + ReLU + MaxPooling + Dropout
    - Block 2: Conv2D(64) + BatchNorm + ReLU + Conv2D(64) + BatchNorm + ReLU + MaxPooling + Dropout
    - Block 3: Conv2D(128) + BatchNorm + ReLU + MaxPooling + Dropout
    - Classification Head: Flatten + Dense(256) + BatchNorm + Dropout + Dense(num_classes, Softmax)
    """
    logger.info(f"Building CNN model with input shape {input_shape} and {num_classes} output classes...")
    
    model = tf.keras.Sequential([
        # Input Layer
        layers.Input(shape=input_shape, name="input_layer"),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', name="conv1_1"),
        layers.BatchNormalization(name="bn1_1"),
        layers.Activation('relu', name="relu1_1"),
        layers.Conv2D(32, (3, 3), padding='same', name="conv1_2"),
        layers.BatchNormalization(name="bn1_2"),
        layers.Activation('relu', name="relu1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(0.25, name="dropout1"),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', name="conv2_1"),
        layers.BatchNormalization(name="bn2_1"),
        layers.Activation('relu', name="relu2_1"),
        layers.Conv2D(64, (3, 3), padding='same', name="conv2_2"),
        layers.BatchNormalization(name="bn2_2"),
        layers.Activation('relu', name="relu2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="dropout2"),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', name="conv3_1"),
        layers.BatchNormalization(name="bn3_1"),
        layers.Activation('relu', name="relu3_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        layers.Dropout(0.4, name="dropout3"),
        
        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(256, name="fc1"),
        layers.BatchNormalization(name="bn_fc1"),
        layers.Activation('relu', name="relu_fc1"),
        layers.Dropout(0.5, name="dropout_fc1"),
        
        # Output Softmax Layer
        layers.Dense(num_classes, activation='softmax', name="output_layer")
    ], name="Handwritten_CNN_Classifier")
    
    return model


class CTCLayer(layers.Layer):
    """Custom CTC Loss Layer to integrate CTC loss calculation in Keras training loop."""
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # Re-broadcast predictions
        return y_pred


def build_crnn_model(num_classes, img_width=128, img_height=32):
    """
    Builds the base CRNN (CNN + Bidirectional LSTM) model for handwritten word recognition.
    y_pred outputs probability distributions of shape (time_steps, num_classes + 1)
    where class index 0 represents the CTC blank token.
    """
    logger.info(f"Building CRNN base model with input shape ({img_height}, {img_width}, 1)...")
    
    inputs = layers.Input(shape=(img_height, img_width, 1), name="image")
    
    # CNN Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x) # Height/Width down by 2 -> (16, 64, 32)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x) # Height/Width down by 2 -> (8, 32, 64)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.MaxPooling2D((2, 1), name="pool3")(x) # Height down by 2, Width unchanged -> (4, 32, 128)
    
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv4")(x)
    x = layers.MaxPooling2D((2, 1), name="pool4")(x) # Height down by 2, Width unchanged -> (2, 32, 256)
    
    # Reshape feature maps to (Width, Height * Channels) for RNN sequence input
    # Shape of pool4 is (batch, H_new, W_new, C_new) = (batch, 2, 32, 256)
    # Permute to (batch, 32, 2, 256)
    x = layers.Permute((2, 1, 3), name="permute")(x)
    
    # Reshape to (batch, 32, 512)
    time_steps = img_width // 4
    features_per_step = (img_height // 16) * 256
    x = layers.Reshape(target_shape=(time_steps, features_per_step), name="reshape")(x)
    
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.25, name="dropout_features")(x)
    
    # Sequence modeling (Bidirectional LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="lstm_1")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="lstm_2")(x)
    
    # Output projection Dense layer
    # num_classes + 1 to account for the CTC blank character (usually index 0)
    outputs = layers.Dense(num_classes + 1, activation="softmax", name="dense_output")(x)
    
    base_model = Model(inputs=inputs, outputs=outputs, name="CRNN_Base")
    return base_model


def build_crnn_training_model(base_model, max_label_len):
    """
    Wraps the CRNN base model with a CTC Loss layer for training.
    """
    logger.info("Wrapping CRNN base model with CTC loss for training...")
    
    # Base model inputs
    image_input = base_model.input
    y_pred = base_model.output
    
    # Inputs required for CTC loss
    labels = layers.Input(name="label", shape=(max_label_len,), dtype="float32")
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")
    
    # CTC loss layer
    output = CTCLayer(name="ctc_loss")(labels, y_pred, input_length, label_length)
    
    # Training model
    training_model = Model(
        inputs=[image_input, labels, input_length, label_length],
        outputs=output,
        name="CRNN_Training_Wrapper"
    )
    return training_model


if __name__ == "__main__":
    # Test model shapes
    cnn = build_cnn_model(47)
    cnn.summary()
    
    base_crnn = build_crnn_model(26)
    train_crnn = build_crnn_training_model(base_crnn, 15)
    train_crnn.summary()
