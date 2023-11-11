import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, Dense, BatchNormalization, Activation,
    GlobalAveragePooling2D, Input, Dropout, Multiply, Add, Reshape
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

# Define the swish activation function
def swish_activation(x):
    return tf.keras.layers.Activation(tf.nn.swish)(x)

# Define the MBConvBlock class
class MBConvBlock(tf.keras.layers.Layer):
    def __init__(self, input_filters, output_filters, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size

        if self.expand_ratio != 1:
            self.expand_conv = Conv2D(filters=input_filters * expand_ratio, kernel_size=1, padding='same', use_bias=False)
            self.expand_bn = BatchNormalization()

        self.depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)
        self.depthwise_bn = BatchNormalization()

        if self.se_ratio:
            num_reduced_filters = max(1, int(input_filters * se_ratio))
            self.se_reduce = Conv2D(num_reduced_filters, 1, activation=swish_activation, padding='same')
            self.se_expand = Conv2D(input_filters * expand_ratio, 1, padding='same')

        self.project_conv = Conv2D(output_filters, 1, padding='same', use_bias=False)
        self.project_bn = BatchNormalization()

    def call(self, inputs, training=False):
        x = inputs
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = swish_activation(self.expand_bn(x, training=training))
        x = self.depthwise_conv(x)
        x = swish_activation(self.depthwise_bn(x, training=training))

        if self.se_ratio:
            se = GlobalAveragePooling2D()(x)
            se = Reshape((1, 1, self.input_filters * self.expand_ratio))(se)
            se = self.se_reduce(se)
            se = swish_activation(se)
            se = self.se_expand(se)
            se = Activation('sigmoid')(se)
            x = Multiply()([x, se])

        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        if self.stride == 1 and self.input_filters == self.output_filters:
            if self.drop_connect_rate:
                x = Dropout(self.drop_connect_rate)(x, training=training)
            x = Add()([x, inputs])
        return x

# Define the EfficientNet function
def EfficientNet(input_shape, dropout_rate=0.2, num_classes=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = swish_activation(x)

    # MBConvBlock arguments: input_filters, output_filters, kernel_size, stride, expand_ratio, se_ratio
    x = MBConvBlock(32, 16, 3, 1, 1, 0.25)(x)
    x = MBConvBlock(16, 24, 3, 2, 6, 0.25)(x)
    x = MBConvBlock(24, 24, 3, 1, 6, 0.25)(x)
    x = MBConvBlock(24, 40, 5, 2, 6, 0.25)(x)
    x = MBConvBlock(40, 40, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(40, 80, 3, 2, 6, 0.25)(x)
    x = MBConvBlock(80, 80, 3, 1, 6, 0.25)(x)
    x = MBConvBlock(80, 80, 3, 1, 6, 0.25)(x)
    x = MBConvBlock(80, 112, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(112, 112, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(112, 112, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(112, 192, 5, 2, 6, 0.25)(x)
    x = MBConvBlock(192, 192, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(192, 192, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(192, 192, 5, 1, 6, 0.25)(x)
    x = MBConvBlock(192, 320, 3, 1, 6, 0.25)(x)

    x = Conv2D(1280, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = swish_activation(x)

    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Load and preprocess the data
def load_preprocess_data():
    # Load the CIFAR10 dataset
    (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Convert labels to one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    val_labels = tf.keras.utils.to_categorical(val_labels, 10)

    # Use `tf.data` to batch and shuffle the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(64)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(64)
    return train_dataset, val_dataset

# Create and compile the model
def create_compile_model(input_shape, dropout_rate, num_classes):
    model = EfficientNet(input_shape=input_shape, dropout_rate=dropout_rate, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model
def train_evaluate_model(model, train_dataset, val_dataset, epochs):
    checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    val_loss, val_accuracy = model.evaluate(val_dataset)
    return val_loss, val_accuracy

# Convert the model to TFLite with quantization
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def populate_save_tflite_model(tflite_model, model_dir, label_file_name, accuracy_percentage, export_dir):
    # Initialize the TFLite metadata writer API.
    model_meta = _metadata_fb.ModelMetadataT()

    # Initialize input and output metadata
    input_meta = _metadata_fb.TensorMetadataT()
    output_meta = _metadata_fb.TensorMetadataT()

    input_meta.name = "input"
    input_meta.description = "Input image to be classified."
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
    input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5, 127.5, 127.5]  # Adjusted mean for [-1,1] normalization
    input_normalization.options.std = [127.5, 127.5, 127.5]   # Adjusted std for [-1,1] normalization
    input_meta.processUnits = [input_normalization]

    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Probabilities of the 10 labels respectively."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(label_file_name)
    label_file.description = "Labels for objects that the model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # Define the correct paths for the model and labels
    tflite_model_path = os.path.join(model_dir, f"efficientnet_cifar10_{accuracy_percentage}.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    # Check if the directories exist, if not, create them
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # Save the TFLite model to file
    tflite_model_path = os.path.join(model_dir, f"efficientnet_cifar10_{accuracy_percentage}.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # Populate the metadata
    populator = _metadata.MetadataPopulator.with_model_file(tflite_model_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([os.path.join(model_dir, label_file_name)])
    populator.populate()

    # Display the metadata
    displayer = _metadata.MetadataDisplayer.with_model_file(tflite_model_path)
    export_json_file = os.path.join(export_dir, "model_metadata.json")
    json_file = displayer.get_metadata_json()

    # Write out the metadata as a json file
    with open(export_json_file, "w") as f:
        f.write(json_file)

    print(f"Model with metadata saved as {tflite_model_path}")


# Main function to run the script
def main():
    # Define paths
    model_dir = "/home/wonseok"  # Change to your preferred directory
    label_file_name = "labels.txt"  # Change to your label file name
    export_dir = "/home/wonseok"  # Change to your preferred export directory

    # Make sure the directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # Check if label file exists
    label_file_path = os.path.join(model_dir, label_file_name)
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"Label file {label_file_path} does not exist.")

    # Load and preprocess the data
    train_dataset, val_dataset = load_preprocess_data()

    # Create and compile the model
    model = create_compile_model(input_shape=(32, 32, 3), dropout_rate=0.2, num_classes=10)

    # Get the number of epochs from the user
    try:
        epochs = int(input("Enter the number of epochs for training: "))
    except ValueError:
        raise ValueError("Invalid input. Please enter an integer for the number of epochs.")

    # Train the model
    train_evaluate_model(model, train_dataset, val_dataset, epochs)

    # Convert the model to TFLite
    tflite_model = convert_to_tflite(model)

    # Evaluate the model to get accuracy
    _, val_accuracy = model.evaluate(val_dataset)
    accuracy_percentage = "{:.1f}".format(val_accuracy * 100)

    # Populate and save the TFLite model with metadata
    populate_save_tflite_model(tflite_model, model_dir, label_file_name, accuracy_percentage, export_dir)

    print(f"Model with metadata saved as {os.path.join(model_dir, f'efficientnet_cifar10_{accuracy_percentage}.tflite')}")

if __name__ == "__main__":
    main()
