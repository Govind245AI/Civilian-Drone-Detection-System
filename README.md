# 🚁 Civilian Drone Detection System

A deep learning system for detecting and classifying aerial objects as **Airplanes**, **Birds**, or **Drones** using Convolutional Neural Networks (CNN). This project includes data preprocessing, model training, and an interactive Gradio interface for real-time predictions.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Notebooks](#notebooks)
9. [Results](#results)
10. [Model Interface](#model-interface)

## Overview

This project implements a CNN-based classification system trained on thousands of images to distinguish between three aerial object classes: **Airplanes**, **Birds**, and **Drones**. 

1. **Data Pipeline**: Automated image validation and preprocessing
2. **Model Training**: Optimized CNN with callbacks for improved training
3. **Interactive Interface**: Gradio-based web interface for easy predictions
4. **Visualization**: Real-time probability distribution charts

### Dataset Details
1. **Total Images**: 2,338 files belonging to 3 classes
2. **Image Format**: 256×256 RGB images
3. **Classes**: 
   - Airplane
   - Birds
   - Drone

## Features

1. **Automated Data Cleaning**: Removes corrupted or invalid images using `imghdr` validation
2. **Image Normalization**: Scales pixel values to [0, 1] range
3. **Data Splitting**: 70% training, 10% validation, 20% testing
4. **Deep CNN Architecture**: Multiple convolutional blocks with pooling and dropout
5. **Advanced Training Techniques**:
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau
   - TensorBoard integration for monitoring
6. **Interactive Gradio UI**: Two-tab interface for quick and detailed predictions
7. **Comprehensive Metrics**: Accuracy and loss tracking with visualizations

## Dataset

### Data Specifications
1. **Input Size**: 256×256 RGB images
2. **Total Samples**: 2,338 images
3. **Classes**: 3 (Airplane, Birds, Drone)
4. **Data Processing Pipeline**:
   1. Remove corrupted images using `imghdr` and `cv2.imread()`
   2. Load dataset with `tf.keras.utils.image_dataset_from_directory()`
   3. Normalize images by dividing by 255
   4. Split into train/validation/test sets

### Directory Structure
```
BirdVsDroneVsAirplane/
├── airplane/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Birds/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── drone/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Model Architecture

The CNN model is built using Keras Sequential API with the following layers:

### Layer Configuration
1. **Conv2D**: 32 filters, 3×3 kernel, ReLU activation (input: 256×256×3)
2. **Conv2D**: 64 filters, 3×3 kernel, ReLU activation
3. **MaxPooling2D**: 2×2 pool size
4. **Conv2D**: 128 filters, 3×3 kernel, ReLU activation
5. **MaxPooling2D**: 2×2 pool size
6. **Conv2D**: 64 filters, 3×3 kernel, ReLU activation
7. **Conv2D**: 128 filters, 3×3 kernel, ReLU activation
8. **MaxPooling2D**: 2×2 pool size
9. **Flatten**: Output shape (107,648)
10. **Dense**: 50 units, ReLU activation
11. **Dropout**: 0.5 dropout rate
12. **Dense**: 3 units, Softmax activation (output layer)

### Training Configuration
1. **Optimizer**: Adam
2. **Loss Function**: Sparse Categorical Crossentropy
3. **Metrics**: Accuracy
4. **Epochs**: 15
5. **Batch Size**: 32
6. **Callbacks**:
   - Early Stopping (monitor val_loss, patience=5)
   - ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6)
   - TensorBoard logging

### Model Summary
- **Total Parameters**: 5,623,499 (21.45 MB)
- **Trainable Parameters**: 5,623,499
- **Non-trainable Parameters**: 0

## Project Structure

```
Civilian-Drone-Detection-System/
├── README.md                        # Project documentation
├── drone.ipynb                      # Main training notebook
├── drone_interface.ipynb            # Gradio interface notebook
├── BirdVsDroneVsAirplane/          # Dataset directory
│   ├── airplane/                    # Airplane images (2,338 total)
│   ├── Birds/                       # Bird images
│   └── drone/                       # Drone images
├── logs/                            # TensorBoard logs
└── Dronedetectionsystem.h5         # Trained model
```

## Installation

### Prerequisites
1. Python 3.7 or higher
2. pip package manager
3. Jupyter Notebook (for running .ipynb files)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Govind245AI/Civilian-Drone-Detection-System.git
cd Civilian-Drone-Detection-System

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Required Dependencies
```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
opencv-python>=4.5.0
pillow>=8.3.0
gradio>=4.0.0
```

## Usage

### Running the Training Pipeline

```bash
# Open Jupyter and run the training notebook
jupyter notebook drone.ipynb
```

**Notebook Steps**:
1. Import required libraries (TensorFlow, NumPy, Pandas, Matplotlib, OpenCV, imghdr)
2. Set data directory to `'BirdVsDroneVsAirplane'`
3. Define supported image extensions: `['jpeg','jpg','bmp','png']`
4. Validate and remove corrupted images:
   ```python
   for image_class in os.listdir(data_dir):
       for image in os.listdir(os.path.join(data_dir,image_class)):
           image_path = os.path.join(data_dir,image_class,image)
           try:
               img = cv2.imread(image_path)
               tip = imghdr.what(image_path)
               if tip not in image_exts:
                   os.remove(image_path)
           except Exception as e:
               print('issue with image {}'.format(image_path))
   ```
5. Load dataset:
   ```python
   data = tf.keras.utils.image_dataset_from_directory('BirdVsDroneVsAirplane')
   # Output: Found 2338 files belonging to 3 classes
   ```
6. Normalize images by dividing by 255
7. Split data:
   ```python
   train_size = int(len(data) * 0.7)  # 70% for training
   test_size = int(len(data) * 0.2) + 1  # 20% for testing
   val_size = int(len(data) * 0.1) + 1  # 10% for validation
   train = data.take(train_size)
   val = data.skip(train_size).take(val_size)
   test = data.skip(train_size + val_size).take(test_size)
   ```
8. Build and compile model
9. Train model with callbacks:
   ```python
   history = model.fit(
       train,
       validation_data=val,
       epochs=15,
       callbacks=[tensorboard_callback, early_stopping, reduce_lr]
   )
   ```
10. Visualize training history (accuracy and loss curves)

### Running the Interactive Interface

```bash
# Open Jupyter and run the interface notebook
jupyter notebook drone_interface.ipynb
```

**Interface Features**:
1. Load trained model from `'Dronedetectionsystem.h5'`
2. Define class names: `['Airplane', 'Birds', 'Drone']`
3. Two main tabs:
   - **Quick Prediction**: Simple image upload and classification
   - **Detailed Analysis**: Comprehensive predictions with visualizations

**Example Usage**:
```python
import gradio as gr
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('Dronedetectionsystem.h5')
class_names = ['Airplane', 'Birds', 'Drone']

# Simple prediction function
def predict_image(image):
    img_array = np.array(image)
    resized = tf.image.resize(img_array, (256, 256))
    scaled = resized / 255.0
    batch = np.expand_dims(scaled, axis=0)
    yhat = model.predict(batch, verbose=0)
    predictions = {
        class_names[i]: float(yhat[0][i])
        for i in range(len(class_names))
    }
    return predictions

# Launch interface
gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3)
).launch()
```

## Notebooks

### 1. `drone.ipynb` - Main Training Notebook

**Purpose**: Complete training pipeline from data preparation to model evaluation

**Workflow**:
1. Import libraries and set configuration
2. Data validation:
   - Read each image using `cv2.imread()`
   - Check image type using `imghdr.what()`
   - Remove files that don't match supported formats
3. Load dataset:
   - Uses `tf.keras.utils.image_dataset_from_directory()`
   - Automatically creates batches of 32 images
   - Infers labels from directory structure
4. Normalize by dividing pixel values by 255
5. Create train/validation/test splits (70/10/20)
6. Build CNN model with 5 convolutional blocks
7. Compile model with Adam optimizer and sparse categorical crossentropy loss
8. Train for 15 epochs with:
   - TensorBoard callback for monitoring
   - Early stopping callback (patience=5)
   - ReduceLROnPlateau callback (factor=0.5, patience=2)
9. Display training history with loss and accuracy curves
10. Evaluate model on test set

**Key Outputs**:
- Trained model weights: `Dronedetectionsystem.h5`
- Training history: Accuracy and loss for all 15 epochs
- Performance curves: Visualization of training progress
- Test set evaluation metrics

### 2. `drone_interface.ipynb` - Gradio Interface Notebook

**Purpose**: Provide interactive web-based interface for predictions

**Components**:

1. **Model Loading**:
   - Load pre-trained model from `'Dronedetectionsystem.h5'`
   - Define class names: `['Airplane', 'Birds', 'Drone']`

2. **Prediction Functions**:
   - `predict_image()`: Simple classification returning confidence scores
   - `predict_with_details()`: Comprehensive analysis with multiple outputs
   - `create_probability_chart()`: Generate bar chart visualization
   - `get_image_metadata()`: Extract image properties

3. **Tab 1: Quick Prediction**:
   - Input: Image upload
   - Output: Class label with probability scores
   - Action: Click "🔍 Classify" button

4. **Tab 2: Detailed Analysis**:
   - Input: Image upload
   - Outputs:
     1. Text results including:
        - Predicted class
        - Confidence percentage
        - Class probabilities with visual bars
        - Top 2 predictions
        - Image metadata (dimensions, channels, size)
        - Prediction timestamp
     2. Probability bar chart visualization
   - Action: Click "📈 Analyze" button
   - Prediction history: Stores last 5 predictions

5. **Tab 3: About**:
   - Model information and architecture details
   - Usage instructions
   - Tips for best results

**Image Processing Pipeline**:
```python
# For each input image
1. Convert to numpy array
2. Resize to 256×256 using tf.image.resize()
3. Scale to [0, 1] by dividing by 255
4. Add batch dimension using np.expand_dims()
5. Pass through model
6. Extract predictions and confidence
```

## Results

### Training Performance (15 Epochs)

**Accuracy Progression**:
1. Epoch 1: Train 44.1% | Val 66.8%
2. Epoch 5: Train 74.2% | Val 83.2%
3. Epoch 10: Train 81.8% | Val 82.0%
4. Epoch 15: Train 85.8% | Val 88.7%

**Loss Progression**:
1. Epoch 1: Train 1.027 | Val 0.725
2. Epoch 5: Train 0.587 | Val 0.489
3. Epoch 10: Train 0.417 | Val 0.407
4. Epoch 15: Train 0.309 | Val 0.317

**Final Metrics**:
1. **Training Accuracy**: 85.84%
2. **Validation Accuracy**: 88.67%
3. **Training Loss**: 0.309
4. **Validation Loss**: 0.317
5. **Learning Rate**: 0.001 (constant throughout training)

**Key Observations**:
1. Model converges well without overfitting
2. Validation accuracy improves consistently
3. Early stopping prevented further training
4. ReduceLROnPlateau maintained optimal learning rate

## Model Interface

### Gradio Web Interface

**Quick Prediction Tab**:
1. Upload image in any format (JPG, PNG, etc.)
2. Click "🔍 Classify" button
3. View predictions for all 3 classes with confidence scores
4. Results show as bar chart with probabilities

**Detailed Analysis Tab**:
1. Upload image
2. Click "📈 Analyze" button
3. View comprehensive results including:
   - Predicted class with confidence percentage
   - All class probabilities with visual progress bars
   - Top 2 ranked predictions
   - Image metadata:
     - Dimensions (width × height pixels)
     - Number of channels (RGB = 3)
     - Data type (uint8)
     - Approximate file size in KB
   - Probability bar chart visualization
   - Timestamp of prediction

**Output Visualization**:
1. Probability chart shows:
   - Horizontal bars for each class
   - Correct prediction highlighted in red
   - Others in teal
   - Percentage labels on bars
2. Text output formatted with emojis and sections
3. Prediction history tracks last 5 predictions

**Tips for Best Results**:
1. Use clear, well-lit images
2. Ensure subject (bird, drone, or airplane) is clearly visible
3. Avoid blurry or heavily cropped images
4. Images are automatically resized to 256×256
5. Works best with images where the subject occupies significant portion of frame

### Running the Interface

**In Jupyter**:
1. Open `drone_interface.ipynb`
2. Run all cells
3. Interface launches at `http://127.0.0.1:7863`

**Standalone**:
```python
# Save as app.py
import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('Dronedetectionsystem.h5')
class_names = ['Airplane', 'Birds', 'Drone']

def predict_image(image):
    img_array = np.array(image)
    resized = tf.image.resize(img_array, (256, 256))
    scaled = resized / 255.0
    batch = np.expand_dims(scaled, axis=0)
    yhat = model.predict(batch, verbose=0)
    return {class_names[i]: float(yhat[0][i]) for i in range(3)}

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3)
).launch()

# Run with: python app.py
```

---

**Framework**: TensorFlow 2.x, Keras, Gradio  
**Language**: Python 3.7+  
**License**: Open Source

For issues or contributions, please open a GitHub issue or pull request.
