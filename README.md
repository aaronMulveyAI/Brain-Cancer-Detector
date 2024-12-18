# Brain Cancer Detector

## Overview
This project implements a **Brain Cancer Detector** using Convolutional Neural Networks (CNNs). The system processes brain scan images to classify them into predefined categories. It supports loading, preprocessing, training, validating, and testing models, as well as making predictions on new images via a graphical interface.

## Project Workflow

1. **Dataset Preparation**
    - Dataset paths for training and testing are specified.
    - Images are loaded from folders, resized to a fixed dimension (320x320), and labeled based on their folder names.

2. **Data Splitting**
    - Data is split into:
      - **Training set** (80%)
      - **Validation set** (10%)
      - **Test set** (10%)

3. **Model Definition**
    - A CNN architecture is defined using TensorFlow/Keras:
        - Convolutional layers
        - MaxPooling layers
        - Dense layers
        - Softmax output layer for classification

4. **Model Training**
    - The model is trained using the training set with categorical cross-entropy loss and SGD optimizer.
    - Validation data is used to monitor the training progress.
    - TensorBoard is integrated for visualizing metrics during training.

5. **Model Evaluation**
    - The model is evaluated on the test set to compute metrics like loss and accuracy.

6. **Prediction Interface**
    - A GUI built with Tkinter allows users to select and classify new images.

## Installation

### Prerequisites
- Python 3.7+
- Required Python libraries:
    - TensorFlow
    - NumPy
    - OpenCV
    - Matplotlib
    - Tkinter

### Steps
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd Brain-Cancer-Detector
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset and place it in the `data` directory with the structure:
    ```
    /data
      /archive
        /Training
          /label1
          /label2
        /Testing
          /label1
          /label2
    ```
4. Run the project:

    ```bash
    python main.py
    ```

## Code Structure

### Data Preparation
- **Loading and Preprocessing**: Images are loaded using OpenCV, resized, and stored along with their labels.
- **Shuffling and Splitting**: Data is randomized and split into training, validation, and test sets.

### Model Definition
- The CNN model consists of:
    - Convolutional layers for feature extraction.
    - MaxPooling layers to reduce dimensions.
    - Dense layers to map features to outputs.
    - Softmax layer for classification into multiple categories.

### Callbacks
- **TensorBoard**: For tracking training metrics.
- **Model Checkpoint**: Saves models at every epoch.

### Evaluation
- The model is evaluated using the test set.
- Metrics include loss and accuracy.

### GUI for Predictions
- Tkinter-based graphical interface:
    - Users can upload an image.
    - The model predicts the class and displays the result along with the image.

## Usage
### Training the Model
Run the script to train the model and save checkpoints:

```bash
python train.py
```

### Predicting New Images

After training the model, run the GUI for predictions:
```bash
python predict.py
```

## Key Features
- **Data Loading**: Automatically loads and processes datasets from folders.
- **Flexible Architecture**: Supports adding more layers or modifying the CNN architecture.
- **Visualization**: TensorBoard integration for detailed analysis.
- **User-Friendly GUI**: Allows non-technical users to classify images easily.

## Future Enhancements
- Incorporate more advanced preprocessing techniques (e.g., data augmentation).
- Use transfer learning with pre-trained models for better accuracy.
- Add support for more complex datasets and classifications.

## Results
- Achieved a test accuracy of ~85% on the provided dataset.
- Demonstrated robust performance on unseen test images.


For any queries or contributions, feel free to raise an issue or contact the project maintainers!

