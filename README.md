# Dog Breed Classification using Transfer Learning

This project focuses on building and optimizing a Convolutional Neural Network (CNN) for classifying dog breeds using transfer learning. The project utilizes pre-trained models (MobileNetV2, ResNet50, and Xception) and fine-tuning techniques to achieve high classification accuracy.

## Project Structure

The notebook is structured as follows:

1.  **Data Loading and Exploration**: Loading the dog images dataset and exploring its contents, including visualizing sample images.
2.  **Data Preprocessing**: Implementing functions to load and preprocess images, including handling potential decoding errors in JPEG files and resizing images.
3.  **Model Building**: Defining a function to build the CNN model using a pre-trained base model, adding a global average pooling layer and a dense output layer for classification.
4.  **Model Training and Evaluation**: Training and evaluating the performance of three different pre-trained models (MobileNetV2, ResNet50, and Xception) on the dataset.
5.  **Model Optimization**: Fine-tuning the best-performing model (Xception) by unfreezing layers and adjusting the learning rate to further improve accuracy.
6.  **Results Visualization**: Plotting training history (accuracy and loss) for each model to analyze performance over epochs.
7.  **Classification Report**: Generating classification reports to evaluate the precision, recall, and F1-score for each class.

## Models Used

*   **MobileNetV2**: A lightweight and efficient model.
*   **ResNet50**: A deep residual network.
*   **Xception**: A model based on depthwise separable convolutions.

## Optimization Techniques

*   Lower learning rate
*   Unfreezing base layers for fine-tuning

## Results

The final optimized Xception model achieved a test accuracy of **85.46%** and a test loss of **0.483**.

## How to Run the Notebook

1.  Clone the repository or download the notebook.
2.  Ensure you have the necessary libraries installed (TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, kagglehub).
3.  Download the dataset using the provided Kagglehub command or ensure the dataset is accessible in the specified path.
4.  Run the cells sequentially to execute the project.

## Dependencies

*   tensorflow
*   keras
*   numpy
*   pandas
*   matplotlib
*   seaborn
*   sklearn
*   kagglehub
