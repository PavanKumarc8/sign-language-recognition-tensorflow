# Sign Language Recognition System using TensorFlow

This project implements a Sign Language Recognition System using TensorFlow in Python. The system is trained on the Sign Language MNIST dataset, which contains images of signs corresponding to each alphabet in the English language.

## Dataset
The dataset consists of two CSV files: `sign_mnist_train.csv` and `sign_mnist_test.csv`. Each row in the CSV files represents a training sample, with the label and pixel values of a 28x28 image. Labels range from 0 to 24, excluding 'J' and 'Z' classes. 

## Data Loading and Preprocessing
The dataset is loaded and preprocessed using Pandas and NumPy libraries. Labels are one-hot encoded, and the pixel values are reshaped into 28x28 images. The training and testing data are then prepared for model training.

## Model Development
A Convolutional Neural Network (CNN) model is developed using TensorFlow's Keras API. The model architecture consists of three Conv2D layers, followed by MaxPooling layers, a Flatten layer, and two fully connected layers. BatchNormalization and Dropout layers are included for improved training stability and prevention of overfitting.

## Model Training and Evaluation
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. Training is performed for 5 epochs, and the model's performance is evaluated on validation data. Training and validation accuracy are visualized with each epoch.

## Conclusion
The trained model achieves an accuracy of 82%, demonstrating its effectiveness in recognizing sign language gestures. This technology has the potential to assist individuals with special needs and contribute to the development of innovative applications.

## Getting Started
1. Clone the repository:
git clone https://github.com/your-username/sign-language-recognition-tensorflow.git


2. Install the required dependencies listed in `requirements.txt`:
pip install -r requirements.txt


3. Run the Python script to train and evaluate the model:
python sign_language_recognition.py


4. Explore the trained model's performance and make improvements as necessary.

## Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request with any improvements or additional features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
