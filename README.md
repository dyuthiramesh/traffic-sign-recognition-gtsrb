# Traffic Sign Recognition using Convolutional Neural Networks (CNNs)

This repository contains code for training and evaluating Convolutional Neural Networks (CNNs) on the task of traffic sign recognition. Four different CNN architectures are implemented and evaluated: LeNet-5, AlexNet, VGG-16, and ResNet-18.

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used for training and testing the models. It contains images of 43 different classes of traffic signs.

## Requirements

- Python 3.x
- PyTorch
- TensorFlow (for visualization purposes)
- scikit-learn
- matplotlib
- seaborn
- pandas
- visualkeras
- Google Colab (for running the notebook)

You can install the required Python packages using the following command:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   - Ensure that you have downloaded the GTSRB dataset and placed it in the appropriate directory (`./data`).
   - The dataset can be downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

2. **Training**:
   - Execute the notebook or script for each CNN architecture to train the models.
   - Adjust hyperparameters and training configurations as needed.

3. **Evaluation**:
   - After training, evaluate the trained models using the provided evaluation functions.
   - The evaluation includes generating confusion matrices, classification reports, ROC curves, and calculating mean Average Precision (mAP).

4. **Model Comparison**:
   - Compare the performance of different CNN architectures based on accuracy, computational efficiency, and other metrics.

## Model Architectures

1. **LeNet-5**:
   - A classic CNN architecture with two convolutional layers followed by max-pooling layers and fully connected layers.
   ![LeNet-5 Architecture](images/lenet5.png)
   
2. **AlexNet**:
   - A deeper CNN architecture compared to LeNet-5, introduced in the AlexNet paper by Krizhevsky et al. It consists of several convolutional and max-pooling layers followed by fully connected layers.
   ![AlexNet Architecture](images/alexnet.jpeg)

3. **VGG-16**:
   - A deeper architecture characterized by a stack of convolutional layers with small 3x3 filters, followed by max-pooling layers and fully connected layers.
   ![VGG-16 Architecture](images/vgg16.jpeg)

4. **ResNet-18**:
   - A CNN architecture with residual connections, addressing the vanishing gradient problem in deeper networks. It consists of several residual blocks, each containing convolutional layers with identity shortcuts.
   ![ResNet-18 Architecture](images/resnet18.png)

## Results and Discussion

- Evaluate the performance of each model architecture on the traffic sign recognition task.
- Discuss the trade-offs between accuracy and computational efficiency for each architecture.
- Provide insights into the suitability of each architecture for real-world applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---