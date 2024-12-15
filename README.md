# Melanoma_detection_assignment
Melanoma_detection

In the realm of cancer, there exist over 200 distinct forms, with melanoma standing out as the most lethal type of skin cancer among them. The diagnostic protocol for melanoma typically initiates with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma skin cancer is pivotal, as it significantly enhances the chances of successful treatment. The initial step in diagnosing melanoma skin cancer involves visually inspecting the affected area of the skin. Dermatologists capture dermatoscopic images of the skin lesions using high-speed cameras, which yield diagnostic accuracies ranging from 65% to 80% for melanoma without supplementary technical assistance. Through further visual assessment by oncologists and dermatoscopic image analysis, the overall predictive accuracy of melanoma diagnosis can be elevated to 75% to 84%. The objective of the project is to construct an automated classification system leveraging image processing techniques to classify skin cancer based on images of skin lesions.

## Problem statement

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Class Imbalance

Class with the least number of samples: seborrheic keratosis
Class that dominates the data: pigmented benign keratosis

To address this we use propotional augmentation with zoom cause images are centralized


## Model Architecture

The break down of the final provided CNN architecture step by step:

1. **Data Augmentation**: The `augmentation_data` variable refers to the augmentation techniques applied to the training data. Data augmentation is used to artificially increase the diversity of the training dataset by applying random transformations such as rotation, scaling, and flipping to the images. This helps in improving the generalization capability of the model.

2. **Normalization**: The `Rescaling(1./255)` layer is added to normalize the pixel values of the input images. Normalization typically involves scaling the pixel values to a range between 0 and 1, which helps in stabilizing the training process and speeding up convergence.

3. **Convolutional Layers**: Three convolutional layers are added sequentially using the `Conv2D` function. Each convolutional layer is followed by a rectified linear unit (ReLU) activation function, which introduces non-linearity into the model. The `padding='same'` argument ensures that the spatial dimensions of the feature maps remain the same after convolution. The number within each `Conv2D` layer (16, 32, 64) represents the number of filters or kernels used in each layer, determining the depth of the feature maps.

4. **Pooling Layers**: After each convolutional layer, a max-pooling layer (`MaxPooling2D`) is added to downsample the feature maps, reducing their spatial dimensions while retaining the most important information. Max-pooling helps in reducing computational complexity and controlling overfitting.

5. **Dropout Layer**: A dropout layer (`Dropout`) with a dropout rate of 0.2 is added after the last max-pooling layer. Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of the neurons during training.

6. **Flatten Layer**: The `Flatten` layer is added to flatten the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.

7. **Fully Connected Layers**: Two fully connected (dense) layers (`Dense`) are added with ReLU activation functions. The first dense layer consists of 128 neurons, and the second dense layer outputs the final classification probabilities for each class label.

8. **Output Layer**: The number of neurons in the output layer is determined by the `target_labels` variable, representing the number of classes in the classification task. The output layer does not have an activation function specified, as it is followed by the loss function during training.

9. **Model Compilation**: The model is compiled using the Adam optimizer (`optimizer='adam'`) and the Sparse Categorical Crossentropy loss function (`loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`), which is suitable for multi-class classification problems. Additionally, accuracy is chosen as the evaluation metric (`metrics=['accuracy']`).

10. **Training**: The model is trained using the `fit` method with the specified number of epochs (`epochs=50`). 

## Model Summary

Epoch 1/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 105s 390ms/step - accuracy: 0.3689 - loss: 2.1975 - val_accuracy: 0.1117 - val_loss: 4.1831
Epoch 2/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 84s 367ms/step - accuracy: 0.5525 - loss: 1.3926 - val_accuracy: 0.3533 - val_loss: 2.2870
Epoch 3/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 102s 446ms/step - accuracy: 0.6431 - loss: 1.1157 - val_accuracy: 0.6572 - val_loss: 0.9464
Epoch 4/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 81s 350ms/step - accuracy: 0.7096 - loss: 0.8754 - val_accuracy: 0.7167 - val_loss: 0.8001
Epoch 5/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 78s 339ms/step - accuracy: 0.7641 - loss: 0.6896 - val_accuracy: 0.7078 - val_loss: 0.8177
Epoch 6/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 79s 344ms/step - accuracy: 0.7916 - loss: 0.6215 - val_accuracy: 0.7006 - val_loss: 0.8686
Epoch 7/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 334ms/step - accuracy: 0.8362 - loss: 0.4764 - val_accuracy: 0.7133 - val_loss: 0.8676
Epoch 8/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 83s 337ms/step - accuracy: 0.8574 - loss: 0.4180 - val_accuracy: 0.6528 - val_loss: 1.0913
Epoch 9/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 78s 340ms/step - accuracy: 0.8643 - loss: 0.3952 - val_accuracy: 0.5783 - val_loss: 1.3990
Epoch 10/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 81s 337ms/step - accuracy: 0.8789 - loss: 0.3526 - val_accuracy: 0.7322 - val_loss: 0.9351
Epoch 11/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 82s 337ms/step - accuracy: 0.9025 - loss: 0.2720 - val_accuracy: 0.7661 - val_loss: 0.6964
Epoch 12/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 83s 338ms/step - accuracy: 0.9030 - loss: 0.2711 - val_accuracy: 0.8022 - val_loss: 0.6949
Epoch 13/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 334ms/step - accuracy: 0.9133 - loss: 0.2519 - val_accuracy: 0.7228 - val_loss: 0.8569
Epoch 14/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 79s 341ms/step - accuracy: 0.9104 - loss: 0.2531 - val_accuracy: 0.7461 - val_loss: 0.8730
Epoch 15/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 336ms/step - accuracy: 0.9174 - loss: 0.2287 - val_accuracy: 0.7017 - val_loss: 1.1125
Epoch 16/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 79s 343ms/step - accuracy: 0.9267 - loss: 0.1987 - val_accuracy: 0.4006 - val_loss: 4.1157
Epoch 17/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 81s 338ms/step - accuracy: 0.9283 - loss: 0.2016 - val_accuracy: 0.6072 - val_loss: 1.3996
Epoch 18/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 79s 341ms/step - accuracy: 0.9225 - loss: 0.2082 - val_accuracy: 0.5683 - val_loss: 2.2099
Epoch 19/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 78s 340ms/step - accuracy: 0.9332 - loss: 0.1872 - val_accuracy: 0.7733 - val_loss: 0.7503
Epoch 20/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 81s 336ms/step - accuracy: 0.9356 - loss: 0.1815 - val_accuracy: 0.7600 - val_loss: 0.8059
Epoch 21/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 78s 341ms/step - accuracy: 0.9371 - loss: 0.1721 - val_accuracy: 0.7761 - val_loss: 0.8611
Epoch 22/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 336ms/step - accuracy: 0.9243 - loss: 0.1890 - val_accuracy: 0.7817 - val_loss: 0.7254
Epoch 23/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 78s 340ms/step - accuracy: 0.9346 - loss: 0.1823 - val_accuracy: 0.7800 - val_loss: 0.7079
Epoch 24/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 76s 332ms/step - accuracy: 0.9422 - loss: 0.1643 - val_accuracy: 0.7517 - val_loss: 0.9936
Epoch 25/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 336ms/step - accuracy: 0.9462 - loss: 0.1525 - val_accuracy: 0.3617 - val_loss: 4.2934
Epoch 26/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 333ms/step - accuracy: 0.9464 - loss: 0.1421 - val_accuracy: 0.7900 - val_loss: 0.7169
Epoch 27/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 76s 333ms/step - accuracy: 0.9476 - loss: 0.1404 - val_accuracy: 0.7750 - val_loss: 0.8505
Epoch 28/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 81s 353ms/step - accuracy: 0.9469 - loss: 0.1431 - val_accuracy: 0.7900 - val_loss: 0.8030
Epoch 29/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 77s 338ms/step - accuracy: 0.9433 - loss: 0.1550 - val_accuracy: 0.4433 - val_loss: 3.3245
Epoch 30/30
225/225 ━━━━━━━━━━━━━━━━━━━━ 78s 340ms/step - accuracy: 0.9534 - loss: 0.1252 - val_accuracy: 0.6850 - val_loss: 1.4141
