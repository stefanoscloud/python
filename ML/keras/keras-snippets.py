# Keras library for Python - Frontend to Tensorflow

# Build CNN models - Sequential() class 
# network = keras.models.Sequential() —This constructs an object that you can use to start building network layers sequentially.
# network.add(keras.layers.Conv2D(filters = 64, kernel_size = (2, 2), input_shape = (50, 50, 1), padding = 'same', activation = 'relu')) —This adds a convolutional layer that can work on two-dimensional images. In this case, the number of output filters is 64, the shape of those filters is 2 × 2, the shape of the input is 50 × 50, the layer will use padding, and the layer will use the standard ReLU activation function.
# network.add(keras.layers.MaxPooling2D((2, 2))) —This adds a pooling layer that will downscale the image by half its original height and width. 
# network.add(keras.layers.Flatten()) —This adds a flattening layer to reduce the dimensionality of the previous layer's output.
# network.add(keras.layers.Dense(3, activation = 'softmax')) —This adds a dense (fully connected) layer to use as the output layer. In this case, there are three possible classes, and the activation function being used is softmax.
# network.compile(optimizer = 'sgd', loss = 'categorical_crossentropy’, metrics = ['accuracy']) —This compiles a sequentially built network. In this case, the network will use a stochastic gradient descent (SGD) solver, a loss function that is commonly used for multi-class classification, and accuracy as the evaluation metric.
# network.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 10) —Fit the training data to the neural network for the specified number of epochs. Optionally, you can supply data to use for validation. 
# network.summary() —Print the general structure of a sequential network you've built.
# keras.utils.plot_model(network, to_file = 'network.png') —Create a more visual representation of the network structure and save it to a file.
# network.evaluate(X_test, y_test) —Evaluate the network's performance on test data. This returns both the loss value and the value of the chosen metric.
# network.predict(X_test) —Use the network to make predictions on test data.

# Build RNN models - Sequential() class 
# network = keras.models.Sequential() —This constructs the sequential network object.
# network.add(keras.layers.Embedding(input_dim = 5000, output_dim = 200, input_length = 1000)) —This constructs an embedding layer. In this case, the vocabulary size is 5,000 words, the word vector is 200 dimensions, and the length of the input text is 1,000 words.
# network.add(keras.layers.LSTM(units = 128)) —This creates an LSTM cell of the specified dimensions.
# You can use the same Flatten() and Dense() layers where necessary.
# You can use the same object methods as with a Keras CNN to compile the model, fit the training data, evaluate the model, predict test data, etc


