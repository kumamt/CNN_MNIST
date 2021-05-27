import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MNIST:
    def __init__(self):
        # list initialisations
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.loadDataset()
        self.normaliseData()

    def loadDataset(self):
        # Loads the MNIST dataset.
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data(path='mnist.npz')

    def normaliseData(self):
        # Normalises the values in a range.
        self.x_train = tf.keras.utils.normalize(self.x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(self.x_test, axis=1)
        self.trainModel()

    def trainModel(self):
        # Creates a sequential model which means it adds layers one by one in a linear manner.
        self.model = tf.keras.models.Sequential()
        # first layer of the model which we want to be flattened.
        self.model.add(tf.keras.layers.Flatten())
        # Adds the Second layer which we want to be dense(fully connected layer) of size 128
        # with activation function relu.
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # Adds the Third layer which we want to be dense of size 128 with activation function relu.
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # Output layer is also dense and the neurons should be equal to number of target class (in this case
        # it is 10), and we want probabilities of at the end so, now we use activation function as softmax.
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        # Here we do our actual learning, using adam optimizer (which is a better option), loss is a parameter
        # which calculates the difference between the output the model gets and what it should be actually,
        # metrics we want is accuracy of our model.
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Here model adjusts the parameters using the training set and the number of epochs.
        self.model.fit(self.x_train, self.y_train, epochs=3)
        self.testModel()

    def testModel(self):
        # Here we test the model as how good it generalises using of test data.
        loss_value, accuracy_value = self.model.evaluate(self.x_test, self.y_test)
        print(loss_value, accuracy_value)
        # We save our model for predictions of unknown data for the model.
        self.model.save('mnist.model')
        self.predictModel()

    def predictModel(self):
        # Loads the model which is created above.
        new_model = tf.keras.models.load_model('mnist.model')
        # Predicts the output using the test data (with no target class).
        prediction = new_model.predict([self.x_test])
        # Using argmax of numpy it chooses the probability which is maximum
        # for a test case belonging to a particular class.
        print(np.argmax(prediction[12]))
        # Shows the actual output.
        plt.imshow(self.x_test[12])
        plt.show()


call_model = MNIST()
