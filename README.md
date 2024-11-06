Objective: The goal of this assignment is to train two models—Naive Bayes and a Perceptron—to predict
the type of diabetes based on input features (age, blood sugar level (Glucose), insulin level, and BMI).

The Naïve Bayes Generative Classifier is a widely-used algorithm in machine learning, it operates on
the principles of Bayes' theorem and assumes independence among features, allowing it to make
predictions quickly and with minimal computational resources. It predicts the class for a given
instance based on the class probabilities computed using Bayes’ theorem. After calculating the
posterior probability of each class given the instance's features, the algorithm selects the class with
the highest probability as the predicted class for that instance. In other words, it assigns the class that
maximizes the posterior probability. 

The Perceptron model is a simple yet powerful algorithm for binary classification, inspired by the structure of a single neuron in the human brain. It operates by taking a linear combination of input features, applying weights to each feature, and passing the result through an activation function (typically a step function). The perceptron model predicts the class of an instance by calculating the weighted sum of its features, and if the result exceeds a certain threshold, it classifies the instance into one class, otherwise into the other class.

During the training process, the model adjusts the weights based on the error between the predicted and actual class labels, using a method called the perceptron learning rule. This rule updates the weights iteratively, ensuring the model minimizes classification errors over time. The perceptron model is efficient and works well for linearly separable data, but it may struggle with more complex, non-linear patterns. Nonetheless, it forms the foundation for more advanced models, such as multi-layer neural networks.
