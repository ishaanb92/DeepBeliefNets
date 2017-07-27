"""
==============================================================
Deep Belief Network features for digit classification
==============================================================

Adapted from http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py

This example shows how to build a classification pipeline with a UnsupervisedDBN
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the UnsupervisedDBN help improve the
classification accuracy.
"""

from __future__ import print_function

print(__doc__)

import numpy as np

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN # use "from dbn.tensorflow import SupervisedDBNClassification" for computations on TensorFlow
#from matplotlib.pyplot import pyplot as plt
import pickle
import io

###############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=20,
                      activation_function='sigmoid')

###############################################################################

# Training RBM-Logistic Pipeline
dbn.fit(X_train)
# Save the training metrics
for layer_wise_error,index in zip(dbn.layer_wise_error,range(len(dbn.layer_wise_error))):
    with io.open("layer_" + str(index), 'wb') as f:
        pickle.dump(layer_wise_error,f)


###############################################################################
# Evaluation for test samples
error = dbn.reconstruction_accuracy(X_test)
# Save the test sample errors
with io.open("test_recon", 'wb') as f:
    pickle.dump(error,f)

