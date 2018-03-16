# MNIST SVM EXAMPLE

c++ 14 of SVM classification example on MNIST dataset using vlfeat and opencv.

Linear SVM is utilized for performance reasons although with some optional preprocessing (PCA decomposition, quadratic interactions terms).

With lambda 0.0002, beta 1, epsilon 0.00005, and 86% of variational variance preserved by PCA, it can attain around 97.47% accuracy on test set. This result is in good match with results described [here](http://yann.lecun.com/exdb/mnist/), as essentialy we are using PCA with quadratic classifier.
