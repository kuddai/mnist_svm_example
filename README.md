# MNIST SVM EXAMPLE

c++ 14 of SVM classification example on MNIST dataset using vlfeat and opencv.

Linear SVM is utilized for performance reasons although with some optional preprocessing (PCA decomposition, quadratic interaction terms).

With lambda 0.0002, beta 1, epsilon 0.00005, and 86% of variational variance preserved by PCA, it can attain around 97.47% accuracy on test set. This result is in good match with results described [here](http://yann.lecun.com/exdb/mnist/), as essentialy we are using PCA with quadratic classifier.


To reproduce results run the following commands:
```bash
# to generate dataset with train and test descriptions (text file containing on each line path to png image and its label). python 2.7 is required, with pypng
python convert_mnist_to_png.py mnist/ mnist_png
# build program (gcc, c++14, opencv are required)
mkdir build; cd build; cmake ..; make;
mv mnist_png build/code
cv build/code
# training
./main train mnist_png/training/description.txt saved_model preprocessed 0.0002 1 0.00005 0.86
# classifcation
./main classify saved_model mnist_png/testing/description.txt predictions.txt preprocessed
# validation
python ../../validate.py --truth mnist_png/testing/description.txt --predictions predictions.txt


```
