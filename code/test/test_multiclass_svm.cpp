#include <vector>
#include <iostream>

#include <catch.hpp>

#include "multiclass_svm.h"
#include "exception.h"


TEST_CASE("empty multiclass model", "multiclass svm") {
    ml::MulticlassSVM svm;
    REQUIRE(svm.GetModels().empty());
    REQUIRE(svm.GetBiases().empty());
    REQUIRE(svm.GetLabels().empty());

    std::vector<std::vector<double>> x = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };

    REQUIRE_THROWS_AS(svm.Predict(x), ml::Exception);
}

TEST_CASE("model training - linearly separable 3 labels", "multiclass svm") {
    ml::MulticlassSVM svm;
    std::vector<std::vector<double>> x = {{1}, {2}, {3}, {4}, {5}, {6}};

    std::vector<int> y = {1, 1, 2, 2, 3, 3};    
    double lambda = 0.000001;
    double bias_multiplier = 5;
    svm.Train(x, y, lambda, bias_multiplier);

    const std::vector<std::vector<double>>& models = svm.GetModels();
    const std::vector<double>& biases = svm.GetBiases();

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] == Approx(y[0]));
    REQUIRE(predictions[1] == Approx(y[1]));
    REQUIRE(predictions[2] == Approx(y[2]));
    REQUIRE(predictions[3] == Approx(y[3]));
    REQUIRE(predictions[4] == Approx(y[4]));
    REQUIRE(predictions[5] == Approx(y[5]));
}


TEST_CASE("initializing multiclass models and biases", "multiclass svm") {
    std::vector<std::vector<double>> models = {
        {2.99943},
        {1.99984},
        {2.99876}
    };
    std::vector<double> biases = {-7.49945, -5.4998, -13.4973};
    std::vector<int> labels = {1, 2, 3};

    ml::MulticlassSVM svm(models, biases, labels);

    std::vector<std::vector<double>> x = {{1}, {2}, {3}, {4}, {5}, {6}};
    std::vector<int> y = {1, 1, 2, 2, 3, 3};    

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] == Approx(y[0]));
    REQUIRE(predictions[1] == Approx(y[1]));
    REQUIRE(predictions[2] == Approx(y[2]));
    REQUIRE(predictions[3] == Approx(y[3]));
    REQUIRE(predictions[4] == Approx(y[4]));
    REQUIRE(predictions[5] == Approx(y[5]));
}

TEST_CASE("checking inconsistent multiclass input", "multiclass svm") {
    std::vector<std::vector<double>> models = {
        {0.9986288825},
        {0.9986288825},
        {0.9986288825}
    };
    std::vector<double> biases = {-4.00736089, -4.00736089, -4.00736089};
    std::vector<int> labels = {1, 2, 3};

    ml::MulticlassSVM svm(models, biases, labels);

    // inconsistent input
    std::vector<std::vector<double>> x = {{1, 2}, {2}, {3}, {4}, {5}, {6}};
    REQUIRE_THROWS_AS(svm.Predict(x), ml::Exception);
}

