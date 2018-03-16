#include <vector>

#include <catch.hpp>

#include "binary_svm.h"
#include "exception.h"


TEST_CASE("empty model", "binary svm") {
    ml::BinarySVM svm;
    REQUIRE(svm.GetModel().empty());
    REQUIRE(svm.GetBias() == 0);

    std::vector<std::vector<double>> x = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };
    REQUIRE_THROWS_AS(svm.Predict(x), ml::Exception);
}

TEST_CASE("model training - linearly separable case", "binary svm") {
    ml::BinarySVM svm;
    std::vector<std::vector<double>> x = {{8}, {7}, {6}, {3}, {2}, {1}};

    std::vector<int> y = {1, 1, 1, -1, -1, -1};    
    double lambda = 0.000001;
    svm.Train(x, y, lambda);

    REQUIRE(svm.GetModel()[0] == Approx(0.9986288825));
    REQUIRE(svm.GetBias() == Approx(-4.00736089));

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] == y[0]);
    REQUIRE(predictions[1] == y[1]);
    REQUIRE(predictions[2] == y[2]);
    REQUIRE(predictions[3] == y[3]);
    REQUIRE(predictions[4] == y[4]);
    REQUIRE(predictions[5] == y[5]);
}

TEST_CASE("model training - binary small dataset", "binary svm") {
    ml::BinarySVM svm;
    std::vector<std::vector<double>> x = {{2}, {3}, {4}, {5}, {6}}; 
    std::vector<int> y = {-1, -1, 1, 1, 1};    
    double lambda = 0.001;
    // without bias_multiplier beta is not big enough
    // as we don't have many dimensions
    double bias_multiplier = 5;
    svm.Train(x, y, lambda, bias_multiplier);

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] == y[0]);
    REQUIRE(predictions[1] == y[1]);
    REQUIRE(predictions[2] == y[2]);
    REQUIRE(predictions[3] == y[3]);
}


TEST_CASE("initializing model and bias", "binary svm") {
    std::vector<double> model = {0.9986288825};
    double bias = -4.00736089;
    ml::BinarySVM svm(model, bias);

    std::vector<std::vector<double>> x = {{8}, {7}, {6}, {3}, {2}, {1}};

    std::vector<int> y = {1, 1, 1, -1, -1, -1};    

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] ==  1);
    REQUIRE(predictions[1] ==  1);
    REQUIRE(predictions[2] ==  1);
    REQUIRE(predictions[3] == -1);
    REQUIRE(predictions[4] == -1);
    REQUIRE(predictions[5] == -1);
}

TEST_CASE("checking inconsistent input", "binary svm") {
    std::vector<double> model = {-0.6220197226};
    double bias = -0.0002974511;
    ml::BinarySVM svm(model, bias);

    std::vector<std::vector<double>> x = {
        {188.0, 666}, // inconsistent input
        {168.0},
        {191.0},
        {150.0},
        {154.0},
        {124.0}
    };

    REQUIRE_THROWS_AS(svm.Predict(x), ml::Exception);
}

TEST_CASE("checking wrong labels", "binary svm") {
    ml::BinarySVM svm;
    std::vector<std::vector<double>> x = {{8}, {7}, {6}, {3}, {2}, {1}};

    // some labels are not in set {1, -1}
    std::vector<int> y = {2, 1, 5, -1, -1, -1};    
    double lambda = 0.000001;

    REQUIRE_THROWS_AS(svm.Train(x, y, lambda), ml::Exception);
}


TEST_CASE("2 demensional case", "binary svm") {
    ml::BinarySVM svm;
    std::vector<std::vector<double>> x = {
        {0.0, -0.5},
        {0.6, -0.3},
        {0.0,  0.5},
        {0.6,  0.0}
    };

    // some labels are not in set {1, -1}
    std::vector<int> y = {1, 1, -1, 1};    
    double lambda = 0.01;
    svm.Train(x, y, lambda);

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] ==  1);
    REQUIRE(predictions[1] ==  1);
    REQUIRE(predictions[2] == -1);
    REQUIRE(predictions[3] ==  1);
}
