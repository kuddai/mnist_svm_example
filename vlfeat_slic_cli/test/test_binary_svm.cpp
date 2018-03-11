#include <vector>

#include <catch.hpp>

#include "binary_svm.h"
#include "exception.h"


TEST_CASE("empty model", "svm") {
    ml::BinarySVM svm;
    REQUIRE(svm.GetModel().empty());
    REQUIRE(svm.GetBias() == 0);

    std::vector<std::vector<double>> x = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };
    REQUIRE_THROWS_AS(svm.Predict(x), ml::Exception);
}

TEST_CASE("model training - perfectly separable case", "svm") {
    // dummpy example taken from here
    // https://stackoverflow.com/questions/33174696/tuning-vlfeat-svm
    ml::BinarySVM svm;
    std::vector<std::vector<double>> x = {
        {8},
        {7},
        {6},
        {3},
        {2},
        {1}
    };

    std::vector<int> y = {1, 1, 1, -1, -1, -1};    
    double lambda = 0.000001;
    svm.Train(x, y, lambda);

    REQUIRE(svm.GetModel()[0] == Approx(0.9986288825));
    REQUIRE(svm.GetBias() == Approx(-4.00736089));

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] == Approx(y[0]));
    REQUIRE(predictions[1] == Approx(y[1]));
    REQUIRE(predictions[2] == Approx(y[2]));
    REQUIRE(predictions[3] == Approx(y[3]));
    REQUIRE(predictions[4] == Approx(y[4]));
    REQUIRE(predictions[5] == Approx(y[5]));
}


TEST_CASE("initializing model and bias", "svm") {
    std::vector<double> model = {0.9986288825};
    double bias = -4.00736089;
    ml::BinarySVM svm(model, bias);

    std::vector<std::vector<double>> x = {
        {8},
        {7},
        {6},
        {3},
        {2},
        {1}
    };

    std::vector<int> y = {1, 1, 1, -1, -1, -1};    

    auto predictions = svm.Predict(x);

    REQUIRE(predictions[0] ==  1);
    REQUIRE(predictions[1] ==  1);
    REQUIRE(predictions[2] ==  1);
    REQUIRE(predictions[3] == -1);
    REQUIRE(predictions[4] == -1);
    REQUIRE(predictions[5] == -1);
}

TEST_CASE("checking inconsistent input", "svm") {
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

