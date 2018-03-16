#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "exception.h"
#include "multiclass_svm.h"
#include "util.h"


void Train(
    const std::string &data_path,
    const std::string &save_path,
    bool preprocessed = false,
    double lambda = 0.01,
    double bias_multiplier = 1,
    double epsilon = 0.02,
    double retain_variance = 0.95
) {
    auto data = ml::ReadData(data_path);
    auto x = std::get<0>(data);
    auto y = std::get<1>(data);

    if (preprocessed && !x.empty()) {
        std::cout << "normalizing input" << std::endl;
        auto out = ml::Normalize(x);
        x = std::get<0>(out);
        double mean = std::get<1>(out);
        double std_dev = std::get<2>(out);

        std::cout << "preprocessing input, retain_variance " << retain_variance << std::endl;
        auto pca = ml::CreatePCA(x, retain_variance);
        x = ml::ProjectPCA(pca, x);
        std::cout << "dimensionality after projection " << x.at(0).size() << std::endl;
        std::cout << "first image after projection: " << std::endl;
        for (size_t i = 0; i < x.at(0).size(); ++i) {
            std::cout << x.at(0).at(i) << " ";
        }
        std::cout << std::endl;

        x = ml::AddQuadraticInteractions(x);
        std::cout << "add quadratic interactions, dimensionality after ";
        std::cout << x.at(0).size() << std::endl;

        std::cout << "saving pca" << std::endl;
        ml::SavePCA(save_path + ".pca", pca);
        ml::SaveNormalizationParams(save_path + ".norm", mean, std_dev);
    }

    std::cout << "start learning\n" << std::endl;
    std::cout << "lambda " << lambda << std::endl;
    std::cout << "bias multiplier " << bias_multiplier << std::endl;
    std::cout << "epsilon " << epsilon << std::endl;

    ml::MulticlassSVM svm;
    svm.Train(x, y, lambda, bias_multiplier, epsilon);

    std::cout << "finish learning\n" << std::endl;
    ml::SaveModel(svm, save_path + ".svm");
}

void Classify(const std::string &model_path, 
              const std::string &input_path, 
              const std::string &output_path, 
              bool preprocessed = false) {
    auto svm = ml::ReadModel(model_path + ".svm");
    auto data = ml::ReadData(input_path, false);
    auto x = std::get<0>(data);

    if (preprocessed && !x.empty()) {
        std::cout << "preprocessing input" << std::endl;
        auto out = ml::LoadNormalizationParams(model_path + ".norm");
        double mean = std::get<0>(out);
        double std_dev = std::get<1>(out);
        auto pca = ml::LoadPCA(model_path + ".pca");
        x = ml::Normalize(x, mean, std_dev);
        x = ml::ProjectPCA(pca, x);
        std::cout << "dimensionality after projection " << x.at(0).size() << std::endl;
        x = ml::AddQuadraticInteractions(x);
        std::cout << "add quadratic interactions, dimensionality after ";
        std::cout << x.at(0).size() << std::endl;
    }

    auto predictions = svm.Predict(x);
    ml::SavePredictions(std::get<2>(data), predictions, output_path);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "the following arguments are expected" << std::endl;
        std::cout << "either: 'train' <data_path> <save_path> ";
        std::cout << "[preprocessed] [lambda] [bias_multiplier] [epsilon]" << std::endl;
        std::cout << "or: 'classify' <model_path>";
        std::cout << " <input_path> <output_path> [preprocessed]" << std::endl;
        return 1;
    }

    std::string mode(argv[1]);
    if (mode == "train" && argc >= 4) {
        try {
            Train(argv[2], 
                  argv[3],
                  argc >= 4 + 1 ? std::string(argv[4]) == "preprocessed" : false, 
                  argc >= 5 + 1 ? atof(argv[5]) : 0.01,
                  argc >= 6 + 1 ? atof(argv[6]) : 1,
                  argc >= 7 + 1 ? atof(argv[7]) : 0.02, 
                  argc >= 8 + 1 ? atof(argv[8]) : 0.95);
        } catch (const ml::Exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        } catch (...) {
            std::cerr << "something went wrong during training" << std::endl;
            return 1;
        }
    }

    if (mode == "classify" && argc >= 5) {
        try {
            Classify(argv[2],
                     argv[3],
                     argv[4],
                     argc >= 5 + 1 ? std::string(argv[5]) == "preprocessed" : false);
        } catch(const ml::Exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        } catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        } catch (...) {
            std::cerr << "something went wrong during classification" << std::endl;
            return 1;
        }
    }

    return 0;
}

