#pragma once

#include <string>
#include <vector>
#include <sstream>

#include <opencv2/core.hpp>

#include "exception.h"
#include "multiclass_svm.h"


namespace ml {

    typedef std::vector<std::vector<double>> Matrix;
    typedef std::tuple<
        std::vector<std::vector<double>>, 
        std::vector<int>,
        std::vector<std::string>
    > Data;

    void ValidateBinaryLabels(const std::vector<int> &y); 

    void ValidateDimensions(size_t expected, size_t got, size_t idx = 0); 

    void ValidateTrainData(const Matrix &x, 
                           const std::vector<int> &y,
                           bool has_binary_labels = true);

    double DotProduct(const std::vector<double> v1, const std::vector<double> v2); 

    Data ReadData(const std::string &data_path, bool load_label = true);

    void SaveModel(const MulticlassSVM &svm, const std::string &save_path);

    MulticlassSVM ReadModel(const std::string &model_path);
    
    void SavePredictions(const std::vector<std::string> &image_paths,
                         const std::vector<int> &predictions, 
                         const std::string &output_path);

    void SaveNormalizationParams(const std::string &path, double mean, double std_dev); 

    std::tuple<double, double> LoadNormalizationParams(const std::string &path); 

    Matrix Normalize(const Matrix &x, double mean, double std_dev);

    std::tuple<Matrix, double, double> Normalize(const Matrix &x);

    void SavePCA(const std::string &path, cv::PCA &pca);

    cv::PCA LoadPCA(const std::string &path);

    cv::PCA CreatePCA(const Matrix &x, double retain_variance = 0.95);

    Matrix ProjectPCA(const cv::PCA &pca, const Matrix &x); 

    Matrix AddQuadraticInteractions(const Matrix &x);
} // namespace ml

