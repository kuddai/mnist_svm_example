#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "exception.h"
#include "multiclass_svm.h"
#include "util.h"


namespace ml {
    const size_t REPORT_THRESHOLD = 1000;

    void ValidateBinaryLabels(const std::vector<int> &y) {
        for (size_t i = 0; i < y.size(); ++i) {
            if (y[i] != 1 && y[i] != -1) {
                throw Exception(
                    "wrong label in " + std::to_string(i) +
                    ": must be 1 or -1, got " + std::to_string(y[i])
                );
            }
        }
    }

    void ValidateDimensions(size_t expected, size_t got, size_t idx) {
        if (expected != got) {
            throw Exception(
                "dimension mismatch in input " + std::to_string(idx) +
                ": expected " + std::to_string(expected) + 
                ", got " + std::to_string(got)
            );
        }
    }

    void ValidateTrainData(const Matrix &x, 
                           const std::vector<int> &y,
                           bool has_binary_labels) {
        if (x.empty()) {
            throw Exception("x is empty");
        }

        if (x.size() != y.size()) {
            throw Exception("x y have different size");         
        }

        if (has_binary_labels) {
            ValidateBinaryLabels(y);
        }

        size_t nb_dim = x[0].size();
        for (size_t i = 0; i < x.size(); ++i) {
            ValidateDimensions(nb_dim, x[i].size(), i);
        }
    }

    double DotProduct(const std::vector<double> v1, const std::vector<double> v2) {
        double dot_product = 0;
        for (size_t i = 0; i < v1.size(); ++i) {
            dot_product += v1[i] * v2[i];
        }
        return dot_product;
    }

    Data ReadData(const std::string &data_path, bool load_label) {
        std::ifstream infile(data_path);  
        std::string image_path, line;
        int label = -1;
        std::vector<std::vector<double>> x; 
        std::vector<int> y;
        std::vector<std::string> image_paths;
        size_t counter = 0;

        while (std::getline(infile, line)) {
            std::istringstream iss(line);

            iss >> image_path;
            if (load_label) {
                iss >> label;
            }

            if (counter % REPORT_THRESHOLD == 0) {
                std::cout << "loaded images " << counter << std::endl;
            }

            cv::Mat mat = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
            std::vector<double> row;
            for (int i = 0; i < mat.rows; ++i) {
                for (int j = 0; j < mat.cols; ++j) {
                    row.push_back((double)mat.at<uchar>(i, j));
                }
            }

            x.push_back(row);
            y.push_back(label);
            image_paths.push_back(image_path);
            counter++;
        }

        std::cout << "upload all images from " << data_path << std::endl;
        std::cout << "number of images " << x.size() << std::endl;
        std::cout << "dimensionality " << x.at(0).size() << std::endl;
        std::cout << "number of labels "  << y.size() << std::endl;
        std::cout << std::endl; 

        return std::make_tuple(x, y, image_paths);
    }

    void SaveModel(const ml::MulticlassSVM &svm, const std::string &save_path) {
        std::cout << "saving model to " << save_path << std::endl;

        std::ofstream output(save_path);

        output << svm.GetModels().size() << " " << svm.GetModels()[0].size() << std::endl;
        for (auto &model : svm.GetModels()) {
            for (auto val : model) {
                output << val << " ";
            }
            output << std::endl;
        }

        output << svm.GetBiases().size() << std::endl;
        for (auto bias : svm.GetBiases()) {
            output << bias << std::endl;
        }

        output << svm.GetLabels().size() << std::endl;
        for (auto label : svm.GetLabels()) {
            output << label << std::endl;
        }
    }

    MulticlassSVM ReadModel(const std::string &model_path) {
        std::string line;
        size_t models_size, nb_dim, nb_biases, nb_labels;
        std::ifstream model_file(model_path);  

        if (!(model_file >> models_size >> nb_dim)) {
            throw std::length_error(
                "incorrect model file " + model_path +
                ", it has wrong number of models"
            );
        }

        std::cout << "number of binarySVM models " << models_size << std::endl;
        std::cout << "number of dimensions in model " << nb_dim << std::endl;

        std::vector<std::vector<double>> models(models_size);
        for (size_t i = 0; i < models_size; ++i) {
            std::vector<double> model(nb_dim);
            double model_value;
            for (size_t j = 0; j < nb_dim; ++j) {
                if (!(model_file >> model_value)) {
                    throw std::length_error(
                        "incorrect model file " + model_path +
                        ", model " + std::to_string(i) + " doesn't have enough values"
                    );
                }
                model[j] = model_value;
            }
            models[i] = model;
        }

        if(!(model_file >> nb_biases)) {
            throw std::length_error(
                "incorrect model file " + model_path +
                ", model file has incorrect number of biases"
            );
        }

        std::vector<double> biases(nb_biases);
        std::cout << "number of biases " << nb_biases << std::endl;
        for (size_t i = 0; i < nb_biases; ++i) {
            double bias_value;
            if (!(model_file >> bias_value)) {
                throw std::length_error(
                    "incorrect model file " + model_path +
                    ", it doesn't have enough biases values"
                );
            }
            biases[i] = bias_value;
        }

        if (!(model_file >> nb_labels)) {
            throw std::length_error(
                "incorrect model file " + model_path +
                ", it has wrong number of biases"
            );
        }

        std::cout << "model has " << nb_labels << " labels" << std::endl;

        std::vector<int> labels(nb_labels);
        for (size_t i = 0; i < nb_labels; ++i) {
            int label;
            if (!(model_file >> label)) {
                throw std::length_error(
                    "incorrect model file " + model_path +
                    ", it has wrong label"
                );
            }
            labels[i] = label;
        }

        std::cout << "model defined for labels:";
        for (auto label : labels) {
            std::cout << " " << label;
        }
        std::cout << std::endl;

        std::cout << "finish reading model file " << model_path << std::endl;

        MulticlassSVM svm(models, biases, labels);
        return svm;
    }

    void SavePredictions(const std::vector<std::string> &images,
                         const std::vector<int> &predictions, 
                         const std::string &output_path) {
        if (images.size() != predictions.size()) {
            throw std::invalid_argument(
                "can't save predictions as number of images doesn't match number of predictions"
            );
        }
        std::cout << "saving predictions to " << output_path << std::endl;

        std::ofstream output(output_path);

        for (size_t i = 0; i < predictions.size(); ++i) {
            output << images[i] << " " << predictions[i] << std::endl;
        }

    }


    void SaveNormalizationParams(const std::string &path, double mean, double std_dev) {
        std::ofstream output(path);
        output << mean << " " << std_dev << std::endl;
    }

    std::tuple<double, double> LoadNormalizationParams(const std::string &path) {
        double mean, std_dev;
        std::ifstream input(path);  

        if (!(input >> mean >> std_dev)) {
            throw std::length_error("incorrect normalization param file " + path);
        }

        return std::make_tuple(mean, std_dev);
    }

    Matrix Normalize(const Matrix &x, double mean, double std_dev) {
        Matrix result(x);

        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result.at(0).size(); ++j) {
                result[i][j] = (result[i][j] - mean) / std_dev;
            }
        }

        return result;
    } 

    std::tuple<Matrix, double, double> Normalize(const Matrix &x) {
        double sum = 0;
        double sum_of_squares = 0;
        double nb = 0;

        for (auto &row : x) {
            for (auto value : row) {
                sum += value;
                // can it overflow ?
                // 255 * 255 * 60 000
                sum_of_squares += value * value;
                ++nb;
            }
        }

        double mean = sum / nb;
        double variance = (sum_of_squares / nb) - mean * mean;
        double std_dev = std::sqrt(variance);

        Matrix result = Normalize(x, mean, std_dev);

        return std::make_tuple(result, mean, std_dev);
    } 

    void SavePCA(const std::string &path, cv::PCA &pca) {
	cv::FileStorage fs(path, cv::FileStorage::WRITE);  
	pca.write(fs);  
	fs.release();  
    }

    cv::PCA LoadPCA(const std::string &path) {
	// read file  
	cv::FileStorage fs2(path, cv::FileStorage::READ);  
	cv::PCA pca2;  
	pca2.read(fs2.root());  
	fs2.release();
	return pca2;
    }

    cv::Mat MatrixToCVMat(const Matrix &x) {
        cv::Mat result(x.size(), x.at(0).size(), CV_64FC1);

        for (int i = 0; i < result.rows; ++i) {
            for (int j = 0; j < result.cols; ++j) {
                result.at<double>(i, j) = x[i][j];
            }
        }
        return result;
    }

    Matrix CVMatToMatrix(const cv::Mat &mat) {
        Matrix x(mat.rows, std::vector<double>(mat.cols));

        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                x[i][j] = mat.at<double>(i, j);
            }
        }

        return x;
    }

    cv::PCA CreatePCA(const Matrix &x, double retain_variance) {
        cv::Mat mat = MatrixToCVMat(x);
        // opencv pcl will normalize input by default
        cv::PCA pca(mat, cv::Mat(), cv::PCA::DATA_AS_ROW,retain_variance); 
        return pca;
    }

    Matrix ProjectPCA(const cv::PCA &pca,  const Matrix &x) {
        cv::Mat mat = MatrixToCVMat(x);
        auto result = pca.project(mat);
        return CVMatToMatrix(result);
    }

    Matrix AddQuadraticInteractions(const Matrix &x) {
        Matrix result(x);
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x.at(0).size(); ++j) {
                for (size_t k = j + 1; k < x.at(0).size(); ++k) {
                    result[i].push_back(x[i][j] * x[i][k]);
                }
            }
        }

        return result;
    }

} // namespace ml

