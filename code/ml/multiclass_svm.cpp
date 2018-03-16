#include <vector>
#include <algorithm>
#include <unordered_map>
#include <iostream>

#include "multiclass_svm.h"
#include "exception.h"
#include "binary_svm.h"
#include "util.h"


namespace ml {
    typedef std::vector<std::vector<double>> matrix;

    std::vector<int> GetUniqueLabels(const std::vector<int> &y) {
        std::vector<int> labels(y);
        std::sort(labels.begin(), labels.end());
        auto it = std::unique(labels.begin(), labels.end());
        labels.resize(std::distance(labels.begin(), it));
        return labels;
    }

    MulticlassSVM:: MulticlassSVM(const matrix &models, 
                                  const std::vector<double> &biases,
                                  const std::vector<int> &labels)
    :models_(models), biases_(biases), labels_(labels) {
        std::vector<int> clone(labels);
        std::sort(clone.begin(), clone.end());
        auto it = std::unique(clone.begin(), clone.end());
        if (it != clone.end()) {
            throw Exception("labels contain duplicates");
        }
    }

    const matrix& MulticlassSVM::GetModels() const {
        return models_;
    }

    const std::vector<double>& MulticlassSVM::GetBiases() const {
        return biases_;
    };

    const std::vector<int>& MulticlassSVM::GetLabels() const {
        return labels_;
    };

    void MulticlassSVM::Train(const matrix &x, 
                              const std::vector<int> &y,
                              double lambda,
                              double bias_multiplier,
                              double epsilon) {
        ValidateTrainData(x, y, false);

        // clear data
        models_.clear();
        biases_.clear();
        labels_ = GetUniqueLabels(y);
        std::cout << "number of unqiue labels " << labels_.size() << std::endl;

        for (size_t i = 0; i < labels_.size(); ++i) {
            for (size_t j = i + 1; j < labels_.size(); ++j) {
                // info
                std::cout << "start svm training one vs one for labels "; 
                std::cout << labels_[i] << " " << labels_[j] << std::endl;

                matrix sub_x;
                std::vector<int> sub_y;
                for (size_t k = 0; k < x.size(); ++k) {
                    if (y[k] == labels_[i]) {
                        sub_x.push_back(x[k]);
                        sub_y.push_back(-1);
                    }

                    if (y[k] == labels_[j]) {
                        sub_x.push_back(x[k]);
                        sub_y.push_back(1);
                    }
                }

                BinarySVM svm;
                svm.Train(sub_x, sub_y, lambda, bias_multiplier, epsilon);
                models_.push_back(svm.GetModel());
                biases_.push_back(svm.GetBias());
            }
        }
    }

    std::vector<int> MulticlassSVM::Predict(const matrix &x) {
        if (x.empty()) {
            return {};
        }

        if (models_.empty()) {
            throw Exception("there are no models");
        }

        // using majority voting 
        size_t idx = 0;
        std::vector<std::vector<int>> raw_predictions;
        for (size_t i = 0; i < labels_.size(); ++i) {
            for (size_t j = i + 1; j < labels_.size(); ++j) {
                BinarySVM svm(models_[idx], biases_[idx]);
                std::vector<int> sub_predictions = svm.Predict(x);
                std::vector<int> raw_prediction;
                for (auto prediction : sub_predictions) {
                    raw_prediction.push_back(prediction == -1 ? labels_[i] : labels_[j]);
                }
                raw_predictions.push_back(raw_prediction);
                ++idx;
            }
        }

        std::vector<int> predictions;
        for (size_t i = 0; i < x.size(); ++i) {
            std::unordered_map<int, int> dict;
            int commonest;
            int maxcount = 0;
            for (auto &raw_prediction : raw_predictions) {
                if (++dict[raw_prediction[i]] > maxcount) {
                    commonest = raw_prediction[i];
                    maxcount = dict[raw_prediction[i]];
                }
            }

            predictions.push_back(commonest);
        }

        return predictions;
    }

} // namespace ml

