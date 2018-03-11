#include <vector>
#include <sstream>
#include <string>
#include <memory>

#include "vl/svm.h"

#include "binary_svm.h"
#include "exception.h"


namespace ml {
    void ValidateLabels(const std::vector<int> &y) {
        for (size_t i = 0; i < y.size(); ++i) {
            if (y[i] != 1 && y[i] != -1) {
                std::stringstream ss;
                ss << "wrong label in " << i;
                ss <<  ": must be 1 or -1, got " << y[i];
                throw Exception(ss.str());
            }
        }
    }

    void ValidateDimensions(size_t expected, size_t got, size_t idx = 0) {
        if (expected != got) {
            std::stringstream ss;
            ss << "dimension mismatch in input " << idx;
            ss <<  ": expected"<< expected << ", got " << got;
            throw Exception(ss.str());
        }
    }

    void ValidateTrainData(const std::vector<std::vector<double>> &x, 
                           const std::vector<int> &y) {
        if (x.empty()) {
            throw Exception("x is empty");
        }

        if (x.size() != y.size()) {
            throw Exception("x y have different size");         
        }

        ValidateLabels(y);

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


    std::vector<double> BinarySVM::GetModel() const {
        return model_;
    }

    double BinarySVM::GetBias() const {
        return bias_;
    }

    void BinarySVM::Train(const std::vector<std::vector<double>> &x, 
                          const std::vector<int> &y,
                          double lambda) {
        ValidateTrainData(x, y);

        // reset model
        model_.clear();
        bias_ = 0;

        const vl_size nb_data = x.size();
        const vl_size nb_dim = x[0].size();

        double raw_x[nb_data * nb_dim];
        double raw_y[nb_data];

        for (vl_size i = 0; i < nb_data; ++i) {
            for (vl_size j = 0; j < nb_dim; ++j) {
                raw_x[i * nb_dim + j] = x[i][j];
            }
            raw_y[i] = y[i];
        }

        auto deleter = [&](VlSvm* ptr) {
            vl_svm_delete(ptr);
        };

        std::unique_ptr<VlSvm, decltype(deleter)> svm(
            vl_svm_new(VlSvmSolverSgd,
                       raw_x, nb_dim, nb_data,
                       raw_y,
                       lambda),
            deleter
        );

        vl_svm_train(svm.get());


        bias_ = vl_svm_get_bias(svm.get());
        const double * raw_model = vl_svm_get_model(svm.get());
        for (vl_size i = 0; i < nb_dim; ++i) {
            model_.push_back(raw_model[i]);
        }
    }

    std::vector<int> BinarySVM::Predict(const std::vector<std::vector<double>> &x) {
        if (x.empty()) {
            return {};
        }

        size_t nb_dim = x[0].size();
        ValidateDimensions(nb_dim, model_.size());

        std::vector<int> predictions;
        for (size_t i = 0; i < x.size(); ++i) {
            auto &input = x[i];
            ValidateDimensions(nb_dim, input.size());
            double dot_product = DotProduct(input, model_);
            int prediction = (dot_product + bias_ > 0) ? 1 : -1;
            predictions.push_back(prediction);
        }

        return predictions;
    }
} // namespace ml

