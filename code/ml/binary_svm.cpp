#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "vl/svm.h"

#include "binary_svm.h"
#include "exception.h"
#include "util.h"


namespace ml {
    typedef std::vector<std::vector<double>> matrix;

    const std::vector<double>& BinarySVM::GetModel() const {
        return model_;
    }

    double BinarySVM::GetBias() const {
        return bias_;
    }

    void BinarySVM::Train(const matrix &x, 
                          const std::vector<int> &y,
                          double lambda,
                          double bias_multiplier, 
                          double epsilon) {
        ValidateTrainData(x, y);

        // reset model
        model_.clear();
        bias_ = 0;


        const vl_size nb_data = x.size();
        const vl_size nb_dim = x[0].size();

        std::vector<double> raw_x(nb_data * nb_dim);
        std::vector<double> raw_y(nb_data);

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
                       raw_x.data(), nb_dim, nb_data,
                       raw_y.data(),
                       lambda),
            deleter
        );

        vl_svm_set_bias_multiplier(svm.get(), bias_multiplier);
        vl_svm_set_epsilon(svm.get(), epsilon);
        vl_svm_train(svm.get());

        std::cout << "svm is learnt in  " << vl_svm_get_statistics(svm.get())->iteration;
        std::cout << " iterations" << std::endl;


        bias_ = vl_svm_get_bias(svm.get());
        const double * raw_model = vl_svm_get_model(svm.get());
        for (vl_size i = 0; i < nb_dim; ++i) {
            model_.push_back(raw_model[i]);
        }
    }

    std::vector<int> BinarySVM::Predict(const matrix &x) {
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

