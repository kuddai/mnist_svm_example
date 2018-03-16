#pragma once

#include <string>
#include <vector>

namespace ml {

class BinarySVM {
public: 
    BinarySVM() {}
    BinarySVM(const std::vector<double> &model, double bias)
        :model_(model), bias_(bias) {}

    const std::vector<double>& GetModel() const;

    double GetBias() const;

    void Train(const std::vector<std::vector<double>> &x, 
               const std::vector<int> &y,
               double lambda = 0.01,
               // http://www.vlfeat.org/api/svm-fundamentals.html
               double bias_multiplier = 1,
               double epsilon = 0.02); 

    std::vector<int> Predict(const std::vector<std::vector<double>> &x);

private:
    std::vector<double> model_;
    double bias_;
};

} // namespace ml

