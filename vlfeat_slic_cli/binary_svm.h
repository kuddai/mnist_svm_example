#pragma once

#include <vector>

namespace ml {

class BinarySVM {
public: 
    BinarySVM() {}
    BinarySVM(const std::vector<double> &model, double bias)
        :model_(model), bias_(bias) {}

    std::vector<double> GetModel() const;

    double GetBias() const;

    void Train(const std::vector<std::vector<double>> &x, 
               const std::vector<int> &y,
               double lambda); 

    std::vector<int> Predict(const std::vector<std::vector<double>> &x);

private:
    std::vector<double> model_;
    double bias_;
};

} // namespace ml

