#pragma once

#include <vector>


namespace ml {

/**
 * Multiclass SVM which utilises "one vs one" scheme
 */
class MulticlassSVM {
public: 
    MulticlassSVM() {}
    MulticlassSVM(const std::vector<std::vector<double>> &models, 
                  const std::vector<double> &biases,
                  const std::vector<int> &labels);

    const std::vector<std::vector<double>>& GetModels() const;
    const std::vector<double>& GetBiases() const;
    const std::vector<int>& GetLabels() const;

    void Train(const std::vector<std::vector<double>> &x, 
               const std::vector<int> &y,
               double lambda = 0.01,
               double bias_multiplier = 1,
               double epsilon = 0.02); 

    std::vector<int> Predict(const std::vector<std::vector<double>> &x);

private:
    std::vector<std::vector<double>> models_;
    std::vector<double> biases_;
    std::vector<int> labels_;
};

} // namespace ml

