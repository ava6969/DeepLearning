//
// Created by dewe on 3/27/22.
//

#ifndef DEEP_NETWORKS_ACCURACY_H
#define DEEP_NETWORKS_ACCURACY_H


#include "utils.h"
#include "mean_metric_wrapper.h"


namespace torch::nn{

    template<bool dict_output>
    class Accuracy : public MeanMetricWrapper<dict_output> {
    public:
            Accuracy(std::optional<std::string> const& name=std::nullopt,
                     std::optional<torch::ScalarType> const& dtype=std::nullopt):
            MeanMetricWrapper<dict_output>(torch::utils::Accuracy(), name.template value_or("accuracy"), dtype) {}
    };

    template<bool dict_output>
    class BinaryAccuracy : public MeanMetricWrapper<dict_output> {
    public:
        BinaryAccuracy(std::optional<std::string> const& name=std::nullopt,
                 std::optional<torch::ScalarType> const& dtype=std::nullopt,
                 float threshold=0.5):
                MeanMetricWrapper<dict_output>(torch::utils::BinaryAccuracy(threshold),
                                               name.template value_or("binary_accuracy"),
                                               dtype){}

    };

    template<bool dict_output>
    class TopKCategoricalAccuracy : public MeanMetricWrapper<dict_output> {
        WrapperFunction fn;

    public:
        TopKCategoricalAccuracy(std::optional<std::string> const& name=std::nullopt,
                       std::optional<torch::ScalarType> const& dtype=std::nullopt,
                       int k=5):
                MeanMetricWrapper<dict_output>(torch::utils::TopKCategoricalAccuracy(k),
                                               name.template value_or("binary_accuracy"),
                                               dtype){}

    };

}


#endif //DEEP_NETWORKS_ACCURACY_H
