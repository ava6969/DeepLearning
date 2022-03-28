//
// Created by dewe on 3/27/22.
//

#ifndef DEEP_NETWORKS_MEAN_METRIC_WRAPPER_H
#define DEEP_NETWORKS_MEAN_METRIC_WRAPPER_H


#include "reduce.h"

namespace torch::nn{


    using WrapperFunction = std::function<torch::Tensor(torch::Tensor const& ,
                                                        torch::Tensor const&)>;

    template<bool dict_output>
    class MeanMetricWrapper : public Mean<dict_output> {
        WrapperFunction fn;


    public:
        MeanMetricWrapper(WrapperFunction const& fn,
                          std::optional<std::string> const& name=std::nullopt,
                          std::optional<torch::ScalarType> const& dtype=std::nullopt):
                          Mean<dict_output>(name, dtype), fn(fn){}

        torch::Tensor updateState(torch::Tensor const& y_true,
                                  torch::Tensor const& y_pred,
                                  std::optional<torch::Tensor> const& sample_weight) override{
            return {};
        }

    };

}

#endif //DEEP_NETWORKS_MEAN_METRIC_WRAPPER_H
