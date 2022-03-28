//
// Created by dewe on 3/27/22.
//

#ifndef DEEP_NETWORKS_REDUCE_H
#define DEEP_NETWORKS_REDUCE_H

#include "metrics.h"

namespace torch::nn {

    template<bool dict_output>
    class Reduce : public Metric<dict_output> {

    private:
        torch::Reduction::Reduction reduction;
        torch::Tensor total, count;

    public:
        Reduce(torch::Reduction::Reduction reduction,
               std::string const& name,
               std::optional<torch::ScalarType> const& _dtype=std::nullopt): Metric<dict_output>(name, _dtype){
            total = this->addWeight("total", "zeros");
            if( reduction == Reduction::END or reduction == Reduction::Mean){
                count = this->addWeight("count", "zeros");
            }
        }

        /*Accumulates statistics for computing the metric.
        Args:
                values: Per-example value.
        sample_weight: Optional weighting of each example. Defaults to 1.
        Returns:
                Update op.
        */
        torch::Tensor updateState(torch::Tensor const& y_true,
                                  torch::Tensor const& y_pred,
                                  std::optional<torch::Tensor> const& sample_weight) override{
            return {};
        }

        torch::Tensor result() override{
            switch( reduction ){
                case Reduction::Sum:
                    return torch::eye( this->total.template item<int>() );
                case Reduction::END:
                case Reduction::Mean:
                    return total / count;
                default:
                    std::stringstream ss;
                    ss << "Reduction " << reduction << " is not implemented. Expected "
                                                       "\"sum\", \"weighted_mean\", or \"sum_over_batch_size\".";
                    throw std::invalid_argument(ss.str());
            }
        }

    };

    template<bool dict_output>
    class Sum : public Reduce<dict_output> {

    public:
        Sum(std::optional<std::string> const& name=std::nullopt,
            std::optional<torch::ScalarType> const& dtype=std::nullopt):
                Reduce<dict_output>(Reduction::Sum,
                                    name.template value_or("sum"), dtype){}
    };

    template<bool dict_output>
    class Mean : public Reduce<dict_output> {

    public:
        Mean(std::optional<std::string> const& name=std::nullopt,
             std::optional<torch::ScalarType> const& dtype=std::nullopt):
             Reduce<dict_output>(Reduction::Mean,
                                 name.template value_or("mean"), dtype){}
    };

}

#endif //DEEP_NETWORKS_REDUCE_H
