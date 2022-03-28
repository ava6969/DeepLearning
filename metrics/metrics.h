//
// Created by dewe on 3/27/22.
//

#pragma once

#include <utility>

#include "optional"
#include "variant"
#include "torch/torch.h"


namespace torch::nn{

    template<bool dict_output>
    class Metric : public Module{
        std::optional< std::string > m_name;
        torch::ScalarType m_dtype;
        std::vector<torch::Tensor> weights;
        bool stateful{true}, built{true};

    public:
        explicit Metric(std::optional<std::string> name = std::nullopt,
               std::optional<torch::ScalarType> dtype= std::nullopt):m_name(std::move(name)),
               m_dtype(dtype.template value_or(torch::kFloat32)){}

        inline auto name() const { return m_name; }
        inline auto dtype() const { return m_dtype; }

        friend std::ostream& operator<<(std::ostream&  os, Metric<dict_output> const& x){
            os << x.m_name.value_or("None") << "=" << x.m_dtype << "\n";
            return os;
        }

        auto toString(){
            std::stringstream ss;
            ss << *this;
            return ss.str();
        }

        virtual torch::Tensor result(){

        }

        void resetState(){

        }

        virtual void updateState( torch::Tensor const& y_true,
                           torch::Tensor const& y_pred,
                           std::optional<torch::Tensor> const& sample_weight) = 0;

        std::vector< Metric<dict_output> >  mergeState( std::vector<Metric<dict_output>> const& metrics){

            std::vector< Metric<dict_output> > assign_add_ops;
            std::transform(metrics.begin(), metrics.end(), [this](auto const& metric){
                if(metric.weights.size() != this->weights.size()){
                    std::stringstream ss;
                    ss << metric << " is not compatible with " << *this;
                    throw std::invalid_argument(ss.str());
                }else{
                    return this->weights + metric.weights;
                }
            });
            return assign_add_ops;
        }

        auto addWeight(std::string const& name,
                       std::optional<std::string> const& initializer=std::nullopt,
                       std::optional<torch::IntArrayRef> const& shape=std::nullopt,
                       std::optional<torch::ScalarType> const& _dtype=std::nullopt){
            if( initializer ){
                if( initializer == "zeros")
                    weights.push_back(
                            register_parameter(name,
                                    torch::zeros(shape.template value_or(1), _dtype.template value_or(kFloat32)),
                                    false));
            }
            else{
                weights.push_back(
                        register_parameter(name,
                                           torch::empty(shape.template value_or(1), _dtype.template value_or(kFloat32)),
                                           false));
            }
            return weights.back();
        }
    };
}
