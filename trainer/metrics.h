//
// Created by dewe on 3/27/22.
//

#pragma once

#include "metrics/accuracy.h"

namespace sam_dn {

    struct InvalidMetricName : SamException{
        const std::string _name;
        explicit InvalidMetricName(std::string  _name):_name(std::move(_name)){}

        std::string msg() override{
            ss << "InvalidMetricName: " << _name << " is an invalid metric.";
            return ss.str();
        }
    };

    template<bool dict_output>
    class Metrics {

        std::function< torch::intrusive_ptr< torch::nn::Metric<dict_output> >() > compiler;

        template<class MetricT>
        static auto make_default_metric_fn(){
            return []() {
                return torch::make_intrusive< MetricT >();
            };
        }

        template<class MetricT, class ... Args>
        static auto make_metric_fn(Args ... opt){
            return [=]() {
                return torch::make_intrusive< MetricT >(opt ...);
            };
        }

    public:
        Metrics()=default;
        ~Metrics()=default;

        explicit Metrics(std::string const& _metric){
            if( _metric == "accuracy"){
                compiler = make_default_metric_fn<torch::nn::Accuracy<dict_output>>();
            }else if( _metric == "top_k_categorical_accuracy"){
                compiler = make_default_metric_fn<torch::nn::TopKCategoricalAccuracy<dict_output>>();
            }else{
                throw InvalidMetricName(_metric);
            }
        }

        explicit Metrics(const char* _metric):Metrics( std::string(_metric) ) {
        }

        Metrics& operator=( std::string const& x){
            *this = Metrics(x);
            return *this;
        }

        [[nodiscard]] inline auto get( std::optional<std::vector<float>> const& weighted_metrics) const{
            return compiler();
        }

//        static torch::intrusive_ptr<Metrics<dict_output>> accuracy() {
//            torch::intrusive_ptr<Metrics<dict_output>> _this = torch::make_intrusive<Metrics<dict_output>>();
//            _this->compiler = make_metric_fn<torch::nn::Accuracy<dict_output>>();
//            return _this;
//        }
//
//        static torch::intrusive_ptr<Metrics<dict_output>> top_k_categorical_accuracy(float k) {
//            torch::intrusive_ptr<Metrics<dict_output>> _this = torch::make_intrusive<Metrics<dict_output>>();
//            _this->compiler = make_metric_fn<torch::nn::TopKCategoricalAccuracy<dict_output>>();
//            return _this;
//        }
    };
}

