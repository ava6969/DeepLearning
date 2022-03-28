//
// Created by dewe on 3/26/22.
//

#ifndef DEEP_NETWORKS_OPTIMIZE_H
#define DEEP_NETWORKS_OPTIMIZE_H


#include <utility>
#include "common/sam_exceptions.h"
#include "string"
#include "sstream"
#include "torch/torch.h"


namespace sam_dn{

    struct InvalidOptimizerName : SamException{
        const std::string optimizer_name;
        explicit InvalidOptimizerName(std::string  optimizer_name):optimizer_name(std::move(optimizer_name)){}

        std::string msg() override{
            ss << "InvalidOptimizerNameError: " << optimizer_name << " is an invalid optimizer.";
            return ss.str();
        }
    };

    class OptimizerOption{

        std::function<std::shared_ptr<torch::optim::Optimizer>(std::vector<torch::Tensor> const&)> compiler;

        template<class OptimizerT>
        auto make_default_optimizer_fn(){
            return [](std::vector<torch::Tensor> const& module_params) {
                return std::make_shared<OptimizerT>(module_params);
            };
        }

        template<class OptimizerT, class OptimizerOptimT>
        auto make_optimizer_fn(OptimizerOptimT opt){
            return [opt](std::vector<torch::Tensor> const& module_params) {
                return std::make_shared<OptimizerT>(module_params, std::move(opt));
            };
        }

    public:

        OptimizerOption()=default;
       ~OptimizerOption()=default;

        explicit OptimizerOption(std::string const& optimizer_name) {
            if( optimizer_name == "adam"){
                compiler = make_default_optimizer_fn<torch::optim::Adam>();
            }else if( optimizer_name == "rms_prop"){
                compiler = make_default_optimizer_fn<torch::optim::RMSprop>();
            }else{
                throw InvalidOptimizerName(optimizer_name);
            }
        }

        explicit OptimizerOption(const char* optimizer_name):OptimizerOption( std::string(optimizer_name) ){}

        OptimizerOption& operator=( std::string const& x){
            *this = OptimizerOption(x);
            return *this;
        }

        OptimizerOption& operator=( const char* x){
            *this = OptimizerOption(x);
            return *this;
        }

        explicit OptimizerOption(torch::optim::AdamOptions const& opt) {
            compiler = make_optimizer_fn<torch::optim::Adam>(opt);
        }

        explicit OptimizerOption(torch::optim::RMSpropOptions const& opt) {
            compiler = make_optimizer_fn<torch::optim::RMSprop>(opt);
        }

        explicit OptimizerOption(torch::optim::SGDOptions const& opt) {
            compiler = make_optimizer_fn<torch::optim::SGD>(opt);
        }

        template<class OptionT>
        OptimizerOption& operator=( OptionT const& opt ){
            *this = OptimizerOption(opt);
            return *this;
        }

        [[nodiscard]] inline auto get(std::vector<torch::Tensor> const& params) const{
            return compiler(params);
        }

    };
}

#endif //DEEP_NETWORKS_OPTIMIZE_H
