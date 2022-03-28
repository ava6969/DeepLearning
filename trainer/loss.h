//
// Created by dewe on 3/26/22.
//

#ifndef DEEP_NETWORKS_LOSS_H
#define DEEP_NETWORKS_LOSS_H

#include "torch/torch.h"
#include "common/sam_exceptions.h"
#include "string"

namespace sam_dn {

    struct InvalidLossName : SamException {
        const std::string name;

        explicit InvalidLossName(std::string loss_name) : name(std::move(loss_name)) {}

        std::string msg() override {
            ss << "InvalidLossNameError: " << name << " is an invalid los function.";
            return ss.str();
        }
    };

    template<class Opt>
    struct LossWrapper{
        Opt opt;
        explicit LossWrapper(Opt opt):opt(opt){}
        LossWrapper()=default;

        torch::Tensor operator()(torch::Tensor const& x, torch::Tensor const& y){
            return opt->forward(x, y);
        }
    };

    class LossOption{

        using LossFunction = std::function<torch::Tensor(torch::Tensor const&, torch::Tensor const&)>;
        std::function<LossFunction()> compiler;

        template<class LossT>
        inline std::function<LossFunction()> make_default_loss_fn(){
            return []() { return LossWrapper<LossT>(); };
        }

        template<class LossT, class LossOptionT>
        inline std::function<LossFunction()>  make_loss_fn(LossOptionT opt){
            return [opt]() { return LossWrapper(LossT(opt)); };
        }

    public:
        LossOption()=default;
        ~LossOption()=default;

        explicit LossOption(std::string const& optimizer_name) {
            if( optimizer_name == "bce"){
                compiler = make_default_loss_fn<torch::nn::BCELoss>();
            }else if( optimizer_name == "mse"){
                compiler = make_default_loss_fn<torch::nn::MSELoss>();
            }else{
                throw InvalidLossName(optimizer_name);
            }
        }

        explicit LossOption(const char* loss_name):LossOption( std::string(loss_name) ) {
        }

        LossOption& operator=( std::string const& x){
            *this = LossOption(x);
            return *this;
        }

        explicit LossOption(torch::nn::BCELossOptions const& opt):compiler(make_loss_fn<torch::nn::BCELoss>(opt)){
        }

        explicit LossOption(torch::nn::MSELossOptions const& opt):compiler(make_loss_fn<torch::nn::MSELoss>(opt)){
        }

        template<class OptionT>
        LossOption& operator=( OptionT const& opt ){
            *this = LossOption(opt);
            return *this;
        }

    };

}

#endif //DEEP_NETWORKS_LOSS_H
