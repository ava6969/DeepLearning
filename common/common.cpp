//
// Created by dewe on 12/9/21.
//

#include "common.h"

namespace sam_dn{

    void paramInit(std::string const& type, float gain, torch::Tensor& param){
        if(type == "orthogonal")
            torch::nn::init::orthogonal_(param, gain);
        else if(type == "xavier_uniform")
            torch::nn::init::xavier_uniform_(param, gain);
        else if(type == "xavier_normal")
            torch::nn::init::xavier_normal_(param, gain);
        else if(type == "constant")
            torch::nn::init::constant_(param, gain);
    }

    void addActivationFunction(std::string const& act_fn, torch::nn::Sequential& model, int idx){
        if(act_fn != "none") {
            auto act = act_fn == "tanh" ? torch::nn::Functional(torch::nn::Tanh()) :
                       act_fn == "relu" ? torch::nn::Functional(torch::nn::ReLU()) :
                       act_fn == "leaky_relu" ? torch::nn::Functional(torch::nn::LeakyReLU()) :
                       torch::nn::Functional(torch::nn::Sigmoid());

            model->push_back(act_fn + "_" + std::to_string(idx), act);
        }
    }

}