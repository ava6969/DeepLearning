//
// Created by dewe on 12/9/21.
//

#ifndef DEEP_NETWORKS_COMMON_H
#define DEEP_NETWORKS_COMMON_H
#include "torch/torch.h"

using TensorDict =  std::unordered_map<std::string, torch::Tensor>;

namespace sam_dn{
    void paramInit(std::string const &type, float gain, torch::Tensor &param);

    void addActivationFunction(std::string const &act_fn, torch::nn::Sequential &model, int idx = 0);

    template<class Module, class BaseModuleOption>
    inline static void initializeWeightBias(Module& sub_module, BaseModuleOption const& option){
        for(auto& weight: sub_module->named_parameters()){
            if( weight.key().find("bias") != std::string::npos )
                paramInit(option.bias_init_type, option.bias_init_param, weight.value());
            else if( weight.key().find("weight") != std::string::npos )
                paramInit(option.weight_init_type, option.weight_init_param, weight.value());
        }
    }

    template<class T>
    inline static auto split(std::vector<T> && data, float train, float valid){
        auto N = data.size();
        int train_end = int(N*train),
        valid_end = train_end + int(N*valid);

        return std::make_tuple(std::vector<T>(data.begin(), data.begin()+train_end),
                               std::vector<T>(data.begin()+train_end, data.begin()+valid_end),
                               std::vector<T>(data.begin()+valid_end, data.end()));
    }

}
#endif //DEEP_NETWORKS_COMMON_H
