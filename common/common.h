//
// Created by dewe on 12/9/21.
//

#ifndef DEEP_NETWORKS_COMMON_H
#define DEEP_NETWORKS_COMMON_H

#include <vision/conv_net.h>
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

    template<class VisionOption>
    std::pair< struct Conv2DInput, torch::ExpandingArray<2>> same_pad(
            int out_filter,
            VisionOption  _opt,
            Conv2DInput _in_shape){
        auto w = _opt.kernel_size()->at(0);
        auto h = _opt.kernel_size()->at(1);

        auto stride_w = _opt.stride()->at(0);
        auto stride_h = _opt.stride()->at(1);

        int top = ceil(h / 2);
        int bottom = floor(h / 2);
        int left = ceil(w / 2);
        int right = floor(w / 2);

        if(left != right and top != bottom){
            throw std::runtime_error("Same Paadding for Conv2D requires equal padding left and right as well as "
                                     "top and bottom");
        }

        return {Conv2DInput{int(((_in_shape.width + right + left - w) / stride_w) + 1),
                            int(((_in_shape.height + top + bottom - h) / stride_h) + 1), out_filter},
                {left, bottom}};
    }

}
#endif //DEEP_NETWORKS_COMMON_H
