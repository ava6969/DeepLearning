//
// Created by dewe on 12/9/21.
//
#include "common/common.h"
#include "residual_block.h"

namespace sam_dn{
    ResidualBlockImpl::ResidualBlockImpl(CNNOption _opt): ModuleWithSizeInfoImpl(_opt)
    {
        auto in_channels = _opt.InputShape().channel;

        torch::nn::Conv2dOptions opt(in_channels, _opt.filters[0], _opt.kernels[0]);
        opt.padding(1);
        opt.stride(1);
        conv1 = register_module("conv1", torch::nn::Conv2d(opt));
        conv2 = register_module("conv2", torch::nn::Conv2d(opt));

        if(_opt.flatten_output)
            m_OutputSize = {  _opt.filters[0] *  _opt.InputShape().height * _opt.InputShape().width  };
        else
            m_OutputSize = m_OutputSize = {_opt.filters[0], _opt.InputShape().height,  _opt.InputShape().width};

        flatten_out = _opt.flatten_output;
        initializeWeightBias(conv1, _opt);
        if(not _opt.activations.empty())
            relu_out = _opt.activations[0] == "relu";

    }
}