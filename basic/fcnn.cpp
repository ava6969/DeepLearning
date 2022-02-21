//
// Created by dewe on 9/17/21.
//

#include "common/sam_exceptions.h"
#include "common/common.h"
#include "fcnn.h"

namespace sam_dn{

    FCNNImpl::FCNNImpl(FCNNOption const& option):BaseModuleImpl(option){

        int actCtr = 0;
        m_BaseModel = {};
        if(option.dims.size() < 2){
            throw InvalidDimSize(static_cast<long>(option.dims.size()) , 2);
        }

        for(auto i = 0UL; i < option.dims.size() - 1; i++){
            torch::nn::LinearOptions opt(option.dims[i], option.dims[i+1]);

            opt.bias(option.new_bias);

            torch::nn::Linear sub_module(opt);

            initializeWeightBias(sub_module, option);

            m_BaseModel->push_back( sub_module );

            addActivationFunction(option.act_fn, m_BaseModel, i);
        }

        m_OutputSize.push_back(option.dims.back());

        register_module("fcnn", m_BaseModel);
    }

    FCNNImpl::FCNNImpl(int64_t input, int64_t output,
                       std::string const& weight_init_type,
                       float weight_init_param,
                       std::string const& bias_init_type,
                       float bias_init_param,
                       bool new_bias): BaseModuleImpl<FCNNOption>() {

            m_BaseModel = {};
            torch::nn::LinearOptions opt(input, output);

            opt.bias(new_bias);

            torch::nn::Linear sub_module(opt);
            FCNNOption option;
            option.weight_init_type = weight_init_type;
            option.weight_init_param = weight_init_param;
            option.bias_init_type = bias_init_type;
            option.bias_init_param = bias_init_param;
            initializeWeightBias(sub_module, option);

            m_BaseModel->push_back(sub_module);

        m_OutputSize.push_back(output);
        this->opt.dims.push_back(input);

        m_BaseModel = register_module("fcnn", m_BaseModel);
    }
}
