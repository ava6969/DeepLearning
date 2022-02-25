//
// Created by dewe on 2/23/22.
//

#pragma once

#include "residual_block.h"
#include "conv_net.h"
#include "optional"
#include "wrappers/basic.h"
#include "common/yaml_options.h"

namespace sam_dn{

    class ImpalaResidualBlockImpl : public ModuleWithSizeInfoImpl{

        ResidualBlock residualBlock1{nullptr};
        ResidualBlock residualBlock2{nullptr};
        torch::nn::BatchNorm2d batch_norm {nullptr};
        torch::nn::Dropout drop_out{nullptr};
        CNN conv_layer{nullptr};
        MaxPool2D max_pool{nullptr};

    public:

        struct Option : BaseModuleOption{
            CNNOption res_option1, res_option2, conv_option;
            bool batch_norm{false};
            std::optional<float> drop_out{};
        };

        explicit ImpalaResidualBlockImpl(Option opt);

        inline torch::Tensor forward ( torch::Tensor const& x) noexcept override{
            auto out = conv_layer(x);
            if(drop_out)
                out = drop_out(out);

            if(batch_norm)
                out = batch_norm(out);

            return residualBlock2(residualBlock1(max_pool(out)));
        }

        inline TensorDict * forwardDict(TensorDict *x) noexcept override{
            x->insert_or_assign( m_Output, forward(x->at(m_Input)));
            return x;
        }
    };
    TORCH_MODULE(ImpalaResidualBlock);


    class ImpalaResnetImpl : public BaseModuleImpl<>{

    public:
        struct Option : BaseModuleOption{
            std::vector<int> filters{};
            bool relu_last{false}, batch_norm{false};
            std::optional<float> drop_out{std::nullopt};
            bool flatten_out{false};
        };

        explicit ImpalaResnetImpl(Option opt): BaseModuleImpl<>(opt){
            int i = 0;
            Conv2DInput inp = {static_cast<int>(opt.dict_opt[m_Input][1]),
                               static_cast<int>( opt.dict_opt[m_Input][2]),
                               static_cast<int>(opt.dict_opt[m_Input][0])};

            this->m_BaseModel = torch::nn::Sequential();
            ImpalaResidualBlockImpl::Option _opt;
            std::for_each(opt.filters.begin(), opt.filters.end()-1, [&](auto sz){
                _opt.conv_option.filters = {sz};
                _opt.conv_option.activations = {"none"};
                _opt.conv_option.setInput(inp);
                _opt.res_option1.filters = {sz};
                _opt.res_option1.kernels = {3};
                _opt.res_option2.filters = {sz};
                _opt.res_option2.kernels = {3};
                _opt.drop_out = opt.drop_out;
                _opt.batch_norm = opt.batch_norm;
                auto block = ImpalaResidualBlock(_opt);
                auto out_sz = block->outputSize();
                inp = {static_cast<int>(out_sz[1]),
                       static_cast<int>(out_sz[2]),
                       static_cast<int>(out_sz[0])};
                this->m_BaseModel->push_back( block );
            });

            _opt.conv_option.filters = {opt.filters.back()};
            _opt.conv_option.setInput(inp);

            if(opt.relu_last)
                _opt.res_option2.activations = {"relu"};
            _opt.res_option2.flatten_output = opt.flatten_out;

            auto block = ImpalaResidualBlock(_opt);
            auto out_sz = block->outputSize();
            inp = {static_cast<int>(out_sz[1]),
                   static_cast<int>(out_sz[2]),
                   static_cast<int>(out_sz[0])};

            m_OutputSize = block->outputSize();
            this->m_BaseModel->push_back(block);
            register_module("impala_resnet", m_BaseModel);
        }
    };
}

namespace YAML{
    CONVERT_WITH_PARENT(BaseModuleOption, ImpalaResidualBlockImpl::Option,
                        SELF(res_option1), SELF(res_option2), SELF(conv_option),
                        SELF(batch_norm), SELF(drop_out))

    CONVERT_WITH_PARENT(BaseModuleOption, ImpalaResnetImpl::Option,
                        SELF(filters), SELF(relu_last), SELF(flatten_out),
                        SELF(batch_norm), SELF(drop_out))
}

