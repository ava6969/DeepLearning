//
// Created by dewe on 2/23/22.
//

#include "impala_residual_block.h"

namespace sam_dn{

    ImpalaResidualBlockImpl::ImpalaResidualBlockImpl(Option opt):ModuleWithSizeInfoImpl(opt){

            opt.conv_option.kernels = {3};
            opt.conv_option.padding = {"same"};
            opt.conv_option.strides = {1};
            assert(opt.conv_option.filters.size() == 1);
            REGISTER_MODULE(conv_layer, CNN(opt.conv_option));

            auto out_sz = conv_layer->outputSize();
            opt.conv_option.setInput( {static_cast<int>(out_sz[1]),
                        static_cast<int>(out_sz[2]),
                        static_cast<int>(out_sz[0])} );
            opt.conv_option.kernels = {3};
            opt.conv_option.padding = {"same"};
            opt.conv_option.strides = {2};
            REGISTER_MODULE(max_pool, MaxPool2D(opt.conv_option));

            out_sz = max_pool->outputSize();
            auto in_channel = out_sz[0];
            opt.res_option1.setInput({static_cast<int>(out_sz[1]),
                        static_cast<int>(out_sz[2]),
                        static_cast<int>(out_sz[0])} );

            opt.res_option2.setInput({static_cast<int>(out_sz[1]),
                        static_cast<int>(out_sz[2]),
                        static_cast<int>(out_sz[0])} );

            REGISTER_MODULE(residualBlock1, ResidualBlock(opt.res_option1));
            REGISTER_MODULE(residualBlock2, ResidualBlock(opt.res_option2));

            if(opt.batch_norm){
                REGISTER_MODULE(batch_norm, torch::nn::BatchNorm2d(in_channel));
            }

            if(opt.drop_out){
                REGISTER_MODULE(drop_out, torch::nn::Dropout(opt.drop_out.value()));
            }

            m_OutputSize = residualBlock2->outputSize();

    }

    torch::Tensor ImpalaResidualBlockImpl::forward ( torch::Tensor const& x) noexcept {
        auto out = conv_layer(x);
        if (drop_out)
            out = drop_out(out);

        if (batch_norm)
            out = batch_norm(out);

        return residualBlock2(residualBlock1(max_pool(out)));
    }

    ImpalaResnetImpl::ImpalaResnetImpl(Option opt): BaseModuleImpl<>(opt) {
        int i = 0;
        Conv2DInput inp = {static_cast<int>(opt.dict_opt[m_Input][1]),
                           static_cast<int>( opt.dict_opt[m_Input][2]),
                           static_cast<int>(opt.dict_opt[m_Input][0])};

        this->m_BaseModel = torch::nn::Sequential();
        ImpalaResidualBlockImpl::Option _opt;
        std::for_each(opt.filters.begin(), opt.filters.end() - 1, [&](auto sz) {
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
            this->m_BaseModel->push_back(block);
        });

        _opt.conv_option.filters = {opt.filters.back()};
        _opt.conv_option.setInput(inp);

        if (opt.relu_last)
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
}