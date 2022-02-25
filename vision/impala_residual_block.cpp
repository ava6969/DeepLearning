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

}