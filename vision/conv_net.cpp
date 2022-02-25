//
// Created by dewe on 10/21/21.
//

#include "conv_net.h"
#include "common/common.h"
#include "basic/fcnn.h"

namespace sam_dn{


    template<>
    int  ConvNetImpl<torch::nn::Conv2d, torch::nn::Conv2dOptions>::outputShape(int side,
                                                                               int padding,
                                                                               int dilation,
                                                                               int kernel,
                                                                               int stride) {
        return ( (side + 2*padding - dilation * (kernel - 1) - 1)/ stride) + 1;
    }


    template<>
    int  ConvNetImpl<torch::nn::ConvTranspose2d, torch::nn::ConvTranspose2dOptions>::outputShape(int side,
                                                                                                 int padding,
                                                                                                 int dilation,
                                                                                                 int kernel,
                                                                                                 int stride) {
        return (side-1)*stride - 2*padding + dilation * (kernel - 1) + 1;
    }

    template<typename Net, typename Option>
    Conv2DInput
    ConvNetImpl<Net, Option>::outputShape(Conv2DInput prev, int padding, int dilation, int kernel, int stride) {
        auto[w, h, _] = prev;
        prev.height = outputShape(h, padding, dilation, kernel, stride);
        prev.width = outputShape(w, padding, dilation, kernel, stride);
        return prev;
    }

    template<typename Net, typename Option>
    ConvNetImpl<Net, Option>::ConvNetImpl(CNNOption const& opt):BaseModuleImpl(opt), in_shape(opt.InputShape()){
        m_BaseModel = {};
        build(opt);
    }

    template<typename Net, typename Option>
    void ConvNetImpl<Net, Option>::build(CNNOption opt) {

        if( (opt.filters.size() != opt.kernels.size()) &&
            (opt.kernels.size() != opt.strides.size()) &&
            (opt.strides.size() != opt.activations.size())){
            // throw
        }

        opt.filters.insert(opt.filters.begin(), opt.InputShape().channel);

        for(int i = 0; i < opt.kernels.size(); i++){
            Option _opt(opt.filters[i], opt.filters[i+1], opt.kernels[i]);
            _opt.stride(opt.strides[i]);

            auto nextShape = outputShape(opt.InputShape(), 0, _opt.dilation()->at(0), opt.kernels[i], opt.strides[i]);

            if(opt.padding[i] == "same") {
                auto w = _opt.kernel_size()->at(0);
                auto h = _opt.kernel_size()->at(1);

                auto stride_w = _opt.stride()->at(0);
                auto stride_h = _opt.stride()->at(1);

                auto _in_shape = opt.InputShape();
                int top = ceil(h / 2);
                int bottom = floor(h / 2);
                int left = ceil(w / 2);
                int right = floor(w / 2);

                torch::nn::ZeroPad2dOptions pad_opt({left, right, top, bottom});
                m_BaseModel->push_back("padding" + std::to_string(i), torch::nn::ZeroPad2d(pad_opt) );

                nextShape = Conv2DInput{int(((_in_shape.width + right + left - w) / stride_w) + 1),
                                        int(((_in_shape.height + top + bottom - h) / stride_h) + 1),
                                        opt.filters[i+1]};
            }

            Net net(_opt);

            initializeWeightBias(net, opt);

            m_BaseModel->push_back("conv" + std::to_string(i), net);

            addActivationFunction(opt.activations[i], m_BaseModel, i);

            opt.setInput(nextShape);
        }

        m_OutputSize = {opt.filters.back(), opt.InputShape().height, opt.InputShape().width};

        if(opt.flatten_output){
            m_BaseModel->push_back("flatten", torch::nn::Flatten());
            m_OutputSize = {  std::accumulate(begin(m_OutputSize), end(m_OutputSize), 1,
                                              [](int64_t x, int64_t other){ return x*other; } )   };
        }

    }

    CNNImpl::CNNImpl(CNNOption const& opt) : ConvNetImpl( opt ) {
        register_module("conv_net", m_BaseModel);
    }

    CNNTransposeImpl::CNNTransposeImpl(CNNOption const& opt):ConvNetImpl( opt ){
            register_module("conv_net_transpose", m_BaseModel);
    }


}