//
// Created by dewe on 9/25/21.
//

#ifndef SAMFRAMEWORK_CONV_NET_H
#define SAMFRAMEWORK_CONV_NET_H

#include <utility>
#include "torch/torch.h"
#include "base.h"
#include "cassert"

#ifdef DEBUG_VISION
#include "vision_debugger.h"
#endif

namespace sam_dn {

    template<typename Net, typename Option>
    class ConvNetImpl : public BaseModuleImpl<CNNOption> {

    protected:
        Conv2DInput in_shape;

        void build(CNNOption opt);

        int outputShape(int side, int padding, int dilation, int kernel, int stride);

        Conv2DInput outputShape(Conv2DInput prev,
                                       int padding,
                                       int dilation, int kernel, int stride);

    public:
        ConvNetImpl()=default;

        explicit ConvNetImpl(CNNOption const& opt);

        inline torch::Tensor forward(const torch::Tensor &x) noexcept override {

#ifndef DEBUG_VISION
           return m_BaseModel->forward(x.view({-1, in_shape.channel, in_shape.height, in_shape.width}));
#else
            auto img = x.view({-1, in_shape.channel, in_shape.height, in_shape.width});
            auto result = m_BaseModel->forward(img);

            torch::NoGradGuard noGradGuard;
            auto _training = this->is_training();

            if(_training)
                this->eval();

            for(auto const& net_pair : m_BaseModel->named_children()){

                if(auto* _net = net_pair.value()->template as<torch::nn::Conv2d>()){
                    img = _net->forward(img);
                    VISION_DEBUGGER.addImages(this->m_Input + "_" + net_pair.key(),
                                              result.flatten(0, 1).unsqueeze(1));
                }else if(auto* p_net = net_pair.value()->template as<torch::nn::ZeroPad2d>()){
                    img = p_net->forward(img);
                    VISION_DEBUGGER.addImages(this->m_Input + "_" + net_pair.key(),
                                              result.flatten(0, 1).unsqueeze(1));
                }
            }
            this->train(_training);
            return result;
#endif
        }
    };

    class CNNImpl : public ConvNetImpl<torch::nn::Conv2d, torch::nn::Conv2dOptions> {

    public:
        CNNImpl()=default;

        explicit CNNImpl(CNNOption const& opt);

    };

    struct CNNTransposeImpl : public ConvNetImpl<torch::nn::ConvTranspose2d, torch::nn::ConvTranspose2dOptions>  {
        explicit CNNTransposeImpl(CNNOption  const& opt);
    };

    TORCH_MODULE(CNN);
    TORCH_MODULE(CNNTranspose);

}

#endif //SAMFRAMEWORK_CONV_NET_H
