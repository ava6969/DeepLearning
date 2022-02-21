//
// Created by dewe on 9/25/21.
//

#ifndef SAMFRAMEWORK_CONV_NET_H
#define SAMFRAMEWORK_CONV_NET_H

#include <utility>
#include "torch/torch.h"
#include "base.h"
#include "cassert"

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
            return m_BaseModel->forward(x.view({-1, in_shape.channel, in_shape.height, in_shape.width}));
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
