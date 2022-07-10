//
// Created by dewe on 9/25/21.
//

#ifndef SAMFRAMEWORK_CNN_VAE_H
#define SAMFRAMEWORK_CNN_VAE_H

#include "torch/torch.h"
#include "cassert"
#include "conv_net.h"

namespace sam_rl {

    struct VAEOutput{
        torch::Tensor z, mu, logvar, decoded;
    };

class CnnVaeImpl : public torch::nn::Module{

    private:
        CNN m_Encoder{nullptr};
        CNNTranspose m_Decoder{nullptr};
        torch::nn::Linear m_ZMean{nullptr}, m_ZLogVar{nullptr}, m_ZDense{nullptr};
        int m_ZDim{};
        int m_Width, m_Height, m_Channel;

    public:
        CnnVaeImpl(Conv2DInput inputShape,
                   int dense_size,
                   int z_dim,
                   CNNOption const& encoderOpt,
                   CNNOption const& decoderOpt);

        VAEOutput forward(torch::Tensor const& encoder_in);

        inline torch::Tensor decode(torch::Tensor const& z) {
            auto batch_sz = z.size(0);
            auto decoder_in =  m_ZDense(z).view({batch_sz, -1, 1, 1});
            return m_Decoder->forward(decoder_in);
        }

        void pretty_print(std::ostream& stream) const override {
            stream  << "sam_rl::CnnVae"
                    << "("
                    << "ZDim=" << this->m_ZDim << ", "
                    << "Height=" << this->m_Height << ", "
                    << "Width=" << this->m_Width << ", "
                    << "Channel=" << this->m_Channel << ", "
                    << ")";
        }
    };

    TORCH_MODULE(CnnVae);
}
#endif //SAMFRAMEWORK_CNN_VAE_H
