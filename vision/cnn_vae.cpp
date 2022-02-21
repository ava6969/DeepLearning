//
// Created by dewe on 9/25/21.
//

#include "common/misc.h"
#include "cnn_vae.h"
#include "torch/torch.h"
#include "conv_net.h"


namespace sam_rl{

    CnnVaeImpl::CnnVaeImpl(Conv2DInput inputShape,
                           int dense_size,
                           int z_dim,
                           CnnOptions const& encoderOpt,
                           CnnOptions const& decoderOpt):
                           m_Encoder(register_module("encoder", CNN(inputShape, encoderOpt))),
                           m_Decoder(register_module("decoder", CNNTranspose(Conv2DInput{1, 1, dense_size}, decoderOpt))),
                           m_ZDense(register_module("dense", torch::nn::Linear(z_dim , dense_size))),
                           m_ZMean(register_module("z_mean", torch::nn::Linear(flatten(m_Encoder->outputSize()) , z_dim))),
                           m_ZLogVar(register_module("z_log_var", torch::nn::Linear(flatten(m_Encoder->outputSize()), z_dim))),
                           m_Width(inputShape.width),
                           m_Height(inputShape.height),
                           m_Channel(inputShape.channel){}

    VAEOutput CnnVaeImpl::forward(const torch::Tensor &encoder_in) {
        assert(encoder_in.dim() == 4 and
               encoder_in.size(1) == m_Channel and
               encoder_in.size(2) == m_Width and
               encoder_in.size(3) == m_Height );

        auto batch_sz = encoder_in.size(0);

        auto z = m_Encoder->forward(encoder_in);
        z = torch::flatten(z, 1); // maintain batch_size

        auto z_mean = m_ZMean(z);
        auto z_log_var = m_ZLogVar(z);

        auto sigma = (z_log_var/2.0).exp();
        auto eps = torch::randn_like(sigma);

        z = z_mean +  sigma * eps;

        auto decoder_in =  m_ZDense(z).view({batch_sz, -1, 1, 1});
        auto decoded = m_Decoder->forward(decoder_in);

        return {z, z_mean, z_log_var, decoded};
    }
}

