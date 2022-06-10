//
// Created by dewe on 6/10/22.
//

#include "relational_module.h"

namespace sam_dn{

    AttentionBlockImpl::AttentionBlockImpl(Option opt):
    norm(register_module("norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.n_features}).elementwise_affine(true) ) ) ),
    dropout(register_module("dropout", torch::nn::Dropout(opt.dropout))),
    attn(register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(opt.n_features, opt.n_heads).dropout(opt.dropout)))),
    ff(register_module("ff", PositionWiseFeedForward(opt.n_features, opt.n_hidden, opt.dropout)))
    {}

    torch::Tensor AttentionBlockImpl::forward ( torch::Tensor const& x,
                                                std::optional<torch::Tensor> const& mask) noexcept{
        auto[attn_output, attn_output_weights] = attn->forward(x, x, x, *mask);
        auto x_norm = dropout(norm(attn_output + x));
        auto z = ff->forward(x_norm);
        return dropout(norm(z));
    }

    RelationalModuleImpl::RelationalModuleImpl(Option opt){

        seq->push_back( PositionalEncoding(opt.n_kernels, opt.attn.n_features));
        for(int i = 0; i < opt.n_blocks; i++)
            seq->push_back(AttentionBlock(opt.attn));

    }

    torch::Tensor RelationalModuleImpl::forward ( torch::Tensor const& x) noexcept {
        return seq->forward(x);
    }
}