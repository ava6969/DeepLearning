//
// Created by dewe on 6/10/22.
//

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "relational_module.h"

namespace sam_dn{

    AttentionBlockImpl::AttentionBlockImpl(SelfAttentionOption opt):
    q_norm(register_module("q_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.n_embed}).elementwise_affine(true) ) ) ),
    k_norm(register_module("k_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.n_embed}).elementwise_affine(true) ) ) ),
    v_norm(register_module("v_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.n_embed}).elementwise_affine(true) ) ) ),
    post_norm(register_module("post_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.features_size}).elementwise_affine(true) ) ) ),
    q(register_module("q", torch::nn::Linear( opt.features_size, opt.n_embed  ) ) ),
    k(register_module("k", torch::nn::Linear( opt.features_size, opt.n_embed  ) ) ),
    v(register_module("v", torch::nn::Linear( opt.features_size, opt.n_embed  ) ) ),
    w1(register_module("w_1", torch::nn::Linear(opt.n_embed, opt.n_embed))),
    w2(register_module("w_2", torch::nn::Linear(opt.n_embed, opt.features_size))),
    logit_scale( std::sqrt( float(opt.n_embed) / float(opt.n_heads) ) ),
    opt( opt ),
    embed_head_ratio( static_cast<int>(std::floor(opt.n_embed/opt.n_heads)) )
    {
        instance_id = global_instance_counter++;
        initializeWeightBias(q, opt);
        initializeWeightBias(k, opt);
        initializeWeightBias(v, opt);
        initializeWeightBias( w1, opt);
        initializeWeightBias( w2, opt);

    }


    torch::Tensor AttentionBlockImpl::forward ( torch::Tensor const& x) noexcept {

        auto [attn_output, attn_weights] = attention_forward(x);

        if (opt.store_state){
            auto cpu = attn_weights.to(torch::kCPU);
            auto nodeH = cpu.size(2);
            auto nodeW = cpu.size(3);
            long N = std::sqrt(nodeH);

            for (int i = 0; i < opt.n_heads; i++) {
                for (int n = 0; n < nodeH; n++) {
                    auto node = (cpu[0][i][n].flatten(0) * 255).to(c10::kByte).view({N, N});

                }
            }
        }

        auto res = torch::relu( w2( torch::relu( w1(attn_output) )));
        auto z = res + x;
        return post_norm(z);
    }

    std::pair<torch::Tensor, torch::Tensor> AttentionBlockImpl::attention_forward(torch::Tensor const& x) {
        auto B = x.size(0);
        auto[Q, K, V] = std::make_tuple( q_norm(q(x)), k_norm(k(x)), v_norm(v(x)) );
        Q = Q.view({B, this->opt.n_features, this->opt.n_heads, embed_head_ratio}).permute({0, 2, 1, 3});
        K = K.view({B, this->opt.n_features, this->opt.n_heads, embed_head_ratio}).permute({0, 2, 3, 1});
        V = V.view({B, this->opt.n_features, this->opt.n_heads, embed_head_ratio}).permute({0, 2, 1, 3});

        auto attn_weights = torch::softmax( torch::matmul(Q, K) / logit_scale, -1 );
        auto A = torch::matmul(attn_weights, V).permute({0, 2, 1, 3}); // ( B, n_output_entities, heads, m_State)
        A = A.contiguous().view({-1, this->opt.n_features, this->opt.n_embed});

        return {A, attn_weights};
    }

    RelationalModuleImpl::RelationalModuleImpl(Option opt){
        seq= torch::nn::Sequential();
        auto in = opt.dict_opt[opt.input];
        Conv2DPositionEncode pos_enc(in[0], in[1], in[2]);
        auto out = pos_enc->outputSize();
        seq->push_back( pos_enc );

        opt.Input(out);
        if (opt.recurrent) {
            AttentionBlock attn(opt.attn);
            for(int i = 0; i < opt.n_blocks; i++) {
                seq->push_back(attn);
            }
        }else{
            for(int i = 0; i < opt.n_blocks; i++) {
                seq->push_back(AttentionBlock(opt.attn));
            }
        }

        REGISTER_MODULE(seq, seq);
        m_OutputSize = std::vector{ opt.attn.features_size };

    }

    torch::Tensor RelationalModuleImpl::forward ( torch::Tensor const& x) noexcept {
        return std::get<0>(torch::max( seq->forward(x), 1));
    }
}