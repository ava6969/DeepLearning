//
// Created by dewe on 6/10/22.
//

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "relational_module.h"

namespace sam_dn{

    AttentionBlockImpl::AttentionBlockImpl(SelfAttentionOption opt):
    qkv_norm(register_module("qkv_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.head_size * 3 * opt.n_heads }).elementwise_affine(true) ) ) ),
    post_norm(register_module("post_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({opt.features_size}).elementwise_affine(true) ) ) ),
    qkv(register_module("qkv", torch::nn::Linear( opt.features_size, 3*opt.head_size*opt.n_heads  ) ) ),
    w1(register_module("w_1", torch::nn::Linear(opt.head_size*opt.n_heads , opt.features_size))),
    w2(register_module("w_2", torch::nn::Linear(opt.features_size, opt.features_size))),
    opt( opt )
    {
        instance_id = global_instance_counter++;
        initializeWeightBias(qkv, opt);
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
        auto qkv_out = qkv_norm( qkv(x) ).view({ B, this->opt.n_features, this->opt.n_heads, 3*opt.head_size  });
        qkv_out = qkv_out.transpose(1, 2);

        auto qkv_split = qkv_out.split_with_sizes({opt.head_size, opt.head_size, opt.head_size}, -1);
        auto[q, k, v] = std::tie(qkv_split[0], qkv_split[1], qkv_split[2]);
        q = q * pow(opt.head_size, -0.5);

        auto dot_product = torch::matmul(q, k.transpose(2, 3)); // [B, H, N, N]
        auto weights = torch::softmax(dot_product, -1);
        auto output = torch::matmul(weights, v); // [B, H, N, V]

        output = output.transpose(1, 2);  // [B, N. H, V]
        output = output.flatten(2); // [B, N, H * V]

        return { output, weights};
    }

    RelationalModuleImpl::RelationalModuleImpl(Option opt): ModuleWithSizeInfoImpl(opt){
        seq= torch::nn::Sequential();
        auto in = opt.dict_opt[opt.input];
        Conv2DPositionEncode pos_enc(in[0], in[1], in[2]);
        auto out = pos_enc->outputSize();
        seq->push_back( pos_enc );

        opt.attn.Input(out);
        if (opt.recurrent) {
            AttentionBlock b(opt.attn);
            for(int i = 0; i < opt.n_blocks; i++) {
                seq->push_back(b);
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