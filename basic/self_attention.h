//
// Created by dewe on 1/11/22.
//

#ifndef DEEP_NETWORKS_SELF_ATTENTION_H
#define DEEP_NETWORKS_SELF_ATTENTION_H

#include "base.h"
#include "common/common.h"
#include "optional"

namespace sam_dn{

    class SelfAttentionImpl : public ModuleWithSizeInfoImpl{

        /// N_EMBED MUST BE DIVISIBLE BY FEATURE_SIZE * heads
    public:
        explicit SelfAttentionImpl(const SelfAttentionOption& opt):ModuleWithSizeInfoImpl(opt),opt(opt),
        qk_scale(  std::sqrt( opt.qk_w / float(opt.features_size) ) ),
        v_scale(  std::sqrt( opt.v_w / float(opt.features_size) ) ),
        logit_scale( std::sqrt( float(opt.n_embed) / float(opt.n_heads) ) ),
        post_scale( std::sqrt( opt.post_w / float(opt.n_embed) ) ),
        qk(register_module("qk_embed", torch::nn::Linear( opt.features_size, opt.n_embed * 2 ) ) ),
        value(register_module("v_embed", torch::nn::Linear( opt.features_size, opt.n_embed ) ) ),
        post_a_mlp(register_module("post_a_mlp", torch::nn::Linear(opt.n_embed, opt.features_size) ) )
        {
            paramInit( opt.weight_init_type == "none" ? "xavier_normal" : opt.weight_init_type,
                       qk_scale, qk->weight);
            paramInit( opt.weight_init_type == "none" ? "xavier_normal" : opt.weight_init_type,
                       v_scale, value->weight);
            paramInit( opt.weight_init_type == "none" ? "xavier_normal" : opt.weight_init_type,
                       post_scale, post_a_mlp->weight);
            embed_head_ratio = static_cast<long>(std::floor(opt.n_embed/opt.n_heads)) ;
            if(opt.layer_norm){
                torch::nn::LayerNormOptions _opt({opt.features_size});
                _opt.elementwise_affine(true);
                norms = register_module("pre_layer_norm", torch::nn::LayerNorm(_opt));
            }
            if(opt.post_layer_norm){
                torch::nn::LayerNormOptions _opt({opt.features_size});
                _opt.elementwise_affine(true);
                post_norm = register_module("post_layer_norm", torch::nn::LayerNorm(_opt));
            }
            m_OutputSize = { opt.features_size };
        }

        auto forward_pass( torch::Tensor const& x){
            auto B = x.size(0);

            TORCH_CHECK(x.dim() == 3,  "SelfAttentionImpl follows openAI Implementation which requires "
                                       "either shape(B NE F)");
            auto inp = norms ? norms.value()(x) : x;

            auto qk_out = qk(inp);
            qk_out = qk_out.view({B, this->opt.n_features, this->opt.n_heads, embed_head_ratio, 2 });
            auto splitted = qk_out.unbind(-1);
            auto query = splitted[0].squeeze(-1).permute({0, 2, 1, 3});
            auto key = splitted[1].squeeze(-1).permute({0, 2, 3, 1});
            auto val = value(x).view({B, this->opt.n_features, this->opt.n_heads, embed_head_ratio })
                    .permute({0, 2, 1, 3});

            return std::make_tuple(query, key, val);
        }

        static torch::Tensor stableMaskedSoftMax(torch::Tensor const& logit, torch::Tensor  mask){
            mask = mask.unsqueeze(2);
            TORCH_CHECK(mask.dim() == 4,  "SelfAttentionImpl follows openAI Implementation which requires mask "
                                       "either shape(B T NE F) or shape(T B NE F)");
            auto logits = logit - ((1.0 - mask) * 1e10);
            logits -= std::get<0>( torch::max(logits, -1, true) );
            auto un_norm_p = logits.exp();
            un_norm_p *= mask;
            auto norm_p = un_norm_p / (un_norm_p.sum(-1, true) + 1e-10);
            norm_p *= mask;
            return norm_p;
        }

        torch::Tensor finish( torch::Tensor const& inp,
                              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> && s){

            auto && [query, key, _value] = s;
            auto logit = torch::matmul(query, key) / logit_scale; // (bs, T, heads, NE, NE)
            auto soft_max = torch::softmax(logit, -1);

            auto att_sum = torch::matmul(soft_max, _value).permute({0, 2, 1, 3}); // ( B, n_output_entities, heads, features)
            att_sum = att_sum.reshape({-1, this->opt.n_features, this->opt.n_embed});
            auto x = inp + post_a_mlp(att_sum);
            x = post_norm ? post_norm.value()(x) : x;
            if(opt.max_pool != std::nullopt)
                x = opt.max_pool.value() ? std::get<0>(torch::max(x, -2)) : torch::mean(x, -2);
            return x;
        }

        inline torch::Tensor forward( torch::Tensor const& x) noexcept override{
            auto res = finish( x, forward_pass(x) );
            return res;
        }

        inline TensorDict* forwardDict( TensorDict* x) noexcept override{
            // TODO: ADD MASK FROM POSITION
//            auto res = finish(x->at(m_Input), stableMaskedSoftMax(forward_pass(x->at(m_Input) ), torch::Tensor()) );
            x->insert_or_assign(m_Output, forward(x->at(m_Input)));
            return x;
        }

    private:
        const float qk_scale, v_scale, logit_scale, post_scale;
        torch::nn::Linear qk{nullptr};
        torch::nn::Linear value{nullptr};
        torch::nn::Linear post_a_mlp{nullptr};
        torch::optional<torch::nn::LayerNorm> norms{torch::nullopt}, post_norm{torch::nullopt};
        SelfAttentionOption opt;
        int64_t embed_head_ratio;
    };

    TORCH_MODULE(SelfAttention);
}

#endif //DEEP_NETWORKS_SELF_ATTENTION_H
