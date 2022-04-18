//
// Created by dewe on 1/11/22.
//

#ifndef DEEP_NETWORKS_SELF_ATTENTION_H
#define DEEP_NETWORKS_SELF_ATTENTION_H

#include "base.h"
#include "common/common.h"
#include "optional"

namespace sam_dn{

    struct SelfAttentionOption: BaseModuleOption{
        int64_t n_heads{}, n_embed{};
        bool layer_norm{}, post_layer_norm{};
        float qk_w{}, v_w{}, post_w{};
        std::optional<bool> max_pool{};
        int64_t n_features{}, features_size{};

        BaseModuleOption& Input(std::vector<int64_t> const& x) override {
            n_features = x.at(0);
            features_size = x.at(1);
            return *this;
        }
    };

    class SelfAttentionImpl : public ModuleWithSizeInfoImpl{

    public:
        explicit SelfAttentionImpl(const SelfAttentionOption& opt);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_pass( torch::Tensor const& x);

        static torch::Tensor stableMaskedSoftMax(torch::Tensor const& logit, torch::Tensor  mask);

        torch::Tensor finish( torch::Tensor const& inp,
                              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> && s);

        torch::Tensor forward( torch::Tensor const& x) noexcept override;

        TensorDict* forwardDict( TensorDict* x) noexcept override;

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

SAM_OPTIONS(BaseModuleOption, SelfAttentionOption,
            SELF(n_embed), SELF(n_heads), SELF(layer_norm), SELF(post_layer_norm),
            SELF(qk_w), SELF(v_w), SELF(max_pool), SELF(post_w));

#endif //DEEP_NETWORKS_SELF_ATTENTION_H
