#pragma once
//
// Created by dewe on 6/10/22.
//
#include "basic/self_attention.h"
#include "wrappers/adapters.h"
#include "base.h"

#ifdef DEBUG_VISION
    #define DEBUG_ATTN
#endif

namespace sam_dn {


    class AttentionBlockImpl : public torch::nn::Module {

        torch::nn::LayerNorm q_norm{nullptr}, k_norm{nullptr}, v_norm{nullptr}, post_norm{nullptr};
        torch::nn::Linear q{nullptr}, k{nullptr}, v{nullptr}, w1{nullptr}, w2{nullptr};
        bool max_out = true;
        std::pair<torch::Tensor, torch::Tensor > attention_forward( torch::Tensor const& x);
        float logit_scale;
        int embed_head_ratio;
        SelfAttentionOption opt;
        inline static int global_instance_counter{}, instance_id;

    public:

        explicit AttentionBlockImpl(SelfAttentionOption opt);
        torch::Tensor forward ( torch::Tensor const& x ) noexcept;


    };
    TORCH_MODULE(AttentionBlock);

    class RelationalModuleImpl : public ModuleWithSizeInfoImpl {
        torch::nn::Sequential seq{nullptr};

    public:

        struct Option : BaseModuleOption{
            SelfAttentionOption attn;
            int64_t n_blocks{};
            bool recurrent{true};

            BaseModuleOption& Input(std::vector<int64_t> const& x) override {
                attn.Input(x);
                return *this;
            }
        };

        explicit RelationalModuleImpl(Option);
        torch::Tensor forward ( torch::Tensor const& x) noexcept override;
    };

    TORCH_MODULE(RelationalModule);

}

SAM_OPTIONS(BaseModuleOption, RelationalModuleImpl::Option, SELF(attn), SELF(n_blocks), SELF(recurrent));