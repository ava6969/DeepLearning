#pragma once
//
// Created by dewe on 6/10/22.
//

#include "fcnn.h"
#include "base.h"

namespace sam_dn {

    class Residual1DBlockImpl : public ModuleWithSizeInfoImpl {

        torch::nn::LayerNorm norm{nullptr};
        torch::nn::Linear w1{nullptr}, w2{nullptr};

    public:

        struct Option : BaseModuleOption{
            int64_t n_features{};
            int64_t hidden_dim{};

            BaseModuleOption & Input(const std::vector<int64_t> & x) override{
                n_features = x[0];
                return *this;
            }
        };

        explicit Residual1DBlockImpl(Option opt );

        torch::Tensor forward ( torch::Tensor const& x) noexcept override;
    };

    class Residual1DBlocksImpl : public ModuleWithSizeInfoImpl {

        torch::nn::Sequential seq{nullptr};

    public:
        struct Option : Residual1DBlockImpl::Option{
            int64_t n_blocks{};
        };

        explicit Residual1DBlocksImpl(Option opt );

        inline torch::Tensor forward ( torch::Tensor const& x) noexcept override{
            return seq->forward(x);
        }
    };

    TORCH_MODULE(Residual1DBlock);
    TORCH_MODULE(Residual1DBlocks);
}

SAM_OPTIONS(BaseModuleOption, Residual1DBlockImpl::Option, SELF(n_features), SELF(hidden_dim))
SAM_OPTIONS(BaseModuleOption, Residual1DBlocksImpl::Option, SELF(n_blocks))