#pragma once
//
// Created by dewe on 6/10/22.
//

#include "base.h"


namespace sam_dn {

    /*
    Adds two extra channels to the feature dimension, indicating the spatial
    position (x and y) of each cell in the feature map using evenly spaced values
    between −1 and 1. Then projects the feature dimension to n_features through a
            linear layer.
    */
    class PositionalEncodingImpl : public torch::nn::Module {

    public:
        PositionalEncodingImpl(int n_kernels, int n_feature) :
                projection(register_module("projection", torch::nn::Linear(n_kernels+2, n_feature))){}

        inline torch::Tensor forward(torch::Tensor const &x) {
            auto z = add_encoding2D(x);
            z = z.view({x.size(0), z.size(1), -1});
            z = projection(z.transpose(2, 1)).transpose(1, 0);
            return z;
        }

        static inline torch::Tensor add_encoding2D(torch::Tensor const& x){
            auto x_ax = x.size(-2);
            auto y_ax = x.size(-1);

            auto x_lin = torch::linspace(-1, 1, x_ax);
            auto xx = x_lin.repeat({x.size(0), y_ax, 1}).view({-1, 1, y_ax, x_ax}).transpose(3, 2);

            auto y_lin = torch::linspace(-1, 1, y_ax).view({-1, 1});
            auto yy = y_lin.repeat({x.size(0), 1, x_ax}).view({-1, 1, y_ax, x_ax}).transpose(3, 2);

            return torch::cat( {x, xx.to(x), yy.to(x)}, 1);
        }

    private:
        torch::nn::Linear projection{nullptr};
    };

    TORCH_MODULE(PositionalEncoding);

    class PositionWiseFeedForwardImpl : public torch::nn::Module {

    public:
        PositionWiseFeedForwardImpl(int d_model, int d_ff, float dropout = 0.1) :
                w_1(register_module("w_1", torch::nn::Linear(d_model, d_ff))),
                w_2(register_module("w_2", torch::nn::Linear(d_ff, d_model))),
                dropout(dropout) {}

        inline auto forward(torch::Tensor const &x) {
            return w_2(dropout(torch::relu(w_1(x))));
        }

    private:
        torch::nn::Linear w_1{nullptr}, w_2{nullptr};
        torch::nn::Dropout dropout{nullptr};
    };

    TORCH_MODULE(PositionWiseFeedForward);

class AttentionBlockImpl : public torch::nn::Module {

        torch::nn::LayerNorm norm{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::MultiheadAttention attn{nullptr};
        PositionWiseFeedForward ff{nullptr};
    public:

        struct Option : BaseModuleOption{
            int n_kernels{}, n_features{}, n_heads{}, n_attn_modules{}, n_hidden{};
            float dropout{};
        };

        explicit AttentionBlockImpl(Option);
        torch::Tensor forward ( torch::Tensor const& x, std::optional<torch::Tensor> const&) noexcept;
    };

    TORCH_MODULE(AttentionBlock);

    class RelationalModuleImpl : public ModuleWithSizeInfoImpl {
        torch::nn::Sequential seq{nullptr};

    public:

        struct Option : BaseModuleOption{
            AttentionBlockImpl::Option attn;
            int64_t n_blocks{}, n_kernels{};

            BaseModuleOption& Input(std::vector<int64_t> const& x) override{
                n_blocks = x[0];
                return *this;
            }
        };

        explicit RelationalModuleImpl(Option);
        torch::Tensor forward ( torch::Tensor const& x) noexcept override;
    };

    TORCH_MODULE(RelationalModule);

}