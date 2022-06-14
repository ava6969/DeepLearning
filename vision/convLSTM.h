#pragma once
//
// Created by dewe on 6/13/22.
//

#include "torch/torch.h"
#include "base.h"
#include "common/common.h"

namespace torch::nn {

    template<size_t N>
    using ConvT = std::conditional_t<N==1, torch::nn::Conv1d,
    std::conditional_t<N==2, torch::nn::Conv2d, torch::nn::Conv3d>>;

    template<size_t N>
    struct XavierNormal {
        int kernel_shape;

        explicit XavierNormal(int kernel_shape) : kernel_shape(kernel_shape) {}

        torch::Tensor operator()(torch::Tensor &x) {
            return torch::nn::init::xavier_normal_(x, sqrt(kernel_shape * N));
        }
    };

    struct Constant {
        torch::Tensor operator()(torch::Tensor &x) {
            return torch::nn::init::constant_(x, 0);
        }
    };

    template<size_t N>
    struct ConvLSTMCellNDOption {
        int kernel_shape{1};
        int output_channel{};
        std::array<long, N + 1> input_shape;
        float forget_bias{0};
        std::function<torch::Tensor(torch::Tensor &)> w_i_init = XavierNormal<N>(kernel_shape),
                w_h_init = XavierNormal<N>(kernel_shape),
                b_init = Constant{};
    };

    struct Conv2DLSTMOption {
        bool batch_first{true};
        int num_layers{1};
        ConvLSTMCellNDOption<2> cell_opt;
    };

class Conv2DLSTMCellImpl : public Module {

    public:

        explicit Conv2DLSTMCellImpl(ConvLSTMCellNDOption<2> const &opt);

        std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward(
                const Tensor &input,
                torch::optional<std::tuple<Tensor, Tensor>> prev_state = {});

        std::tuple<Tensor, Tensor> zero_states(int batch_size, torch::Device const &d);

    private:
    ConvLSTMCellNDOption<2> opt;
        ConvT<2> input_to_hidden{nullptr};
        ConvT<2> hidden_to_hidden{nullptr};
        torch::Tensor b;

};

    class Conv2DLSTMImpl : public Module {

    public:

        explicit Conv2DLSTMImpl(Conv2DLSTMOption opt);

        std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward(
                const Tensor &input,
                torch::optional<std::tuple<Tensor, Tensor>> prev_state = {});

        std::tuple<Tensor, Tensor> zero_states(int batch_size, torch::Device const &d);

    private:
        torch::nn::ModuleList cells;
        Conv2DLSTMOption opt;
    };

    TORCH_MODULE(Conv2DLSTMCell);
    TORCH_MODULE(Conv2DLSTM);
}