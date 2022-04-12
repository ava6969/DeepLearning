//
// Created by dewe on 3/27/22.
//
#include <torch/torch.h>

#include "common/summary.h"
#include "basic/fcnn.h"
#include "vision/conv_net.h"

int main()
{
    sam_dn::FCNNOption fcnn_opt;
    fcnn_opt.act_fn = "relu";
    fcnn_opt.dims = {784, 64, 32, 10};
    fcnn_opt.weight_init_type = "orthogonal";
    fcnn_opt.weight_init_param = std::sqrt(2.f);
    fcnn_opt.bias_init_param = 1;
    fcnn_opt.bias_init_type = "constant";

    sam_dn::CNNOption conv_net_opt;
    conv_net_opt.filters = {6, 2, 3};
    conv_net_opt.kernels = {3, 2, 6};
    conv_net_opt.weight_init_type = "orthogonal";
    conv_net_opt.weight_init_param = std::sqrt(2.f);
    conv_net_opt.bias_init_param = 1;
    conv_net_opt.bias_init_type = "constant";
    torch::nn::Sequential sequential(
        // Libtorch layers
        torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 8, { 3, 4 })),
        torch::nn::Flatten(),
        torch::nn::Linear(torch::nn::LinearOptions(8, 3)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(3, 6)),
        torch::nn::Sigmoid(),
        // Custom layers
        FCNNImpl(fcnn_opt),
    );
    sam_dn::print_summary(*sequential.ptr(), std::cout);
}