//
// Created by dewe on 3/27/22.
//
#include <torch/torch.h>
#include "vision/conv_net.h"
#include "basic/fcnn.h"
#include "common/summary.h"
#include "basic/fcnn.h"
#include "vision/conv_net.h"
#include "vision/cnn_vae.h"
#include "vision/impala_residual_block.h"

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
    conv_net_opt.filters = {3, 3, 3};
    conv_net_opt.kernels = {3, 3, 3};
    conv_net_opt.weight_init_type = "orthogonal";
    conv_net_opt.weight_init_param = std::sqrt(2.f);
    conv_net_opt.bias_init_param = 1;
    conv_net_opt.bias_init_type = "constant";

    torch::nn::Sequential sequential(
        // Libtorch layers
        torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 8, {3, 4})),
        torch::nn::Flatten(),
        torch::nn::Linear(torch::nn::LinearOptions(8, 3)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(3, 6)),
        torch::nn::Sigmoid(),
        // Custom layers
        FCNNImpl(fcnn_opt)
        // These layers have pretty print methods defined,
        // but I can't instantiate them for some reason
        // sam_rl::CnnVaeImpl(
        //     Conv2DInput{3, 3, 3},
        //     3,
        //     5,
        //     conv_net_opt,
        //     conv_net_opt
        // ),
        // CNNImpl(conv_net_opt),
    );
    sam_dn::print_summary(sequential, std::cout);

    torch::nn::Sequential seq2{
        sam_dn::CNN(
                sam_dn::CNNOption({16, 32, 64})
                .setKernels({3, 3, 3})
                .flattenOut(true)
                .setStrides({1, 1, 1})
                .setActivations({"relu", "relu", "relu"})
                .setPaddings({"same", "same", "same"})
                .Input({3, 128, 128})),
        sam_dn::FCNN(
                sam_dn::FCNNOption({ 128, 64})
                .Input({256}))
    };
    sam_dn::print_summary(seq2, std::cout);
}
