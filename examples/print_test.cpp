//
// Created by dewe on 3/27/22.
//
#include <torch/torch.h>
#include "vision/conv_net.h"
#include "basic/fcnn.h"
#include "common/summary.h"

int main()
{
    torch::nn::Sequential sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 8, { 3, 4 })),
        torch::nn::Flatten(),
        torch::nn::Linear(torch::nn::LinearOptions(8, 3)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(3, 6)),
        torch::nn::Sigmoid()
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