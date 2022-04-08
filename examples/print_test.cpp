//
// Created by dewe on 3/27/22.
//
#include <torch/torch.h>

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
    sam_dn::print_summary(*sequential.ptr(), std::cout);
}