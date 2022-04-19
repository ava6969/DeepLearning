#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <iostream>
#include <vector>

constexpr auto BYTE_COUNT = 1024;
constexpr auto LABEL_COUNT = 4;
static const char* BYTE_LABELS[] = { "B", "MB", "GB", "TB" };

namespace sam_dn {

    /// Prints out the properties of each layer.
    void print_summary(torch::nn::SequentialImpl &sequential, std::ostream& out);

    inline static void print_summary(torch::nn::Sequential &sequential, std::ostream& out) {
        print_summary(*sequential.ptr(), out);
    }

}