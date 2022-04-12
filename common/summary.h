#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <tabulate/table.hpp>
#include <iostream>
#include <vector>

const auto BYTE_COUNT = 1024;
const auto LABEL_COUNT = 4;
static const char* BYTE_LABELS[] = { "B", "MB", "GB", "TB" };

namespace sam_dn {

    /// Prints out the properties of each layer.
    void print_summary(torch::nn::SequentialImpl &sequential, std::ostream& out) {
        tabulate::Table table;
        table.add_row({"Options", "Output Shape", "Param #"});

        uint32_t param_count = 0;

        for (auto it = sequential.begin(); it != sequential.end(); it++) {
            auto params = it->ptr()->parameters();
            std::string shape_str;
            uint32_t layer_param_count = 1;

            // Write out shape dimensions
            if (params.size() > 0) {
                shape_str += "[";
                auto param = params[0];
                auto param_size_count = param.sizes().size();
                for (int i = param_size_count - 2; i >= 0; i--) {
                    auto dim = param.sizes()[i];
                    layer_param_count *= dim;
                    shape_str += std::to_string(dim);
                    if (i > 0) {
                        shape_str += ", ";
                    }
                }
                shape_str += "]";
            }
            else {
                layer_param_count = 0;
            }
            param_count += layer_param_count;

            std::stringstream opt_stream;
            it->ptr()->pretty_print(opt_stream);
            table.add_row({ opt_stream.str(), shape_str, std::to_string(layer_param_count) });
        }
        out << table << std::endl;

        out << "Parameter Count: " << std::to_string(param_count) << std::endl;

        auto model_size = param_count * sizeof(0.0);
        auto label_index = 0;
        out << "Model Size: ";
        while (model_size >= BYTE_COUNT && label_index < LABEL_COUNT) {
            model_size /= BYTE_COUNT;
            label_index += 1;
        }
        out << std::to_string(model_size) << " " << BYTE_LABELS[label_index] << std::endl;
    }
}