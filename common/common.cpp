//
// Created by dewe on 12/9/21.
//

#include <tabulate/table.hpp>
#include "vision/conv_net.h"
#include "summary.h"
#include "common.h"

namespace sam_dn{

    void paramInit(std::string const& type, float gain, torch::Tensor& param){
        if(type == "orthogonal")
            torch::nn::init::orthogonal_(param, gain);
        else if(type == "xavier_uniform")
            torch::nn::init::xavier_uniform_(param, gain);
        else if(type == "xavier_normal")
            torch::nn::init::xavier_normal_(param, gain);
        else if(type == "constant")
            torch::nn::init::constant_(param, gain);
    }

    void addActivationFunction(std::string const& act_fn, torch::nn::Sequential& model, int idx){
        if(act_fn != "none") {
            auto act = act_fn == "tanh" ? torch::nn::Functional(torch::nn::Tanh()) :
                       act_fn == "relu" ? torch::nn::Functional(torch::nn::ReLU()) :
                       act_fn == "leaky_relu" ? torch::nn::Functional(torch::nn::LeakyReLU()) :
                       torch::nn::Functional(torch::nn::Sigmoid());

            model->push_back(act_fn + "_" + std::to_string(idx), act);
        }
    }

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