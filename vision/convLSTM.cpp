//
// Created by dewe on 6/13/22.
//

#include "convLSTM.h"

namespace torch::nn {

    Conv2DLSTMCellImpl::Conv2DLSTMCellImpl(ConvLSTMCellNDOption<2> const &_opt) :
            opt(_opt),
            input_to_hidden(register_module("input_to_hidden",
                                            Conv2d( Conv2dOptions (_opt.input_shape[0], 4 * _opt.output_channel,
                                                                   _opt.kernel_shape)
                                                                     .bias(false).padding( enumtype::kSame() ) ))),
            hidden_to_hidden(register_module("hidden_to_hidden",
                                             Conv2d( Conv2dOptions(
                                                     _opt.output_channel,
                                                     4 * _opt.output_channel,
                                                     _opt.kernel_shape).bias(false).padding( enumtype::kSame() ) )) ) {
        auto biases = torch::tensor_split(torch::zeros({4 * _opt.output_channel}), 4, 0);
        biases[1] += _opt.forget_bias;
        b = _opt.b_init(register_parameter(std::string("b"), torch::cat(biases).view({-1, 1, 1})));
    }

    std::tuple<Tensor, std::tuple<Tensor, Tensor>> Conv2DLSTMCellImpl::forward(
            const Tensor &input,
            torch::optional<std::tuple<Tensor, Tensor>> prev_state) {

        auto B = input.size(0);
        if(not prev_state)
            prev_state = zero_states(B, input.device());
        auto gates = input_to_hidden(input);
        gates += hidden_to_hidden(std::get<0>(*prev_state));
        gates += b;

        auto s = torch::tensor_split(gates, 4, 1);
        auto [i, f, g, o] = std::tie(s[0], s[1], s[2], s[3]);

        auto next_cell = torch::sigmoid(f) * std::get<1>(*prev_state);
        next_cell += torch::sigmoid(i) * torch::tanh(g);
        auto next_hidden = torch::sigmoid(o) * torch::tanh(next_cell);

        return {next_hidden, std::make_tuple(next_hidden, next_cell)};
    }

    std::tuple<Tensor, Tensor> Conv2DLSTMCellImpl::zero_states(int batch_size, torch::Device const &d) {

        std::vector<long> shape(4);
        for(int i = 1;i < 4; i++)
            shape[i] = opt.input_shape[i-1];
        shape[0] = batch_size;
        shape[1] = opt.output_channel;
        return {torch::zeros(shape, d), torch::zeros(shape, d)};
    }

    Conv2DLSTMImpl::Conv2DLSTMImpl(Conv2DLSTMOption opt): opt(opt){

        for(int i = 0 ; i < opt.num_layers; i++){
            auto sz = opt.cell_opt.input_shape;
            opt.cell_opt.input_shape = i == 0 ? sz : std::array<long, 3>{opt.cell_opt.output_channel, sz[1], sz[2]};
            cells->push_back(Conv2DLSTMCell(opt.cell_opt));
        }
        cells = register_module("cells", cells);
    }

    std::tuple<Tensor, std::tuple<Tensor, Tensor>>
    Conv2DLSTMImpl::forward(const Tensor &input,
                            torch::optional<std::tuple<Tensor, Tensor>> prev_state) {

        int64_t T =  opt.batch_first ? input.size(1) : input.size(0);
        int64_t B =  opt.batch_first ? input.size(0) : input.size(1);
        auto input_tensor = opt.batch_first ? input : input.transpose(0, 1);

        if(not prev_state)
            prev_state = zero_states(B, input.device());

        std::tuple< std::vector< torch::Tensor >, std::vector<  torch::Tensor > > last_state_list;
        int cell_layer_idx = 0;

        for( auto& cell: *cells){
            auto hc =
                    std::make_tuple( std::get<0>((*prev_state))[cell_layer_idx],
                            std::get<1>((*prev_state))[cell_layer_idx]);

            std::vector<torch::Tensor>  output_inner(T);
            for(int64_t t = 0; t < T; t++){
                std::cout << input_tensor.sizes() << "\n";
                auto it =  input_tensor.index_select(1, torch::tensor(t) ).squeeze(1);
                std::cout << it.sizes() << "\n";
                std::tie(it, hc) = cell->as<Conv2DLSTMCellImpl>()->forward( it, hc);
                output_inner[t] = it;
            }

            input_tensor = torch::stack(output_inner, 1);
            std::get<0>(last_state_list).emplace_back(std::get<0>(hc).clone());
            std::get<1>(last_state_list).emplace_back(std::get<1>(hc).clone());
            cell_layer_idx++;
        }

        return { opt.batch_first ? input_tensor : input_tensor.transpose(0, 1),
                 std::tuple( torch::stack( std::get<0>(last_state_list), 0),
                             torch::stack( std::get<1>(last_state_list), 0) ) };
    }

    std::tuple<Tensor, Tensor> Conv2DLSTMImpl::zero_states(int batch_size, const Device &dev) {
        auto d = opt.cell_opt.input_shape;
        auto hidden_dim = opt.cell_opt.output_channel;

        return std::make_tuple(
                torch::zeros({opt.num_layers, batch_size, hidden_dim, d[1], d[2]}, dev),
                torch::zeros({opt.num_layers, batch_size, hidden_dim, d[1], d[2]}, dev));


    }
}