#pragma once
//
// Created by dewe on 6/14/22.
//

#include "torch/torch.h"

namespace sam_dn{

    class PositionalEncoding : torch::nn::Module{

    public:
        PositionalEncoding(int d_model, float drop_out, int max_len=1024)
        :drop_out(register_module("drop_out", torch::nn::Dropout(drop_out))){
            pe = register_buffer("position_encoder", torch::zeros(max_len));
            auto position = torch::arange(max_len).unsqueeze(1);
            auto div_term = torch::exp( torch::arange(0, d_model, 2) * (-std::log(10000.0)/d_model));
            pe.slice(1, 0, c10::nullopt, 2) = torch::sin(position * div_term);
            pe.slice(1, 1, c10::nullopt, 2) = torch::cos(position * div_term);
            pe = pe.unsqueeze(0).transpose(0, 1);
        }

        torch::Tensor forward(torch::Tensor const& x) noexcept{
            auto z = x + pe.slice(0, 0, x.size(0));
            return drop_out(z);
        }

    private:
        torch::nn::Dropout drop_out{nullptr};
        torch::Tensor pe;

    };



}