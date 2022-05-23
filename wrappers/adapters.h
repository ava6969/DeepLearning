//
// Created by dewe on 2/15/22.
//

#ifndef SAM_RL_TRADER_ADAPTERS_H
#define SAM_RL_TRADER_ADAPTERS_H

#include "base.h"

namespace sam_dn{

    class Conv2DPositionEncodeImpl: public ModuleWithSizeInfoImpl{

        torch::Tensor spatial_coords;
    public:

        explicit Conv2DPositionEncodeImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){
            auto in = opt.dict_opt[opt.input];
            auto cH = in[1];
            auto cW = in[2];

            auto x_coord = torch::arange(cW).repeat({cH, 1}).to(c10::kFloat) / cW;
            auto y_coord = torch::arange(cH).repeat({cW, 1}).transpose(1, 0).to(c10::kFloat) / cH;
            spatial_coords = torch::stack({x_coord, y_coord}).unsqueeze(0);

            m_OutputSize = {cH*cW, in[0] + 2};
        }

//        torch::Tensor forward(torch::Tensor const& x) noexcept override{
//            auto n_cols = x.size(-1);
//            auto out = x.flatten(-2).transpose(-1, -2);
//            auto c = torch::arange(out.size(-2)).expand(out.sizes().slice(0, out.dim() -1)).to(out);
//
//            auto sz = out.sizes().slice(0, out.dim()-2).vec();
//            sz.insert(sz.end(), {-1, 1});
//
//            auto x_coord = (c % n_cols).view(sz);
//            auto y_coord = ( c.div(n_cols, "trunc") ).view(sz);
//            std::cout << x_coord << y_coord << "\n";
//
//            out = torch::cat({out, x_coord, y_coord}, -1);
//
//            return out;
//        }

        torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return torch::cat({x, spatial_coords.to(x.device()).repeat({ x.size(0) , 1, 1, 1})},
                              1).permute({0, 2, 3, 1}).flatten(1, 2);
        }

        inline TensorDict * forwardDict(TensorDict *  x) noexcept override{
            x->insert_or_assign(m_Output, forward(x->at(m_Input)));
            return x;
        }
    };

    TORCH_MODULE(Conv2DPositionEncode);
}
#endif //SAM_RL_TRADER_ADAPTERS_H
