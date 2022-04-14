//
// Created by dewe on 2/15/22.
//

#ifndef SAM_RL_TRADER_ADAPTERS_H
#define SAM_RL_TRADER_ADAPTERS_H

#include "base.h"

namespace sam_dn{

    class Conv2DPositionEncodeImpl: public ModuleWithSizeInfoImpl{

    public:
        Conv2DPositionEncodeImpl()=default;

        explicit Conv2DPositionEncodeImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){
            auto in = opt.dict_opt[opt.input];
            auto r = in[1];
            auto c = in[2];
            m_OutputSize = {r*c, in[0] + 2};
        }

        torch::Tensor forward(torch::Tensor const& x) noexcept override{
            auto n_cols = x.size(-1);
            auto out = x.flatten(-2).transpose(-1, -2);
            auto c = torch::arange(out.size(-2)).expand(out.sizes().slice(0, out.dim() -1)).to(out);

            auto sz = out.sizes().slice(0, out.dim()-2).vec();
            sz.insert(sz.end(), {-1, 1});

            auto x_coord = (c % n_cols).view(sz);
            auto y_coord = ( c.div(n_cols, "trunc") ).view(sz);

            out = torch::cat({out, x_coord, y_coord}, -1);

            return out;
        }

        inline TensorDict * forwardDict(TensorDict *  x) noexcept override{
            x->insert_or_assign(m_Output, forward(x->at(m_Input)));
            return x;
        }
    };

    TORCH_MODULE(Conv2DPositionEncode);
}
#endif //SAM_RL_TRADER_ADAPTERS_H
