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
        explicit Conv2DPositionEncodeImpl(int channel, int x_ax, int y_ax){
            registerModule(channel, x_ax, y_ax);
        }

        explicit Conv2DPositionEncodeImpl(BaseModuleOption opt):
        ModuleWithSizeInfoImpl(opt){
            auto in = opt.dict_opt[opt.input];
            registerModule(in[0], in[1], in[2]);
        }

        void registerModule(int channel, int x_ax, int y_ax){
            auto x_lin = torch::linspace(-1,1,x_ax);
            auto x_coord = x_lin.repeat({1, y_ax,1}).view({-1, 1, y_ax, x_ax}).transpose(3,2).squeeze(0).squeeze(0);

            auto y_lin = torch::linspace(-1,1,y_ax).view({-1, 1});
            auto y_coord =  y_lin.repeat({1,1,x_ax}).view({-1, 1, y_ax, x_ax}).transpose(3,2).squeeze(0).squeeze(0);

            spatial_coords = register_buffer("spatial_coords", torch::stack({x_coord, y_coord}, 0));

            m_OutputSize = {x_ax*y_ax, channel + 2};
        }

        inline torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return torch::cat({x, spatial_coords.repeat({ x.size(0), 1, 1, 1})}, 1).permute({0, 2, 3, 1}).flatten(1, 2);
        }

        inline TensorDict * forwardDict(TensorDict *  x) noexcept override{
            x->insert_or_assign(m_Output, forward(x->at(m_Input)));
            return x;
        }
    };

    TORCH_MODULE(Conv2DPositionEncode);
}
#endif //SAM_RL_TRADER_ADAPTERS_H
