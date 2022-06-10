//
// Created by dewe on 6/10/22.
//

#include <common/common.h>
#include "residual_layer.h"


namespace sam_dn{

    Residual1DBlockImpl::Residual1DBlockImpl(Option opt ):
    norm( torch::nn::LayerNormOptions({opt.n_features}).elementwise_affine(true) ),
    w1( opt.n_features, opt.hidden_dim),
    w2( opt.hidden_dim, opt.n_features){
        initializeWeightBias(norm, opt);
        initializeWeightBias(w1, opt);
        initializeWeightBias(w2, opt);

        m_OutputSize = { opt.n_features };
        REGISTER_MODULE(norm, norm);
        REGISTER_MODULE(w1, w1);
        REGISTER_MODULE(w2, w2);
    }

    torch::Tensor Residual1DBlockImpl::forward ( torch::Tensor const& x) noexcept{
        auto out = torch::relu(w1(norm(x)));
        out = w2(out);
        return out + x;
    }

    Residual1DBlocksImpl::Residual1DBlocksImpl(Option opt) {

        for(int i = 0; i < opt.n_blocks; i++){
            seq->push_back(Residual1DBlock(opt));
        }
        REGISTER_MODULE(seq, seq);
        m_OutputSize = { opt.n_features };

    }
}