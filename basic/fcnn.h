//
// Created by dewe on 9/17/21.
//

#ifndef SAMFRAMEWORK_FCNN_H
#define SAMFRAMEWORK_FCNN_H

#include "base.h"

namespace sam_dn{

    class FCNNImpl : public BaseModuleImpl<FCNNOption> {

    public:
        FCNNImpl()=default;

        explicit FCNNImpl(FCNNOption const& option);

        explicit FCNNImpl(int64_t input, int64_t output,
                          std::string const& weight_init_type="none",
                          float weight_init_param=std::sqrt(2.f),
                          std::string const& bias_init_type="none",
                          float bias_init_param=0,
                          bool new_bias=true);

        inline torch::Tensor forward(const torch::Tensor &x) noexcept override {
            return m_BaseModel->forward(x.reshape({-1, opt.dims[0]}));
        }

    };
    TORCH_MODULE(FCNN);
}


#endif //SAMFRAMEWORK_FCNN_H
