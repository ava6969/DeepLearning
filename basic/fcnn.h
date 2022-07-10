//
// Created by dewe on 9/17/21.
//

#ifndef SAMFRAMEWORK_FCNN_H
#define SAMFRAMEWORK_FCNN_H

#include <utility>

#include "base.h"


namespace sam_dn{

    struct FCNNOption : BaseModuleOption{
        std::vector<int64_t> dims;
        std::string act_fn{"relu"};

        FCNNOption()=default;
        FCNNOption( std::vector<int64_t> const& dims, std::string act_fn="relu"):dims( dims), act_fn(std::move(act_fn)){}

        BaseModuleOption& Input(std::vector<int64_t> const& x) override {
            dims.insert(dims.begin(), x.begin(), x.end());
            return *this;
        }
    };

    class FCNNImpl : public BaseModuleImpl<FCNNOption> {

    public:
        FCNNImpl()=default;

        explicit FCNNImpl(BaseModuleOption& opt):FCNNImpl( dynamic_cast<FCNNOption&>(opt) ) {}
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
        
        void pretty_print(std::ostream& stream) const override {
            stream  << "sam_dn::FCNN"
                    << "("
                    << "act_fn=" << this->opt.act_fn << ", "
                    << "weight_init_param=" << this->opt.weight_init_param << ", "
                    << "bias_init_param=" << this->opt.bias_init_param << ", "
                    << "new_bias=" << this->opt.new_bias
                    << ")";
        }
    };
    TORCH_MODULE(FCNN);
}

SAM_OPTIONS(BaseModuleOption, FCNNOption, SELF(dims), SELF(act_fn));

#endif //SAMFRAMEWORK_FCNN_H
