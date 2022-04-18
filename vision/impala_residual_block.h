//
// Created by dewe on 2/23/22.
//

#pragma once

#include "residual_block.h"
#include "optional"
#include "wrappers/basic.h"

namespace sam_dn{

    class ImpalaResidualBlockImpl : public ModuleWithSizeInfoImpl{

    public:

        struct Option : BaseModuleOption{
            CNNOption res_option1, res_option2, conv_option;
            bool batch_norm{false};
            std::optional<float> drop_out{};
        };

        explicit ImpalaResidualBlockImpl(Option opt);

        torch::Tensor forward ( torch::Tensor const& x) noexcept override;

        inline TensorDict * forwardDict(TensorDict *x) noexcept override{
            x->insert_or_assign( m_Output, forward(x->at(m_Input)));
            return x;
        }

        void pretty_print(std::ostream& stream) const override {
            stream  << "sam_dn::ImpalaResidualBlock"
                    << "("
                    << "batch_norm=" << this->batch_norm
                    << ")";
        }

    private:
        ResidualBlock residualBlock1{nullptr};
        ResidualBlock residualBlock2{nullptr};
        torch::nn::BatchNorm2d batch_norm {nullptr};
        torch::nn::Dropout drop_out{nullptr};
        CNN conv_layer{nullptr};
        MaxPool2D max_pool{nullptr};
    };


    class ImpalaResnetImpl : public BaseModuleImpl<>{

    public:
        struct Option : BaseModuleOption{
            std::vector<int> filters{};
            bool relu_last{false}, batch_norm{false};
            std::optional<float> drop_out{std::nullopt};
            bool flatten_out{false};
        };

        explicit ImpalaResnetImpl(Option opt);
    };

    TORCH_MODULE(ImpalaResidualBlock);
    TORCH_MODULE(ImpalaResnet);

}

SAM_OPTIONS(BaseModuleOption, ImpalaResidualBlockImpl::Option,
            SELF(res_option1), SELF(res_option2), SELF(conv_option),
            SELF(batch_norm), SELF(drop_out))

SAM_OPTIONS(BaseModuleOption, ImpalaResnetImpl::Option,
            SELF(filters), SELF(relu_last), SELF(flatten_out),
            SELF(batch_norm), SELF(drop_out))


