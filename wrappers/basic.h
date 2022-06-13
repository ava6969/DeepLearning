//
// Created by dewe on 12/12/21.
//

#ifndef DEEP_NETWORKS_BASIC_H
#define DEEP_NETWORKS_BASIC_H

#include "base.h"
#include "vision/conv_net.h"
#include "boost/algorithm/string.hpp"

namespace sam_dn{


    struct AxisOption : BaseModuleOption{
        std::vector<int64_t> axis;
    };

    enum class Joiner{
        Cat,
        Vstack,
        Stack
    };

    template<Joiner joiner>
    struct JoinAtAxisImpl : public ModuleWithSizeInfoImpl{
        std::vector<std::string> inputs;
        std::vector<torch::Tensor> input_tensors;
        int64_t Axis1{};
        JoinAtAxisImpl()=default;

        explicit JoinAtAxisImpl(AxisOption opt);

        inline TensorDict * forwardDict(TensorDict *x)  noexcept final;

        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            std::cerr << "Using Unimplmented concat -> make sure concat is called with forwardDict(x)\n";
            return {};
        }
    };

    struct SqueezeAtAxisImpl : ModuleWithSizeInfoImpl{
        int64_t Axis1{};
        SqueezeAtAxisImpl()=default;
        explicit SqueezeAtAxisImpl(AxisOption opt): ModuleWithSizeInfoImpl(opt),Axis1(opt.axis[0]) {}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::squeeze(x, Axis1);
        }
    };

    struct SplitAndStackImpl : ModuleWithSizeInfoImpl{
        int64_t splitAxis{}, stackAxis{}, splitSize{};

        struct Option: BaseModuleOption{
            int64_t split_axis{}, stack_axis{}, split_size{};
        };

        SplitAndStackImpl()=default;

        explicit SplitAndStackImpl(Option opt);

        torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::stack(torch::tensor_split(x, splitSize, splitAxis), stackAxis);;
        }
    };

    struct FlattenImpl : ModuleWithSizeInfoImpl{
        int64_t _from{}, _to{};
        FlattenImpl()=default;
        explicit FlattenImpl(AxisOption opt): ModuleWithSizeInfoImpl(opt),
                                              _from(opt.axis[0]),
                                              _to(opt.axis[1]){}

        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::flatten(x, _from, _to);
        }
    };

    struct UnSqueezeAtAxisImpl : ModuleWithSizeInfoImpl{
        int64_t Axis1{};
        UnSqueezeAtAxisImpl()=default;
        explicit UnSqueezeAtAxisImpl(AxisOption opt): ModuleWithSizeInfoImpl(opt),Axis1(opt.axis[0]) {}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::unsqueeze(x, Axis1);
        }
    };

    struct TransposeImpl : ModuleWithSizeInfoImpl{
        int64_t Axis1{}, Axis2{};
        TransposeImpl()=default;
        explicit TransposeImpl(AxisOption opt): ModuleWithSizeInfoImpl(opt),
        Axis1(opt.axis[0]),
        Axis2(opt.axis[1]) {}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final {
            return torch::transpose(x, Axis1, Axis2);
        }
    };

    template<class ImplType>
    struct ConditionalImpl : ModuleWithSizeInfoImpl{
        AxisOption _opt;
        std::shared_ptr<ImplType> impl = nullptr;
        bool (*condition) (torch::Tensor const&, AxisOption const&);

        explicit ConditionalImpl(AxisOption opt,
                                 bool (*condition) (torch::Tensor const&, AxisOption const&) ):ModuleWithSizeInfoImpl(opt),
                                 condition(condition), _opt(opt){}

        inline torch::Tensor forward(const torch::Tensor &x) noexcept override {
            return condition(x, _opt) ? impl->forward(x) : x;
        }
    };

    struct ExpandIfDimEqualImpl : ConditionalImpl< UnSqueezeAtAxisImpl >{

        explicit ExpandIfDimEqualImpl(AxisOption opt):
        ConditionalImpl< UnSqueezeAtAxisImpl >(opt, [](torch::Tensor const& x, AxisOption const& _opt) -> bool {
            return x.dim() == _opt.axis[1];
        }){}
    };

    struct LayerNorm1dImpl: public ModuleWithSizeInfoImpl{
        torch::nn::LayerNorm norm{nullptr};
    public:
        struct Option : BaseModuleOption{
            int64_t input_sz{};
            int64_t axis{};
            BaseModuleOption& Input(const std::vector<int64_t> & x) override{
                input_sz = x[axis];
                return *this;
            }
        };

        explicit LayerNorm1dImpl(Option const& opt):ModuleWithSizeInfoImpl(opt){
            torch::nn::LayerNormOptions _opt({opt.input_sz});
            norm = register_module("layer_norm", torch::nn::LayerNorm(_opt));
        }

        torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return norm(x);
        }
        TensorDict* forwardDict(TensorDict* x) noexcept override{
            x->insert_or_assign(m_Output, norm(x->at(m_Input))); return x;
        }
    };

    struct DropoutImpl: public ModuleWithSizeInfoImpl{
        torch::nn::Dropout drop_out{nullptr};
    public:
        struct Option : BaseModuleOption{
            float prob{};
        };

        explicit DropoutImpl(Option const& opt):ModuleWithSizeInfoImpl(opt){
            drop_out = register_module("dropout", torch::nn::Dropout(opt.prob));
        }

        inline torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return drop_out(x);
        }
        inline TensorDict* forwardDict(TensorDict* x) noexcept override{
            x->insert_or_assign(m_Output, drop_out(x->at(m_Input))); return x;
        }
    };

    struct MaxPool2DImpl: public ModuleWithSizeInfoImpl{
        torch::nn::MaxPool2d pool_2d{nullptr};
        bool flatten_out;

    public:
        explicit MaxPool2DImpl(CNNOption opt);

        inline torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return flatten_out ? torch::flatten( pool_2d(x) ) : pool_2d(x);
        }
        inline TensorDict* forwardDict(TensorDict* x) noexcept override{
            x->insert_or_assign(m_Output, forward(x->at(m_Input))); return x;
        }
    };

    extern template class JoinAtAxisImpl<Joiner::Stack>;
    extern template class JoinAtAxisImpl<Joiner::Cat>;
    extern template class JoinAtAxisImpl<Joiner::Vstack>;

    TORCH_MODULE(LayerNorm1d);
    TORCH_MODULE(Dropout);
    TORCH_MODULE(MaxPool2D);
    TORCH_MODULE(ExpandIfDimEqual);
    TORCH_MODULE(Flatten);
    TORCH_MODULE(SplitAndStack);
    TORCH_MODULE_IMPL(Stack, JoinAtAxisImpl<Joiner::Stack>);
    TORCH_MODULE_IMPL(Concat, JoinAtAxisImpl<Joiner::Cat>);
    TORCH_MODULE_IMPL(Vstack, JoinAtAxisImpl<Joiner::Vstack>);
    TORCH_MODULE(SqueezeAtAxis);
    TORCH_MODULE(Transpose);

}

SAM_OPTIONS(BaseModuleOption, LayerNorm1dImpl::Option, SELF(axis));
SAM_OPTIONS(BaseModuleOption, DropoutImpl::Option, SELF(prob));
SAM_OPTIONS(BaseModuleOption, AxisOption, SELF(axis));
SAM_OPTIONS(BaseModuleOption, SplitAndStackImpl::Option, SELF(split_size), SELF(stack_axis), SELF(split_axis));
#endif //DEEP_NETWORKS_BASIC_H
