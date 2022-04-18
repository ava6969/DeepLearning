//
// Created by dewe on 12/12/21.
//

#ifndef DEEP_NETWORKS_BASIC_H
#define DEEP_NETWORKS_BASIC_H

#include "base.h"
#include "boost/algorithm/string.hpp"

namespace sam_dn{

    template<bool cat, int64_t Axis1=-1>
    struct JoinAtAxisImpl : public ModuleWithSizeInfoImpl{
        std::vector<std::string> inputs;
        std::vector<torch::Tensor> input_tensors;
        JoinAtAxisImpl()=default;

        explicit JoinAtAxisImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){
            boost::split(inputs, this->m_Input, boost::is_any_of(";"));
            input_tensors.resize(inputs.size());
        }

        inline TensorDict * forwardDict(TensorDict *x)  noexcept final {
            std::transform(inputs.begin(), inputs.end(), input_tensors.begin(), [&x](auto const& in) {
                return x->at(in);
            });
            if constexpr(cat)
                x->template insert_or_assign( this->m_Output, torch::cat(input_tensors, Axis1));
            else
                x->template insert_or_assign( this->m_Output, torch::vstack(input_tensors));

            return x;
        }

        torch::Tensor forward(const torch::Tensor &x) noexcept final{
            std::cerr << "Using Unimplmented concat -> make sure concat is called with forwardDict(x)\n";
            return {};
        }
    };

    template<int64_t Axis1>
    struct SqueezeAtAxisImpl : ModuleWithSizeInfoImpl{
        SqueezeAtAxisImpl()=default;
        explicit SqueezeAtAxisImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::squeeze(x, Axis1);
        }
    };

    template<int64_t _from, int64_t _to=-1>
    struct FlattenImpl : ModuleWithSizeInfoImpl{
        FlattenImpl()=default;
        explicit FlattenImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::flatten(x, _from, _to);
        }
    };

    template<int64_t Axis1>
    struct UnSqueezeAtAxisImpl : ModuleWithSizeInfoImpl{
        UnSqueezeAtAxisImpl()=default;
        explicit UnSqueezeAtAxisImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::unsqueeze(x, Axis1);
        }
    };

    template<int64_t Axis1, int64_t Axis2>
    struct TransposeImpl : ModuleWithSizeInfoImpl{
        TransposeImpl()=default;
        explicit TransposeImpl(BaseModuleOption opt): ModuleWithSizeInfoImpl(opt){}
        inline torch::Tensor forward(const torch::Tensor &x) noexcept final {
            return torch::transpose(x, Axis1, Axis2);
        }
    };

    template<class ImplType>
    struct ConditionalImpl : ModuleWithSizeInfoImpl{

        std::shared_ptr<ImplType> impl = nullptr;
        bool (*condition) (torch::Tensor const&);

        explicit ConditionalImpl(bool (*condition) (torch::Tensor const&) ):condition(condition){}
        explicit ConditionalImpl(BaseModuleOption opt,
                                 bool (*condition) (torch::Tensor const&) ):ModuleWithSizeInfoImpl(opt),
                                 condition(condition){}

        inline torch::Tensor forward(const torch::Tensor &x) noexcept override {
            return condition(x) ? impl->forward(x) : x;
        }
    };

    template<int64_t Axis, int64_t Dim>
    struct ExpandIfDimEqualImpl : ConditionalImpl< UnSqueezeAtAxisImpl<Axis> >{

        explicit ExpandIfDimEqualImpl(BaseModuleOption opt):
        ConditionalImpl< UnSqueezeAtAxisImpl<Axis> >(opt, [](torch::Tensor const& x) -> bool {
            return x.dim() == Dim;
        }){}

        explicit ExpandIfDimEqualImpl():ConditionalImpl< UnSqueezeAtAxisImpl<Axis> >([](torch::Tensor const& x) -> bool {
            return x.dim() == Dim;
        }){
        }
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

        torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return drop_out(x);
        }
        TensorDict* forwardDict(TensorDict* x) noexcept override{
            x->insert_or_assign(m_Output, drop_out(x->at(m_Input))); return x;
        }
    };

    struct MaxPool2DImpl: public ModuleWithSizeInfoImpl{
        torch::nn::MaxPool2d pool_2d{nullptr};
        torch::nn::ZeroPad2d m_padding{nullptr};

        bool flatten_out;
    public:
        explicit MaxPool2DImpl(CNNOption opt):ModuleWithSizeInfoImpl(opt){

            auto _opt = torch::nn::MaxPool2dOptions(opt.kernels[0]).stride(opt.strides[0]);
            pool_2d = register_module("pool_2d", torch::nn::MaxPool2d(_opt));

            auto size_fn = [_opt](int s, int i) {
                return (
                        (s + 2*_opt.padding()->at(i) - _opt.dilation()->at(i) * (_opt.kernel_size()->at(i) - 1) - 1 )
                / _opt.stride()->at(i)) + 1; };
            auto _in_shape = opt.InputShape();

            if(opt.flatten_output)
                m_OutputSize = {  size_fn(_in_shape.height, 1) * size_fn(_in_shape.width, 0) * _in_shape.channel   };
            else if(not opt.padding.empty() and  opt.padding[0] == "same"){

                auto valid_w = size_fn(_in_shape.height, 1);
                auto valid_h = size_fn(_in_shape.width, 0);

                int w = floorl( (_in_shape.width - 1) / _opt.stride()->at(0)) + 1;
                int h = floorl( (_in_shape.height - 1) / _opt.stride()->at(1)) + 1;

                if(  valid_h < h and valid_w < w){
                    auto l_r = w-valid_w;
                    auto t_b = h-valid_h;
                    torch::nn::ZeroPad2dOptions pad_opt({l_r, l_r, t_b, t_b});
                    REGISTER_MODULE(m_padding,  torch::nn::ZeroPad2d(pad_opt) );
                    m_OutputSize = { _in_shape.channel, w, h};
                }else{
                    std::cerr << "Cant use Same padding for maxpool2d, resize\n";
                    m_OutputSize = { _in_shape.channel, valid_h, valid_w};
                }
            }
            else
                m_OutputSize = { _in_shape.channel, size_fn(_in_shape.width, 0), size_fn(_in_shape.height, 1)};

            flatten_out = opt.flatten_output;

        }

        inline torch::Tensor forward(torch::Tensor const& x) noexcept override{
            auto out = m_padding ? m_padding(x) : x;
            return flatten_out ? torch::flatten( pool_2d(out) ) : pool_2d(out);
        }
        inline TensorDict* forwardDict(TensorDict* x) noexcept override{
            x->insert_or_assign(m_Output, forward(x->at(m_Input))); return x;
        }
    };

    TORCH_MODULE(LayerNorm1d);
    TORCH_MODULE(Dropout);
    TORCH_MODULE(MaxPool2D);

    using ConditionalSqueezeAtAxis0Impl = ConditionalImpl<SqueezeAtAxisImpl<0>>;
    using ConditionalSqueezeAtAxis1Impl = ConditionalImpl<SqueezeAtAxisImpl<1>>;
    using ConditionalSqueezeAtAxis2Impl = ConditionalImpl<SqueezeAtAxisImpl<2>>;
    using ConditionalUnSqueezeAtAxis0Impl = ConditionalImpl<UnSqueezeAtAxisImpl<0>>;
    using ConditionalUnSqueezeAtAxis1Impl = ConditionalImpl<UnSqueezeAtAxisImpl<1>>;
    using ConditionalUnSqueezeAtAxis2Impl = ConditionalImpl<UnSqueezeAtAxisImpl<2>>;

    using ExpandAtAxis0IfDimEqual1Impl = ExpandIfDimEqualImpl<0, 1>;
    using ExpandAtAxis0IfDimEqual2Impl = ExpandIfDimEqualImpl<0, 2>;
    using ExpandAtAxis0IfDimEqual3Impl = ExpandIfDimEqualImpl<0, 3>;

    using Transpose01Impl = TransposeImpl<0, 1>;
    using Transpose12Impl = TransposeImpl<1, 2>;
    using Transpose23Impl = TransposeImpl<2, 3>;

    using ConcatEndImpl = JoinAtAxisImpl<true>;
    using Concat0Impl = JoinAtAxisImpl<true, 0>;
    using Concat1Impl = JoinAtAxisImpl<true, 1>;
    using Concat2Impl = JoinAtAxisImpl<true, 2>;
    using Concat3Impl = JoinAtAxisImpl<true, 3>;

    using StackEndImpl = JoinAtAxisImpl<true>;
    using Stack0Impl = JoinAtAxisImpl<false, 0>;
    using Stack1Impl = JoinAtAxisImpl<false, 1>;
    using Stack2Impl = JoinAtAxisImpl<false, 2>;
    using Stack3Impl = JoinAtAxisImpl<false, 3>;

    using FlattenEndImpl = FlattenImpl<0, -1>;
    using Flatten01Impl = FlattenImpl<0, 1>;

    TORCH_MODULE(ConditionalSqueezeAtAxis0);
    TORCH_MODULE(ConditionalSqueezeAtAxis1);
    TORCH_MODULE(ConditionalUnSqueezeAtAxis0);
    TORCH_MODULE(ConditionalUnSqueezeAtAxis1);
    TORCH_MODULE(ExpandAtAxis0IfDimEqual1);
    TORCH_MODULE(ExpandAtAxis0IfDimEqual2);
    TORCH_MODULE(ExpandAtAxis0IfDimEqual3);

    TORCH_MODULE(Transpose01);
    TORCH_MODULE(Transpose12);
    TORCH_MODULE(Transpose23);

    TORCH_MODULE(ConcatEnd);
    TORCH_MODULE(Concat0);
    TORCH_MODULE(Concat1);
    TORCH_MODULE(Concat2);
    TORCH_MODULE(Concat3);

    TORCH_MODULE(FlattenEnd);
    TORCH_MODULE(Flatten01);

    TORCH_MODULE(StackEnd);
    TORCH_MODULE(Stack0);
    TORCH_MODULE(Stack1);
    TORCH_MODULE(Stack2);
    TORCH_MODULE(Stack3);

}

SAM_OPTIONS(BaseModuleOption, LayerNorm1dImpl::Option, SELF(axis));
SAM_OPTIONS(BaseModuleOption, DropoutImpl::Option, SELF(prob));

#endif //DEEP_NETWORKS_BASIC_H
