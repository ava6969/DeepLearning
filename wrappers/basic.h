//
// Created by dewe on 12/12/21.
//

#ifndef DEEP_NETWORKS_BASIC_H
#define DEEP_NETWORKS_BASIC_H

#include "base.h"
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

        explicit JoinAtAxisImpl(AxisOption opt): ModuleWithSizeInfoImpl(opt){
            boost::split(inputs, this->m_Input, boost::is_any_of(";"));
            input_tensors.resize(inputs.size());

            if constexpr(joiner != Joiner::Vstack )
                Axis1 = opt.axis[0];
            if constexpr( joiner == Joiner::Cat )
                m_OutputSize = { std::accumulate(inputs.begin(), inputs.end(), 0L, [&opt](auto accum, auto k){
                    return accum + opt.dict_opt[k][0];
                }) };
            else{
                auto front = opt.dict_opt[inputs[0]];
                if( std::all_of(inputs.begin(), inputs.end(), [&](auto const& k){
                    return opt.dict_opt[k] == front;
                })){
                    m_OutputSize = front;
                }else{
                    throw std::runtime_error("JoinAtAxisImplError: Cannot stack dims of unequal shape");
                }
            }
        }

        inline TensorDict * forwardDict(TensorDict *x)  noexcept final {
            std::transform(inputs.begin(), inputs.end(), input_tensors.begin(), [&x](auto const& in) {
                return x->at(in);
            });
            if constexpr(joiner == Joiner::Cat )
                x->template insert_or_assign( this->m_Output, torch::cat(input_tensors, Axis1));
            else{
                if (joiner == Joiner::Vstack ){
                    x->template insert_or_assign( this->m_Output, torch::vstack(input_tensors));
                }else{
                    x->template insert_or_assign( this->m_Output, torch::stack(input_tensors, Axis1));
                }
            }
            return x;
        }

        torch::Tensor forward(const torch::Tensor &x) noexcept final{
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

        explicit SplitAndStackImpl(Option opt): ModuleWithSizeInfoImpl(opt),
                                                    splitAxis(opt.split_axis),
                                                    stackAxis(opt.stack_axis),
                                                    splitSize(opt.split_size){
            LOG(WARNING) << "SplitAndStackImpl output size cannot be correctly calculated\n";
        }

        inline torch::Tensor forward(const torch::Tensor &x) noexcept final{
            return torch::stack( torch::tensor_split(x, splitSize, splitAxis) , stackAxis );
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
    TORCH_MODULE(ExpandIfDimEqual);
    TORCH_MODULE(Flatten);
    TORCH_MODULE(SplitAndStack);
    TORCH_MODULE_IMPL(Stack, JoinAtAxisImpl<Joiner::Stack>);
    TORCH_MODULE_IMPL(Concat, JoinAtAxisImpl<Joiner::Cat>);
    TORCH_MODULE_IMPL(Vstack, JoinAtAxisImpl<Joiner::Vstack>);
    TORCH_MODULE(SqueezeAtAxis);
    TORCH_MODULE(Transpose);

}

//namespace YAML {
//    template<> struct convert<sam_dn::AxisOption> {
//        static bool decode(YAML::Node &x, sam_dn::AxisOption &out) {
//            out.axis = x["axis"].as<std::vector<int64_t>>();
//            return true;
//        }
//    };
//}

SAM_OPTIONS(BaseModuleOption, LayerNorm1dImpl::Option, SELF(axis));
SAM_OPTIONS(BaseModuleOption, DropoutImpl::Option, SELF(prob));
SAM_OPTIONS(BaseModuleOption, AxisOption, SELF(axis));
SAM_OPTIONS(BaseModuleOption, SplitAndStackImpl::Option, SELF(split_size), SELF(stack_axis), SELF(split_axis));
#endif //DEEP_NETWORKS_BASIC_H
