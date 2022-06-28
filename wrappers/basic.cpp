//
// Created by dewe on 5/27/22.
//

#include "basic.h"

namespace sam_dn{

    SplitAndStackImpl::SplitAndStackImpl(Option opt): ModuleWithSizeInfoImpl(opt),
    splitAxis(opt.split_axis),
    stackAxis(opt.stack_axis),
    splitSize(opt.split_size){
        LOG(WARNING) << "SplitAndStackImpl output size cannot be correctly calculated\n";
    }

    template<Joiner joiner>
    JoinAtAxisImpl<joiner>::JoinAtAxisImpl(AxisOption opt): ModuleWithSizeInfoImpl(opt){
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

    template<Joiner joiner>
     TensorDict * JoinAtAxisImpl<joiner>::forwardDict(TensorDict *x)  noexcept {
         std::transform(inputs.begin(), inputs.end(), input_tensors.begin(), [&x](auto const &in) {
             return x->at(in);
         });

        if constexpr(joiner == Joiner::Cat)
             x->template insert_or_assign(this->m_Output, torch::cat(input_tensors, Axis1));
         else {
             if (joiner == Joiner::Vstack) {
                 x->template insert_or_assign(this->m_Output, torch::vstack(input_tensors));
             } else {
                 x->template insert_or_assign(this->m_Output, torch::stack(input_tensors, Axis1));
             }
         }
         return x;
     }

    MaxPool2DImpl::MaxPool2DImpl(CNNOption opt):ModuleWithSizeInfoImpl(opt){

            auto _opt = torch::nn::MaxPool2dOptions(opt.kernels[0]).stride(opt.strides[0]);

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

                int w = int( std::floor( float(_in_shape.width - 1) / float(_opt.stride()->at(0)) ) ) + 1;
                int h = int( std::floor( float(_in_shape.height - 1) / float(_opt.stride()->at(1)) ) ) + 1;

                if(  valid_h < h and valid_w < w){
                    auto l_r = w-valid_w;
                    auto t_b = h-valid_h;
                    _opt = _opt.padding({l_r, t_b});
                    m_OutputSize = { _in_shape.channel, w, h};
                }else{
                    std::cerr << "Cant use Same padding for maxpool2d, resize\n";
                    m_OutputSize = { _in_shape.channel, valid_h, valid_w};
                }
            }
            else
            m_OutputSize = { _in_shape.channel, size_fn(_in_shape.width, 0), size_fn(_in_shape.height, 1)};

            pool_2d = register_module("pool_2d", torch::nn::MaxPool2d(_opt));
            flatten_out = opt.flatten_output;

    }

    template class JoinAtAxisImpl<Joiner::Stack>;
    template class JoinAtAxisImpl<Joiner::Cat>;
    template class JoinAtAxisImpl<Joiner::Vstack>;
}