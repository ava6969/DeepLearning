//
// Created by dewe on 3/24/22.
//

#include <basic/embedding.h>
#include <basic/fcnn.h>
#include <memory/recurrent_net.h>
#include "base.h"

namespace sam_dn{

    BASEMODULE_IMPL_T
    BASEMODULET::BaseModuleImpl(BaseModuleT const& impl) : m_BaseModel( impl ) {
        if constexpr( std::is_same_v<BaseModuleT, torch::nn::Sequential> ){

            auto modules = static_cast<torch::nn::Sequential>(impl)->modules(); // TODO: change to .children()

            auto ptr = std::find_if(modules.begin(), modules.end(),
                                    [](const std::shared_ptr<torch::nn::Module>& module){
                                        return module->as<ModuleWithSizeInfoImpl>() != nullptr;
                                    });

            if(ptr != end(modules)){
                auto castedModule = std::dynamic_pointer_cast<ModuleWithSizeInfoImpl>(*ptr);
                this->m_Input = castedModule->input();
                this->m_Output = castedModule->output();
            }
        }
    }

    BASEMODULE_IMPL_T
    torch::Tensor BASEMODULET::forward(torch::Tensor const& x) noexcept {

        if constexpr(std::is_same_v<StateType, NoState>) {
            auto y = m_BaseModel->forward(x);
            return y;
        } else {
            torch::Tensor out;
            if constexpr (BatchFirst)
                out = x.view({opt.batch_size, -1, x.size(-1)});
            else
                out = x.view({-1, opt.batch_size, x.size(-1)});

            std::tie(out, m_States) = this->m_BaseModel->forward(out, m_States);
            return opt.return_all_seq ? out.contiguous().flatten(0, 1) :
                   out.slice(int(BatchFirst), -1).contiguous().view({opt.batch_size, -1});
        }
    }

    BASEMODULE_IMPL_T
    TensorDict* BASEMODULET::forwardDict(TensorDict *x) noexcept {
        if constexpr(std::is_same_v<BaseModuleT, torch::nn::Sequential> and parseRecurseDict) {
            for (auto &module: this->m_BaseModel->children())
                if (auto _m = module->template as<ModuleWithSizeInfoImpl>()) {
                    x = _m->forwardDict(x);
                }
        } else if constexpr(std::is_same_v<StateType, NoState>) {
            x->insert_or_assign(m_Output,
                                this->m_BaseModel->forward(x->at(m_Input)));
        } else {
            std::tie(x->at(m_Output), m_States) =
                    this->m_BaseModel->forward(x->at(m_Input), m_States);
        }
        return x;
    }

    BASEMODULE_IMPL_T
    void BASEMODULET::initialState(StateType const& new_state) noexcept {
        if constexpr( std::is_same_v<StateType, NoState>) {
            return;
        }

        if constexpr (std::is_same_v<StateType, TensorTuple>){
            auto state1 = std::get<0>(new_state);
            auto state2 = std::get<1>(new_state);
            this->m_States = std::make_tuple(state1.data().view({this->opt.num_layers, -1, this->opt.hidden_size}),
                                             state2.data().view({this->opt.num_layers, -1, this->opt.hidden_size}));
        }else if constexpr( std::is_same_v<StateType, torch::Tensor> ){
            this->m_States = std::move(new_state.data().view({this->opt.num_layers, -1, this->opt.hidden_size}));
        }
    }

    template class BaseModuleImpl<sam_dn::EmbeddingOption, torch::nn::Embedding, sam_dn::NoState, false, false >;
    template class BaseModuleImpl<sam_dn::FCNNOption, torch::nn::Sequential, sam_dn::NoState, false, false>;

    template class BaseModuleImpl<RecurrentNetOption, torch::nn::RNN, at::Tensor, false, false>;
    template class BaseModuleImpl<RecurrentNetOption, torch::nn::LSTM, std::tuple<at::Tensor, at::Tensor>, false, false>;
    template class BaseModuleImpl<RecurrentNetOption, torch::nn::GRU, at::Tensor, false, false>;

    template class BaseModuleImpl<RecurrentNetOption, torch::nn::RNN, at::Tensor, true, false>;
    template class BaseModuleImpl<RecurrentNetOption, torch::nn::LSTM, std::tuple<at::Tensor, at::Tensor>, true, false>;
    template class BaseModuleImpl<RecurrentNetOption, torch::nn::GRU, at::Tensor, true, false>;

    template class BaseModuleImpl<sam_dn::BaseModuleOption, torch::nn::Sequential, sam_dn::NoState, false, true>;
    template class BaseModuleImpl<sam_dn::BaseModuleOption, torch::nn::Sequential, sam_dn::NoState, false, false>;
    template class BaseModuleImpl<sam_dn::CNNOption, torch::nn::Sequential, sam_dn::NoState, false, false>;

}