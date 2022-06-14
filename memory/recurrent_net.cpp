//
// Created by dewe on 3/24/22.
//

#include "recurrent_net.h"

namespace sam_dn{

    RecurrentNetTemplate
    std::string REC_IMPL_T::description() const {
        if constexpr(batchFirst)
            return type == 'l' ? "bf_lstm" : type == 'g' ? "bf_gru" : "bf_rnn";
        else{
            return type == 'l' ? "tf_lstm" : type == 'g' ? "tf_gru" : "tf_rnn";
        }
    }

    RecurrentNetTemplate
    REC_IMPL_T::RecurrentNetImpl(RecurrentNetOption opt):
            BaseModuleImpl<RecurrentNetOption, MemoryType, StateType, batchFirst>(opt)
                    ,device(opt.device)
            {

        opt.type = description() + std::to_string( opt.id.template value_or(instance_count) );
        instance_id = opt.type;

        OptionType implOpt(opt.InputSize(), opt.hidden_size);
        implOpt.dropout(opt.drop_out).bias(opt.new_bias).num_layers(opt.num_layers).batch_first(batchFirst);
        this->m_BaseModel = this->register_module(opt.type, MemoryType(implOpt));

        this->m_OutputSize = std::vector<int64_t>{opt.hidden_size};
        initializeWeightBias(this->m_BaseModel, opt);

        device = this->m_BaseModel->all_weights()[0].device();

        initialStateWithBatchSizeAndDevice(opt.batch_size, device);
        opt.batch_first = batchFirst;
        baseBatchSz = opt.batch_size;
        instance_count++;
    }

    RecurrentNetTemplate
    void REC_IMPL_T::initialStateWithBatchSizeAndDevice(int batch_size, torch::Device _device) noexcept{
        auto shape = std::vector<int64_t>{this->opt.num_layers, batch_size, this->opt.hidden_size};

        if constexpr (std::is_same_v<StateType, TensorTuple>){
            this->m_States  = {torch::zeros(shape, _device), torch::zeros(shape, _device)};
        }else{
            this->m_States   = torch::zeros(shape, _device);
        }
    }

    RecurrentNetTemplate
    void REC_IMPL_T::initialStateWithBatchSize(int batch_size) noexcept{
        auto shape = std::vector<int64_t>{this->opt.num_layers, batch_size, this->opt.hidden_size};

        if constexpr (std::is_same_v<StateType, TensorTuple>){
            this->m_States  = { torch::zeros(shape, this->device), torch::zeros(shape, this->device) };
        }else{
            this->m_States   = torch::zeros(shape, this->device);
        }
    }
    RecurrentNetTemplate
    void REC_IMPL_T::to(torch::Device _device, bool non_blocking) {
        this->m_BaseModel->to(_device, non_blocking);
        this->device = _device;
        if constexpr(type == 'l') {
            std::get<0>(this->m_States).to(_device, non_blocking);
            std::get<1>(this->m_States).to(_device, non_blocking);
        } else {
            this->m_States.to(_device, non_blocking);
        }
    }

    RecurrentNetTemplate StateType REC_IMPL_T::clone_states() noexcept{
        if constexpr (std::is_same_v<StateType, TensorTuple>){
            auto state1 = std::get<0>(this->m_States);
            auto state2 = std::get<1>(this->m_States);
            return std::make_tuple( std::move(state1.data()), std::move(state2.data()));
        }else{
            return std::move(this->m_States.data());
        }
    }

    RecurrentNetTemplate void REC_IMPL_T::clone_states(std::unordered_map<std::string, std::pair< torch::Tensor, ModuleWithSizeInfoImpl*>>& x) noexcept{
        if constexpr (std::is_same_v<StateType, TensorTuple>){
            x[this->instance_id + "_hx"] = { std::get<0>(this->m_States).data(), this};
            x[this->instance_id + "_cx"] = { std::get<1>(this->m_States).data(), this};
        }else{
            x[instance_id] = { this->m_States.data(), this};
        }
    }

    RecurrentNetTemplate
    torch::Tensor REC_IMPL_T::zero_states(int _batch_size) noexcept{
        return torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
    }

    RecurrentNetTemplate
    torch::Tensor RL_REC_IMPL_T::pass(torch::Tensor const& _mask, torch::Tensor const& x, TensorDict const& hxs){
        StateType maskedState;
        torch::Tensor out;

        if constexpr(type == 'l'){

            auto&& hx = hxs.at(this->instance_id + "_hx").view({num_layers, -1, hidden_size});
            auto&& cx = hxs.at(this->instance_id + "_cx").view({num_layers, -1, hidden_size});
            maskedState = reset_states ? std::make_tuple( hx * _mask, cx * _mask) : std::make_tuple( hx, cx);
        } else{
            auto&& hx = hxs.at(this->instance_id).view({num_layers, -1, hidden_size});
            maskedState = reset_states ? hx * _mask : hx;

        }
        std::tie(out, this->m_States) = this->m_BaseModel(x, maskedState);

        return out;
    }

    RecurrentNetTemplate
    RL_REC_IMPL_T::RLRecurrentNetImpl(RecurrentNetOption opt):
            RecurrentNetImpl<StateType, MemoryType, OptionType, batchFirst, type>(opt){
        num_layers = opt.num_layers;
        hidden_size = opt.hidden_size;
        reset_states = opt.reset_hidden;
        if(this->m_Input.empty())
            this->m_Input = "observation";
    }

    RecurrentNetTemplate
    std::pair<std::vector<int64_t>, std::vector<torch::Tensor>>
    RL_REC_IMPL_T::terminatedTransitionsIndicesFromAnyWorker( torch::Tensor const& env_mask, int T ) const noexcept{
        std::vector<int64_t> terminalTransitionIndices{0};
        int  i = 1;
        auto _mask = env_mask.unbind();
        std::for_each(_mask.begin() + 1, _mask.begin() + T, [&terminalTransitionIndices, &i] (torch::Tensor & _in){
            if( torch::any(_in.view(-1) == 0.0, -1).sum(0).template item<int>() > 0 )
                terminalTransitionIndices.push_back(i);
            i++;
        });
        terminalTransitionIndices.push_back(T);
        return std::make_pair(terminalTransitionIndices, _mask);
    }

    RecurrentNetTemplate
    std::vector<torch::Tensor> RL_REC_IMPL_T::rnnScores(std::vector<int64_t>  const& terminalTransitionIndices,
                          std::vector<torch::Tensor>  const& mask_vec,
                          TensorDict const& x,
                          torch::Tensor const& input) noexcept{
        torch::Tensor rnn_scores;
        std::vector<torch::Tensor> output;
        output.reserve(terminalTransitionIndices.size() - 1);
        std::transform(begin(terminalTransitionIndices), end(terminalTransitionIndices) - 1,
                       begin(terminalTransitionIndices) + 1,
                       std::back_inserter(output),
                       [this, x, mask_vec , input](int64_t start_index, int64_t end_index)  {
                           return pass( mask_vec.at(start_index).view({1, -1, 1}),
                                        input.slice(0, start_index, end_index), x );
                       });
        return output;
    }

    RecurrentNetTemplate
    void RL_REC_IMPL_T::fillState(TensorDict* x) const noexcept{
        // flattent hxs incase of num_layer > 1
        if constexpr(type == 'l'){

            x->template insert_or_assign(this->instance_id + "_hx", std::get<0>(this->m_States).flatten(0, 1).data());
            x->template insert_or_assign(this->instance_id + "_cx", std::get<1>(this->m_States).flatten(0, 1).data());
        }else{
            x->template insert_or_assign(this->instance_id , this->m_States.flatten(0, 1).data());

        }
    }

    RecurrentNetTemplate
    TensorDict* RL_REC_IMPL_T::forwardDict(TensorDict *x) noexcept {

        auto const &input = x->at(this->m_Input);
        auto const &env_mask = reset_states ? x->at("mask") : torch::Tensor{};
        torch::Tensor out;
        if (input.size(0) == size_hx(x, 0) ) {
            out = pass( reset_states ? env_mask.unsqueeze(0) : env_mask,
                        input.unsqueeze(0), *x).squeeze(0);
        }
        else {
            const auto L = input.size(-1);
            const auto N = this->size_hx(x, 0);
            const auto T = int(input.size(0) / N);
            out = input.view({T, N, L});

            if( not ReturnAllSeq::set() ){
                if(reset_states){
                    auto[indices, mask_vec] = terminatedTransitionsIndicesFromAnyWorker(env_mask, T);
                    auto output = rnnScores(indices, mask_vec, *x, out);
                    out = torch::cat(output, 0);
                }else{
                    out = pass(env_mask, out, *x).squeeze(0);
                }

                out = out.view({T * N, -1});
            }else{
                std::conditional_t<type == 'l', std::array< std::vector<torch::Tensor>, 2>,
                        std::vector<torch::Tensor> > all_state;

                torch::Tensor result;
                for(int t = 0; t < T; t++){
                    result = pass(env_mask[t].unsqueeze(0), out[t].unsqueeze(0), *x);
                    if constexpr(type == 'l' ){
                        all_state[0].push_back( std::get<0>(this->m_States).data() );
                        all_state[1].push_back( std::get<1>(this->m_States).data()  );
                    }else{
                        all_state.push_back( this->m_States.data()  );
                    }
                    fillState(x);
                }
                out = result;

                if constexpr(type == 'l' ){
                    this->m_States = TensorTuple ( torch::stack(all_state[0], -1), torch::stack(all_state[1], -1) );
                }else{
                    this->m_States = torch::stack(all_state[0], -1);
                }

            }
        }
        x->template insert_or_assign(this->m_Output, out.squeeze(0));
        fillState(x);

        return x;
    }

    template class RecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, true, 'l'>;
    template class RecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, false, 'l'>;
    template class RecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, true, 'g'>;
    template class RecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, false, 'g'>;
    template class RecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, true, 'r'>;
    template class RecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, false, 'r'>;
    template class RLRecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, false, 'l'>;
    template class RLRecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, false, 'g'>;
    template class RLRecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, false, 'r'>;

}