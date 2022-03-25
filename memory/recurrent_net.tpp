//
// Created by dewe on 3/24/22.
//


namespace sam_dn{

    RL_REC_T
    std::string REC_IMPL_T::fullType() const {
        if constexpr(batchFirst)
            return type == 'l' ? "bf_lstm" : type == 'g' ? "bf_gru" : "bf_rnn";
        else{
            return type == 'l' ? "tf_lstm" : type == 'g' ? "tf_gru" : "tf_rnn";
        }
    }

    RL_REC_T
    REC_IMPL_T::RecurrentNetImpl(RecurrentNetOption opt):
            BaseModuleImpl<RecurrentNetOption, MemoryType, StateType, batchFirst>(opt),
            device(opt.device){

        opt.type = fullType() + std::to_string(ctr);
        hidden_out_key = opt.type;
        OptionType implOpt(opt.InputSize(), opt.hidden_size);

        implOpt.dropout(opt.drop_out).bias(opt.new_bias).num_layers(opt.num_layers).batch_first(batchFirst);

        this->m_BaseModel = this->register_module(opt.type, MemoryType(implOpt));

        this->m_OutputSize = std::vector<int64_t>{opt.hidden_size};
        initializeWeightBias(this->m_BaseModel, opt);

        device = this->m_BaseModel->all_weights()[0].device();

        initialStateWithBatchSizeAndDevice(opt.batch_size, device);
        opt.batch_first = batchFirst;
        baseBatchSz = opt.batch_size;
        ctr++;
    }

    RL_REC_T
    void REC_IMPL_T::initialStateWithBatchSizeAndDevice(int batch_size, torch::Device _device) noexcept{
        auto shape = std::vector<int64_t>{this->opt.num_layers, batch_size, this->opt.hidden_size};

        if constexpr (std::is_same_v<StateType, TensorTuple>){
            this->m_States  = {torch::zeros(shape, _device), torch::zeros(shape, _device)};
        }else{
            this->m_States   = torch::zeros(shape, _device);
        }
    }

    RL_REC_T
    void REC_IMPL_T::initialStateWithBatchSize(int batch_size) noexcept{
        auto shape = std::vector<int64_t>{this->opt.num_layers, batch_size, this->opt.hidden_size};

        if constexpr (std::is_same_v<StateType, TensorTuple>){
            this->m_States  = { torch::zeros(shape, this->device), torch::zeros(shape, this->device) };
        }else{
            this->m_States   = torch::zeros(shape, this->device);
        }
    }
    RL_REC_T
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

    RL_REC_T StateType REC_IMPL_T::clone_states() noexcept{
        if constexpr (std::is_same_v<StateType, TensorTuple>){
            auto state1 = std::get<0>(this->m_States);
            auto state2 = std::get<1>(this->m_States);
            return std::make_tuple( std::move(state1.data()), std::move(state2.data()));
        }else{
            return std::move(this->m_States.data());
        }
    }

    RL_REC_T
    StateType REC_IMPL_T::zero_states(int _batch_size) noexcept{
        if constexpr (std::is_same_v<StateType, TensorTuple>){
            auto&& state1 = torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
            auto&& state2 = torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
            return std::make_tuple( std::move(state1), std::move(state2));
        }else{
            return torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
        }
    }

    RL_REC_T
    torch::Tensor RL_REC_IMPL_T::pass(torch::Tensor const& _mask, torch::Tensor const& x, TensorDict const& hxs){
        StateType maskedState;
        torch::Tensor out;

        if constexpr(type == 'l'){
            maskedState = std::make_tuple(hxs.at(hidden_state_key.first).view({num_layers, -1, hidden_size}) * _mask,
                                          hxs.at(hidden_state_key.second).view({num_layers, -1, hidden_size}) * _mask);
        } else{
            maskedState =  hxs.at(hidden_state_key) * _mask;
        }

        std::tie(out, this->m_States) = this->m_BaseModel(x, maskedState);
        return out;
    }

    RL_REC_T
    RL_REC_IMPL_T::RLRecurrentNetImpl(RecurrentNetOption opt):
            RecurrentNetImpl<StateType, MemoryType, OptionType, batchFirst, type>(opt){
        auto id = counter;
        if constexpr(type == 'l') {
            hidden_state_key = {"__lstm_h"s.append(std::to_string(id)), "__lstm_c"s.append(std::to_string(id))};
        }else{
            hidden_state_key = "__gru_h"s.append(std::to_string(id));
        }
        num_layers = opt.num_layers;
        hidden_size = opt.hidden_size;
        if(this->m_Input.empty())
            this->m_Input = "observation";
    }

    RL_REC_T
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

    RL_REC_T
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

    RL_REC_T
    void RL_REC_IMPL_T::fillState(TensorDict* x) const noexcept{
        // flattent hxs incase of num_layer > 1
        if constexpr(type == 'l'){
            x->template insert_or_assign(hidden_state_key.first, std::get<0>(this->m_States).flatten(0, 1).data());
            x->template insert_or_assign(hidden_state_key.second, std::get<1>(this->m_States).flatten(0, 1).data());
        }else{
            x->template insert_or_assign(hidden_state_key, this->m_States.flatten(0, 1).data());
        }
    }

    RL_REC_T
    TensorDict* RL_REC_IMPL_T::forwardDict(TensorDict *x) noexcept {

        auto const &input = x->at(this->m_Input);
        auto const &env_mask = x->at("mask");
        torch::Tensor out;

        if (input.size(0) == size_hx(x, 0)) {
            out = pass(env_mask.unsqueeze(0), input.unsqueeze(0), *x).squeeze(0);
        } else {
            const auto L = input.size(-1);
            const auto N = size_hx(x, 0);
            const auto T = int(input.size(0) / N);
            out = input.view({T, N, L});

            auto[indices, mask_vec] = terminatedTransitionsIndicesFromAnyWorker(env_mask, T);

            auto output = rnnScores(indices, mask_vec, *x, out);

            out = torch::cat(output, 0);
            out = out.view({T * N, -1});
        }
        x->template insert_or_assign(this->m_Output, out.squeeze(0));
        fillState(x);

        return x;
    }
}