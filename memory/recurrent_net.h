//
// Created by dewe on 9/25/21.
//

#ifndef SAMFRAMEWORK_RECURRENT_NET_H
#define SAMFRAMEWORK_RECURRENT_NET_H

#include "base.h"
#include "common/common.h"
#include "string"

using namespace std::string_literals;

namespace sam_dn{

    template<typename StateType, typename MemoryType, class OptionType, bool batchFirst, char type>
    class RecurrentNetImpl  : public BaseModuleImpl<RecurrentNetOption, MemoryType, StateType, batchFirst>{

        inline std::string fullType() const {
            if constexpr(batchFirst)
                return type == 'l' ? "bf_lstm" : type == 'g' ? "bf_gru" : "bf_rnn";
            else{
                return type == 'l' ? "tf_lstm" : type == 'g' ? "tf_gru" : "tf_rnn";
            }
        }

    protected:
        int ctr = 0 ;
        std::string hidden_out_key;
        int64_t baseBatchSz;
        std::queue<StateType> snapShot;
        torch::Device device;

    public:
        explicit RecurrentNetImpl(RecurrentNetOption opt):
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

        inline void initialStateWithBatchSizeAndDevice(int batch_size, torch::Device device) noexcept{
            auto shape = std::vector<int64_t>{this->opt.num_layers, batch_size, this->opt.hidden_size};

            if constexpr (std::is_same_v<StateType, TensorTuple>){
                this->m_States  = {torch::zeros(shape, device), torch::zeros(shape, device)};
            }else{
                this->m_States   = torch::zeros(shape, device);
            }
        }

        inline void initialStateWithBatchSize(int batch_size) noexcept{
            auto shape = std::vector<int64_t>{this->opt.num_layers, batch_size, this->opt.hidden_size};

            if constexpr (std::is_same_v<StateType, TensorTuple>){
                this->m_States  = { torch::zeros(shape, this->device), torch::zeros(shape, this->device) };
            }else{
                this->m_States   = torch::zeros(shape, this->device);
            }
        }

        void batchSize(int sz){
            this->opt.batch_size = sz;
        }

        void defaultBatchSize(){
            this->opt.batch_size = baseBatchSz;
        }

        auto SnapShot() const { return snapShot; }

        inline void cacheHiddenState() noexcept{
            snapShot.push( clone_states() );
        }

        inline void resetState() noexcept{
            this->m_States = snapShot.front();
            snapShot.pop();
        }

        inline auto clone_states() noexcept{
            if constexpr (std::is_same_v<StateType, TensorTuple>){
                auto state1 = std::get<0>(this->m_States);
                auto state2 = std::get<1>(this->m_States);
                return std::make_tuple( std::move(state1.data()), std::move(state2.data()));
            }else{
                return std::move(this->m_States.data());
            }
        }

        inline auto zero_states(int _batch_size) noexcept{
            if constexpr (std::is_same_v<StateType, TensorTuple>){
                auto&& state1 = torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
                auto&& state2 = torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
                return std::make_tuple( std::move(state1), std::move(state2));
            }else{
                return torch::zeros( {this->opt.num_layers, _batch_size, this->opt.hidden_size}, device );
            }
        }

        inline auto getHiddenKey() const{
            return hidden_out_key;
        }
    };

    template<typename StateType, typename MemoryType, class OptionType, bool batchFirst, char type>
    class RLRecurrentNetImpl  : public RecurrentNetImpl<StateType, MemoryType, OptionType, batchFirst, type> {

        int counter{};
        std::conditional_t<type == 'l', std::pair<std::string, std::string>, std::string> hidden_state_key{};
        int num_layers = 0, hidden_size=0;

        auto pass(torch::Tensor const& _mask, torch::Tensor const& x, TensorDict const& hxs){
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

    public:
        explicit RLRecurrentNetImpl(RecurrentNetOption opt):
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

        inline auto size_hx(int axis=1){
            if  constexpr(type == 'l'){
                return std::get<0>(this->m_States).size(axis);
            } else
                return this->m_States.size(axis);
        }

        inline auto size_hx(TensorDict* x, int axis){
            if  constexpr(type == 'l'){
                return x->at( std::get<0>( hidden_state_key ) ).size(axis);
            } else
                return x->at( hidden_state_key ).size(axis);
        }

        inline auto terminatedTransitionsIndicesFromAnyWorker( torch::Tensor const& env_mask, int T ) const noexcept{
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

        inline auto rnnScores(std::vector<int64_t>  const& terminalTransitionIndices,
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

        inline void fillState(TensorDict* x) const noexcept{
            // flattent hxs incase of num_layer > 1
            if constexpr(type == 'l'){
                x->template insert_or_assign(hidden_state_key.first, std::get<0>(this->m_States).flatten(0, 1).data());
                x->template insert_or_assign(hidden_state_key.second, std::get<1>(this->m_States).flatten(0, 1).data());
            }else{
                x->template insert_or_assign(hidden_state_key, this->m_States.flatten(0, 1).data());
            }
        }

        TensorDict * forwardDict(TensorDict *x) noexcept override{

            auto const& input = x->at(this->m_Input);
            auto const& env_mask = x->at("mask");
            torch::Tensor out;

            if (input.size(0) == size_hx(x, 0)) {
                out = pass( env_mask.unsqueeze(0), input.unsqueeze(0), *x).squeeze(0);
            }
            else {
                const auto L = input.size(-1);
                const auto N = size_hx(x, 0);
                const auto T = int( input.size(0) / N );
                out = input.view({T, N, L});

                auto [indices, mask_vec] = terminatedTransitionsIndicesFromAnyWorker( env_mask, T );

                auto output = rnnScores(indices, mask_vec, *x, out);

                out = torch::cat(output, 0);
                out = out.view({T * N, -1});
            }
            x->template insert_or_assign(this->m_Output, out.squeeze(0));
            fillState(x);

            return x;
        }
    };

    using LSTMBatchFirstImpl = RecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, true, 'l'>;
    using LSTMTimeFirstImpl = RecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, false, 'l'>;
    using GRUBatchFirstImpl = RecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, true, 'g'>;
    using GRUTimeFirstImpl = RecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, false, 'g'>;
    using RNNBatchFirstImpl = RecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, true, 'r'>;
    using RNNTimeFirstImpl = RecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, false, 'r'>;

    using RLLSTMTimeFirstImpl = RLRecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, false, 'l'>;
    using RLGRUTimeFirstImpl = RLRecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, false, 'g'>;
    using RLRNNTimeFirstImpl = RLRecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, false, 'r'>;

    TORCH_MODULE(LSTMBatchFirst);
    TORCH_MODULE(LSTMTimeFirst);
    TORCH_MODULE(GRUBatchFirst);
    TORCH_MODULE(GRUTimeFirst);
    TORCH_MODULE(RNNBatchFirst);
    TORCH_MODULE(RNNTimeFirst);
}

#endif //SAMFRAMEWORK_RECURRENT_NET_H
