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
    class ReturnAllSeq{
        inline static bool flag{false};
    public:
        ReturnAllSeq(){
            flag = true;
        }
        ~ReturnAllSeq(){
            flag = false;
        }

        inline static bool set() { return flag; }
    };


#define RecurrentNetTemplate template<typename StateType, typename MemoryType, class OptionType, bool batchFirst, char type>
#define RL_REC_IMPL_T RLRecurrentNetImpl<StateType, MemoryType, OptionType, batchFirst, type>
#define REC_IMPL_T RecurrentNetImpl<StateType, MemoryType, OptionType, batchFirst, type>

    struct RecurrentNetOption : BaseModuleOption {
    public:
        int64_t hidden_size{512}, num_layers{1}, batch_size{};
        bool return_all_seq{false};
        bool batch_first{false};
        float drop_out{0};
        std::string device;
        std::string type;
        bool reset_hidden{true};
        std::optional<int> id{std::nullopt};

        BaseModuleOption& Input(std::vector<int64_t> const& rnn_in_size) override{
            assert(rnn_in_size.size() == 1);
            input_size = rnn_in_size[0];
            return *this;
        }

        RecurrentNetOption set_type(std::string const& _type){
            this->type = _type;
            return *this;
        }

        [[nodiscard]] auto SeqLength() const { return sequence_length; }
        [[nodiscard]] auto InputSize() const { return input_size; }

    private:
        int64_t input_size{}, sequence_length{};
    };

    template<typename StateType, typename MemoryType, class OptionType, bool batchFirst, char type>
    class RecurrentNetImpl  : public BaseModuleImpl<RecurrentNetOption, MemoryType, StateType, batchFirst>{

        [[nodiscard]] std::string description() const;

    protected:
        static inline int instance_count = 0 ;
        std::string instance_id;
        int64_t baseBatchSz;
        std::queue<StateType> snapShot;
        torch::Device device;

    public:
        explicit RecurrentNetImpl(RecurrentNetOption opt);

        void initialStateWithBatchSizeAndDevice(int batch_size, torch::Device device) noexcept;

        void initialStateWithBatchSize(int batch_size) noexcept;

        void to(torch::Device _device, bool non_blocking) override;

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

        void clone_states(std::unordered_map<std::string, std::pair< torch::Tensor, ModuleWithSizeInfoImpl*>>& ) noexcept override;

        StateType clone_states() noexcept;

        torch::Tensor zero_states(int _batch_size) noexcept override;

        inline auto getHiddenKey() const { return instance_id; }
    };

    template<typename StateType, typename MemoryType, class OptionType, bool batchFirst, char type>
    class RLRecurrentNetImpl  : public RecurrentNetImpl<StateType, MemoryType, OptionType, batchFirst, type> {

    public:
        explicit RLRecurrentNetImpl(RecurrentNetOption opt);

        inline auto size_hx(int axis=1){
            if  constexpr(type == 'l'){
                return std::get<0>(this->m_States).size(axis);
            } else
                return this->m_States.size(axis);
        }

        inline auto size_hx(TensorDict* x, int axis){
            if  constexpr(type == 'l'){
                return x->at( this->instance_id + "_hx" ).size(axis);
            } else
                return x->at( this->instance_id ).size(axis);
        }

        [[nodiscard]] std::pair<std::vector<int64_t>, std::vector<torch::Tensor>>
        terminatedTransitionsIndicesFromAnyWorker( torch::Tensor const& env_mask, int T ) const noexcept;

        std::vector<torch::Tensor> rnnScores(std::vector<int64_t>  const& terminalTransitionIndices,
                              std::vector<torch::Tensor>  const& mask_vec,
                              TensorDict const& x,
                              torch::Tensor const& input) noexcept;

        void fillState(TensorDict* x) const noexcept;

        TensorDict * forwardDict(TensorDict *x) noexcept override;

    private:

        int num_layers = 0, hidden_size=0;
        bool reset_states{false};
        torch::Tensor pass(torch::Tensor const& _mask, torch::Tensor const& x, TensorDict const& hxs);
    };

    extern template class RecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, true, 'l'>;
    extern template class RecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, false, 'l'>;
    extern template class RecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, true, 'g'>;
    extern template class RecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, false, 'g'>;
    extern template class RecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, true, 'r'>;
    extern template class RecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, false, 'r'>;
    extern template class RLRecurrentNetImpl<TensorTuple , torch::nn::LSTM, torch::nn::LSTMOptions, false, 'l'>;
    extern template class RLRecurrentNetImpl<torch::Tensor , torch::nn::GRU, torch::nn::GRUOptions, false, 'g'>;
    extern template class RLRecurrentNetImpl<torch::Tensor , torch::nn::RNN, torch::nn::RNNOptions, false, 'r'>;

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

SAM_OPTIONS(BaseModuleOption, RecurrentNetOption, SELF(hidden_size), SELF(num_layers), SELF(batch_size), SELF(id),
            SELF(return_all_seq), SELF(batch_first), SELF(drop_out), SELF(device), SELF(type), SELF(reset_hidden))

#endif //SAMFRAMEWORK_RECURRENT_NET_H
