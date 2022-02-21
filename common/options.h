//
// Created by dewe on 11/11/21.
//

#ifndef SAM_RL_OPTIONS_H
#define SAM_RL_OPTIONS_H

#include <utility>
#include <stdexcept>
#include <cassert>
#include "cmath"
#include "string"
#include "vector"
#include "unordered_map"

namespace sam_dn{

    struct BaseModuleOption{
        std::string input{}, output{}, weight_init_type{"none"}, bias_init_type{"none"};
        float weight_init_param{ std::sqrt(2.f) }, bias_init_param{ 0 };
        bool new_bias{};
        std::unordered_map<std::string, std::vector<int64_t>> dict_opt;

        virtual void Input(std::vector<int64_t> const&){

        }
    };

    struct RecurrentNetOption : BaseModuleOption{
    public:
        int64_t hidden_size{512}, num_layers{1}, batch_size{};
        bool return_all_seq{false};
        bool batch_first{false};
        float drop_out{0};
        std::string device;
        std::string type;

        void Input(std::vector<int64_t> const& rnn_in_size) override{
            assert(rnn_in_size.size() == 1);
            input_size = rnn_in_size[0];
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

    struct FCNNOption : BaseModuleOption{
        std::vector<int64_t> dims;
        std::string act_fn;

        void Input(std::vector<int64_t> const& x) override {
            dims.insert(dims.begin(), x.begin(), x.end());
        }
    };

    struct Conv2DInput {
        int width{}, height{}, channel{};
    };

    struct CNNOption : BaseModuleOption{
    private:
        Conv2DInput inputShape;

    public:
        std::vector<int> filters{}, kernels{}, strides{};
        std::vector <std::string> padding{}, activations{};
        bool flatten_output = {false};

        void Input(std::vector<int64_t> const& x) override {
            switch (x.size()) {
                case 1:
                    inputShape = Conv2DInput{1, 1, static_cast<int>(x[0])};
                    break;
                case 2:
                    inputShape =  Conv2DInput{static_cast<int>(x[0]), static_cast<int>(x[1]), 1};
                    break;
                case 3:
                    inputShape =  Conv2DInput{static_cast<int>(x[1]), static_cast<int>(x[2]), static_cast<int>(x[0])};
                    break;
                default:
                    throw std::runtime_error("Conv2d input size requires 1 <= Observation Size <= 3");
            }
        }

        [[nodiscard]] auto InputShape() const { return inputShape; }

        void setInput(Conv2DInput const& in){
            inputShape = in;
        }
    };

    struct EmbeddingOption: BaseModuleOption{
        int64_t embed_dim{}, embed_num{};
        void Input(std::vector<int64_t> const& x) override {
            embed_num = x.at(0);
        }
    };

    struct SelfAttentionOption: BaseModuleOption{
        int64_t n_heads{}, n_embed{};
        bool layer_norm{}, post_layer_norm{};
        float qk_w{}, v_w{}, post_w{};
        bool max_pool{};
        int64_t n_features{}, features_size{};

        void Input(std::vector<int64_t> const& x) override {
            n_features = x.at(0);
            features_size = x.at(1);
        }
    };

    struct EntityAwareOption : sam_dn::BaseModuleOption{
        sam_dn::FCNNOption feature_option;
        std::string pool_type;
    };
}
#endif //SAM_RL_OPTIONS_H
