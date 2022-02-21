//
// Created by dewe on 9/25/21.
//

#ifndef SAMFRAMEWORK_MDN_H
#define SAMFRAMEWORK_MDN_H
#include "torch/torch.h"
#include "cassert"
#include "recurrent_net.h"

namespace sam_rl{

    class MDNImpl : public torch::nn::Module{

        public:
            MDNImpl(int num_mixtures, int input_size, int z_dim):
                    m_WeightBias(register_module("mdn_dense",
                                                 torch::nn::Linear(input_size, num_mixtures*3*z_dim))),
                    m_numMixtures(num_mixtures){
            }

            inline
            std::array<torch::Tensor, 4> forward(torch::Tensor x){

                assert(x.dim() == 2);

                x = m_WeightBias(x);
                auto _x = x.view({-1, m_numMixtures*3});

                auto dist  = _x.split(3, 1);
                auto[pi, mu, sigma] = std::make_tuple(dist[0], dist[1], dist[2]);

                pi = pi - pi.logsumexp(-1, true);
                return {pi, mu, sigma, x};
            }

    private:
        torch::nn::Linear m_WeightBias = nullptr;
        int m_numMixtures;
    };

    class MDNLSTMImpl : public BaseModuleImpl{

        std::shared_ptr<MDNImpl> m_MDN;
        std::shared_ptr<LSTMImpl> m_LSTM;
        std::optional<float> m_InputDropOut{}, m_OutputDropOut{};
        bool m_UseLayerNorm{false};
        int m_LSTMStateSize;
        TensorDict fullResult;

    public:
        MDNLSTMImpl(MDNLSTMOption const& opt): BaseModuleImpl(opt),
        m_MDN(std::make_shared<MDNImpl>(num_mixtures, opt.hidden_size(), z_dim)),
        m_LSTM(std::make_shared<BatchFirstLSTMImpl>(opt, maxSeqLength, true)),
        m_LSTMStateSize(opt.hidden_size()){

            register_module("mdn", m_MDN);
            register_module("lstm", m_LSTM);
        }

        MDNLSTMImpl(MDNLSTMOption const& opt),
                    m_MDN(std::make_shared<MDNImpl>(num_mixtures, opt.hidden_size(), outputSeqWidth)),
                    m_LSTM(std::make_shared<BatchFirstLSTMImpl>(opt, maxSeqLength, true)),
                m_InputDropOut(inputDropOut),
                m_OutputDropOut(outputDropout),
                m_UseLayerNorm(useLayerNorm),
                m_LSTMStateSize(opt.hidden_size()){

            register_module("mdn", m_MDN);
            register_module("lstm", m_LSTM);
        }

        inline auto initial_state(int batch_size, torch::Device const& device) {
            return m_LSTM->initial_state(batch_size, device);
        }

        auto set_state(TensorTuple const& x) {
            return m_LSTM->initial_state(x);
        }

        auto get_state() const{
            return m_LSTM->get_state();
        }

        torch::Tensor forward_simple(const torch::Tensor &x) override{

            auto N = x.size(0);

            auto output = m_UseLayerNorm ? torch::layer_norm(x, x.size(1)) : x;
            output = m_InputDropOut ? torch::dropout(x, m_InputDropOut.value(), is_training()) : output;
            output = m_LSTM->forward(output); // (N, L, Hin)
            output = m_OutputDropOut ? torch::dropout(x, m_OutputDropOut.value(), is_training()) : output;

//            auto lastState = m_LSTM->get_state(); // (n_layers, Hin)

            output = output.view({-1, m_LSTMStateSize});

            auto[pi, mu, sigma, output_] = m_MDN->forward(output);

            fullResult["mu"] = mu;
            fullResult["sigma"] = sigma;
            fullResult["pi"] = pi;

            return output_;
        }

        TensorDict forward(TensorDict x) override{
            fullResult = std::move(x);
            fullResult["state"] = forward(fullResult.at(m_Input));
            return fullResult;
        }
    };

    TORCH_MODULE(MDN);
    TORCH_MODULE(MDNLSTM);
}
#endif //SAMFRAMEWORK_MDN_H
