//
// Created by dewe on 5/21/22.
//

#ifndef REINFORCEMENT_LEARNING_DEEPMINDSELFATTENTIONIMPL_H
#define REINFORCEMENT_LEARNING_DEEPMINDSELFATTENTIONIMPL_H

#include "torch/torch.h"


namespace sam_dn {

    struct MultiHeadAttention : torch::nn::Module {
        MultiHeadAttention(int n_heads, int N, int node_size):
        sqrt_emb_size( static_cast<int>(std::sqrt(emb_size)) ),
        q_lin( torch::nn::Linear(node_size, N) ),
        k_lin( torch::nn::Linear(node_size, N) ),
        a_lin( torch::nn::Linear(N, N) ),
        q_norm( torch::nn::LayerNorm( torch::nn::LayerNormOptions({n_heads, N, node_size}).elementwise_affine(true))),
        k_norm( torch::nn::LayerNorm( torch::nn::LayerNormOptions({n_heads, N, node_size}).elementwise_affine(true))),
        v_norm( torch::nn::LayerNorm( torch::nn::LayerNormOptions({n_heads, N, node_size}).elementwise_affine(true))){}

        torch::Tensor forward( torch::Tensor const& x){
            auto Q = q_lin(q_norm(x));
            auto K = k_lin(k_norm(x));
            auto A = torch::nn::functional::elu(Q + K);
            A = a_lin(A);
            A = torch::nn::functional::softmax(A, 3);
            auto V = vln(v_norm(x));

            auto softmax = torch::softmax( torch::bmm(Q, K.transpose(1, 2)) / sqrt_emb_size, -1);
            return torch::bmm(softmax, V);
        }

    private:
        torch::nn::Linear q_lin{nullptr}, k_lin{nullptr}, a_lin{nullptr};
        torch::nn::LayerNorm q_norm{nullptr}, k_norm{nullptr}, v_norm{nullptr};
        int sqrt_emb_size{};
    };

    class DeepMindSelfAttentionImpl {

    public:

        struct Option{
            int elem_dim;
            int node_size;
            int n_elems;
            int elem_size;
            int n_heads;
        };

        DeepMindSelfAttentionImpl(Option opt):
        linear1(opt.n_heads*opt.elem_size, opt.elem_size),
        linear2(opt.elem_size, opt.elem_size),
        ln( torch::nn::LayerNorm( torch::nn::LayerNormOptions({opt.elem_size}).elementwise_affine(true)))
        {
            for(int i = 0 ; i < opt.n_heads; i++){
                heads->push_back(AttentionHead(opt.n_elems, opt.elem_size, opt.embed_size));
            }
        }

        torch::Tensor forward( torch::Tensor const& x){

        }


    private:
        torch::nn::Linear linear1{nullptr}, linear2{nullptr};
        torch::nn::LayerNorm ln{nullptr};
        torch::nn::Linear k_proj{nullptr}, q_proj{nullptr}, v_proj{nullptr};

    };
}

#endif //REINFORCEMENT_LEARNING_DEEPMINDSELFATTENTIONIMPL_H
