#ifndef SAMFRAMEWORK_GTRXL_H
#define SAMFRAMEWORK_GTRXL_H

#include "base.h"
#include "common/common.h"
#include <vector>

#define GTRXLTemplate 
#define GTRXL_IMPL_T GTRXLImpl

namespace sam_dn {
    /**
     * The Gated Transformer-XL (GTrXL) is a transformer-based architecture
     * used for temporal sequences. It provides significant benefits over vanilla
     * transformers, due to its usage of gated layers over residual layers, and
     * its reordering of layer normalization layers, which allows for an
     * identity map from input to output.
     * 
     * Based on https://github.com/dhruvramani/Transformers-RL.
     * 
     * \param embedding_size    The size of each input embedding.
     * \param num_heads         The number of heads to use for multi-headed attention.
     * \param num_layers        The number of transformer blocks to stack.
     * \param batch_size        Batch size.
     * \param bg                Bias term. Setting this to a value greater than 0 can
     *                          greatly speed up learning.
     * \param d_inner_head      Dimension of each head.
     * \param d_inner_ff        Hidden dimension of feedforward layer.
     * \param dropout_o         Dropout applied after multihead attention.
     * \param dropout_a         Dropout applied after attention mechanism.
     */
    struct GTRXLOption : public BaseModuleOption {
        int64_t embedding_size;
        int64_t num_heads;
        int64_t num_layers;
        int64_t batch_size;
        float_t bg;
        int64_t d_inner_head;
        int64_t d_inner_ff;
        float_t dropout_o;
        float_t dropout_a;
    };

    GTRXLTemplate
    class GTRXLImpl : public BaseModuleImpl<GTRXLOption> {
        private:
            std::optional<std::vector<torch::Tensor>> memory;

        public:
            explicit GTRXLImpl(GTRXLOption opt);
            
            torch::Tensor forward(torch::Tensor input);
    };

    class PositionalEmbedding : public torch::nn::Module {
        private:
            int64_t dim;
            torch::Tensor inv_freq;

        public:
            explicit PositionalEmbedding(int64_t dim) {
                this->dim = dim;
                this->inv_freq = 1.0 / (torch::pow(10000, torch::arange(0.0, dim, 2.0) / (float_t)dim));
            };

            torch::Tensor forward(torch::Tensor positions) {
                auto sinusoid_inp = torch::outer(positions.toType(c10::ScalarType::Float), this->inv_freq);
                auto pos_emb = torch::cat({sinusoid_inp.sin(), sinusoid_inp.cos()}, -1);
                return pos_emb.unsqueeze(1);
            };
    };

    class PositionwiseFF : public torch::nn::Module {
        private:
            int64_t d_input;
            int64_t d_inner;
            int64_t dropout;
            torch::nn::Sequential ff;

        public:
            explicit PositionwiseFF(int64_t d_input, int64_t d_inner, float_t dropout) {
                this->d_input = d_input;
                this->d_inner = d_inner;
                this->dropout = dropout;
                this->ff = torch::nn::Sequential();
                this->ff->push_back(torch::nn::Linear(d_input, d_inner));
                this->ff->push_back(torch::nn::ReLU(true));
                this->ff->push_back(torch::nn::Dropout(dropout));
                this->ff->push_back(torch::nn::Linear(d_inner, d_input));
                this->ff->push_back(torch::nn::Dropout(dropout));
            };

            torch::Tensor forward(torch::Tensor input) {
                return this->ff.get()->forward(input);
            };
    };

    class GatedUnit : public torch::nn::Module {
        private:
            torch::nn::Linear wr, wz, wg;
            torch::nn::Linear ur, uz, ug;
            float_t bg;

        public:
            explicit GatedUnit(int64_t d_input, float_t bg) {
                this->wr = torch::nn::Linear(d_input, d_input);
                this->ur = torch::nn::Linear(d_input, d_input);
                this->wz = torch::nn::Linear(d_input, d_input);
                this->uz = torch::nn::Linear(d_input, d_input);
                this->wg = torch::nn::Linear(d_input, d_input);
                this->ug = torch::nn::Linear(d_input, d_input);
            };

            torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
                auto r = torch::sigmoid(this->wr(y) + this->ur(x));
                auto z = torch::sigmoid(this->wz(y) + this->uz(x));
                auto h = torch::tanh(this->wg(y) + this->ug(torch::mul(r, x)));
                auto g = torch::mul(1.0 - z, x) + torch::mul(z, h);
                return g;
            };
    };

    class MultiHeadAttention : public torch::nn::Module {
        private:
            int64_t d_input;
            int64_t d_inner;
            int64_t num_heads;
            float_t scale;
            torch::nn::Linear linear_kv;
            torch::nn::Linear linear_q;
            torch::nn::Linear linear_p;
            torch::nn::Dropout drop_a;
            torch::nn::Linear l_out;
            torch::nn::Dropout drop_o;

            torch::Tensor rel_shift(torch::Tensor x) {
                auto zero_pad = torch::zeros({x.size(0), 1, x.size(2), x.size(3)});
                return torch::cat({zero_pad, x}, 1)
                    .view({x.size(1) + 1, x.size(0), x.size(2), x.size(3)})
                    .index({1, torch::indexing::Slice()})
                    .view_as(x);
            };

        public:
            explicit MultiHeadAttention(
                int64_t d_input,
                int64_t d_inner,
                int64_t num_heads,
                float_t dropout,
                float_t dropout_a
            ) {
                this->d_input = d_input;
                this->d_inner = d_inner;
                this->num_heads = num_heads;

                this->linear_kv = torch::nn::Linear(d_input, d_inner * num_heads * 2, false);
                this->linear_q = torch::nn::Linear(d_input, d_inner * num_heads, false);

                this->linear_p = torch::nn::Linear(d_input, d_inner * num_heads, false);
                this->scale = 1.0 / std::pow(d_inner, 0.5);
                this->drop_a = torch::nn::Dropout(dropout_a);

                this->l_out = torch::nn::Linear(d_inner * num_heads, d_input, false);
                this->drop_o = torch::nn::Dropout(dropout);
            };

            torch::Tensor forward(
                torch::Tensor input,
                torch::Tensor pos_embs,
                torch::Tensor memory,
                torch::Tensor u,
                torch::Tensor v,
                std::optional<torch::Tensor> mask
            ) {
                auto cur_seq = input.size(0);
                auto prev_seq = memory.size(0);
                auto h = this->num_heads;
                auto d = this->d_inner;

                auto input_with_memory = torch::cat({memory, input}, 0);

                auto kv_chunks = torch::chunk(
                    this->linear_kv(input_with_memory),
                    2,
                    -1
                );
                auto k_tfmd = kv_chunks[0];
                auto v_tfmd = kv_chunks[1];
                auto q_tfmd = this->linear_q(input);

                auto bs = q_tfmd.size(1);
                assert(q_tfmd.size(1) == k_tfmd.size(1));

                auto content_attn = torch::einsum(
                    "ibhd,jbhd->ijbh",
                    {
                        q_tfmd.view({cur_seq, bs, h, d}) + u,
                        k_tfmd.view({cur_seq + prev_seq, bs, h, d})
                    }
                );

                auto p_tfmd = this->linear_p(pos_embs);
                auto position_attn = torch::einsum(
                    "ibhd,jhd->ijbh",
                    {
                        q_tfmd.view({cur_seq, bs, h, d}) + v,
                        p_tfmd.view({cur_seq + prev_seq, h, d})
                    }
                );
                position_attn = this->rel_shift(position_attn);
                auto attn = content_attn + position_attn;

                if (mask.has_value()) {
                    attn = attn.masked_fill(mask->index({"...", torch::indexing::None}), -INFINITY);
                }

                attn = torch::softmax(attn * this->scale, 1);
                attn = this->drop_a(attn);

                auto attn_weighted_values =
                    torch::einsum(
                        "ijbh,jbhd->ibhd",
                        {
                            attn,
                            v_tfmd.view({cur_seq + prev_seq, bs, h, d})
                        }
                    )
                    .contiguous()
                    .view({cur_seq, bs, h * d});

                return this->drop_o(this->l_out(attn_weighted_values));
            };
    };

    class StableTransformerEncoderLayerXL : public torch::nn::Module {
        private:
            GatedUnit gate1;
            GatedUnit gate2;
            MultiHeadAttention mha;
            PositionwiseFF ff;
            torch::nn::LayerNorm norm1;
            torch::nn::LayerNorm norm2;
        
        public:
            explicit StableTransformerEncoderLayerXL(
                int64_t num_heads,
                int64_t d_input,
                int64_t d_head_inner,
                int64_t d_ff_inner,
                float_t dropout,
                float_t dropout_a,
                float_t bg
            ):
                gate1(GatedUnit(d_input, bg)),
                gate2(GatedUnit(d_input, bg)),
                mha(MultiHeadAttention(
                    d_input,
                    d_head_inner,
                    num_heads,
                    dropout,
                    dropout_a
                )),
                ff(PositionwiseFF(d_input, d_ff_inner, dropout))
            {
                this->norm1 = torch::nn::LayerNorm(d_input);
                this->norm2 = torch::nn::LayerNorm(d_input);
            };

            torch::Tensor forward(
                torch::Tensor input,
                torch::Tensor pos_embs,
                torch::Tensor u,
                torch::Tensor v,
                std::optional<torch::Tensor> mask,
                torch::Tensor mems
            ) {
                auto src2 = this->norm1(input);
                src2 = this->mha.forward(src2, pos_embs, mems, u, v, mask);
                auto src = this->gate1.forward(input, src2);
                src2 = this->ff.forward(this->norm2(src));
                src = this->gate2.forward(src, src2);
                return src;
            };
    };

    class StableTransformerXL : public torch::nn::Module {
        private:
            int64_t num_layers;
            int64_t num_heads;
            int64_t d_input;
            int64_t d_head_inner;
            int64_t d_ff_inner;
            torch::Tensor u, v;
            PositionalEmbedding pos_embs;
            torch::nn::Dropout drop;
            torch::nn::ModuleList layers;

        public:
            explicit StableTransformerXL(
                int64_t d_input,
                int64_t num_layers,
                int64_t num_heads,
                int64_t d_head_inner,
                int64_t d_ff_inner,
                float_t dropout,
                float_t dropout_a,
                float_t bg
            ):
                pos_embs(PositionalEmbedding(d_input))
            {
                this->num_layers = num_layers;
                this->num_heads = num_heads;
                this->d_input = d_input;
                this->d_head_inner = d_head_inner;
                this->d_ff_inner = d_ff_inner;
                this->drop = torch::nn::Dropout(dropout);
                std::vector<StableTransformerEncoderLayerXL> layers_list;
                for (int i = 0; i < num_layers; i++) {
                    auto layer = StableTransformerEncoderLayerXL(
                        num_heads,
                        d_input,
                        d_head_inner,
                        d_ff_inner,
                        dropout,
                        dropout_a,
                        bg
                    );
                    layers_list.push_back(layer);
                }
                this->layers = torch::nn::ModuleList(&layers_list);

                this->u = this->register_parameter("u", torch::empty({this->num_heads, this->d_head_inner}));
                this->v = this->register_parameter("v", torch::empty({this->num_heads, this->d_head_inner}));
            };

            std::vector<torch::Tensor> init_memory() {
                std::vector<torch::Tensor> memory;
                for (int i = 0; i < this->num_layers + 1; i++) {
                    memory.push_back(torch::zeros({20, 5, 8}));
                }
                return memory;
            };

            std::vector<torch::Tensor> update_memory(
                std::vector<torch::Tensor> previous_memory,
                std::vector<torch::Tensor> hidden_states
            ) {
                assert(hidden_states.size() == previous_memory.size());
                auto mem_len = previous_memory[0].size(0);
                auto seq_len = hidden_states[0].size(0);

                std::vector<torch::Tensor> new_memory;
                {
                    torch::NoGradGuard no_grad;
                    int64_t end_idx = mem_len + seq_len;
                    auto beg_idx = std::max((int64_t)0, end_idx - mem_len);
                    for (int i = 0; i < previous_memory.size(); i++) {
                        auto m = previous_memory[i];
                        auto h = hidden_states[i];
                        auto cat = torch::cat({m, h}, 0);
                        new_memory.push_back(cat.index({beg_idx, end_idx}).detach());
                    }
                }
                return new_memory;
            };

            std::tuple<torch::Tensor, std::vector<torch::Tensor>> pass(torch::Tensor inputs, std::optional<std::vector<torch::Tensor>> _memory) {
                auto memory = _memory.value_or(this->init_memory());

                assert(memory.size() == this->layers->children().size() + 1);

                auto cur_seq = inputs.size(2);
                auto bs = inputs.size(3);
                auto prev_seq = memory[0].size(0);

                auto dec_attn_mask = torch::triu(
                        torch::ones({cur_seq, cur_seq + prev_seq}),
                        1 + prev_seq
                    )
                    .toType(c10::ScalarType::Bool)
                    .index({"...", torch::indexing::None});

                auto pos_ips = torch::arange(cur_seq + prev_seq - 1, -1, -1.0);
                auto pos_embs = this->drop(this->pos_embs.forward(pos_ips));
                if (this->d_input % 2 != 0) {
                    pos_embs = pos_embs.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1)});
                }

                std::vector<torch::Tensor> hidden_states {inputs};
                auto layer_out = inputs;
                for (int i = 0; i < this->layers->children().size(); i++) {
                    auto mem = memory[i];
                    auto layer = this->layers->children()[i];
                    layer_out = ((StableTransformerEncoderLayerXL*)&layer)->forward(
                        layer_out,
                        pos_embs,
                        this->u,
                        this->v,
                        dec_attn_mask,
                        mem
                    );
                    hidden_states.push_back(layer_out);
                }

                memory = this->update_memory(memory, hidden_states);
                return std::make_tuple(layer_out, memory);
            };
    };
}

#include "gtrxl.tpp"

SAM_OPTIONS(
    BaseModuleOption,
    GTRXLOption,
    SELF(embedding_size),
    SELF(num_heads),
    SELF(num_layers),
    SELF(batch_size),
    SELF(bg),
    SELF(dropout_o),
    SELF(dropout_a),
    SELF(d_inner_head),
    SELF(d_inner_ff)
)

#endif //SAMFRAMEWORK_GTRXL_H