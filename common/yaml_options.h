//
// Created by dewe on 11/11/21.
//

#pragma once

#include "options.h"
#include "common/yaml_helper.h"
#include "wrappers/basic.h"

namespace YAML{

    using namespace sam_dn;

    template<>
    struct convert<BaseModuleOption>{
        static Node encode(const BaseModuleOption& rhs) {
            Node node;
            node.push_back(rhs.input);
            node.push_back(rhs.output);
            node.push_back(rhs.weight_init_param);
            node.push_back(rhs.bias_init_param);
            node.push_back(rhs.weight_init_type);
            node.push_back(rhs.bias_init_type);
            node.push_back(rhs.new_bias);
            return node;
        }

        static bool decode(const Node& node, BaseModuleOption& rhs) {

            DEFINE_REQUIRED(rhs, input);
            DEFINE_REQUIRED(rhs, output);
            DEFINE(rhs, weight_init_type, "none");
            DEFINE(rhs, bias_init_type, "none");
            DEFINE(rhs, new_bias, true);

            if(rhs.weight_init_type != "none"){
                DEFINE_REQUIRED(rhs, weight_init_param);
            }

            if(rhs.bias_init_type != "none"){
                DEFINE_REQUIRED(rhs, bias_init_param);
            }
            return true;
        }
    };

    template<> 
    struct convert<FCNNOption>{
        static Node encode(const FCNNOption& rhs) {
            Node node;
            node.push_back(rhs.input);
            node.push_back(rhs.output);
            node.push_back(rhs.dims);
            node.push_back(rhs.act_fn);
            node.push_back(rhs.weight_init_param);
            node.push_back(rhs.bias_init_param);
            node.push_back(rhs.weight_init_type);
            node.push_back(rhs.bias_init_type);
            node.push_back(rhs.new_bias);
            return node;
        }

        static bool decode(const Node& node, FCNNOption& rhs) {
            if(convert<BaseModuleOption>::decode(node, rhs)){
                DEFINE_REQUIRED(rhs, dims);
                DEFINE(rhs, act_fn, "relu");
                return true;
            }
            return false;
        }
    };

    template<>
    struct convert<EmbeddingOption>{
        static Node encode(const EmbeddingOption& rhs) {
            Node node;
            node.push_back(rhs.input);
            node.push_back(rhs.output);
            node.push_back(rhs.embed_dim);
            node.push_back(rhs.embed_num);
            node.push_back(rhs.weight_init_param);
            node.push_back(rhs.bias_init_param);
            node.push_back(rhs.weight_init_type);
            node.push_back(rhs.bias_init_type);
            node.push_back(rhs.new_bias);
            return node;
        }

        static bool decode(const Node& node, EmbeddingOption& rhs) {
            if(convert<BaseModuleOption>::decode(node, rhs)){
                DEFINE_REQUIRED(rhs, embed_dim);
                return true;
            }
            return false;
        }
    };

    template<>
    struct convert<Conv2DInput>{
        static Node encode(const Conv2DInput& rhs) {
            Node node;
            node.push_back(rhs.width);
            node.push_back(rhs.height);
            node.push_back(rhs.channel);
            return node;
        }

        static bool decode(const Node& node, Conv2DInput& rhs) {
            DEFINE_REQUIRED(rhs, width);
            DEFINE_REQUIRED(rhs, height);
            DEFINE_REQUIRED(rhs, channel);
            return true;
        }
    };

    template<>
    struct convert<CNNOption>{
        static Node encode(const CNNOption& rhs) {
            Node node;
            node.push_back(rhs.input);
            node.push_back(rhs.output);
            node.push_back(rhs.InputShape());
            node.push_back(rhs.filters);
            node.push_back(rhs.kernels);
            node.push_back(rhs.strides);
            node.push_back(rhs.padding);
            node.push_back(rhs.activations);
            return node;
        }

        static bool decode(const Node& node, CNNOption& rhs) {
            if(convert<BaseModuleOption>::decode(node, rhs)){
                DEFINE(rhs, filters, {});
                DEFINE(rhs, kernels, {});
                DEFINE(rhs, strides, {});
                DEFINE(rhs, padding, {});
                DEFINE(rhs, activations, {});
                DEFINE(rhs, flatten_output, false);
                return true;
            }
            return false;
        }
    };

    template<>
    struct convert<RecurrentNetOption>{
        static Node encode(const RecurrentNetOption& rhs) {
            Node node;
            node.push_back(rhs.input);
            node.push_back(rhs.output);
            node.push_back(rhs.hidden_size);
            node.push_back(rhs.num_layers);
            node.push_back(rhs.return_all_seq);
            node.push_back(rhs.batch_first);
            node.push_back(rhs.drop_out);
            node.push_back(rhs.type);
            return node;
        }

        static bool decode(const Node& node, RecurrentNetOption& rhs) {

            DEFINE_REQUIRED(rhs, input);
            DEFINE_REQUIRED(rhs, output);
            DEFINE_REQUIRED(rhs, hidden_size);
            DEFINE_REQUIRED(rhs, type);
            DEFINE_REQUIRED(rhs, batch_size);
            DEFINE_REQUIRED(rhs, device);
            DEFINE(rhs, num_layers, 1);
            DEFINE(rhs, return_all_seq, false);
            DEFINE(rhs, drop_out, 0);
            DEFINE(rhs, weight_init_type, "none");
            DEFINE(rhs, bias_init_type, "none");
            DEFINE(rhs, new_bias, true);

            if(rhs.weight_init_type != "none"){
                DEFINE_REQUIRED(rhs, weight_init_param);
            }

            if(rhs.bias_init_type != "none"){
                DEFINE_REQUIRED(rhs, bias_init_param);
            }

            return true;
        }
    };

    CONVERT_WITH_PARENT(BaseModuleOption, SelfAttentionOption,
                        SELF(n_embed), SELF(n_heads), SELF(layer_norm), SELF(post_layer_norm),
                        SELF(qk_w), SELF(v_w), SELF(max_pool), SELF(post_w));

    CONVERT_WITH_PARENT(BaseModuleOption, EntityAwareOption, SELF(feature_option), SELF(pool_type))

    CONVERT_WITH_PARENT(BaseModuleOption, sam_dn::LayerNorm1dImpl::Option, SELF(axis))

    CONVERT_WITH_PARENT(BaseModuleOption, sam_dn::DropoutImpl::Option, SELF(prob))

}
