//
// Created by dewe on 11/11/21.
//

#include "basic/fcnn.h"
#include "vision/conv_net.h"
#include "memory/recurrent_net.h"
#include "basic/embedding.h"
#include "basic/self_attention.h"
#include "basic/entity_aware.h"
#include "yaml_options.h"
#include "wrappers/basic.h"
#include "vision/residual_block.h"
#include "vision/resnet.h"
#include "wrappers/adapters.h"
#include "builder.h"


namespace sam_dn{

    std::unordered_map<std::string, torch::nn::Sequential> Builder::compile(const std::filesystem::path & path,
                                                                            std::unordered_map<std::string,
                                                                            std::vector<int64_t> >& shapeMap) {
        auto root = YAML::LoadFile(path.string());
        return compile(root, shapeMap);
    }

    Builder::Builder() {

        registerCallBack<sam_dn::FCNNImpl, sam_dn::FCNNOption>("fcnn");
        registerCallBack<sam_dn::EmbeddingImpl, sam_dn::EmbeddingOption>("embed");

        registerCallBack<sam_dn::CNNImpl, sam_dn::CNNOption>("cnn");
        registerCallBack<sam_dn::Conv2DPositionEncodeImpl>("conv2d_pos_encode");
        registerCallBack<sam_dn::ResidualBlockImpl, sam_dn::CNNOption>("resnet_block");
        registerCallBack<sam_dn::ResnetHolderImpl, sam_dn::ResnetHolderImpl::Option>("resnet");
        registerCallBack<sam_dn::MaxPool2DImpl, sam_dn::CNNOption>("max_pool2d");

        registerCallBack<sam_dn::LSTMBatchFirstImpl , sam_dn::RecurrentNetOption>("bf_lstm");
        registerCallBack<sam_dn::RNNBatchFirstImpl , sam_dn::RecurrentNetOption>("bf_rnn");
        registerCallBack<sam_dn::GRUBatchFirstImpl , sam_dn::RecurrentNetOption>("bf_gru");

        registerCallBack<sam_dn::LSTMTimeFirstImpl , sam_dn::RecurrentNetOption>("tf_lstm");
        registerCallBack<sam_dn::RNNTimeFirstImpl , sam_dn::RecurrentNetOption>("tf_rnn");
        registerCallBack<sam_dn::GRUTimeFirstImpl , sam_dn::RecurrentNetOption>("tf_gru");

        registerCallBack<sam_dn::RLLSTMTimeFirstImpl , sam_dn::RecurrentNetOption>("rl_lstm");
        registerCallBack<sam_dn::RLRNNTimeFirstImpl , sam_dn::RecurrentNetOption>("rl_rnn");
        registerCallBack<sam_dn::RLGRUTimeFirstImpl , sam_dn::RecurrentNetOption>("rl_gru");

        registerCallBack<sam_dn::SelfAttentionImpl, sam_dn::SelfAttentionOption>("self_attn");
        registerCallBack<sam_dn::EntityAwareImpl, sam_dn::EntityAwareOption>("entity_aware");

        registerCallBack<sam_dn::DropoutImpl, sam_dn::DropoutImpl::Option >("dropout");
        registerCallBack<sam_dn::LayerNorm1dImpl, sam_dn::LayerNorm1dImpl::Option >("layernorm");

        registerCallBack<sam_dn::Transpose01Impl >("transpose01");
        registerCallBack<sam_dn::Transpose12Impl>("transpose12");
        registerCallBack<sam_dn::Transpose23Impl>("transpose23");

        registerCallBack<sam_dn::ExpandAtAxis0IfDimEqual2Impl >("expand_axis0_if_dim2");
        registerCallBack<sam_dn::ExpandAtAxis0IfDimEqual1Impl >("expand_axis0_if_dim1");

        registerCallBack<sam_dn::ConcatEndImpl>("concat_end");
        registerCallBack<sam_dn::Concat0Impl>("concat0");
        registerCallBack<sam_dn::Concat1Impl>("concat1");
        registerCallBack<sam_dn::Concat2Impl>("concat2");
        registerCallBack<sam_dn::Concat3Impl>("concat3");
    }

    SequentialMap Builder::compile(const YAML::Node & root, InputShapes & shapeMap) {

        std::set<std::string> feature_names;
        SequentialMap result;
        std::shared_ptr< sam_dn::ModuleWithSizeInfoImpl > loaded_module;

        bool isMultiModalData = shapeMap.size() > 1;

        auto toString = [](auto const& x) { return x.template as<std::string>(); };
        auto toVector = [](auto const& x) { return x.template as<std::vector<int64_t>>(); };
        auto merge = [](auto  x, auto const& y) { return x.append("::").append(y); };
        auto contains = [](auto const& _map, auto const& key) { return _map.find( key ) == _map.end(); };

        auto get_size = [&isMultiModalData, &shapeMap](auto const& x) {
            return isMultiModalData ? at::nullopt : at::make_optional(shapeMap[x]);
        };

        auto get_output_size = [&loaded_module, &shapeMap](auto const& shape_key){
            return loaded_module->outputSize().empty() ?  shapeMap[shape_key] : loaded_module->outputSize();
        };

        for(auto const& features : shapeMap){
            feature_names.insert(features.first);
        }

        featureShapeInfo = shapeMap;

        for (auto module: root) {
            m_Model = torch::nn::Sequential();
            std::string feature = "observation";
            auto module_name = toString(module.first);

            if(module_name != "__ignore__") {
                for (auto sub_module_entry: module.second) {
                    auto sub_module = sub_module_entry.second;
                    sub_module["input"] = get(sub_module["input"], feature);

                    if (sub_module["input_shape"])
                        shapeMap[feature] = toVector(sub_module["input_shape"]);

                    feature = toString(sub_module_entry.first);
                    if (contains(feature_names, feature))
                        feature = merge(module_name, feature);

                    sub_module["output"] = feature;

                    auto type = try_get<std::string>(sub_module, "type"s);
                    try {
                        auto shape_key = toString(sub_module["input"]);
                        auto input_shape = get_size(shape_key);
                        isMultiModalData = false;
                        loaded_module = m_CallBacks[type](sub_module, input_shape);
                        shapeMap[feature] = get_output_size(shape_key);
                        featureShapeInfo = shapeMap;
                    } catch (std::exception const &) {
                        std::cerr << "Invalid Attribute Type: [" << type << "]\n";
                        throw;
                    }
                }
                result.insert_or_assign(module_name, std::move(m_Model) );
            }
        }
        return result;
    }

}
