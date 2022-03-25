//
// Created by dewe on 1/14/22.
//

#pragma once

#include "fcnn.h"

namespace sam_dn {

    enum class PoolType {
        SUM,
        MAX,
        AVG,
        NONE
    };

    struct EntityAwareOption : sam_dn::BaseModuleOption{
        sam_dn::FCNNOption feature_option;
        std::string pool_type;
    };

    using Filter = bool(*)(std::string const&);

class EntityAwareImpl : public ModuleWithSizeInfoImpl {

public:
    EntityAwareImpl() = default;

    explicit EntityAwareImpl(EntityAwareOption const& opt, Filter const& filter=nullptr);

    static PoolType getPoolType(std::string p_type);

    static void fromPoolType(PoolType pool_type, torch::Tensor& stacked_out);

    TensorDict* complete(TensorDict* );

    TensorDict* forwardDict(TensorDict* ) noexcept override;

protected:
    std::unordered_map<std::string, FCNN> m_featureEmbeddings;
    PoolType pool_type{PoolType::NONE};
    std::vector<torch::Tensor> result;
};

    TORCH_MODULE(EntityAware);
}

SAM_OPTIONS(BaseModuleOption, EntityAwareOption, SELF(feature_option), SELF(pool_type));

