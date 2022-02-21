//
// Created by dewe on 1/14/22.
//

#ifndef SAM_RL_TRADER_ENTITY_AWARE_H
#define SAM_RL_TRADER_ENTITY_AWARE_H

#include <optional>
#include "torch/torch.h"
#include "base.h"
#include "common/options.h"
#include "boost/algorithm/string.hpp"

namespace sam_dn {
    enum class PoolType {
        SUM,
        MAX,
        AVG,
        NONE
    };

class EntityAwareImpl : public ModuleWithSizeInfoImpl {

public:
    EntityAwareImpl() = default;

    explicit EntityAwareImpl(EntityAwareOption const &opt,
                             bool(*filter)(std::string const&)=nullptr) : ModuleWithSizeInfoImpl(opt) {
        pool_type = getPoolType(opt.pool_type);

        if (pool_type == PoolType::NONE)
            m_OutputSize = {static_cast<long>(opt.dict_opt.size()), opt.feature_option.dims.back()};
        else
            m_OutputSize = {opt.feature_option.dims.back()};

        for (auto const &[f, v]: opt.dict_opt) {
            if (filter(f))
                continue;

            auto f_opt = opt.feature_option;
            f_opt.Input(v);
            featuresDict[f] = register_module(f, FCNN(f_opt));
        }

        result.resize(opt.dict_opt.size());
    }

    static PoolType getPoolType(std::string p_type) {
        boost::to_lower(p_type);
        PoolType pool_type = PoolType::SUM;
        if (p_type == "none")
            pool_type = PoolType::NONE;
        else if (p_type == "max")
            pool_type = PoolType::MAX;
        else if (p_type == "avg")
            pool_type = PoolType::AVG;
        else if (p_type == "sum")
            pool_type = PoolType::SUM;
        else {
            std::stringstream ss;
            ss << "Pool type Error: " << p_type << " is not a a valid pool type, choose either "
                                                   "[none, max, avg, sum]";
            throw std::runtime_error(ss.str());
        }
        return pool_type;
    }

    static void fromPoolType(PoolType pool_type, torch::Tensor &stacked_out) {
        switch (pool_type) {
            case PoolType::SUM:
                stacked_out = torch::sum(stacked_out, -2);
                break;
            case PoolType::MAX:
                stacked_out = std::get<0>(torch::max(stacked_out, -2));
                break;
            case PoolType::AVG:
                stacked_out = torch::mean(stacked_out, -2);
                break;
            case PoolType::NONE:
                break;
        }
    }

    TensorDict *complete(TensorDict *d) {
        auto stacked_out = torch::stack(result, -2);

        fromPoolType(pool_type, stacked_out);

        d->insert_or_assign(m_Output, stacked_out);
        return d;
    }

    inline TensorDict *forwardDict(TensorDict *d) noexcept override {
        int i = 0;

        for (auto&[k, model]: featuresDict)
            result[i++] = model->forward(d->at(k));

        return complete(d);
    }

protected:
    std::unordered_map<std::string, sam_dn::FCNN> featuresDict;
    PoolType pool_type{PoolType::NONE};
    std::vector<torch::Tensor> result;
};

    TORCH_MODULE(EntityAware);
}

#endif //SAM_RL_TRADER_ENTITY_AWARE_H
