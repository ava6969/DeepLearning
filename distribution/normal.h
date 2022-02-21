//
// Created by dewe on 1/15/22.
//

#ifndef SAM_RL_TRADER_NORMAL_H
#define SAM_RL_TRADER_NORMAL_H

#include "distribution.h"

namespace sam_dn{
    class Normal : public Distribution {

    private:
        torch::Tensor loc, scale;

    public:
        Normal()=default;

        inline void load(torch::Tensor const& _loc, torch::Tensor const& _scale)  {
            auto broad_casted_tensors = torch::broadcast_tensors({_loc, _scale});
            this->loc = broad_casted_tensors[0];
            this->scale = broad_casted_tensors[1];
        }

        [[nodiscard]] inline torch::Tensor entropy() const noexcept override{
            return (0.5 + 0.5 * std::log(2 * M_PI) + torch::log(this->scale)).sum(-1);
        }

        [[nodiscard]] inline torch::Tensor logProb(torch::Tensor const& action) const noexcept override{
            auto variance = this->scale.pow(2);
            auto log_scale = this->scale.log();
            return (-(action - this->loc).pow(2) /
                    (2 * variance) -
                    log_scale -
                    std::log(std::sqrt(2 * M_PI)));
        }

        [[nodiscard]] inline torch::Tensor sample() const noexcept override{
            torch::NoGradGuard guard;
            return at::normal(this->loc, this->scale);
        }

        [[nodiscard]] inline torch::Tensor rsample() const noexcept{
            auto eps = at::normal( torch::zeros({}).to(loc), torch::ones({}).to(loc));
            return this->loc + eps*scale;
        }

        [[nodiscard]] inline auto mean() const noexcept { return loc; }
        [[nodiscard]] inline auto stddev() const noexcept { return scale; }
        [[nodiscard]] inline auto var() const noexcept { return scale.pow(2); }
    };
}


#endif //SAM_RL_TRADER_NORMAL_H
