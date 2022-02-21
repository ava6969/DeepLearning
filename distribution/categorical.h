//
// Created by dewe on 9/20/21.
//

#ifndef SAMFRAMEWORK_CATEGORICAL_H
#define SAMFRAMEWORK_CATEGORICAL_H

#include "distribution.h"

namespace sam_dn{
    class Categorical : public Distribution{

    protected:
        torch::Tensor logit, prob;

    public:
        Categorical()=default;

        virtual inline void load(torch::Tensor const& _logits)  {
            if (_logits.dim() < 1)
            {
                throw std::runtime_error("Logit tensor must have at least one dimension");
            }
            this->logit = _logits - _logits.logsumexp(-1, true);
            this->prob = torch::softmax(this->logit, -1);
        }

        [[nodiscard]] inline torch::Tensor entropy() const noexcept override{
            assert(this->prob.dim() > 0);
            return -(logit * this->prob).sum(-1);
        }

        [[nodiscard]] inline torch::Tensor logProb(torch::Tensor const& action) const noexcept override{
            auto longAction = action.view(-1).toType(c10::kLong).unsqueeze(-1);
            auto broadcasted_tensors = torch::broadcast_tensors({longAction, logit});
            longAction = broadcasted_tensors[0].narrow(-1, 0, 1);
            return broadcasted_tensors[1].gather(-1, longAction).squeeze(-1);
        }

        [[nodiscard]] inline torch::Tensor sample() const noexcept override{
            torch::Tensor&& probs_2d = this->prob.view({-1, logit.size(-1)});
            return torch::multinomial(probs_2d, 1, true).contiguous();
        }

        [[nodiscard]] inline auto logits() const noexcept { return logit; }
        [[nodiscard]] inline auto probs() const noexcept { return prob; }
    };
}
#endif //SAMFRAMEWORK_CATEGORICAL_H
