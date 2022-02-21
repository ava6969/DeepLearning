//
// Created by dewe on 9/20/21.
//

#ifndef SAMFRAMEWORK_DISTRIBUTION_H
#define SAMFRAMEWORK_DISTRIBUTION_H

#include <torch/torch.h>
#include "vector"

namespace sam_dn{

    class Distribution
    {

    public:
        Distribution()=default;
        virtual ~Distribution() = default;

        [[nodiscard]] virtual torch::Tensor entropy() const noexcept = 0;
        [[nodiscard]] virtual torch::Tensor logProb(torch::Tensor const& action) const noexcept  = 0;
        [[nodiscard]] virtual torch::Tensor sample() const noexcept = 0;

    };
}

#endif //SAMFRAMEWORK_DISTRIBUTION_H
