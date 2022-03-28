//
// Created by dewe on 3/27/22.
//

#include "any"
#include "torch/torch.h"


namespace torch::utils{

    struct Accuracy{
        torch::Tensor operator()(torch::Tensor const& y_true,
                               torch::Tensor  const& y_pred){
            AT_ASSERTM(y_pred.sizes() == y_true.sizes(),
                       "AccuracyError: y_true shape not compatible with y_pred shape");
            auto _new_t = y_pred;
            if (y_pred.dtype() != y_true.dtype())
                _new_t = y_pred.to(y_true);
            return torch::eq(_new_t, y_true).to(kFloat32);
        }
    };

    struct BinaryAccuracy{
        float threshold;

        explicit BinaryAccuracy(float threshold=0.5): threshold(threshold){}

        torch::Tensor operator()(torch::Tensor const& y_true, torch::Tensor const& y_pred) {
            auto _y_pred = (y_pred > threshold).to(y_pred.dtype());
            return torch::eq(_y_pred, y_true).mean(-1);
        }
    };

    struct CategoricalAccuracy{
        inline torch::Tensor operator()(torch::Tensor const& y_true, torch::Tensor const& y_pred) {
            return torch::eq( y_true.argmax(-1), y_pred.argmax(-1) ).to(kFloat32);
        }
    };

    struct TopKCategoricalAccuracy{
        int k;

        explicit TopKCategoricalAccuracy(int _k=5):k(5){
        }

        inline torch::Tensor operator()(torch::Tensor const& y_true, torch::Tensor const& y_pred) {
            auto[top_p, top_class] = torch::topk(y_pred, k);
            auto _equals = top_class.eq(y_true.view(top_class.sizes()));
            return _equals.to(kFloat32);
        }
    };

}