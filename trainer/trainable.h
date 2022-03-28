//
// Created by dewe on 3/26/22.
//

#ifndef DEEP_NETWORKS_TRAINABLE_H
#define DEEP_NETWORKS_TRAINABLE_H


#include "variant"
#include "functional"
#include "optional"
#include "loss.h"
#include "optimizer.h"
#include "utility"
#include "common/builder.h"
#include "common.h"
#include "metrics.h"
#include "torch/torch.h"


namespace sam_dn {

    using OptionalTensor = std::optional<torch::Tensor>;
    #define NONE std::nullopt

    template<bool dict_output=false>
    struct CompileOption {
        LossOption loss{"mse"};
        OptimizerOption optimizer{"adam"};
        std::vector<Metrics<dict_output>> metrics{};
        std::conditional< dict_output, std::unordered_map<std::string, float> , std::vector<float>> loss_weights;
        std::optional< std::vector<Metrics<dict_output>> > weighted_metrics = std::nullopt;
    };

    struct EvaluationOption {
        std::optional<int> batch_size{};
        int max_queue_size{10}, workers{1};
        bool use_multi_threading=false;
        std::optional<int> steps{};
        std::optional<int> verbose;
        std::vector< std::function<void()> > callbacks;
    };

    struct TrainingOption : EvaluationOption{
        float validation_split{};
        int initial_epoch{}, validation_freq{1}, epochs{1};
        bool shuffle{true};
        std::optional<int> batch_size{}, steps_per_epoch{}, validation_steps{}, validation_batch_size{};
        std::optional<std::vector<float>> class_weight, sample_weight;
    };

    template<bool dict_output=false>
    class TrainableModel : public torch::nn::Module{

        using TensorDictVariant = std::conditional_t<dict_output,
        std::unordered_map<std::string, torch::Tensor>, torch::Tensor>;

        std::shared_ptr<torch::optim::Optimizer> optimizer;
        torch::nn::Sequential module;
        std::function<std::optional<torch::Tensor>(std::optional<torch::Tensor>,
                                                   std::optional<torch::Tensor>,
                                                   std::optional<torch::Tensor>,
                                                   std::optional<torch::Tensor>)> compiledLoss;
        Metrics<dict_output> compiledMetrics;
        std::vector<Metrics<dict_output>> metrics;
        torch::Tensor losses;
        LossOption loss;
        bool isCompiled{false};

    protected:

        virtual OptionalTensor compute_loss(OptionalTensor const& x=NONE,
                                            OptionalTensor const& y=NONE,
                                            OptionalTensor const& y_pred=NONE,
                                            OptionalTensor const& sample_weight=NONE){
            return this->compiledLoss(y, y_pred, sample_weight, this->losses);
        }

        virtual std::unordered_map<std::string, TensorDictVariant> compute_metrics(
                OptionalTensor const& x=NONE,
                OptionalTensor const& y=NONE,
                OptionalTensor const& y_pred=NONE,
                OptionalTensor const& sample_weight=NONE){

            this->compiledMetrics.updateState(y.value(), y_pred.value(), sample_weight);

            std::unordered_map<std::string, TensorDictVariant> return_metrics;

            for(auto const& metric : this->metrics){
                auto && result = metric.result();
                if constexpr(dict_output){
                    return_metrics.template merge(result);
                }else{
                    return_metrics[metric.name()] = result;
                }
            }

            return return_metrics;
        }

    public:
        explicit TrainableModel( torch::nn::Sequential module) : module(std::move(module)) {}

        template<class ModelBuilderOverride=sam_dn::Builder>
        explicit TrainableModel( std::filesystem::path const& config_path,
                                 std::string const& output_name,
                                 std::optional<torch::Device> const& device,
                                 bool print_model=false) {
            sam_dn::InputShapes shapes;
            auto modules = ModelBuilderOverride().compile(config_path, shapes);
            module = this->template register_module(output_name,
                                                    try_get(modules, output_name));

            if(device){
                module->to(device.value());
            }else{
                torch::DeviceType device_type = torch::kCPU;
                if (torch::cuda::is_available()) {
                    std::cout << "CUDA available! Training on GPU." << std::endl;
                    device_type = torch::kCUDA;
                } else {
                    std::cout << "Training on CPU." << std::endl;
                    device_type = torch::kCPU;
                }
                module->to(device_type);
            }

            if( print_model )
                std::cout << module << "\n";

        }

        void compile(CompileOption<dict_output> const& option) {
            this->optimizer = option.optimizer.get( this->module->parameters() );

//            this->compileLoss = option.loss_fn.get( option.loss_weights );

//            this->compiledMetrics = option.metrics.get( option.weighted_metrics );

            this->isCompiled = true;

//            this->loss = option.loss_fn;

        }

        void evaluate(EvaluationOption const& option,
                      OptionalTensor const& x=NONE,
                      OptionalTensor const& y=NONE){

        }

        void fit(TrainingOption const& option,
                 OptionalTensor const& x=NONE,
                 OptionalTensor const& y=NONE,
                 OptionalTensor const& validation_data=NONE){

        }

    };
}

#endif //DEEP_NETWORKS_TRAINABLE_H
