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
        std::optional<int64_t> batch_size{}, steps_per_epoch{}, validation_steps{}, validation_batch_size{};
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
        torch::Device m_Device{c10::kCPU};

    protected:

//        virtual OptionalTensor compute_loss(OptionalTensor const& x=NONE,
//                                            OptionalTensor const& y=NONE,
//                                            OptionalTensor const& y_pred=NONE,
//                                            OptionalTensor const& sample_weight=NONE){
//            return this->compiledLoss(y, y_pred, sample_weight, this->losses);
//        }
//
//        [[maybe_unused]] virtual std::unordered_map<std::string, TensorDictVariant> compute_metrics(
//                OptionalTensor const& x=NONE,
//                OptionalTensor const& y=NONE,
//                OptionalTensor const& y_pred=NONE,
//                OptionalTensor const& sample_weight=NONE){
//
//            this->compiledMetrics.updateState(y.value(), y_pred.value(), sample_weight);
//
//            std::unordered_map<std::string, TensorDictVariant> return_metrics;
//
//            for(auto const& metric : this->metrics){
//                auto && result = metric.result();
//                if constexpr(dict_output){
//                    return_metrics.template merge(result);
//                }else{
//                    return_metrics[metric.name()] = result;
//                }
//            }
//
//            return return_metrics;
//        }

    public:
        explicit TrainableModel( torch::nn::Sequential module) : module(std::move(module)) {}

        template<class ModelBuilderOverride=sam_dn::Builder>
        explicit TrainableModel( std::filesystem::path const& config_path,
                                 std::string const& output_name,
                                 std::optional<torch::Device>  device,
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
                device = device_type;
            }

            if( print_model )
                std::cout << module << "\n";

            m_Device = device.value();
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

        template<class DataSet>
        auto make_data_loader(DataSet const& data_set, bool shuffle, TrainingOption opt){

            torch::data::DataLoaderOptions _opt(opt.batch_size.template value_or(16));
            _opt.workers(opt.workers);
            _opt.max_jobs( opt.max_queue_size );

//            if(shuffle){
                return torch::data::make_data_loader<torch::data::samplers::RandomSampler>( data_set, _opt);
//            }else{
//                return torch::data::make_data_loader<torch::data::samplers::SequentialSampler>( data_set, _opt);
//            }
        }

        template<class DataSet>
        void fit(TrainingOption const& option,
                 DataSet const& train_set,
                 std::optional<DataSet> const& validation_data=NONE){

            module->train();

            torch::data::DataLoaderOptions train_opt(option.batch_size.template value_or(16));
            train_opt.workers(option.workers);
            train_opt.max_jobs( option.max_queue_size );

            auto train_loader = make_data_loader(train_set, option.shuffle, option);
            decltype( train_loader ) validation_loader;
            if(validation_data){
                validation_loader = make_data_loader( *validation_data, false, option);
            }

            for(int epoch = 0; epoch < option.epochs; epoch++){
                train_per_epoch(train_loader, epoch,  train_set.size().value(), 1);

                if(validation_loader){
                    validate_per_epoch(validation_loader, epoch, validation_data->size().value(), 1 );
                }
            }
        }

        template<class DataLoaderT>
        void train_per_epoch(DataLoaderT& data_loader, int epoch, int dataset_size, int log_interval){

            size_t batch_idx = 0;
            for (auto& batch : *data_loader) {
                auto data = batch.data.to( m_Device ), targets = batch.target.to( m_Device );
                optimizer->zero_grad();
                auto output = module->forward(data);
                torch::Tensor _loss;
//                auto _loss = compiledLoss(output, targets);
                AT_ASSERT(!std::isnan(_loss.template item<float>()));
                _loss.backward();
                optimizer->step();

                if (batch_idx++ % log_interval == 0) {
                    std::printf(
                            "\rTrain Epoch: %d [%5ld/%5d] Loss: %.4f",
                            epoch,
                            batch_idx * batch.data.size(0),
                            dataset_size,
                            _loss.template item<float>());
                }
            }
        }

        template<class DataLoaderT>
        void validate_per_epoch(DataLoaderT& data_loader, int epoch, int dataset_size, int log_interval){

            torch::NoGradGuard no_grad;
            module->eval();
            double test_loss = 0;
            int32_t correct = 0;

            for (const auto& batch : *data_loader) {
                auto data = batch.data.to(m_Device), targets = batch.target.to(m_Device);
                auto output = module->forward(data);
//                test_loss += compiledLoss(output, targets);
                auto pred = output.argmax(1);
                correct += pred.eq(targets).sum().template item<int64_t>();
            }

            test_loss /= dataset_size;
            std::printf(
                    "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
                    test_loss,
                    static_cast<double>(correct) / dataset_size);

        }

    };
}

#endif //DEEP_NETWORKS_TRAINABLE_H
