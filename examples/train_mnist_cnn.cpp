//
// Created by dewe on 12/9/21.
//

#define DEBUG_VISION

#include "trainer/trainable.h"
#include "vision/conv_net.h"
#include "basic/fcnn.h"


// Where to find the MNIST dataset.
const char* kDataRoot = "examples/data/mnist";


// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


template <typename DataLoader>
void train(
        size_t epoch,
        torch::nn::Sequential model,
        torch::Device device,
        DataLoader& data_loader,
        torch::optim::Optimizer& optimizer,
        size_t dataset_size) {
    model->train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::cross_entropy_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        if (batch_idx++ % kLogInterval == 0) {
            std::printf(
                    "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                    epoch,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>());
        }
    }
}

template <typename DataLoader>
void test(
        torch::nn::Sequential model,
        torch::Device device,
        DataLoader& data_loader,
        size_t dataset_size) {
    torch::NoGradGuard no_grad;
    model->eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto output = model->forward(data);
        test_loss += torch::cross_entropy_loss(
                output,
                targets,
                /*weight=*/{},
                torch::Reduction::Sum)
                .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / dataset_size);
}


int main(){

    torch::manual_seed(1);

    TrainingOption trainingOption;
    trainingOption.batch_size = 64;
    trainingOption.shuffle = true;
    trainingOption.use_multi_threading = true;
    trainingOption.epochs = 5;

    EvaluationOption evaluationOption;
    evaluationOption.verbose = 2;

    CompileOption opt;
    opt.loss = "mse";
//    opt.optimizer = torch::optim::SGDOptions(0.01).momentum(0.5);
    opt.optimizer = "sgd";
    opt.metrics = {"accuracy", "mae"};
//    opt.metrics.emplace_back( torch::nn::TopKCategoricalAccuracy<false>() );

    TrainableModel model("examples/model/mnist_resnet.yaml",
                         "mnist",
                         c10::kCUDA,
                         true);

    model.compile(opt);

    model.fit(trainingOption);

    model.evaluate(evaluationOption);

//    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
//            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//            .map(torch::data::transforms::Stack<>());
//    const size_t train_dataset_size = train_dataset.size().value();
//    auto train_loader =
//            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//                    std::move(train_dataset), kTrainBatchSize);
//
//    auto test_dataset = torch::data::datasets::MNIST(
//            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
//            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//            .map(torch::data::transforms::Stack<>());
//    const size_t test_dataset_size = test_dataset.size().value();
//    auto test_loader =
//            torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
//
//    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
//        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
//        test(model, device, *test_loader, test_dataset_size);
//    }

}