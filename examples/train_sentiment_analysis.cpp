//
// Created by dewe on 12/11/21.
//

#include <utility>
#include "wrappers/basic.h"
#include "memory/recurrent_net.h"
#include "basic/embedding.h"

constexpr auto vocab_size = 74073+1; // +1 for the 0 padding + our word tokens
constexpr auto output_size = 1;
constexpr auto embedding_dim = 400;
constexpr auto hidden_dim = 256;
constexpr auto n_layers = 2;
constexpr auto lr=0.001;
constexpr auto epochs = 4; // 3-4 is approx where I noticed the validation loss stop decreasing
constexpr auto print_every = 100ul;
constexpr auto clip=5; // gradient clipping
constexpr auto batch_size=50;
constexpr auto device = c10::kCUDA;

class MyDataset : public torch::data::Dataset<MyDataset>
{
private:
    torch::Tensor states_, labels_;

public:
    explicit MyDataset(torch::Tensor states_, torch::Tensor labels_)
            : states_(std::move(states_)), labels_(std::move(labels_)) {   };

    inline torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override{
        return {states_[index].to(device, torch::kLong), labels_[index].to(device, torch::kLong)};
    }

    [[nodiscard]] inline torch::optional<size_t> size() const override{
        return states_.size(0);
    }
};
template<class T=int>
std::vector<T> readFile(const char* filename)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    std::vector<T> vec;
    vec.resize(fileSize / sizeof (T));

    file.read( reinterpret_cast<char*>( vec.data() ), fileSize);

    return vec;
}

template< typename MemoryType, typename DataLoaderType>
void train(torch::nn::Sequential& net, DataLoaderType& data_loader, DataLoaderType& valid_loader){

    auto criterion = torch::nn::BCELoss();
    auto optimizer = torch::optim::Adam(net->parameters(), lr);
    auto counter = 0UL;
    net->train();

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        MemoryType* mem = nullptr;
        int i = 0;
        while (not mem)
            mem = net[i++]->template as<MemoryType>();
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            counter++;

            mem->clone_states();

            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.

            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = criterion(prediction, batch.target.toType(c10::kFloat));
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            torch::nn::utils::clip_grad_norm_(net->parameters(), clip);

            optimizer.step();
            auto stateBeforeVal = mem->getState();
            // Output the loss and checkpoint every 100 batches.
            if (counter % print_every == 0) {
                mem->initial_state(batch_size, device);
                net->eval();
                auto val_losses = std::vector<float>{};
                for (auto& inputs : * valid_loader) {
                    mem->initial_state(mem->getState());
                    prediction = net->forward(inputs.data);
                    loss = criterion(prediction, inputs.target.toType(c10::kFloat));
                    val_losses.push_back(loss.template item<float>());
                }

                net->train();
                std::cout << "Epoch: " << epoch+1 << " | Steps: " << counter
                          << " | Loss: " << loss.item<float>() << " | Val Loss: " <<
                                  torch::tensor(val_losses).mean().item().toFloat()<< std::endl;

            }
        }

        std::cout << std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count()  << " seconds elapsed \n";
    }
}

template< typename MemoryType, typename DataLoaderType>
void test(torch::nn::Sequential& net, DataLoaderType& test_loader) {
    int i = 0;
    MemoryType* mem = nullptr;
    while (not mem)
        mem = net[i++]->template as<MemoryType>();
    int numCorrect = 0;
    auto criterion = torch::nn::BCELoss();
    mem->initial_state(batch_size, device);
    net->eval();
    auto test_losses = std::vector<float>{};
    float timer{};
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& inputs : * test_loader) {
        mem->clone_states();

        start = std::chrono::high_resolution_clock::now();
        auto prediction = net->forward(inputs.data);
        timer += std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();

        auto loss = criterion(prediction, inputs.target.toType(c10::kFloat));
        test_losses.push_back(loss.template item<float>());
        prediction = torch::round(prediction);

        auto correctTensor = prediction.eq( inputs.target.toType(c10::kFloat).view_as(prediction));
        numCorrect += correctTensor.squeeze().sum().item().toInt();
    }

    float data_loader_size = std::distance(test_loader->begin(), test_loader->end());
    float _size =  data_loader_size * batch_size;
    std::cout << "Test Loss: " << torch::tensor(test_losses).mean().item().toFloat()
    << " | Test Accuracy: " << float(numCorrect) / _size << " | Inference FPS: " << (data_loader_size / timer) << std::endl;

}

int main(){

    auto result = readFile("examples/data/sentiment_analysis/features.bin");

    torch::Tensor features = torch::tensor(readFile("examples/data/sentiment_analysis/features.bin")).view({25000, -1});
    torch::Tensor labels = torch::tensor(readFile<int>("examples/data/sentiment_analysis/labels.bin")).view({25000, -1});
    torch::manual_seed(1);
    torch::set_num_threads(1);
    auto split_frac = 0.8;
    auto split_idx = long(double(features.size(0))*split_frac);
    auto&& [ train_x, remaining_x] = std::make_pair(features.slice(0, 0, split_idx), features.slice(0, split_idx));
    auto&& [train_y, remaining_y] = std::make_pair(labels.slice(0, 0, split_idx), labels.slice(0, split_idx));

    auto test_idx = long(double(remaining_x.size(0))*0.5);
    auto&& [ val_x, test_x] = std::make_pair(remaining_x.slice(0, 0, test_idx), remaining_x.slice(0, test_idx));
    auto&& [val_y, test_y] = std::make_pair(remaining_y.slice(0, 0, test_idx), remaining_y.slice(0, test_idx));

    printf("\t\t\tFeature Shapes:\n");
    std::cout << "Train set: \t\t" << train_x.sizes() << "\nValidation set: \t" << val_x.sizes()
    << "\nTest set: \t\t" << test_x.sizes() << "\n";

    torch::data::DataLoaderOptions opt;
    opt.batch_size(batch_size);

    auto train_loader = torch::data::make_data_loader(MyDataset(train_x, train_y).
            map(torch::data::transforms::Stack<>()), opt);
    auto valid_loader = torch::data::make_data_loader(MyDataset(val_x, val_y).
            map(torch::data::transforms::Stack<>()), opt);
    auto test_loader = torch::data::make_data_loader(MyDataset(test_x, test_y).
            map(torch::data::transforms::Stack<>()), opt);

    sam_dn::RecurrentNetOption netOption;
    netOption.drop_out=0.5;
    netOption.num_layers =n_layers;
    netOption.device = DeviceTypeName(device, true);
    netOption.batch_size = opt.batch_size();
    netOption.hidden_size=hidden_dim;
    netOption.weight_init_param = sqrt(2);
    netOption.weight_init_type = "orthogonal";
    netOption.Input({embedding_dim});

    sam_dn::EmbeddingOption option;
    option.embed_dim = embedding_dim;
    option.Input({vocab_size});

    torch::nn::Sequential net( sam_dn::Embedding( option ),
                              sam_dn::ExpandAtAxis0IfDimEqual2(),
                              sam_dn::Transpose01(),
                              sam_dn::GRUTimeFirst(netOption),
                              torch::nn::Dropout(0.3),
                              torch::nn::Linear(hidden_dim, output_size),
                              torch::nn::Sigmoid());
    net->to(device);
    std::cout << net << "\n";

    train<sam_dn::GRUTimeFirstImpl>(net, train_loader, valid_loader);
    test<sam_dn::GRUTimeFirstImpl>(net, test_loader);
}