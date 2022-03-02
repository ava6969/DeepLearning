//
// Created by dewe on 12/9/21.
//

#include <torch/torch.h>
#include "basic/fcnn.h"

int main(){

    sam_dn::FCNNOption opt;
    opt.act_fn = "relu";
    opt.dims = {784, 64, 32, 10};
    opt.weight_init_type = "orthogonal";
    opt.weight_init_param = std::sqrt(2.f);
    opt.bias_init_param = 1;
    opt.bias_init_type = "constant";

    sam_dn::FCNN net(opt);

    auto data_loader = torch::data::make_data_loader(
            torch::data::datasets::MNIST("examples/data").map(
                    torch::data::transforms::Stack<>()),
            /*batch_size=*/64);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::cross_entropy_loss(prediction, batch.target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }
}