//
// Created by dewe on 6/11/22.
//

#define DEBUG_VISION
#include "vision/convLSTM.h"


int main(){
    torch::nn::Conv2DLSTMOption c_lstm;
    c_lstm.cell_opt.input_shape = {3, 4, 4};
    c_lstm.cell_opt.kernel_shape = 3;
    c_lstm.cell_opt.output_channel = 64;
    c_lstm.batch_first = true;

    torch::nn::Conv2DLSTM c1(c_lstm);
    std::cout << c1 << "\n";

    auto test = torch::randint(255, {5, 7, 3, 4, 4}) / 255;
    auto [out, s] = c1(test);

    std::cout << out.sizes() << "\n";
    std::cout << std::get<0>(s).sizes() << "\n";
    std::cout << std::get<1>(s).sizes() << "\n";

    c_lstm.batch_first = false;
    test = torch::randint(255, {7, 5, 3, 4, 4}) / 255;
    torch::nn::Conv2DLSTM c2(c_lstm);
    std::tie(out, s) = c2(test);

    std::cout << out.sizes() << "\n";
    std::cout << std::get<0>(s).sizes() << "\n";
    std::cout << std::get<1>(s).sizes() << "\n";

}