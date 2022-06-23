//
// Created by dewe on 6/11/22.
//

#define DEBUG_VISION
#include "vision/relational_module.h"


int main(){

    torch::manual_seed(1);
    sam_dn::RelationalModuleImpl::Option opt;
    opt.recurrent = true;
    opt.n_blocks = 2;
    opt.attn.n_heads = 2;
    opt.attn.head_size = 64;
    opt.input = "test";
    opt.dict_opt["test"] = {3, 5, 5};
    opt.weight_init_param = sqrt(2);
    opt.weight_init_type = "orthogonal";

    RelationalModule encode(opt);
    std::cout << encode << "\n";

    auto test = torch::randint(255, {6, 3, 5, 5}) / 255;
    std::cout << encode(test) << "\n";


}