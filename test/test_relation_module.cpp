//
// Created by dewe on 6/11/22.
//

#define DEBUG_VISION
#include "vision/relational_module.h"


int main(){
    sam_dn::RelationalModuleImpl::Option opt;
    opt.recurrent = true;
    opt.n_blocks = 2;
    opt.attn.n_heads = 4;
    opt.attn.n_embed = 256;
    opt.input = "test";
    opt.dict_opt["test"] = {64, 14, 14};

    RelationalModule encode(opt);
    std::cout << encode << "\n";

    auto test = torch::randint(255, {1, 64, 14, 14}) / 255;
    std::cout << encode(test).sizes() << "\n";

    for(auto const& n: encode->named_parameters())
        std::cout << n.key() << "\t" <<  n.value().data_ptr() << "\n";
}