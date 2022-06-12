//
// Created by dewe on 6/11/22.
//

#include "wrappers/adapters.h"


int main(){
    BaseModuleOption opt("in", "out");
    opt.fill({6, 4, 4});

    Conv2DPositionEncode encode(opt);
    std::cout << encode << "\n";

    auto test = torch::rand({1, 6, 4, 4});
    std::cout << encode(test) << "\n";
}