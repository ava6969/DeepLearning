//
// Created by dewe on 3/26/22.
//

#ifndef DEEP_NETWORKS_DATASET_H
#define DEEP_NETWORKS_DATASET_H


#include "torch/torch.h"


namespace sam_dn{

    struct ExampleLabel{
        torch::Tensor example, label;
    };

    struct BasicDataSplit{
        ExampleLabel train, test;
    };

    struct AdvancedDataSplit{
        ExampleLabel train, valid, test;
    };

    class Dataset{

        virtual BasicDataSplit load_data() = 0;

    };
}

#endif //DEEP_NETWORKS_DATASET_H
