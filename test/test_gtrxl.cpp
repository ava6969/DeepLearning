#include "memory/gtrxl.h"


const auto EMBEDDING_SIZE = 32;
const auto EMBEDDINGS_COUNT = 8;
const auto BATCH_SIZE = 1;

int main() {
    sam_dn::GTRXLOption opt;
    opt.embedding_size = EMBEDDING_SIZE;
    opt.num_heads = 2;
    opt.num_layers = 2;
    opt.batch_size = 1;
    opt.bg = 0.1;

    sam_dn::GTRXLImpl gtrxl { opt };
    std::cout << gtrxl << "\n";

    auto test = torch::rand({BATCH_SIZE, EMBEDDINGS_COUNT, EMBEDDING_SIZE});
    std::cout << gtrxl.forward(test).sizes() << "\n";

    for(auto const& n: gtrxl.named_parameters())
        std::cout << n.key() << "\t" <<  n.value().data_ptr() << "\n";
}