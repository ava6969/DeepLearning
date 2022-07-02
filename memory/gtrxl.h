#ifndef SAMFRAMEWORK_GTRXL_H
#define SAMFRAMEWORK_GTRXL_H

#include "base.h"
#include "common/common.h"

#define GTRXLTemplate template<typename StateType, typename MemoryType, bool batchFirst>
#define GTRXL_IMPL_T GTRXLImpl<StateType, MemoryType, batchFirst>

namespace sam_dn {
    /**
     * The Gated Transformer-XL (GTrXL) is a transformer-based architecture
     * used for temporal memory. It provides significant benefits over vanilla
     * transformers, due to its usage of gated layers over residual layers, and
     * its reordering of layer normalization layers, which allows for an
     * identity map from input to output.
     * 
     * \param embedding_size    The size of each input embedding.
     * \param num_heads         The number of heads to use for multi-headed attention.
     * \param num_layers        The number of transformer blocks to stack.
     * \param batch_size        Batch size.
     * \param bg                Bias term. Setting this to a value greater than 0 can
     *                          greatly speed up learning.
     */
    struct GTRXLOption : BaseModuleOption {
        int64_t embedding_size;
        int64_t num_heads;
        int64_t num_layers;
        int64_t batch_size;
        float_t bg;
    };

    GTRXLTemplate
    class GTRXLImpl : public BaseModuleImpl<GTRXLOption, MemoryType, StateType, batchFirst> {
        protected:
            torch::Device device;
            int64_t batch_base_sz;

        public:
            explicit GTRXLImpl(GTRXLOption opt);

            void to(torch::Device _device, bool non_blocking) override;
    };
}

#include "gtrxl.tpp"

SAM_OPTIONS(
    BaseModuleOption,
    GTRXLOption,
    SELF(embedding_size),
    SELF(num_heads),
    SELF(num_layers),
    SELF(batch_size),
    SELF(bg)
)

#endif //SAMFRAMEWORK_GTRXL_H