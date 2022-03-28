//
// Created by dewe on 12/12/21.
//

#ifndef DEEP_NETWORKS_EMBEDDING_H
#define DEEP_NETWORKS_EMBEDDING_H

#include "base.h"

namespace sam_dn{

    struct EmbeddingOption: BaseModuleOption{
        int64_t embed_dim{}, embed_num{};

        void Input(std::vector<int64_t> const& x) override {
            embed_num = x.at(0);
        }
    };

    class EmbeddingImpl : public BaseModuleImpl<EmbeddingOption, torch::nn::Embedding > {
    public:

        explicit EmbeddingImpl(const EmbeddingOption& opt): BaseModuleImpl<EmbeddingOption, torch::nn::Embedding>(opt){
            torch::nn::EmbeddingOptions opts(opt.embed_num, opt.embed_dim);
            this->m_BaseModel = register_module("embed", torch::nn::Embedding(opts));
            m_OutputSize = { opts.embedding_dim() };
        }

//        void pretty_print(std::ostream& stream) const override{
//            stream << "Embedding Layer(embed=10, embed_dim=11)";
//
//            auto params = this->parameters();
//            size_t n_trainable{}, n_non_trainable{};
//
//            for(auto const& p: params){
//                if(p.requires_grad()){
//                    n_trainable += p.flatten().size(0);
//                }else{
//                    n_non_trainable += p.flatten().size(0);
//                }
//            }
//            //this->m_OutputSize;
//        }
//
//        TORCH_API friend std::ostream& operator<<(
//                std::ostream& stream,
//                const EmbeddingImpl& module) {
//            stream << "Embedding Layer";
//            return stream;
//        }

    };

    TORCH_MODULE(Embedding);
}

SAM_OPTIONS(BaseModuleOption, EmbeddingOption, SELF(embed_num), SELF(embed_dim))

#endif //DEEP_NETWORKS_EMBEDDING_H
