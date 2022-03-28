//
// Created by dewe on 3/26/22.
//

#ifndef DEEP_NETWORKS_COMMON_H
#define DEEP_NETWORKS_COMMON_H

#include "exception"
#include "string"
#include "sstream"
#include "unordered_map"

namespace sam_dn{

    template<class K, class V>
    inline auto try_get(std::unordered_map<K, V> const& _dict,
                       K const& k,
                       std::string const& info=""){
        try{
           return _dict.at(k);
        } catch (std::runtime_error const& x) {
            std::stringstream ss;
            ss << "KeyNotFoundError: " << k << info << "\n";
            throw std::runtime_error(ss.str());
        }
    }

    std::vector<torch::Tensor> operator+(std::vector<torch::Tensor> const& x,
                                         std::vector<torch::Tensor> const& y){
        std::vector<torch::Tensor> z( std::min(x.size(), y.size()));
        std::transform(x.begin(), x.end(), y.begin(), z.begin(), [](auto const& a, auto const& b){
            return a + b;
        });
        return z;
    }
}

#endif //DEEP_NETWORKS_COMMON_H
