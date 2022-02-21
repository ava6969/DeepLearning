//
// Created by dewe on 11/11/21.
//

#ifndef SAM_RL_BUILDER_H
#define SAM_RL_BUILDER_H

#include "yaml-cpp/yaml.h"
#include "filesystem"
#include "torch/torch.h"
#include "base.h"
#include "unordered_map"

using namespace std::string_literals;

namespace sam_dn{

    struct NOOption;
    using InputShapes = std::unordered_map<std::string, std::vector<int64_t> >;
    using SequentialMap = std::unordered_map<std::string, torch::nn::Sequential>;

    class Builder {

    protected:
        torch::nn::Sequential m_Model;
        std::unordered_map<std::string, std::vector<int64_t>> featureShapeInfo;
        std::unordered_map<std::string, std::function<std::shared_ptr< ModuleWithSizeInfoImpl >(YAML::Node const&,
                                                           at::optional< std::vector<int64_t> > const&)> > m_CallBacks{};

        template<class Impl, class Option>
        inline std::shared_ptr< ModuleWithSizeInfoImpl > push(std::string const& name, Option option){
            std::shared_ptr< Impl > ptr = std::make_shared<Impl>( option );
            m_Model->push_back(name,  ptr);
            return ptr;
        }

    public:

        Builder();

        template<class Impl, class Option>
        inline std::shared_ptr< ModuleWithSizeInfoImpl > make(YAML::Node const& node,
                         at::optional< std::vector<int64_t> > const& in_shape) {
            auto opt = node.as<Option>();

            if(in_shape)
                opt.Input( in_shape.value() );
            opt.dict_opt = featureShapeInfo;

            return push<Impl>(opt.output, opt);
        }

        template<class Impl, class Option=BaseModuleOption>
        inline void registerCallBack(std::string const& name){
            m_CallBacks[name] = [this](YAML::Node const& node,
                    at::optional< std::vector<int64_t> >  const& in_shape) -> std::shared_ptr< ModuleWithSizeInfoImpl >{
                return make<Impl, Option>(node, in_shape);
            };
        }

        virtual std::unordered_map<std::string, torch::nn::Sequential> compile(std::filesystem::path const&,
                                                                               InputShapes & x);

        virtual std::unordered_map<std::string, torch::nn::Sequential> compile(YAML::Node const&,
                                                                               InputShapes &);

    };
}


#endif //SAM_RL_BUILDER_H
