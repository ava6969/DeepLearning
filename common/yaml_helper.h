//
// Created by dewe on 11/13/21.
//

#ifndef SAM_RL_YAML_HELPER_H
#define SAM_RL_YAML_HELPER_H

#include "yaml-cpp/yaml.h"
#include <optional>
#include <torch/torch.h>
#include "variant"

using namespace std::string_literals;

namespace YAML {

    template<typename ScalarT, typename MapT>
    struct convert< std::variant<ScalarT, MapT> >{
        static Node encode(const std::variant<ScalarT, MapT> & rhs) {
            Node node;
            return node;
        }

        static bool decode(const Node& node, std::variant<ScalarT, MapT> & self) {

            if(node.IsMap()){
                self = node.as<MapT>();
            }else if(node.IsScalar()){
                self = node.as<ScalarT>();
            }

            return true;
        }
    };

    template<>
    struct convert< c10::Device > {
        static Node encode(const c10::Device &rhs) {
            Node node;
            node.push_back(rhs.str());
            return node;
        }

        static bool decode(const Node &node, c10::Device &rhs) {
            rhs = c10::Device( node.as<std::string>() );
            return true;
        }
    };

    template<typename T>
    struct convert< std::set<T> > {
        static Node encode(const  std::set<T> &rhs) {
            Node node;
            for (auto const &it: rhs) {
                node.push_back(it);
            }
            return node;
        }

        static bool decode(const Node &node,  std::set<T> & rhs) {
            for (auto const &it: node) {
                rhs.insert( it.template as<T>() );
            }
            return true;
        }
    };

    template<typename T>
    struct convert<std::unordered_map<std::string, T> > {
        static Node encode(const std::unordered_map<std::string, T> &rhs) {
            Node node;
            for (auto const &it: rhs) {
                node[it.first] = it.second;
            }
            return node;
        }

        static bool decode(const Node &node, std::unordered_map<std::string, T> &rhs) {
            for (auto const &it: node) {
                auto key = it.begin()->first.as<std::string>();
                auto value = it.begin()->second.as<T>();
                rhs[key] = value;
            }
            return true;
        }
    };

    template<typename T>
    struct convert<std::optional<T> > {
        static Node encode(const std::optional<T> &rhs) {
            Node node;
            if (rhs.value())
                node.template push_back(rhs.value());
            else {
                node.template push_back("none");
            }
            return node;
        }

        static bool decode(const Node &node, std::optional<T> &rhs) {
            if(node.IsDefined())
                rhs = std::optional<T>(node.template as<T>());
            else
                rhs = std::nullopt;
            return true;
        }
    };
}

template<class T>
struct NodePair{
    T& value;
    std::string key;
};

template<class T>
inline static T get(YAML::Node const &val, T default_value){
    return val.IsDefined() ? val.as<T>() : default_value;
}

template<class T>
inline static T try_get(YAML::Node const & node, std::string  key){
    if( node[key].IsDefined() )
        return  node[key].as<T>();
    else{
        throw std::runtime_error("attribute ["s.append(key).append("] is required."));
    }
}

template<class T>
inline static T push_back_optional(YAML::Node &node, std::optional<T> rhs){
    node.push_back( rhs.has_value() ? std::to_string( rhs.value() ) : "~" );
}

#define DEFINE_REQUIRED(opt, input) opt. input = try_get<decltype(opt. input)>(node, #input)
#define DEFINE(opt, input, value) opt. input = get<decltype(opt. input)>(node[#input], value)
#define DEFINE_DEFAULT(opt, input) opt. input = get<decltype(opt. input)>(node[#input], opt. input)
#define DEFINE_DEFAULT_SUB_NODE(node, opt, input) opt. input = get<decltype(opt. input)>(node[#input], opt. input)
#define DEFINE_DEFAULT_RAW_NODE(opt, input) opt. input = get<decltype(opt. input)>(node, opt. input)

#define BEGIN_EXCEPTION_CHECK try{
#define END_EXCEPTION_CHECK  } catch (std::exception const& exp) { std::cerr << exp.what() << "\n"; return false; } return true;

template<class T>
inline static void push(YAML::Node& node, T attribute){
    node.push_back(attribute);
}

template<class T>
inline static void push(YAML::Node& node, NodePair<T> attribute){
    node.push_back(attribute.value);
}

template<typename ... Args>
inline static void push(YAML::Node& node, Args ... Fargs){
    push(node, Fargs...);
}

inline static void get_t(const YAML::Node &node, NodePair< std::optional<c10::Device> > arg){
    arg.value = node["device"].IsDefined() ? torch::Device(node["device"].as<std::string>()) : c10::kCPU;
}

template<class T>
inline static void get_t(const YAML::Node &node, NodePair<T> arg){
    if(arg.key.empty())
        throw std::runtime_error("NodePair arg.key is empty , which means an attribute doesnt match the text passed "
                                 "into SELF(...)");
    arg.value = get<T>(node[arg.key], arg.value);
}

template <std::size_t I, typename Tuple>
static void get_helper(const YAML::Node &node, Tuple&& tuple)
{
    get_t(node, std::get<I>(std::forward<Tuple>(tuple)));

    if constexpr (I != 0) {
        get_helper<I - 1>(node, std::forward<Tuple>(tuple));
    }
}

template<class ... Targs>
inline static void DECODE(const YAML::Node &node, Targs ... args){
    get_helper<sizeof...(Targs) - 1>( node,  std::forward_as_tuple( std::forward<Targs>(args)... ) );
}

template<class ... Args>
inline static YAML::Node ENCODE(Args ... Fargs){
    YAML::Node node;
    push(node, Fargs...);
    return node;
}

#define CONVERT(class_type, ...)     \
template<> \
struct convert<class_type> { \
    static YAML::Node encode(const class_type &self) { \
        return ENCODE(__VA_ARGS__); \
    } \
    static bool decode(const YAML::Node &node, class_type &self) { \
            BEGIN_EXCEPTION_CHECK    \
            DECODE(node, __VA_ARGS__);                 \
        END_EXCEPTION_CHECK          \
        }                            \
        };                              \


#define CONVERT_WITH_PARENT(parent_class_type, class_type, ...)     \
template<> \
struct convert<class_type> { \
    static YAML::Node encode(const class_type &self) { \
        return ENCODE(__VA_ARGS__); \
    } \
    static bool decode(const YAML::Node &node, class_type &self) {  \
        if(! convert<parent_class_type>::decode(node, self))      \
            return false;                                           \
            BEGIN_EXCEPTION_CHECK    \
        DECODE(node, __VA_ARGS__);                                  \
           END_EXCEPTION_CHECK                                                                          \
    }}; \

#define SELF(attr) NodePair<decltype(self. attr)>{const_cast< decltype(self. attr) & >(self. attr),std::string(#attr)}
#endif //SAM_RL_YAML_HELPER_H
