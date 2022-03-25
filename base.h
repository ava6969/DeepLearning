//
// Created by dewe on 9/17/21.
//

#pragma once

#include "common/yaml_helper.h"
#include "torch/torch.h"

#define REGISTER_MODULE(attr, inst) this-> attr = register_module(#attr, inst )

using TensorDict =  std::unordered_map<std::string, torch::Tensor>;
using TensorTuple = std::tuple<torch::Tensor, torch::Tensor>;

namespace sam_dn{

    struct BaseModuleOption{
        std::string input{}, output{}, weight_init_type{"none"}, bias_init_type{"none"};
        float weight_init_param{ std::sqrt(2.f) }, bias_init_param{ 0 };
        bool new_bias{};
        std::unordered_map<std::string, std::vector<int64_t>> dict_opt;

        virtual void Input(std::vector<int64_t> const&) {}
    };

    struct ModuleWithSizeInfoImpl :  torch::nn::Module{
        [[nodiscard]] inline auto outputSize() const { return m_OutputSize; }

        ModuleWithSizeInfoImpl()=default;

        ModuleWithSizeInfoImpl(ModuleWithSizeInfoImpl const&)=default;

        explicit ModuleWithSizeInfoImpl(BaseModuleOption const &opt) :
        m_Input(opt.input), m_Output(opt.output) {}

        virtual inline TensorDict * forwardDict(TensorDict *x) noexcept { return x; }
        virtual inline torch::Tensor forward(const torch::Tensor &x) noexcept { return x; }

        void input( std::string const& x) { m_Input = x; }
        void output( std::string const& x) { m_Output = x; }

        auto input( ) { return m_Input; }
        auto output( ) { return m_Output; }

        protected:
            std::string m_Input{}, m_Output{};
            std::vector<int64_t> m_OutputSize{};
    };

    struct NoState{};

#define BASEMODULE_IMPL_T template<class ModuleOptionT, class BaseModuleT, typename StateType, bool BatchFirst, bool parseRecurseDict>
#define BASEMODULET BaseModuleImpl<ModuleOptionT, BaseModuleT, StateType, BatchFirst, parseRecurseDict>

    template<class ModuleOptionT = BaseModuleOption,
            class BaseModuleT=torch::nn::Sequential,
            typename StateType=NoState,
            bool BatchFirst=false,
            bool parseRecurseDict=false>
    class BaseModuleImpl : public ModuleWithSizeInfoImpl {

    public:
        BaseModuleImpl() = default;

        explicit BaseModuleImpl(BaseModuleT const& impl);

        explicit BaseModuleImpl(ModuleOptionT const &opt):ModuleWithSizeInfoImpl(opt),
        opt(opt){}

        torch::Tensor forward(torch::Tensor const& x) noexcept override;

        TensorDict* forwardDict(TensorDict *x) noexcept override;

        inline void registerModule(std::string const& name) noexcept{
            assert(m_BaseModel);
            register_module(name, m_BaseModel);
        }

        inline auto getState() const{
            return m_States;
        }

        void initialState(StateType const& new_state) noexcept;

        bool constexpr isRecurrent(){ return not std::is_same_v<StateType, NoState>; }

        inline auto getModule() {
            return m_BaseModel;
        }

        template<class ChildImpl, class ... Args> inline static
        std::shared_ptr<BaseModuleImpl> make(Args ... arg) {
            return std::make_shared<ChildImpl>(arg...);
        }

        inline std::string inputKey() const  { return m_Input; }

    protected:
        BaseModuleT m_BaseModel{nullptr};
        ModuleOptionT opt{};
        StateType m_States;
    };

    using DefaultRecurseModule  = sam_dn::BaseModuleImpl<
            sam_dn::BaseModuleOption, torch::nn::Sequential,
            sam_dn::NoState, false, true>;

    TORCH_MODULE(ModuleWithSizeInfo);
}

#include "base.tpp"

using namespace sam_dn;

namespace YAML {
    template<>
    struct convert<sam_dn::BaseModuleOption> {
        static Node encode(const sam_dn::BaseModuleOption &self) {
            return ENCODE(SELF(input), SELF(output),
                          SELF(weight_init_param), SELF(bias_init_param),
                          SELF(weight_init_type), SELF(bias_init_type), SELF(new_bias));
        }

        static bool decode(const Node &node, sam_dn::BaseModuleOption &rhs) {

            DEFINE_REQUIRED(rhs, input);
            DEFINE_REQUIRED(rhs, output);
            DEFINE(rhs, weight_init_type, "none");
            DEFINE(rhs, bias_init_type, "none");
            DEFINE(rhs, new_bias, true);

            if (rhs.weight_init_type != "none") {
                DEFINE_REQUIRED(rhs, weight_init_param);
            }

            if (rhs.bias_init_type != "none") {
                DEFINE_REQUIRED(rhs, bias_init_param);
            }
            return true;
        }
    };
}