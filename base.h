//
// Created by dewe on 9/17/21.
//

#ifndef SAMFRAMEWORK_BASE_H
#define SAMFRAMEWORK_BASE_H

#include "common/options.h"
#include "torch/torch.h"

#define REGISTER_MODULE(attr, inst) this-> attr = register_module(#attr, inst )

using TensorDict =  std::unordered_map<std::string, torch::Tensor>;
using TensorTuple = std::tuple<torch::Tensor, torch::Tensor>;

namespace sam_dn{

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

    struct NoState{

    };

    template<class ModuleOptionT = BaseModuleOption,
            class BaseModuleT=torch::nn::Sequential,
            typename StateType=NoState,
            bool BatchFirst=false,
            bool parseRecurseDict=false>
    class BaseModuleImpl : public ModuleWithSizeInfoImpl {

    public:
        BaseModuleImpl() = default;
        explicit BaseModuleImpl(BaseModuleT const& impl) : m_BaseModel( impl ) {
            if constexpr( std::is_same_v<BaseModuleT, torch::nn::Sequential> ){

                auto modules = static_cast<torch::nn::Sequential>(impl)->modules(); // TODO: change to .children()

                auto ptr = std::find_if(modules.begin(), modules.end(),
                                        [](const std::shared_ptr<torch::nn::Module>& module){
                    return module->as<ModuleWithSizeInfoImpl>() != nullptr;
                });

                if(ptr != end(modules)){
                    auto castedModule = std::dynamic_pointer_cast<ModuleWithSizeInfoImpl>(*ptr);
                    this->m_Input = castedModule->input();
                    this->m_Output = castedModule->output();
                }
            }
        }

        explicit BaseModuleImpl(ModuleOptionT const &opt):ModuleWithSizeInfoImpl(opt),
        opt(opt){}

        inline torch::Tensor forward(torch::Tensor const& x) noexcept override {

            if constexpr(std::is_same_v<StateType, NoState>) {
                auto y = m_BaseModel->forward(x);
                return y;
            } else {
                torch::Tensor out;
                if constexpr (BatchFirst)
                    out = x.view({opt.batch_size, -1, x.size(-1)});
                else
                    out = x.view({-1, opt.batch_size, x.size(-1)});

                std::tie(out, m_States) = this->m_BaseModel->forward(out, m_States);
                return opt.return_all_seq ? out.contiguous().flatten(0, 1) :
                       out.slice(int(BatchFirst), -1).contiguous().view({opt.batch_size, -1});
            }
        }

        inline TensorDict * forwardDict(TensorDict *x) noexcept override{
            if constexpr( std::is_same_v<BaseModuleT, torch::nn::Sequential> and parseRecurseDict){
                for(auto& module : this->m_BaseModel->children())
                    if( auto _m = module->template as<ModuleWithSizeInfoImpl>() ){
                        x = _m->forwardDict(x);
                    }
            }
            else if constexpr( std::is_same_v<StateType, NoState>) {
                x->insert_or_assign(m_Output,
                                    this->m_BaseModel->forward( x->at(m_Input) ));
            }else {
                std::tie(x->at(m_Output), m_States) =
                                    this->m_BaseModel->forward( x->at(m_Input), m_States );
            }
            return x;
        }

        void registerModule(std::string const& name) noexcept{
            assert(m_BaseModel);
            register_module(name, m_BaseModel);
        }

        inline auto getState() const{
            return m_States;
        }

        inline void initialState(StateType const& new_state) noexcept {
            if constexpr( std::is_same_v<StateType, NoState>) {
                return;
            }

            if constexpr (std::is_same_v<StateType, TensorTuple>){
                auto state1 = std::get<0>(new_state);
                auto state2 = std::get<1>(new_state);
                this->m_States = std::make_tuple(state1.data().view({this->opt.num_layers, -1, this->opt.hidden_size}),
                                                 state2.data().view({this->opt.num_layers, -1, this->opt.hidden_size}));
            }else{
                this->m_States = std::move(new_state.data().view({this->opt.num_layers, -1, this->opt.hidden_size}));
            }
        }

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

#endif //SAMFRAMEWORK_BASE_H
