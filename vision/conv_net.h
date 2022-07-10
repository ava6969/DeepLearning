//
// Created by dewe on 9/25/21.
//

#ifndef SAMFRAMEWORK_CONV_NET_H
#define SAMFRAMEWORK_CONV_NET_H

#include <utility>
#include "torch/torch.h"
#include "base.h"
#include "cassert"

//#ifdef DEBUG_VISION
//#include "vision_debugger.h"
//#endif

namespace sam_dn {

    struct Conv2DInput {
        int width{}, height{}, channel{};
    };

    struct CNNOption : BaseModuleOption{
    private:
        Conv2DInput inputShape;

    public:
        std::vector<int> filters{}, kernels{}, strides{};
        std::vector <std::string> padding{}, activations{};
        bool flatten_output = {false};

        CNNOption()=default;
        CNNOption(std::vector<int>  filters):filters(std::move( filters)){}

        CNNOption& setKernels(std::vector<int> const& x) { kernels = x; return *this; }
        CNNOption& setStrides(std::vector<int> const& _strides) { strides = _strides; return *this; }
        CNNOption& setPaddings(std::vector <std::string> const& p) { padding = p; return *this; }
        CNNOption& setActivations(std::vector <std::string> const& a) { activations = a; return *this; }
        CNNOption& flattenOut(bool _flatten_out) { flatten_output = _flatten_out; return *this; }

        BaseModuleOption& Input(std::vector<int64_t> const& x) override;

        [[nodiscard]] auto InputShape() const { return inputShape; }

        void setInput(Conv2DInput const& in){
            inputShape = in;
        }
    };

    template<typename Net, typename Option>
    class ConvNetImpl : public BaseModuleImpl<CNNOption> {

    protected:
        Conv2DInput in_shape;

        void build(CNNOption opt);

        int outputShape(int side, int padding, int dilation, int kernel, int stride);

        Conv2DInput outputShape(Conv2DInput prev,
                                       int padding,
                                       int dilation, int kernel, int stride);

    public:
        ConvNetImpl()=default;

        explicit ConvNetImpl(CNNOption const& opt);

        inline torch::Tensor forward(const torch::Tensor &x) noexcept override {

//#ifndef DEBUG_VISION
           return m_BaseModel->forward(x.view({-1, in_shape.channel, in_shape.height, in_shape.width}));
//#else
//            auto img = x.view({-1, in_shape.channel, in_shape.height, in_shape.width});
//            auto result = m_BaseModel->forward(img);
//
//            torch::NoGradGuard noGradGuard;
//            auto _training = this->is_training();
//
//            if(_training)
//                this->eval();
//
//            for(auto const& net_pair : m_BaseModel->named_children()){
//
//                if(auto* _net = net_pair.value()->template as<torch::nn::Conv2d>()){
//                    img = _net->forward(img);
//                    VisionDebugger::ptr()->addImages(this->m_Input + "_" + net_pair.key(),
//                                              result.flatten(0, 1).unsqueeze(1));
//                }else if(auto* p_net = net_pair.value()->template as<torch::nn::ZeroPad2d>()){
//                    img = p_net->forward(img);
//                    VisionDebugger::ptr()->addImages(this->m_Input + "_" + net_pair.key(),
//                                              result.flatten(0, 1).unsqueeze(1));
//                }
//            }
//            this->train(_training);
//            return result;
//#endif
        }
    };

    class CNNImpl : public ConvNetImpl<torch::nn::Conv2d, torch::nn::Conv2dOptions> {

    public:
        CNNImpl()=default;

        explicit CNNImpl(CNNOption const& opt);

        void pretty_print(std::ostream& stream) const override {
            stream  << "sam_dn::ConvNet"
                    << "("
                    << "bias_init_param=" << this->opt.bias_init_param << ", "
                    << "bias_init_type=" << this->opt.bias_init_type << ", "
                    << "flatten_output=" << this->opt.flatten_output << ", "
                    << "weight_init_param=" << this->opt.weight_init_param << ", "
                    << "weight_init_type=" << this->opt.weight_init_type
                    << ")";
        }
        explicit CNNImpl(BaseModuleOption& opt):CNNImpl( dynamic_cast<CNNOption&>(opt) ) {}
    };

    struct CNNTransposeImpl : public ConvNetImpl<torch::nn::ConvTranspose2d, torch::nn::ConvTranspose2dOptions>  {
        explicit CNNTransposeImpl(CNNOption  const& opt);
    };

    TORCH_MODULE(CNN);
    TORCH_MODULE(CNNTranspose);

}

SAM_OPTIONS2(Conv2DInput, SELF(width), SELF(height), SELF(channel))

SAM_OPTIONS(BaseModuleOption, CNNOption, SELF(filters), SELF(kernels), SELF(strides),
            SELF(padding), SELF(activations), SELF(flatten_output))

#endif //SAMFRAMEWORK_CONV_NET_H
