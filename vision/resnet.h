//
// Created by dewe on 2/14/22.
//

#ifndef SAM_RL_TRADER_RESNET_H
#define SAM_RL_TRADER_RESNET_H

#include <torch/torch.h>
#include "common/yaml_options.h"
#include "common/common.h"

namespace sam_dn{

    static auto
    create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                        int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,
                        int64_t dilation = 1, bool bias = false)
    {
        return torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size)
                .stride(stride)
                .padding(padding)
                .bias(bias)
                .groups(groups)
                .dilation(dilation);
    }

    static auto
    conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride = 1,
                           int64_t groups = 1, int64_t dilation = 1){
        return torch::nn::Conv2d(create_conv_options(
                in_planes, out_planes, /*kernel_size = */ 3, stride,
                /*padding = */ dilation, groups, /*dilation = */ dilation, false));
    }

    static auto
    conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride = 1){
        return torch::nn::Conv2d(create_conv_options(
                in_planes, out_planes,
                /*kerner_size = */ 1, stride,
                /*padding = */ 0, /*groups = */ 1, /*dilation = */ 1, false));
    }

    class BasicBlockImpl : public torch::nn::Module
    {
    private:

        torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr};
        torch::nn::Sequential m_bn1{nullptr}, m_bn2{nullptr};
        torch::nn::ReLU m_relu{nullptr};
        at::optional<torch::nn::Sequential> m_down_sample{at::nullopt};

        int64_t m_stride;

    public:
        static constexpr int64_t m_expansion = 1;
        BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride = 1,
                   at::optional<torch::nn::Sequential> down_sample = at::nullopt,
                   int64_t groups = 1, int64_t base_width = 64,
                   int64_t dilation = 1,
                   std::function<torch::nn::Sequential(int)> norm_layer=nullptr){

            if(! norm_layer)
                norm_layer = [](int planes) { return torch::nn::Sequential(torch::nn::BatchNorm2d(planes)); };

            if ((groups != 1) or (base_width != 64))
            {
                throw std::invalid_argument{"BasicBlock only supports groups=1 and base_width=64"};
            }
            if (dilation > 1)
            {
                throw std::invalid_argument{"Dilation > 1 not supported in BasicBlock"};
            }

            m_conv1 = register_module("conv1", conv3x3(in_planes, planes, stride));
            m_bn1 = register_module("bn1", norm_layer(planes));
            m_relu = register_module("relu", torch::nn::ReLU{true});
            m_conv2 = register_module("conv2", conv3x3(planes, planes));
            m_bn2 = register_module("bn2", norm_layer(planes));

            if (down_sample)
            {
                m_down_sample = register_module("downsample", down_sample.value());
            }
            m_stride = stride;
        }

        torch::Tensor forward(torch::Tensor const& x)
        {
            auto identity = x;

            auto out = m_conv1(x);
            out = m_bn1->forward(out);
            out = m_relu(out);

            out = m_conv2(out);
            out = m_bn2->forward(out);

            if (m_down_sample)
            {
                identity = m_down_sample.value()->forward(x);
            }

            out += identity;
            out = m_relu(out);

            return out;
        }

    };

    class BottleneckImpl : public torch::nn::Module
    {
        torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr}, m_conv3{nullptr};
        torch::nn::Sequential m_bn1{nullptr}, m_bn2{nullptr}, m_bn3{nullptr};
        torch::nn::ReLU m_relu{nullptr};
        at::optional<torch::nn::Sequential> m_down_sample{at::nullopt};

        int64_t m_stride;

    public:
        static constexpr int64_t m_expansion = 4;
        BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride = 1,
                       at::optional<torch::nn::Sequential> down_sample = at::nullopt,
                       int64_t groups = 1, int64_t base_width = 64,
                       int64_t dilation = 1,
                       std::function<torch::nn::Sequential(int)> norm_layer=nullptr)
        {
            if(! norm_layer)
                norm_layer = [](int planes) { return torch::nn::Sequential(torch::nn::BatchNorm2d(planes)); };

            int64_t width = planes * (base_width / 64) * groups;

            m_conv1 = register_module("conv1", conv1x1(inplanes, width));
            m_bn1 = register_module("bn1", norm_layer(width));
            m_conv2 = register_module("conv2", conv3x3(width, width, stride, groups, dilation));
            m_bn2 = register_module("bn2", norm_layer(width));
            m_conv3 = register_module("conv3", conv1x1(width, planes * m_expansion));
            m_bn3 = register_module("bn3", norm_layer(planes * m_expansion));
            m_relu = register_module("relu", torch::nn::ReLU{true});
            if (down_sample)
            {
                m_down_sample = register_module("downsample", down_sample.value());
            }
            m_stride = stride;
        }

        torch::Tensor forward(torch::Tensor x)
        {
            torch::Tensor identity = x;

            torch::Tensor out = m_conv1(x);
            out = m_bn1->forward(out);
            out = m_relu(out);

            out = m_conv2(out);
            out = m_bn2->forward(out);
            out = m_relu(out);

            out = m_conv3(out);
            out = m_bn3->forward(out);

            if (m_down_sample)
            {
                identity = (*m_down_sample)->forward(x);
            }

            out += identity;
            out = m_relu(out);

            return out;
        }
    };


    class ResNetImpl : public torch::nn::Module
    {

    private:
        std::function<torch::nn::Sequential(int)> norm_layer{nullptr};
        int64_t m_inplanes = 64;
        int64_t m_dilation = 1;
        int64_t m_groups = 1;
        int64_t m_base_width = 64;

        torch::nn::Conv2d m_conv1{nullptr};
        torch::nn::Sequential m_bn1{nullptr};
        torch::nn::ReLU m_relu{nullptr};
        torch::nn::MaxPool2d m_maxpool{nullptr};
        torch::nn::Sequential m_layer1{nullptr}, m_layer2{nullptr},
                m_layer3{nullptr}, m_layer4{nullptr};
        torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
        torch::nn::Linear m_fc{nullptr};

        template<class BlockType>
        torch::nn::Sequential  make_layer(int64_t planes, int64_t blocks, int64_t stride = 1, bool dilate = false)
        {
            if(blocks < 1)
                return nullptr;

            auto _norm_layer = this->norm_layer;
            at::optional<torch::nn::Sequential> downsample{at::nullopt};
            int64_t previous_dilation = m_dilation;

            if (dilate)
            {
                m_dilation *= stride;
                stride = 1;
            }
            if ((stride != 1) || (m_inplanes != planes * BlockType::m_expansion))
            {
                torch::nn::Sequential modules = _norm_layer(planes * BlockType::m_expansion);
                modules->push_back(conv1x1(m_inplanes, planes * BlockType::m_expansion, stride));
                std::reverse(modules->begin(), modules->end());
                downsample = modules;
            }

            torch::nn::Sequential layers{};

            layers->push_back(std::make_shared<BlockType>(m_inplanes, planes, stride, downsample,
                                    m_groups, m_base_width, previous_dilation, _norm_layer));

            m_inplanes = planes * BlockType::m_expansion;
            for (int64_t i = 0; i < blocks; i++)
            {
                layers->push_back( std::make_shared<BlockType>(m_inplanes, planes, 1, at::nullopt,
                                                               1, m_base_width, m_dilation,
                                                               norm_layer) );
            }

            return layers;
        }

    public:
        ResNetImpl()=default;

        template <typename BlockType>
        ResNetImpl(at::optional<BlockType> ,
                   const std::vector<int64_t>& layers,
                   std::optional<int64_t> num_classes = std::nullopt,
                   bool zero_init_residual = false, int64_t groups = 1,
                   int64_t width_per_group = 64,
                   at::optional<std::vector<bool>> replace_stride_with_dilation = at::nullopt,
                   std::function<torch::nn::Sequential(int)> norm_layer=nullptr)
        {
            if(! norm_layer)
                norm_layer = [](int planes) { return torch::nn::Sequential(torch::nn::BatchNorm2d(planes)); };
            this->norm_layer = norm_layer;

            if (not replace_stride_with_dilation)
            {
                // Each element in the tuple indicates if we should replace
                // the 2x2 stride with a dilated convolution instead.
                replace_stride_with_dilation = {false, false, false};
            }
            if (replace_stride_with_dilation->size() != 3)
            {
                throw std::invalid_argument{
                        "replace_stride_with_dilation should be empty or have exactly "
                        "three elements."};
            }

            m_groups = groups; // previous code m_groups = m_groups;
            m_base_width = width_per_group;

            m_conv1 = register_module("conv1",
                                      torch::nn::Conv2d{create_conv_options(
                            /*in_planes = */ 3, /*out_planes = */ m_inplanes,
                            /*kerner_size = */ 7, /*stride = */ 2, /*padding = */ 3,
                            /*groups = */ 1, /*dilation = */ 1, /*bias = */ false)});

            m_bn1 = register_module("bn1", norm_layer(m_inplanes));
            m_relu = register_module("relu", torch::nn::ReLU{true});
            m_maxpool = register_module("maxpool", torch::nn::MaxPool2d{
                            torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})});

            if(layers.at(0)>0)
                m_layer1 = register_module("layer1", make_layer<BlockType>(64, layers.at(0)));
            if(layers.at(1)>0)
                m_layer2 = register_module("layer2", make_layer<BlockType>(128, layers.at(1), 2,
                                                                       replace_stride_with_dilation->at(0)));
            if(layers.at(2)>0)
                m_layer3 = register_module("layer3", make_layer<BlockType>(256, layers.at(2), 2,
                                                                           replace_stride_with_dilation->at(1)));
            if(layers.at(3)>0)
                m_layer4 = register_module("layer4", make_layer<BlockType>(512, layers.at(3), 2,
                                                                           replace_stride_with_dilation->at(2)));

            if(num_classes){
                m_avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
                        torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
                m_fc = register_module("fc", torch::nn::Linear(512 * BlockType::m_expansion, num_classes.value()));
            }


            // auto all_modules = modules(false);
            // https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#_CPPv4NK5torch2nn6Module7modulesEb
            for (auto const& m : modules(false))
            {
                if (auto conv_net = m->template as<torch::nn::Conv2dImpl>())
                {
                    torch::nn::init::kaiming_normal_(conv_net->weight, 0, torch::kFanOut, torch::kReLU);
                }
                else if (auto batch2d = m->template as<torch::nn::BatchNorm2dImpl>()){
                    torch::nn::init::constant_(batch2d->weight, 1.0);
                    torch::nn::init::constant_(batch2d->bias, 0.0);
                }
                else if (auto group_norm = m->template as<torch::nn::GroupNormImpl>()){
                    torch::nn::init::constant_(group_norm->weight, 1.0);
                    torch::nn::init::constant_(group_norm->bias, 0.0);
                }
            }

            if (zero_init_residual)
            {
                for (auto m : modules(false))
                {
                    if (auto bottle_neck = m->template as<BottleneckImpl>())
                    {
                        torch::nn::init::constant_(*(bottle_neck->named_modules()["bn3"]->named_parameters(false)
                        .find("weight")), 0.0);
                    }
                    else if (auto basic_block = m->template as<BasicBlockImpl>())
                    {
                        torch::nn::init::constant_(*(basic_block->named_modules()["bn2"]->named_parameters(false)
                                .find("weight")), 0.0);
                    }
                }
            }
        }

        torch::Tensor _forward_impl(torch::Tensor x)
        {

            x = m_conv1(x);
            x = m_bn1->forward(x);
            x = m_relu(x);
            x = m_maxpool(x);
            if(m_layer1)
                x = m_layer1->forward(x);
            if(m_layer2)
                x = m_layer2->forward(x);
            if(m_layer3)
                x = m_layer3->forward(x);
            if(m_layer4)
                x = m_layer4->forward(x);

            if(m_fc){
                x = m_avgpool(x);
                x = torch::flatten(x, 1);
                x = m_fc(x);
            }

            return x;
        }

        torch::Tensor forward(torch::Tensor const& x) { return _forward_impl(x); }
    };

    TORCH_MODULE(ResNet);

    template <typename BlockType, class ... Args>
    static auto _resnet(const std::vector<int64_t>& layers, Args ... args)
    {
       return ResNet(at::optional<BlockType>{at::nullopt},
                     layers, args ...);
    }

    template <class ... Args>
    static auto resnet18(Args ... args)
    {
        return _resnet<BasicBlockImpl>(std::vector<int64_t>{2, 2, 2, 2}, args ...);
    }

    template <class ... Args>
    static auto resnet34(Args ... args)
    {
        return _resnet<BasicBlockImpl>(std::vector<int64_t>{3, 4, 6, 3}, args ...);
    }

    template <class ... Args>
    static auto resnet50(Args ... args)
    {
        return _resnet<BottleneckImpl>(std::vector<int64_t>{3, 4, 6, 3}, args ...);
    }

    template <class ... Args>
    static auto resnet101(Args ... args)
    {
        return _resnet<BottleneckImpl>(std::vector<int64_t>{3, 4, 23, 3}, args ...);
    }

    template <class ... Args>
    static auto resnet152(Args ... args)
    {
        return _resnet<BottleneckImpl>(std::vector<int64_t>{3, 4, 36, 3}, args ...);
    }

    class ResnetHolderImpl : public ModuleWithSizeInfoImpl{
    public:
            struct Option: BaseModuleOption{
                std::string res_type{"resnet43"};
                std::optional<int> num_classes{std::nullopt};
                std::string act_fn{"none"};
                std::vector<int64_t> blocks{};
                bool basic{true};
            };
        explicit ResnetHolderImpl(ResnetHolderImpl::Option opt):
                ModuleWithSizeInfoImpl(opt),opt(opt){
            ResNet base{};

            if(opt.res_type == "resnet18"){
                base = resnet18(opt.num_classes);
            }else if(opt.res_type == "resnet34"){
                base = resnet34(opt.num_classes);
            }else if(opt.res_type == "resnet50"){
                base = resnet50(opt.num_classes);
            }else if(opt.res_type == "resnet101"){
                base = resnet101(opt.num_classes);
            }else if(opt.res_type == "resnet152"){
                base = resnet152(opt.num_classes);
            }else if(opt.res_type == "custom"){
                if(opt.blocks.size() != 4){
                    throw std::runtime_error("For Custom Resnet specify(blocks: must equal be of length 4)\n");
                }
                if(opt.basic)
                    base = _resnet<BasicBlockImpl>(opt.blocks);
                else
                    base = _resnet<BottleneckImpl>(opt.blocks);
            }else{
                throw std::runtime_error(opt.res_type.append(" is invalid choose resnet(18|34|50|101|152) or custom\n"));
            }

            if(opt.num_classes){
                m_OutputSize = {opt.num_classes.value()};
            }else{
                auto in_shape = opt.dict_opt[opt.input];
                in_shape.insert(in_shape.begin(), 2);
                m_OutputSize = base->forward(torch::zeros(in_shape).requires_grad_(false)).sizes().slice(1).vec();
            }

            model = torch::nn::Sequential(base);
            addActivationFunction(opt.act_fn, model);

            register_module(opt.res_type, model);
        }

        torch::Tensor forward(torch::Tensor const& x) noexcept override{
            return model->forward(x);
        }

        inline TensorDict * forwardDict(TensorDict *  x) noexcept override{
            x->insert_or_assign(m_Output, model->forward(x->at(m_Input)));
            return x;
        }

    private:
        ResnetHolderImpl::Option opt;
            torch::nn::Sequential model{nullptr};
        };

    TORCH_MODULE(ResnetHolder);
}

namespace YAML{
    using namespace sam_dn;
    CONVERT_WITH_PARENT(BaseModuleOption, ResnetHolderImpl::Option, SELF(res_type), SELF(num_classes), SELF(act_fn),
                        SELF(blocks), SELF(basic))
}



#endif //SAM_RL_TRADER_RESNET_H
