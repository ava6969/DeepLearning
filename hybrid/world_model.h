//
// Created by dewe on 9/29/21.
//

#ifndef SAMFRAMEWORK_WORLD_MODEL_H
#define SAMFRAMEWORK_WORLD_MODEL_H

#include <filesystem>
#include <utility>
#include "common/environments/sam_env.h"
#include "base.h"
#include "vision/cnn_vae.h"
#include "memory/mdn.h"
#include "tuple"

namespace sam_rl{


    struct SerializableExperienceImpl : torch::nn::Module {
        torch::Tensor observation, action, done;

        SerializableExperienceImpl()=default;
        SerializableExperienceImpl(torch::Tensor observation,
                               torch::Tensor action,
                               torch::Tensor done):
                               observation(register_buffer("observation", std::move(observation))),
                               action(register_buffer("action", std::move(action))),
                               done(register_buffer("done", std::move(done))){

        }

        void save(torch::serialize::OutputArchive &archive) const override;

        void load(torch::serialize::InputArchive &archive) override;
    };

    TORCH_MODULE(SerializableExperience);

    class ImageDataset : public torch::data::datasets::Dataset<ImageDataset, torch::Tensor>{
        std::vector<torch::Tensor> data;

    public:
        explicit ImageDataset(std::vector<torch::Tensor>  _data):data(std::move(_data)){}

        ExampleType get(size_t index) override{
            return data[index];
        }

        [[nodiscard]] torch::optional<size_t> size() const override{
            return data.size();
        }

    };

    class ImageActionDataset : public torch::data::datasets::Dataset<ImageDataset,
            std::pair<torch::Tensor, torch::Tensor>>{
        std::vector<torch::Tensor> images, actions;

    public:
        explicit ImageActionDataset(std::vector<torch::Tensor> images,
                                    std::vector<torch::Tensor> actions):
                images(std::move(images)),
                actions(std::move(actions)){}

        ExampleType get(size_t index) override{
            return std::make_pair(images[index], actions[index]);
        }

        [[nodiscard]] torch::optional<size_t> size() const override{
            return images.size();
        }

    };


    class WorldModelImpl : public BaseModuleImpl{

    private:
        CnnVae m_VAE;
        MDNLSTM m_RNN;
        bool m_FullEpisode=false;
        bool m_Render=false;
        int m_Seed=0;
        EnvCreator m_EnvMaker;
        std::tuple<torch::Tensor, torch::Tensor> m_LSTMState;
        c10::Device m_Device;
        std::string m_ImgKey;

        inline auto createDataLoader(std::filesystem::path const& directory, int batch_size,int N, int M){
            std::vector<torch::Tensor> data;
            createDataSet(data, directory, N, M);
            auto dataSet = ImageDataset(data);
            return torch::data::make_data_loader(std::move(dataSet), batch_size);
        }

        inline auto createDataLoaderWithAction(std::filesystem::path const& directory, int batch_size,int N, int M){
            std::vector<torch::Tensor> images, actions;
            createDataSet(images, directory, N, M, nullptr, &actions);
            auto dataSet = ImageActionDataset(images, actions);
            return torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataSet), batch_size);
        }

    public:
        WorldModelImpl(EnvCreator  envMaker,
                       CnnVae vae,
                       MDNLSTM mdnlstm,
                       c10::Device device,
                       bool fullEpisode,
                       bool renderMode,
                       std::string  img_key="observation",
                       int seed=0):
                       m_VAE(std::move(vae)),
                       m_RNN(std::move(mdnlstm)),
                       m_FullEpisode(fullEpisode),
                       m_Render(renderMode),
                       m_Seed(seed),
                       m_EnvMaker(std::move(envMaker)),
                       m_Device(device),
                       m_ImgKey(std::move(img_key)){

            register_module("vae", m_VAE);
            register_module("mdn_rnn", m_RNN);
            this->to(m_Device);
        }

       inline void reset(int batchSize=1){
           m_LSTMState = m_RNN->initial_state(batchSize, m_Device);
        }

        auto getState() const { return m_RNN->get_state(); }


        TensorDict forward(TensorDict x) override{
            return {};
        }

        torch::Tensor forward_simple(const torch::Tensor &x) override{

        }

        torch::Tensor process(torch::Tensor const& obs);

        void record(std::function<torch::Tensor(WorldModelImpl*, torch::Tensor const&)> const& actionSampler,
                    std::filesystem::path const& saveDIR={},
                    int MAX_FRAMES=1000,
                    int MAX_TRIAL=200);

        void trainVAE(std::filesystem::path const& directory,
                      std::filesystem::path const& save_to, int batch_size, int n_epochs,
                      float learning_rate, float kl_tolerance, int N, int M);

        void record(std::filesystem::path const& directory,
                         std::filesystem::path const& vae_path,
                            std::filesystem::path const& saveDIR,
                         int batch_size, int N, int M);

        void
        createDataSet(vector<torch::Tensor> &observations, const std::filesystem::path &directory, uint64_t N, int M,
                      vector<torch::Tensor> *dones=nullptr, vector<torch::Tensor> *actions=nullptr);
    };

    TORCH_MODULE(WorldModel);
}

#endif //SAMFRAMEWORK_WORLD_MODEL_H
