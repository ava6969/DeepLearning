//
// Created by dewe on 9/29/21.
//

#include "world_model.h"
#include "opencv2/opencv.hpp"

namespace sam_rl{
    void WorldModelImpl::record(
            const std::function<torch::Tensor(WorldModelImpl *, const torch::Tensor &)> &actionSampler,
            const std::filesystem::path &saveDIR, int MAX_FRAMES, int MAX_TRIAL) {

        auto serialize_recorded = not saveDIR.empty();
        if(serialize_recorded and not std::filesystem::exists(saveDIR)){
            std::filesystem::create_directory(saveDIR);
        }

        auto env = m_EnvMaker();
        int totalFrames = 0;
        eval();
        for(int i = 0; i < MAX_TRIAL; i++){

            std::default_random_engine rd(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::vector<torch::Tensor> recording_obs, recording_action, recording_done;
            reset();
            auto _seed = std::uniform_int_distribution<uint64_t>()(rd);
            env->seed(_seed);
            torch::cuda::manual_seed(_seed);
            torch::manual_seed(_seed);
            torch::cuda::manual_seed_all(_seed);

            for(auto& params : parameters()){
                auto loc = torch::zeros_like(params);
                params = loc.cauchy_() * torch::rand({1}).item<double>()*sqrt(0.01);
            }

            auto filename = "/vae_record_" + std::to_string(_seed) + ".pt";

            auto obs = env->reset();

            int frame;
            for(frame = 0; frame < MAX_FRAMES; frame++){
                if(m_Render)
                    env->render();

                auto img = process(obs[m_ImgKey]).to(m_Device);
                recording_obs.emplace_back(img.cpu());

                auto[z, mu, logvar, _] = m_VAE->forward(img.unsqueeze(0) / 255.0f);

                auto action = actionSampler(this, z);
                recording_action.emplace_back(action.cpu());

                auto stepOutput = env->step(action.cpu());
                recording_done.emplace_back(torch::tensor(stepOutput.done).toType(torch::kBool));
                obs = stepOutput.m_NextObservation;

                m_RNN->forward(torch::cat({z, action.view({1, -1})}, -1).unsqueeze(0).to(m_Device));

                if(stepOutput.done)
                    break;
            }
            totalFrames += (frame+1);
            std::cout << "dead at " <<  frame+1 << " total recorded frames for this worker " << totalFrames << "\n";
            SerializableExperience serializableExperience(torch::stack(recording_obs),
                                                          torch::stack(recording_action),
                                                          torch::stack(recording_done));
            auto _dir = saveDIR;
            const auto& save_path = _dir.concat(filename);
            torch::save(serializableExperience, save_path);
        }
    }

    void WorldModelImpl::createDataSet(vector<torch::Tensor> &observations,
                                       std::filesystem::path const& directory,
                                       uint64_t N,
                                       int M,
                                       std::vector<torch::Tensor>* dones,
                                       std::vector<torch::Tensor>* actions) {

        int i = 0;
        std::vector<std::filesystem::path> filePaths;

        for(auto const& file : std::filesystem::directory_iterator(directory)){
            filePaths.push_back(file.path());
        }

        for(auto const& path: filePaths){
            SerializableExperience exp;
            torch::load(exp, path);
            if(exp->observation.size(0) < M)
                continue;
            auto splitted = torch::tensor_split(exp->observation, M, 0);
            observations.insert(observations.end(), splitted.begin(), splitted.end());

            if(actions){
                splitted = torch::tensor_split(exp->action, M, 0);
                actions->insert(actions->end(), splitted.begin(), splitted.end());
            }

            if(dones){
                splitted = torch::tensor_split(exp->action, M, 0);
                dones->insert(dones->end(), splitted.begin(), splitted.end());
            }

            if( (i+1) % 100 == 0){
                std::cout << "loading file " << i+1 << "\n";
            }
            i++;
            if(i > N)
                break;
        }
    }

    void
    WorldModelImpl::trainVAE(const std::filesystem::path &directory,
                             std::filesystem::path const& save_to, int batch_size, int n_epochs, float learning_rate,
                             float kl_tolerance, int N, int M) {

        torch::serialize::OutputArchive archive;
        auto dataLoader = createDataLoader(directory, batch_size, N, M);
        auto optimizer = torch::optim::Adam(parameters(), learning_rate);
        int train_step = 0;
        std::cout << "epoch " << 0 << "/" << n_epochs << ", step " << train_step + 1 <<
                  "\t" << "loss" << "\t" << "r_loss" << "\t" << "kl_loss"
                  << "\n";
        for(auto epoch = 0; epoch < n_epochs; epoch++) {

            for (auto const &batch: *dataLoader) {
                auto B = static_cast<long>(batch.size());
                auto x = torch::vstack(batch).to(this->m_Device); // already normalized
                auto x_norm = x/255;
                auto[z, mu, logvar, y] = m_VAE->forward(x_norm);

                auto z_size = z.size(-1);

                auto r_loss = torch::sum((x_norm - y).pow(2), {1, 2, 3}).mean();

                auto kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1);
                kl_loss = torch::maximum(kl_loss, torch::ones_like(kl_loss) * kl_tolerance * z_size).mean();

                auto loss = r_loss + kl_loss;

                optimizer.zero_grad();

                loss.backward();
                optimizer.step();

                if ((train_step + 1) % 500 == 0) {
                    std::cout << "epoch " << epoch << "/" << n_epochs << ", step " <<
                              train_step + 1 << "\t" << (loss/B).item<float>() << "\t"
                              << (r_loss/B).item<float>() << "\t" << (kl_loss/B).item<float>()
                              << "\n";
                }
                if ((train_step + 1) % 500 == 0) {
                    torch::save(m_VAE, save_to);
                }
                train_step++;
            }
        }
        torch::save(m_VAE, save_to);
    }

    void WorldModelImpl::record(const std::filesystem::path &directory, const std::filesystem::path &vae_path,
                                const std::filesystem::path &saveDIR, int batch_size, int N, int M) {

        auto dataLoader = createDataLoaderWithAction(directory, batch_size, N, M);

        auto serialize_recorded = not saveDIR.empty();
        if(serialize_recorded and not std::filesystem::exists(saveDIR)){
            std::filesystem::create_directory(saveDIR);
        }

        torch::load(m_VAE, vae_path);

        std::vector<torch::Tensor> mu, log_var, actions_;
        for (auto const &batch: *dataLoader) {
            auto B = static_cast<long>(batch.size());
            std::vector<torch::Tensor> img_v(B), actions_v(B);
            std::transform(batch.begin(), batch.end(), img_v.begin(), actions_v.begin(), [](auto& b, auto&i){
                i = b.first;
                return b.second;
            });

            auto img = torch::vstack(img_v).to(this->m_Device); // already normalized
            auto actions = torch::vstack(actions_v).to(this->m_Device); // already normalized

            auto simple_obs = img/255;
            auto[z, _mu, _logvar, _] = m_VAE->forward(simple_obs);

            mu.emplace_back(_mu.toType(torch::kFloat16));
            log_var.emplace_back(_logvar.toType(torch::kFloat16));
            actions_.emplace_back(actions);
        }
        torch::serialize::OutputArchive a;
        a.write("series", std::make_tuple(torch::vstack(actions_), torch::vstack(mu), torch::vstack(log_var)));
        auto filename = saveDIR / "/series.pt";
        a.save_to(filename);

    }

    torch::Tensor WorldModelImpl::process(const torch::Tensor &obs)  {
        cv::Mat img(obs.size(0), obs.size(1), CV_8UC3, obs.data_ptr<uint8_t>());
        cv::resize(img, img, {64, 64}, 0, 0, cv::INTER_AREA);
        return torch::from_blob(img.data, {64*64*3}, torch::kUInt8)
        .view({64, 64, 3})
        .permute({2, 0, 1})
        .toType(torch::kUInt8);
    }

    void SerializableExperienceImpl::save(torch::serialize::OutputArchive &archive) const {
            archive.write("observation", observation);
            archive.write("action", action);
            archive.write("done", done);
        }

    void SerializableExperienceImpl::load(torch::serialize::InputArchive &archive) {
        archive.read("observation", observation);
        archive.read("action", action);
        archive.read("done", done);
    }
}
