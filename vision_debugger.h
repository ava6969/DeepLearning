//
// Created by dewe on 2/24/22.
//

#pragma once
#include "opencv2/opencv.h"
#include "tensorboard_logger.h"
#include "filesystem"
#include "mutex"
#include "torch/torch.h"
#include "unordered_map"
#include <boost/asio/ip/host_name.hpp>

namespace sam_dn {

    class VisionDebugger {

        mutable std::mutex mtx;
        mutable std::unique_ptr<TensorBoardLogger> m_TLogger{nullptr};

        void encode(torch::Tensor processed, std::string const &tag, int step, int h, int w, int c)
        const {
            if (processed.dtype() != c10::kByte)
                processed = (processed * 255).to(c10::kByte);

            cv::Mat img()
//            std::ostringstream ss;
//            torch::save(processed, ss);
//            std::string img = ss.str();
//            m_TLogger->add_image(tag, step, img, h, w, c);
        }

    public:
        VisionDebugger() {
            GOOGLE_PROTOBUF_VERIFY_VERSION;
            std::filesystem::path const & _path = "vision_debug_run";

            std::stringstream ss;
            ss << "events.out.tfevents."
               << std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::high_resolution_clock::now().time_since_epoch()).count()
               << "." << boost::asio::ip::host_name() << "_vision_debug.pb";
            auto result = _path / ss.str();
            m_TLogger = std::make_unique<TensorBoardLogger>(result.string().c_str());
        }

        void addImage(std::string const &tag,
                      torch::Tensor const &image) {

            std::lock_guard lck(mtx);
            auto sz = image.sizes();
            auto[c, w, h] = std::make_tuple(sz[0], sz[1], sz[2]);
            auto processed = image.permute({1, 2, 0}).flatten();
            encode(processed, tag, 0, h, w, c);

        }

        void addImages(std::string const &tag,
                       torch::Tensor const & images) const {
            std::lock_guard lck(mtx);
            auto sz = images.sizes();
            auto[n, c, w, h] = std::make_tuple(sz[0], sz[1], sz[2], sz[3]);
            auto processed = images.permute({0, 2, 3, 1}).flatten(1);

            for (int i = 0; i < n; ++i) {
                auto _t = tag + std::to_string(i);
                encode(images[i].permute({1, 2, 0}).flatten(), _t, 1, h, w, c);
            }

        }
    };

#ifdef DEBUG_VISION
    const VisionDebugger VISION_DEBUGGER;
#endif
}
