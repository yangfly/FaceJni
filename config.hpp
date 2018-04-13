#ifndef OCEAN_AI_CONFIG_HPP_
#define OCEAN_AI_CONFIG_HPP_

#include <iostream>
#include <string>
#include "rapidjson/document.h"
#include "common.hpp"

namespace ocean_ai {

	struct Config {
		struct Options {
			bool detection;
			bool recognition;
			Options() {}
			Options(const rapidjson::Value& v) :
				detection(v["detection"].GetBool()),
				recognition(v["recognition"].GetBool()) {}
		} options;

		struct Settings {
			int K_ctx_per_GPU;

			struct Glog {
				int level;
				std::string dir;
				Glog() {}
				Glog(const rapidjson::Value& v) : 
					level(v["level"].GetInt()), 
					dir(v["dir"].GetString()) {}
			} glog;

			struct Mtcnn {
				std::string model_dir;
				float factor;
				int min_size;
				cv::Vec3f thresholds;
				bool precise_landmark;

				struct Limitation {
					bool enable;
					int size;
					Limitation() {}
					Limitation(const rapidjson::Value& v) :
						enable(v["enable"].GetBool()),
						size(v["size"].GetInt()) {}
					Limitation(const Limitation& limit) :
						enable(limit.enable),
						size(limit.size) {}
				} limitation;

				Mtcnn() {}
				Mtcnn(const rapidjson::Value& v) :
					model_dir(v["model_dir"].GetString()),
					factor(v["factor"].GetFloat()),
					min_size(v["min_size"].GetInt()),
					precise_landmark(v["precise_landmark"].GetBool()),
					limitation(v["limitation"]) {
					if (v["thresholds"].Capacity() < 3)
						throw std::invalid_argument("thresholds are not enough in json config.");
					thresholds = cv::Vec3f(
						v["thresholds"][0].GetFloat(),
						v["thresholds"][1].GetFloat(),
						v["thresholds"][2].GetFloat());
				}
			} mtcnn;

			struct Center {
				std::string deploy;
				std::string model;
				struct Mirror {
					bool enable;
					std::string mode;
					Mirror() {}
					Mirror(const rapidjson::Value& v) :
						enable(v["enable"].GetBool()),
						mode(v["mode"].GetString()) {
						if (mode != "concat" && mode != "add" && mode != "max" && mode != "min")
							throw std::invalid_argument("Unsupported mode of mirror in json config.");
					}
				} mirror;
				struct Pca {
					bool enable;
					std::string model;
					Pca() {}
					Pca(const rapidjson::Value& v) :
						enable(v["enable"].GetBool()),
						model(v["model"].GetString()) {}
				} pca;
				FPoints ref_points;
				Center() {}
				Center(const rapidjson::Value& v) :
					deploy(v["deploy"].GetString()),
					model(v["model"].GetString()),
					mirror(v["mirror"]),
					pca(v["pca"]) {

					for (int i = 0; i < 5; ++i) {
						if (v["ref_points"].Capacity() < 10)
							throw std::invalid_argument("ref_points are not enough in json config.");
						ref_points.emplace_back(
							v["ref_points"][2 * i].GetFloat(),
							v["ref_points"][2 * i + 1].GetFloat());
					}
				}
			} center;

			Settings() {}
			Settings(const rapidjson::Value& v) :
				K_ctx_per_GPU(v["K_ctx_per_GPU"].GetInt()),
				glog(v["glog"]),
				mtcnn(v["mtcnn"]),
				center(v["center"]) {}
		} settings;

		Config() {}
		Config(const char* config_path) {
			/* string buf = fileToString(configPath); */
			char* json = readFile(config_path);
			/* cout << json << endl; */
			rapidjson::Document doc;
			doc.Parse(json);
			assert(!doc.HasParseError());
			options = Options(doc["options"]);
			settings = Settings(doc["settings"]);
			free(json);
		}

	 private:
		static char* readFile(const char* filename) {
			FILE* fp = fopen(filename, "rb");
			if (!fp)
				return 0;
			fseek(fp, 0, SEEK_END);
			size_t length = static_cast<size_t>(ftell(fp));
			fseek(fp, 0, SEEK_SET);
			char* json = reinterpret_cast<char*>(malloc(length + 1));
			size_t readLength = fread(json, 1, length, fp);
			json[readLength] = '\0';
			fclose(fp);
			return json;
		}
	};

}; // ocean_ai

#endif // OCEAN_AI_CONFIG_HPP_