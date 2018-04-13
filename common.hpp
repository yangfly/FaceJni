#ifndef OCEAN_AI_COMMON_HPP_
#define OCEAN_AI_COMMON_HPP_

#include <chrono>
#include <opencv2/opencv.hpp>

namespace ocean_ai {

	// Right Value Reference
	#define R(x) std::move(x)

	/************************************************************************/
	/*                        Ocean AI Face Types                           */
	/************************************************************************/
	// Bounding box: x1, y1, x2, y2
	typedef cv::Vec4f BBox;
	// Facial landmarks: left/right eye, nose, left/right mouth.
	typedef std::vector<cv::Point2f> FPoints;

	class FaceInfo {
	public:
		BBox bbox;		// bounding box
		float score;	// face confidence 	
		FPoints fpts;	// facial landmarks
		FaceInfo(BBox&& bbox, float score)
			: bbox(R(bbox)),
			score(score) {
		}
		FaceInfo(BBox&& bbox, float score, FPoints&& fpts)
			: bbox(R(bbox)),
			score(score),
			fpts(R(fpts)) {
		}
	};

	/************************************************************************/
	/*                         A Tool Timer                                 */
	/************************************************************************/
	/*! \brief Timer */
	class Timer {
		using Clock = std::chrono::high_resolution_clock;
	public:
		/*! \brief start or restart timer */
		inline void Tic() {
			start_ = Clock::now();
		}
		/*! \brief stop timer */
		inline void Toc() {
			end_ = Clock::now();
		}
		/*! \brief return time in ms */
		inline double Elasped() {
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
			return duration.count();
		}

	private:
		Clock::time_point start_, end_;
	};

} // ocean_ai


#endif // OCEAN_AI_COMMON_HPP_