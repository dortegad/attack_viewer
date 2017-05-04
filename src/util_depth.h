#include "opencv2\objdetect.hpp"

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"


class Util_Depth
{
public: static void normalize(cv::Mat &depthImage, cv::Mat &depthNorImg);
};

