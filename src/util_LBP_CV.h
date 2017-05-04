#include "opencv2\objdetect.hpp"

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"


class Util_LBP_CV
{
public: static int LBP_RGB(cv::Mat &img, cv::Mat_<double> &features);
public: static int LBP_Depth(cv::Mat &imgDepth, cv::Mat_<double> &features);
};

