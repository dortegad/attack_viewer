#pragma once

//REALSENSE
//#include "pxcsensemanager.h"
//#include "pxcmetadata.h"
#include "util_cmdline.h"
#include "util_render.h"

//OPENCV
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\video\video.hpp"
#include "opencv2\objdetect\objdetect.hpp"


enum ImgType { RGB, DEPTH, IR };
enum ImgAdjustType { REAL, ADJUST_RGB, ADJUST_DEPTH };

class CamSenz3D
{
private:
	cv::VideoCapture capture;

	void imshowDepth(const char *winname, cv::Mat &depth, cv::VideoCapture &capture);
	void normalize(cv::Mat &depthImage, cv::Mat &depthNorImg);
	bool detectFace(const cv::Mat &img, cv::Rect &rect);
public:
	CamSenz3D();
	~CamSenz3D();
	int init();
	int isAttack();
	int stop();
};

