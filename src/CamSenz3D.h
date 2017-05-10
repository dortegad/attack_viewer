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
#include "opencv2\ml\ml.hpp"


enum ImgType { RGB, DEPTH, IR };
enum ImgAdjustType { REAL, ADJUST_RGB, ADJUST_DEPTH };

class CamSenz3D
{
private:
	cv::VideoCapture capture;
	cv::Ptr<cv::ml::SVM> svm_rgb_attack_01;
	cv::Ptr<cv::ml::SVM> svm_rgb_attack_02;
	cv::Ptr<cv::ml::SVM> svm_rgb_attack_03;
	cv::Ptr<cv::ml::SVM> svm_rgb_attack_04;
	cv::Ptr<cv::ml::SVM> svm_rgb_attack_05;

	cv::Ptr<cv::ml::SVM> svm_depth_attack_01;
	cv::Ptr<cv::ml::SVM> svm_depth_attack_02;
	cv::Ptr<cv::ml::SVM> svm_depth_attack_03;
	cv::Ptr<cv::ml::SVM> svm_depth_attack_04;
	cv::Ptr<cv::ml::SVM> svm_depth_attack_05;

	void printStreamProperties(cv::VideoCapture &capture);

	void imshowDepth(const char *winname, cv::Mat &depth, cv::VideoCapture &capture);
	bool detectFace(const cv::Mat &img, cv::Rect &rect);
	float evalue(cv::Ptr<cv::ml::SVM> svm, cv::Mat &features, float umbral, const std::string &msg);
public:
	CamSenz3D();
	~CamSenz3D();
	int init();
	int isAttack();
	int stop();
};

