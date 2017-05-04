#include "util_LBP_CV.h"


#include "opencv2\face.hpp"

#include "util_depth.h"

//------------------------------------------------------------------------------
int Util_LBP_CV::LBP_RGB(cv::Mat &img, cv::Mat_<double> &features)
{	
	cv::cvtColor(img, img, CV_BGR2GRAY);
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	images.push_back(img); labels.push_back(0);

	
	cv::Ptr<cv::face::LBPHFaceRecognizer> model2 = cv::face::createLBPHFaceRecognizer();
	model2->train(images, labels);
	std::vector<cv::Mat> histogramsS = model2->getHistograms();
	features =  histogramsS[0];

	return histogramsS[0].cols;
}

//------------------------------------------------------------------------------
int Util_LBP_CV::LBP_Depth(cv::Mat &imgDepth, cv::Mat_<double> &features)
{
	cv::Mat_<double> imgDepthNorm;
	Util_Depth::normalize(imgDepth, imgDepthNorm);

	cv::Mat depthGrayNormalicedFace;
	imgDepthNorm.convertTo(depthGrayNormalicedFace, CV_8UC1);
	cv::equalizeHist(depthGrayNormalicedFace, depthGrayNormalicedFace);

	std::vector<cv::Mat> images;
	std::vector<int> labels;
	images.push_back(depthGrayNormalicedFace); labels.push_back(0);

	cv::Ptr<cv::face::LBPHFaceRecognizer> model2 = cv::face::createLBPHFaceRecognizer();
	model2->train(images, labels);
	std::vector<cv::Mat> histogramsS = model2->getHistograms();
	features = histogramsS[0];

	return histogramsS[0].cols;
}