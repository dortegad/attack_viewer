#include "util_depth.h"


//------------------------------------------------------------------------------
void Util_Depth::normalize(cv::Mat &depthImage, cv::Mat &depthNorImg)
{
	/*
	//NORMALIZACION SIMPLE
	double min;
	double max;
	cv::minMaxIdx(faceDepth, &min, &max);
	cv::Mat adjFace;
	float scale = 255 / (max - min);
	faceDepth.convertTo(adjFace, CV_8UC1, scale, -min*scale);
	cv::equalizeHist(adjFace, adjFace);
	cv::resize(adjFace, adjFace, imgSize);
	cv::cvtColor(adjFace, adjFace, CV_GRAY2BGR);
	cv::drawKeypoints(adjFace, keypoints_1, adjFace);
	cv::imshow("Depth face_00", adjFace);//cv::imwrite("c:\\img.jpg", adjFace);
	*/

	cv::Mat_<double> depthNorm(depthImage.size());

	std::vector <ushort> cleanDepth;
	for (int col = 0; col < depthImage.cols; col++)
	{
		for (int row = 0; row < depthImage.rows; row++)
		{
			ushort depth = depthImage.at<ushort>(row, col);
			if (depth > 0)
				cleanDepth.push_back(depth);
		}
	}
	cv::Scalar mean;
	cv::Scalar stddev;
	cv::meanStdDev(cleanDepth, mean, stddev);
	double cleanMin = mean.val[0] - 1.5 * stddev.val[0];
	double cleanMax = mean.val[0] + 1.5 * stddev.val[0];
	//std::cout << cleanMax << "-" << cleanMin << std::endl;
	for (int col = 0; col < depthImage.cols; col++)
	{
		for (int row = 0; row < depthImage.rows; row++)
		{
			double depth = depthImage.at<ushort>(row, col);
			if ((depth < cleanMin) || (depth > cleanMax))
			{
				depthNorm.at<double>(row, col) = 0;
			}
			else
			{
				depthNorm.at<double>(row, col) = (1 - ((depth - cleanMin) / (cleanMax - cleanMin))) * 255;
			}
		}
	}

	depthNorImg = depthNorm;

	// PARA VER LA IMAGEN NORMALIZADA
	//cv::Mat depthGrayNormalicedFace;
	//depthNorImg.convertTo(depthGrayNormalicedFace, CV_8UC1);
	//cv::equalizeHist(depthGrayNormalicedFace, depthGrayNormalicedFace);
	//cv::imshow("imagenDepth Normalize", depthGrayNormalicedFace);
	//cv::waitKey();
}