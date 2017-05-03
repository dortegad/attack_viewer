#include "CamSenz3D.h"

#include "pxcsession.h"
#include "pxcsmartptr.h"
#include "pxccapture.h"
#include "util_render.h"
#include "util_capture_file.h"

//---------------------------------------------------------------------------------------
CamSenz3D::CamSenz3D(){}

//---------------------------------------------------------------------------------------
CamSenz3D::~CamSenz3D(){}

//---------------------------------------------------------------------------------------
void CamSenz3D::normalize(cv::Mat &depthImage, cv::Mat &depthNorImg)
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
	double cleanMin = mean.val[0] - 3 * stddev.val[0];
	double cleanMax = mean.val[0] + 3 * stddev.val[0];
	//std::cout << cleanMax << "-" << cleanMin << std::endl;
	for (int col = 0; col < depthImage.cols; col++)
	{
		for (int row = 0; row < depthImage.rows; row++)
		{
			float depth = depthImage.at<ushort>(row, col);
			if ((depth < cleanMin) || (depth > cleanMax))
			{
				depthNorm.at<double>(row, col) = 0;
			}
			else
			{
				depthNorm.at<double>(row, col) = (1 - ((depthImage.at<ushort>(row, col) - cleanMin) / (cleanMax - cleanMin))) * 255;
			}
		}
	}

	depthNorImg = depthNorm;

	// PARA VER LA IMAGEN NORMALIZADA
	cv::Mat depthGrayNormalicedFace;
	depthNorImg.convertTo(depthGrayNormalicedFace, CV_8UC1);
	cv::equalizeHist(depthGrayNormalicedFace, depthGrayNormalicedFace);
	cv::imshow("imagenDepth Normalize", depthGrayNormalicedFace);
	//cv::waitKey();
}

//---------------------------------------------------------------------------------------
void CamSenz3D::imshowDepth(const char *winname, cv::Mat &depth, cv::VideoCapture &capture)
{
	short lowValue = (short)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE);
	short saturationValue = (short)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE);

	cv::Mat image;
	//depth = depth(cv::Rect(100, 100,100,100));
	cv::Mat depthNorm;
	normalize(depth, depthNorm);

	image.create(depth.rows, depth.cols, CV_8UC1);
	for (int row = 0; row < depth.rows; row++)
	{
		uchar* ptrDst = image.ptr(row);
		short* ptrSrc = (short*)depth.ptr(row);
		for (int col = 0; col < depth.cols; col++, ptrSrc++, ptrDst++)
		{
			if ((lowValue == (*ptrSrc)) || (saturationValue == (*ptrSrc)))
				*ptrDst = 0;
			else
				*ptrDst = (uchar)((*ptrSrc) >> 2);
		}
	}
	cv::imshow(winname, image);
}

//---------------------------------------------------------------------------------------
bool CamSenz3D::detectFace(const cv::Mat &img, cv::Rect &rect)
{
	cv::Mat imgGray = img.clone();

	cv::cvtColor(imgGray, imgGray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(imgGray, imgGray);

	//-- Detect faces
	std::vector<cv::Rect> faces;
	cv::CascadeClassifier cascade;
	cascade.load("haarcascade_frontalface_alt.xml");
	cascade.detectMultiScale(imgGray, faces);// , 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	imgGray.release();

	if (faces.size() > 0)
	{
		rect = faces[0];
		return true;
	}
	else
		return false;
}

//---------------------------------------------------------------------------------------
int CamSenz3D::init()
{
	this->capture.open(cv::CAP_INTELPERC);
	if (!this->capture.isOpened())
	{
		//std::cerr << "Can not open a capture object." << std::endl;
		return -1;
	}

	if (!this->capture.set(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_IDX, (double)1))
	{
		//std::cerr << "Can not setup a image stream." << std::endl;
		return -1;
	}

	if (!this->capture.set(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_IDX, (double)1))
	{
		//std::cerr << "Can not setup a depth stream." << std::endl;
		return -1;
	}

	return 0;
}


//---------------------------------------------------------------------------------------
int CamSenz3D::isAttack()
{
	if (!this->capture.grab())
	{
		//std::cout << "Can not grab images." << std::endl;
		return -1;
	}

	cv::Mat bgrImage;
	if (this->capture.retrieve(bgrImage, cv::CAP_INTELPERC_IMAGE))
	{
		//MOSTRAR EL FRAME DEPTH CAPTURADO
		//cv::imshow("color image", bgrImage);
	}

	cv::Mat depthImage;
	if (this->capture.retrieve(depthImage, cv::CAP_INTELPERC_DEPTH_MAP))
	{
		cv::Mat uvMap;
		if (this->capture.retrieve(uvMap, cv::CAP_INTELPERC_UVDEPTH_MAP))
		{
			/*PARA REGITRAR EL FRAME DEPTH CAPTURADO REGISTRADO AL TAMAÑO DEL FRAME COLOR
			cv::Mat_<ushort> depth2BGR(bgrImage.size());
			int allImg = (uvMap.cols*uvMap.rows);
			for (int i = 0; i < allImg; i++)
			{
				float *uvmap = (float *)uvMap.ptr() + 2 * i;
				int x = (int)((*uvmap) * bgrImage.cols); uvmap++;
				int y = (int)((*uvmap) * bgrImage.rows);
				if ((x > 0) && (x < bgrImage.cols) && (y < bgrImage.rows) && (y > 0))
				{
					ushort *depthValue = (ushort *)depthImage.ptr() + i;;
					depth2BGR.at<ushort>(y, x) = *depthValue;
				}
			}
			//imshowDepth("depth image", depth2BGR, capture);
			*/
			
			//MOSTRAR EL FRAME DEPTH CAPTURADO
			//this->imshowDepth("depthImage", depthImage, capture);

			//Mat bgrImageGray;
			//cv::cvtColor(bgrImage, bgrImageGray, CV_BGR2GRAY);
			cv::Mat bgr2Depth(uvMap.rows, uvMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
			for (int x = 0; x < uvMap.cols - 1; x++)
			{
				for (int y = 0; y < uvMap.rows - 1; y++)
				{
					float *uvmap = (float *)uvMap.ptr() + 2 * ((y*uvMap.cols) + x);
					int color_x = (int)((*uvmap) * bgrImage.cols); uvmap++;
					int color_y = (int)((*uvmap) * bgrImage.rows);
					if ((color_x > 0) && (color_x < bgrImage.cols) && (color_y < bgrImage.rows) && (color_y > 0))
					{
						bgr2Depth.at<cv::Vec3b>(y, x) = bgrImage.at<cv::Vec3b>(color_y, color_x);
					}
				}
			}
			//MOSTRAR EL FRAME RGB CAPTURADO REGISTRADO CON LA IAMGEN DEPTH CAPTURADA
			//cv::imshow("bgr2Depth", bgr2Depth);

			cv::Rect rectFace;
			if (detectFace(bgr2Depth, rectFace))
			{
				cv::Mat matFace = bgr2Depth(rectFace);
				cv::imshow("face bgr2Depth", matFace);

				cv::Mat matFaceDepth = depthImage(rectFace);
				this->imshowDepth("face depthImage", matFaceDepth, capture);
			}

			
		}
	}

	return 0;
}

//---------------------------------------------------------------------------------------
int CamSenz3D::stop()
{
	return 0;
}
