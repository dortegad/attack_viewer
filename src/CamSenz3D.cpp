#include "CamSenz3D.h"

#include "util_LBP_CV.h"
#include "util_depth.h"

#include <iostream>
#include <stdio.h>
#include <stddef.h>
//---------------------------------------------------------------------------------------
CamSenz3D::CamSenz3D(){}

//---------------------------------------------------------------------------------------
CamSenz3D::~CamSenz3D(){}

//---------------------------------------------------------------------------------------
void CamSenz3D::imshowDepth(const char *winname, cv::Mat &depth, cv::VideoCapture &capture)
{
	short lowValue = (short)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE);
	short saturationValue = (short)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE);

	cv::Mat image;
	//depth = depth(cv::Rect(100, 100,100,100));
	cv::Mat depthNorm;
	Util_Depth::normalize(depth, depthNorm);

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
void CamSenz3D::printStreamProperties(cv::VideoCapture &capture)
{
	int profilesCount = capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_COUNT);
	std::cout << "Image stream." << std::endl;
	std::cout << "  Brightness = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_BRIGHTNESS) << std::endl;
	std::cout << "  Contrast = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_CONTRAST) << std::endl;
	std::cout << "  Saturation = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_SATURATION) << std::endl;
	std::cout << "  Hue = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_HUE) << std::endl;
	std::cout << "  Gamma = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_GAMMA) << std::endl;
	std::cout << "  Sharpness = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_SHARPNESS) << std::endl;
	std::cout << "  Gain = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_GAIN) << std::endl;
	std::cout << "  Backligh = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_BACKLIGHT) << std::endl;
	std::cout << "Image streams profiles:" << std::endl;
	for (int i = 0; i < profilesCount; i++)
	{
		capture.set(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_IDX, (double)i);
		std::cout << "  Profile[" << i << "]: ";
		std::cout << "width = " << (int)capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_FRAME_WIDTH);
		std::cout << ", height = " << (int)capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_FRAME_HEIGHT);
		std::cout << ", fps = " << capture.get(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_FPS);
		std::cout << std::endl;
	}

	profilesCount = (size_t)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_COUNT);
	std::cout << "Depth stream." << std::endl;
	std::cout << "  Low confidence value = " << capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE) << std::endl;
	std::cout << "  Saturation value = " << capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE) << std::endl;
	std::cout << "  Confidence threshold = " << capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD) << std::endl;
	std::cout << "  Focal length = (" << capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ) << ", "
		<< capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT) << ")" << std::endl;
	std::cout << "Depth streams profiles:" << std::endl;
	for (int i = 0; i < profilesCount; i++)
	{
		capture.set(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_IDX, (double)i);
		std::cout << "  Profile[" << i << "]: ";
		std::cout << "width = " << (int)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_FRAME_WIDTH);
		std::cout << ", height = " << (int)capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_FRAME_HEIGHT);
		std::cout << ", fps = " << capture.get(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_FPS);
		std::cout << std::endl;
	}
	cv::waitKey();
}

//---------------------------------------------------------------------------------------
int CamSenz3D::init()
{
	this->capture.open(cv::CAP_INTELPERC);
	//printStreamProperties(capture);
	if (!this->capture.isOpened())
	{
		//std::cerr << "Can not open a capture object." << std::endl;
		return -1;
	}

	if (!this->capture.set(cv::CAP_INTELPERC_IMAGE_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_IDX, (double)5))
	{
		//std::cerr << "Can not setup a image stream." << std::endl;
		return -1;
	}

	if (!this->capture.set(cv::CAP_INTELPERC_DEPTH_GENERATOR | cv::CAP_PROP_INTELPERC_PROFILE_IDX, (double)3))
	{
		//std::cerr << "Can not setup a depth stream." << std::endl;
		return -1;
	}


	svm_depth_attack_01 = cv::ml::SVM::load(".\\SVM_LBP_DEPTH\\svm_attack_01.svm");
	svm_depth_attack_02 = cv::ml::SVM::load(".\\SVM_LBP_DEPTH\\svm_attack_02.svm");
	svm_depth_attack_03 = cv::ml::SVM::load(".\\SVM_LBP_DEPTH\\svm_attack_03.svm");
	svm_depth_attack_04 = cv::ml::SVM::load(".\\SVM_LBP_DEPTH\\svm_attack_04.svm");
	svm_depth_attack_05 = cv::ml::SVM::load(".\\SVM_LBP_DEPTH\\svm_attack_05.svm");

	svm_rgb_attack_01 = cv::ml::SVM::load(".\\SVM_LBP_RGB\\svm_attack_01.svm");
	svm_rgb_attack_02 = cv::ml::SVM::load(".\\SVM_LBP_RGB\\svm_attack_02.svm");
	svm_rgb_attack_03 = cv::ml::SVM::load(".\\SVM_LBP_RGB\\svm_attack_03.svm");
	svm_rgb_attack_04 = cv::ml::SVM::load(".\\SVM_LBP_RGB\\svm_attack_04.svm");
	svm_rgb_attack_05 = cv::ml::SVM::load(".\\SVM_LBP_RGB\\svm_attack_05.svm");
	
	
	return 0;
}

//---------------------------------------------------------------------------------------
float CamSenz3D::evalue(cv::Ptr<cv::ml::SVM> svm, cv::Mat &features, float umbral, const std::string &msg)
{
	cv::Mat_<float> sample = features;
	float result = svm->predict(sample, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);
	int preditClass = svm->predict(sample, cv::noArray());
	float confidence = 1.0 / (1.0 + exp(-result));
	std::cout << msg << " : " << result << " - " << confidence << " - " << preditClass << std::endl;
	if (confidence < umbral)
		std::cout << msg << " : BONA_FIDE" << std::endl;
	else
		std::cout << msg << " : ATTACK" << std::endl;

	return confidence;
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
			if (detectFace(bgrImage, rectFace))
			{
				cv::Mat matFace = bgrImage(rectFace);
				cv::resize(matFace, matFace, cv::Size(100, 100));
				cv::imshow("face bgr", matFace);
				cv::Mat_<double> featuresRGB;
				Util_LBP_CV::LBP_RGB(matFace, featuresRGB);

				float score_attack_01_rgb = this->evalue(svm_rgb_attack_01, featuresRGB, 0.7, "RGB Attack 1");
				float score_attack_02_rgb = this->evalue(svm_rgb_attack_02, featuresRGB, 0.7, "RGB Attack 2");
				float score_attack_03_rgb = this->evalue(svm_rgb_attack_03, featuresRGB, 0.7, "RGB Attack 3");
				float score_attack_04_rgb = this->evalue(svm_rgb_attack_04, featuresRGB, 0.7, "RGB Attack 4");
				float score_attack_05_rgb = this->evalue(svm_rgb_attack_05, featuresRGB, 0.7, "RGB Attack 5");
			}

			cv::Rect rectFacedDepth;
			if (detectFace(bgr2Depth, rectFacedDepth))
			{
				cv::Mat matFaceDepth = depthImage(rectFacedDepth);
				cv::resize(matFaceDepth, matFaceDepth, cv::Size(100, 100));
				this->imshowDepth("face depthImage", matFaceDepth, capture);
				cv::Mat_<double> featureDepth;
				Util_LBP_CV::LBP_Depth(matFaceDepth, featureDepth);

				float score_attack_01_depth = this->evalue(svm_depth_attack_01, featureDepth, 0.7, "DEPTH Attack 1");
				float score_attack_02_depth = this->evalue(svm_depth_attack_02, featureDepth, 0.7, "DEPTH Attack 2");
				float score_attack_03_depth = this->evalue(svm_depth_attack_03, featureDepth, 0.7, "DEPTH Attack 3");
				float score_attack_04_depth = this->evalue(svm_depth_attack_04, featureDepth, 0.7, "DEPTH Attack 4");
				float score_attack_05_depth = this->evalue(svm_depth_attack_05, featureDepth, 0.7, "DEPTH Attack 5");
			}

			
		}
	}

	return 0;
}

//---------------------------------------------------------------------------------------
int CamSenz3D::stop()
{
	svm_rgb_attack_01.release();
	svm_rgb_attack_02.release();
	svm_rgb_attack_03.release();
	svm_rgb_attack_04.release();
	svm_rgb_attack_05.release();

	svm_depth_attack_01.release();
	svm_depth_attack_02.release();
	svm_depth_attack_03.release();
	svm_depth_attack_04.release();
	svm_depth_attack_05.release();
	return 0;
}
