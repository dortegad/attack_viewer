#include "CamSenz3D.h"

int main(int argc, char* argv[])
{
	CamSenz3D cam;

	if (cam.init() != 0)
		return 0;

	int frame = 0;
	for (;; frame++)
	{
		cam.isAttack();

		cv::waitKey(30);
	}

	cam.stop();

	return 0;
}