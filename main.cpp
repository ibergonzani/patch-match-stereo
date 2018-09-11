#include <iostream>
#include <opencv2/opencv.hpp>
#include <pm.h>


bool check_image(const cv::Mat &image, std::string name="Image")
{
	if(!image.data)
	{
		std::cerr <<name <<" data not loaded.\n";
		return false;
	}
	return true;
}


bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2)
{
	if(img1.cols != img2.cols or img1.rows != img2.rows)
	{
		std::cerr << "Images' dimensions do not corresponds.";
		return false;
	}
	return true;
}


int main(int argc, char** argv)
{
	const float alpha =  0.9f;
	const float gamma = 10.0f;
	const float tau_c = 10.0f;
	const float tau_g =  2.0f;
	
	// Reading images
	cv::Mat3b img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat3b img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
	
	// Image loading check
	if(!check_image(img1, "Image 1") or !check_image(img2, "Image 2"))
		return 1;
	
	// Image sizes check
	if(!check_dimensions(img1, img2))
		return 1;
	
	// processing images
	pm::PatchMatch patch_match(alpha, gamma, tau_c, tau_g);
	patch_match.set(img1, img2);
	patch_match.process(3);
	patch_match.postProcess();
	
	cv::Mat1f disp1 = patch_match.getLeftDisparityMap();
	cv::Mat1f disp2 = patch_match.getRightDisparityMap();
	
	cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);
	cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);
	
	try
	{
		cv::imwrite("left_disparity.png", disp1);
		cv::imwrite("right_disparity.png", disp2);
	} 
	catch(std::exception &e)
	{
		std::cerr << "Disparity save error.\n" <<e.what();
		return 1;
	}

	return 0;
}
