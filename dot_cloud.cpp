#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/fm/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "/temple0102.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + "/temple0107.png", CV_LOAD_IMAGE_COLOR);

cv::Point image0Points[8] = {
	cv::Point(578, 2040),
	cv::Point(647, 2269),
	cv::Point(759, 2223),
	cv::Point(746, 2457),
	cv::Point(567, 1082),
	cv::Point(548, 1375),
	cv::Point(632, 2038),
	cv::Point(895, 1500)
};

cv::Point image1Points[8] = {
	cv::Point(932, 2545),
	cv::Point(953, 2619),
	cv::Point(988, 2604),
	cv::Point(981, 2671),
	cv::Point(408, 473),
	cv::Point(365, 781),
	cv::Point(473, 1436),
	cv::Point(734, 926)
};

// 8 points algorithm normalized
cv::Mat getNormalizedFundamentalMatrix(cv::Point* points1, cv::Point* points2) {
	std::cout << "\n" << "[getNormalizedFundamentalMatrix] Starting" << "\n";
	cv::Mat s, u, vt, F, t, T;
	cv::Mat A = cv::Mat::zeros(8, 9, CV_64F);
	
	cv::Size im0size = image0.size();
	cv::Size im1size = image1.size();
	
	double tTmp[3][3] = {
		{ im0size.width + im0size.height, 0, im0size.width / 2 },
		{ 0, im0size.width + im0size.height, im0size.height / 2 },
		{ 0, 0, 1 },
	};
	t = cv::Mat(3, 3, CV_64F, &tTmp);
	
	double TTmp[3][3] = {
		{ im1size.width + im1size.height, 0, im1size.width / 2 },
		{ 0, im1size.width + im1size.height, im1size.height / 2 },
		{ 0, 0, 1 },
	};
	T = cv::Mat(3, 3, CV_64F, &TTmp);
	
	for (int i = 0; i < 8; i+=1) {
		
	}
	
	
	return A;
}

// 8 points algorithm
cv::Mat getFundamentalMatrix(cv::Point* points1, cv::Point* points2) {
	std::cout << "\n" << "[getFundamentalMatrix] Starting" << "\n";
	
	cv::Mat s, u, vt, F;
	cv::Mat A = cv::Mat::zeros(8, 9, CV_64F);
 
	std::cout << "\n" << "[getFundamentalMatrix] Computing Points" << "\n";
	for (int i = 0; i < 8; i+=1) {
		cv::Point p1 = points1[i];
		cv::Point p2 = points2[i];
		
		double li[9] = {p1.x * p2.x, p1.y * p2.x, p2.x, p2.y * p1.x, p2.y * p1.y, p2.y, p1.x, p1.y, 1};
		
		cv::Mat(1, 9, CV_64F, &li).copyTo(A.row(i));
	}

	std::cout << "\n" << "[getFundamentalMatrix] A SIZE: " << A.size() << "\n";
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing SVD" << "\n";
	cv::SVD::compute(A, u, s, vt, cv::SVD::FULL_UV);
	std::cout << "\n" << "[getFundamentalMatrix] SVD Computed" << "\n";
	
	std::cout << "\n" << vt.row(7) << "\n";
	std::cout << "\n" << "[getFundamentalMatrix] Reshaping F" << "\n";
	
	// First version of F
	F = vt.row(7).reshape(1, 3);
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing Second SVD" << "\n";
	cv::SVD::compute(F, u, s, vt, cv::SVD::FULL_UV);
	
	u.at<double>(2, 2) = 0;
	
	std::cout << u << "\n\n";
	std::cout << s << "\n\n";
	std::cout << vt << "\n\n";
	std::cout << "\n" << "[gnetFundamentalMatrix] Computing Second F" << "\n";
	// Second version of F
	F = s.mul(vt);

	std::cout << F << "\n";
	return F;
}

int main() {
	clock_t begin = clock();
	//	TODO: CODE
	getFundamentalMatrix(image0Points, image1Points);
	getNormalizedFundamentalMatrix(image0Points, image1Points);
	// CODE
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds\n";
	return 0;
}