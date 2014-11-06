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
	cv::Point(202, 250),
	cv::Point(353, 208),
	cv::Point(108, 460),
	cv::Point(326, 177),
	cv::Point(275, 454),
	cv::Point(275, 454),
	cv::Point(275, 454),
	cv::Point(275, 454)
};

cv::Point image1Points[8] = {
	cv::Point(223, 274),
	cv::Point(335, 194),
	cv::Point(110, 463),
	cv::Point(312, 169),
	cv::Point(279, 442),
	cv::Point(275, 454),
	cv::Point(275, 454),
	cv::Point(275, 454)
};

void fundamentalMatrix() {
	Eigen::MatrixXd A(8, 9);
	int length = ARRAY_SIZE(image0Points), i;

	for (i = 0; i < length; i++) {
		A(i, 0) = image1Points[i].x * image0Points[i].x; // x' * x
		A(i, 1) = image1Points[i].x * image0Points[i].y; // x' * y
		A(i, 2) = image1Points[i].x; //  x'
		A(i, 3) = image1Points[i].y * image0Points[i].x; // y'* x
		A(i, 4) = image1Points[i].y * image0Points[i].y; // y' * y
		A(i, 5) = image1Points[i].y; // y'
		A(i, 6) = image0Points[i].x; // x
		A(i, 7) = image1Points[i].y; // y
		A(i, 8) = 1; // 1
	}
	
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	std::cout << svd.matrixV().rightCols(1) << "\n\n";
	
	//	Reshaping the matrix
	Eigen::MatrixXd F(3, 3);
	int width = 3;
	for (i = 0; i < 9; i++) {
		int row = i / width;
		int column = i % width;
		F(row, column) = svd.matrixV().rightCols(1)(i);
	}
	F(3, 3) = 0;
	
	std::cout << F << "\n";
}

int main() {
	clock_t begin = clock();
	//	TODO: CODE
	fundamentalMatrix();
	// CODE
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds\n";
	return 0;
}