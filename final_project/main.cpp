#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define PI 3.14159265
#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/final_project/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "/input_images/1.jpg", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + "/input_images/2.jpg", CV_LOAD_IMAGE_COLOR);
cv::Mat output = cv::Mat::zeros(image1.size(), CV_64F);

cv::Point image1Points[10] = {
	cv::Point(618, 163),
	cv::Point(762, 157),
	cv::Point(613, 340),
	cv::Point(752, 346),
	cv::Point(930, 217),
	cv::Point(1000, 216),
	cv::Point(924, 316),
	cv::Point(993, 317),
	cv::Point(478, 262),
	cv::Point(477, 332)
};

cv::Point image0Points[10] = {
	cv::Point(213, 125),
	cv::Point(360, 131),
	cv::Point(217, 313),
	cv::Point(362, 318),
	cv::Point(513, 196),
	cv::Point(572, 197),
	cv::Point(511, 288),
	cv::Point(569, 290),
	cv::Point(58, 226),
	cv::Point(59, 306)
};


cv::Mat getHomography(cv::Point* points1, cv::Point* points2) {
	cv::Mat w, u, vt, tmp, H;
	cv::Mat A = cv::Mat::zeros(20, 9, CV_64F);
	
	for (int i = 0; i < 20; i+=2) {
		cv::Point p1 = points1[i/2];
		cv::Point p2 = points2[i/2];
		
		double li[9] = {p1.x, p1.y, 1, 0, 0, 0, p1.x * -p2.x, p1.y * -p2.x, -p2.x};
		double li1[9] = {0, 0, 0, p1.x, p1.y, 1, -p2.y * p1.x, -p2.y * p1.y, -p2.y};

		cv::Mat(1, 9, CV_64F, &li).copyTo(A.row(i));
		cv::Mat(1, 9, CV_64F, &li1).copyTo(A.row(i + 1));
	}
	
	cv::SVD::compute(A, w, u, vt);
	H = vt.row(8).reshape(1, 3);
	H = H / H.at<double>(2, 2);
	return H;
}

void changeHomographyCoords(cv::Mat H, cv::Mat thetaMatrix) {
	double c;
	c = sqrt(pow(H.at<double>(2, 0), 2) + pow(H.at<double>(2, 1), 2));
	
	double tmpMatrix[2][2] = {
		{H.at<double>(0, 0), H.at<double>(0, 1)},
		{H.at<double>(1, 0), H.at<double>(1, 1)}
	};
	
	cv::Mat tmp = cv::Mat(2, 2, CV_64F, &tmpMatrix);
	tmp = tmp * thetaMatrix;
	
	H.at<double>(0, 0) = tmp.at<double>(0, 0);
	H.at<double>(0, 1) = tmp.at<double>(0, 1);
	H.at<double>(1, 0) = tmp.at<double>(1, 0);
	H.at<double>(1, 1) = tmp.at<double>(1, 1);
	H.at<double>(2, 0) = -c;
	H.at<double>(2, 1) = 0;
	H.at<double>(2, 2) = 1;
}

int main() {
	clock_t begin = clock();
	// CODE
	
	cv::Mat thetaMatrix;
	cv::Mat H = getHomography(image0Points, image1Points);
	
	double theta = atan2(-H.at<double>(2, 1), -H.at<double>(2, 0));
	double thetaMatrixTmp[2][2] = {
		{cos(theta), -sin(theta)},
		{sin(theta), cos(theta)}
	};

	thetaMatrix = cv::Mat(2, 2, CV_64F, &thetaMatrixTmp);
	changeHomographyCoords(H, thetaMatrix);
	cv::warpPerspective(image1, output, H, image1.size());
	
	
	// CODE
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds\n";
	
	cv::namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray image", output);
	cv::waitKey(0);
	return 0;
}