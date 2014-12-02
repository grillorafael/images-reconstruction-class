#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <ctime>

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/final_project/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "/input_images/1.jpg", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + "/input_images/2.jpg", CV_LOAD_IMAGE_COLOR);

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
	
	// --- std homography
//	std::vector<cv::Point2f> p1tmp(10);
//	std::vector<cv::Point2f> p2tmp(10);
//	std::cout << "\n" << "[123123123]" << "\n";
//	for(int i = 0; i < 10; i++) {
//		p1tmp[i] = cv::Point2f(image0Points[i].x, image0Points[i].y);
//		p2tmp[i] = cv::Point2f(image1Points[i].x, image1Points[i].y);
//	}
//	return cv::findHomography(p1tmp, p2tmp);
	// --- std homography
	
	return H;
}

void changeHomographyCoords(cv::Mat H, cv::Mat thetaMatrix) {
	double c;
	cv::Mat tmp;
	c = sqrt(pow(H.at<double>(2, 0), 2) + pow(H.at<double>(2, 1), 2));
	
	double tmpMatrix[2][2] = {
		{H.at<double>(0, 0), H.at<double>(0, 1)},
		{H.at<double>(1, 0), H.at<double>(1, 1)}
	};
	
	cv::Mat(2, 2, CV_64F, &tmpMatrix).copyTo(tmp);
	tmp = tmp * thetaMatrix;
	
	H.at<double>(0, 0) = tmp.at<double>(0, 0);
	H.at<double>(0, 1) = tmp.at<double>(0, 1);
	H.at<double>(1, 0) = tmp.at<double>(1, 0);
	H.at<double>(1, 1) = tmp.at<double>(1, 1);
	H.at<double>(2, 0) = -c;
	H.at<double>(2, 1) = 0;
	H.at<double>(2, 2) = 1;
}

cv::Mat applyS(cv::Mat H, double u1, cv::Mat thetaMatrix, cv::Mat p, bool inverse = false) {
//	std::cout << "\n" << "[applyS] Start" << "\n";
	p = p / p.at<double>(2, 0);

	cv::Mat transformedPoint = cv::Mat::zeros(2, 1, CV_64F);
	transformedPoint.at<double>(0, 0) = p.at<double>(0, 0);
	transformedPoint.at<double>(1, 0) = p.at<double>(1, 0);
	
	transformedPoint = thetaMatrix.inv() * transformedPoint;
//	std::cout << "\n" << "[applyS] Changing point coords" << "\n";
	
	cv::Mat leftyMatrix;
	double leftyMatrixTmp[2][2] = {
		{H.at<double>(1, 1), H.at<double>(0, 1)},
		{-H.at<double>(0, 1), H.at<double>(1, 1)}
	};
	
	cv::Mat(2, 2, CV_64F, &leftyMatrixTmp).copyTo(leftyMatrix);
	
	cv::Mat rightMatrix;
	double rightMatrixTmp[2][1] = {
		{ (H.at<double>(0, 0) - H.at<double>(1, 1)) * u1 + H.at<double>(0, 2) },
		{ (H.at<double>(1, 0) + H.at<double>(0, 1)) * u1 + H.at<double>(1, 2) }
	};
	cv::Mat(2, 1, CV_64F, &rightMatrixTmp).copyTo(rightMatrix);
	
	double multiplier = 1 / (1 + H.at<double>(2, 0) * u1);
	
	if(inverse) {
		leftyMatrix.at<double>(1,1) *= 0.7673;
		leftyMatrix = leftyMatrix.inv();
		rightMatrix = rightMatrix * -1;
		
		multiplier = (1 / multiplier);
	}
	
	
	cv::Mat output = multiplier * ((leftyMatrix * transformedPoint) + rightMatrix);
	output = thetaMatrix * output;
	
	return output;
}

cv::Mat halfProjectiveWarp(cv::Mat H, cv::Mat image0, int u1, cv::Mat thetaMatrixTmp, bool left) {
	std::cout << "\n" << "[halfProjectiveWarp] Start" << "\n";
	cv::Mat thetaMatrix = cv::Mat::zeros(3, 3, CV_64F);
	thetaMatrix.at<double>(0, 0) = thetaMatrixTmp.at<double>(0, 0);
	thetaMatrix.at<double>(1, 0) = thetaMatrixTmp.at<double>(1, 0);
	thetaMatrix.at<double>(0, 1) = thetaMatrixTmp.at<double>(0, 1);
	thetaMatrix.at<double>(1, 1) = thetaMatrixTmp.at<double>(1, 1);
	
	thetaMatrix.at<double>(2, 0) = 0;
	thetaMatrix.at<double>(2, 1) = 0;
	thetaMatrix.at<double>(2, 2) = 1;
	thetaMatrix.at<double>(0, 2) = 0;
	thetaMatrix.at<double>(1, 2) = 0;
	
	cv::Size imageSize = image0.size();
	cv::Mat output;
	
	std::cout << "\n" << "[halfProjectiveWarp] Calculating bounds" << "\n";
	cv::Mat p00 = cv::Mat::zeros(3, 1, CV_64F);
	p00.at<double>(2, 0) = 1;
	p00 = H * p00;
	
	cv::Mat p01 = cv::Mat::zeros(3, 1, CV_64F);
	p01.at<double>(1, 0) = imageSize.height - 1;
	p01.at<double>(2, 0) = 1;
	p01 = H * p01;
	
	cv::Mat p10 = cv::Mat::zeros(3, 1, CV_64F);
	p10.at<double>(0, 0) = imageSize.width - 1;
	p10.at<double>(2, 0) = 1;
	p10 = applyS(H, u1, thetaMatrixTmp, p10);

	cv::Mat p11 = cv::Mat::zeros(3, 1, CV_64F);
	p11.at<double>(0, 0) = imageSize.width - 1;
	p11.at<double>(1, 0) = imageSize.height - 1;
	p11.at<double>(2, 0) = 1;
	p11 = applyS(H, u1, thetaMatrixTmp, p11);
	
	std::cout << "\n" <<  "[halfProjectiveWarp] Calculating new image size" << "\n";
	int height = std::max(p01.at<double>(1, 0), p11.at<double>(1, 0)) - std::min(p00.at<double>(1, 0), p10.at<double>(1, 0));
	int width = std::max(p10.at<double>(0, 0), p11.at<double>(0, 0)) - std::min(p00.at<double>(0, 0), p10.at<double>(0, 0));

	int minX = std::min(p00.at<double>(0, 0), p10.at<double>(0, 0));
	int maxX = std::max(p10.at<double>(0, 0), p11.at<double>(0, 0));
	
	int minY = std::min(p00.at<double>(1, 0), p10.at<double>(1, 0));
	int maxY = std::max(p01.at<double>(1, 0), p11.at<double>(1, 0));
	
	cv::Size newSize = cv::Size(width, height);
	std::cout << "\n" <<  newSize << "\n";
	output = cv::Mat::zeros(newSize, CV_8UC3);
	
	cv::Mat pointTmp = cv::Mat::zeros(2, 1, CV_64F);
	cv::Mat pointTmp3d = cv::Mat::zeros(3, 1, CV_64F);
	pointTmp3d.at<double>(2, 0) = 1;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			pointTmp.at<double>(0, 0) = x + minX;
			pointTmp.at<double>(1, 0) = y + minY;
			
			pointTmp = thetaMatrix.inv() * pointTmp;
			
			pointTmp3d.at<double>(0, 0) = pointTmp.at<double>(0, 0);
			pointTmp3d.at<double>(1, 0) = pointTmp.at<double>(1, 0);

			cv::Mat correspondingPoint;
			if(pointTmp.at<double>(0, 0) < u1) {
				// Apply H
				correspondingPoint = H.inv() * pointTmp3d;
				correspondingPoint = correspondingPoint / correspondingPoint.at<double>(2, 0);
				correspondingPoint = thetaMatrix * correspondingPoint;
			}
			else {
				// Apply S inverse
				correspondingPoint = applyS(H, u1, thetaMatrixTmp, thetaMatrix * pointTmp3d, true);
			}
			
			if(correspondingPoint.at<double>(0, 0) > 0 && correspondingPoint.at<double>(0, 0) < image0.size().width &&
			   correspondingPoint.at<double>(1, 0) > 0 && correspondingPoint.at<double>(1, 0) < image0.size().height) {
				output.at<cv::Vec3b>(y, x) = image0.at<cv::Vec3b>(correspondingPoint.at<double>(1, 0), correspondingPoint.at<double>(0, 0));
			}
		}
	}
	
//	for (int x = 0; x < image0.size().width; x++) {
//		for (int y = 0; y < image0.size().height; y++) {
//			pointTmp.at<double>(0, 0) = x;
//			pointTmp.at<double>(1, 0) = y;
//			
//			pointTmp = thetaMatrix.inv() * pointTmp;
//			
//			pointTmp3d.at<double>(0, 0) = pointTmp.at<double>(0, 0);
//			pointTmp3d.at<double>(1, 0) = pointTmp.at<double>(1, 0);
//			
//			cv::Mat correspondingPoint = applyS(H, u1, thetaMatrixTmp, pointTmp3d);
//			correspondingPoint.at<double>(0, 0) += minX;
//			correspondingPoint.at<double>(1, 0) += minY;
//
//			cv::Mat correspondingPointUV = thetaMatrix.inv() * correspondingPoint;
//			
//			if(correspondingPointUV.at<double>(0, 0) >= u1) {
//				
//				if(correspondingPoint.at<double>(0, 0) > 0 && correspondingPoint.at<double>(1, 0) > 0) {
//					output.at<cv::Vec3b>(correspondingPoint.at<double>(1, 0), correspondingPoint.at<double>(0, 0)) = image0.at<cv::Vec3b>(y, x);
//				}
//			}
//		}
//	}
	
	return output;
}

int main() {
	clock_t begin = clock();
	// CODE
	
	int u = 1125;
	cv::Mat thetaMatrix, output;
	cv::Mat H = getHomography(image0Points, image1Points);
	
	double theta = atan2(-H.at<double>(2, 1), -H.at<double>(2, 0));
	double thetaMatrixTmp[2][2] = {
		{cos(theta), -sin(theta)},
		{sin(theta), cos(theta)}
	};

	cv::Mat(2, 2, CV_64F, &thetaMatrixTmp).copyTo(thetaMatrix);
	std::cout << "H\n" << H << "\n";
	changeHomographyCoords(H, thetaMatrix);
	std::cout << "H'\n" << H << "\n";
	output = halfProjectiveWarp(H, image1, u, thetaMatrix, true);
	
	// CODE
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds\n";
	
	cv::namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray image", output);
	cv::waitKey(0);
	imwrite("output.jpg", output);
	return 0;
}