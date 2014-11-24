#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <gsl/gsl_poly.h>
#include <iostream>
#include <fstream>
#include <cmath>

#define numberOfPoints 8
#define WINDOW_SIZE 7
#define ENABLE_DEBUG 0
#define F_MODE "n" // "n" or "r"
#define L_TRESHOLD 20

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/fm/";

cv::Mat image0RGB = cv::imread(IMAGES_PATH + "/temple0102.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image1RGB = cv::imread(IMAGES_PATH + "/temple0110.png", CV_LOAD_IMAGE_COLOR);

cv::Mat image0 = cv::imread(IMAGES_PATH + "/temple0102.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + "/temple0110.png", CV_LOAD_IMAGE_COLOR);

cv::Point image0Points[numberOfPoints] = {
	cv::Point(107, 461),
	cv::Point(141, 148),
	cv::Point(197, 468),
	cv::Point(193, 243),
	cv::Point(127, 457),
	cv::Point(167, 363),
	cv::Point(162, 120),
	cv::Point(299, 446)
};

cv::Point image1Points[numberOfPoints] = {
	cv::Point(128, 495),
	cv::Point(162, 183),
	cv::Point(264, 470),
	cv::Point(265, 264),
	cv::Point(155, 483),
	cv::Point(181, 381),
	cv::Point(217, 152),
	cv::Point(291, 415)
};

double k0[3][3] = {
	{1520.4, 0, 302.32},
	{0, 1525.9, 246.87},
	{0, 0, 1}
};

double k1[3][3] = {
	{1520.4, 0, 302.32},
	{0, 1525.9, 246.87},
	{0, 0, 1}
};

double r0[3][3] = {
	{0.19834669516517861000, 0.89752906429360457000, -0.39382758570889664000},
	{0.88684804433022724000, 0.00674039737427117440, 0.46201202723618323000},
	{0.41732377692231076000, -0.44090378291809157000, -0.79463495985503529000}

};
double r1[3][3] = {
	{0.48431095960713322000, 0.83534948970494449000, -0.26006561567055902000},
	{0.43169237343254652000, 0.03038031678189489700, 0.90150914087013334000},
	{0.76097607657835897000, -0.54887909998643680000, -0.34590048348258595000}
};

double t0[3] = {-0.05157154578564796000, 0.00120001566069944090, 0.60066423290325455000};
double t1[3] = {-0.04497259796363829700, -0.00470967854812120830, 0.61062211974380132000};

double rt0[3][4] = {
	{0.19834669516517861000, 0.89752906429360457000, -0.39382758570889664000, -0.05157154578564796000},
	{0.88684804433022724000, 0.00674039737427117440, 0.46201202723618323000, 0.00120001566069944090},
	{0.41732377692231076000, -0.44090378291809157000, -0.79463495985503529000, 0.60066423290325455000}
};

double rt1[3][4] = {
	{0.48431095960713322000, 0.83534948970494449000, -0.26006561567055902000, -0.04497259796363829700},
	{0.43169237343254652000, 0.03038031678189489700, 0.90150914087013334000, -0.00470967854812120830},
	{0.76097607657835897000, -0.54887909998643680000, -0.34590048348258595000, 0.61062211974380132000}
};

double distanceBetween(cv::Point p1, cv::Point p2) {
	int x1 = p1.x;
	int x2 = p2.x;
	
	int y1 = p1.y;
	int y2 = p2.y;
	
	return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

double ssdValue(cv::Point currentPosition, cv::Point position) {
	double value = 0;
	int row, column;
	cv::Point from, to;
	
	for(row = -WINDOW_SIZE; row <= WINDOW_SIZE; row++) {
		for(column = -WINDOW_SIZE; column <= WINDOW_SIZE; column++) {
			cv::Vec3b image0Value = image0.at<cv::Vec3b>(currentPosition.y + row, currentPosition.x + column);
			cv::Vec3b image1Value = image1.at<cv::Vec3b>(position.y + row, position.x + column);
			
			from = cv::Point(image0Value[1], image0Value[2]);
			to = cv::Point(image1Value[1], image1Value[2]);
			
			double grey0 = image0Value[0];
			double grey1 = image1Value[0];
			
			
			value += pow(distanceBetween(from, to), 2.0);
		}
	}
	
	return value;
}

cv::Mat getFundamentalMatrix(cv::Point* points1, cv::Point* points2) {
	std::cout << "\n" << "[getFundamentalMatrix] Starting" << "\n";
	
	cv::Mat s, u, vt, F;
	cv::Mat A = cv::Mat::zeros(numberOfPoints, 9, CV_64F);
	cv::Mat tmp = cv::Mat::zeros(3, 3, CV_64F);
 
	std::cout << "\n" << "[getFundamentalMatrix] Computing Points " << numberOfPoints;
	for (int i = 0; i < numberOfPoints; i += 1) {
		cv::Point p1 = points1[i];
		cv::Point p2 = points2[i];
		 
		double li[9] = {
			p1.x * p2.x,
			p1.y * p2.x,
			p2.x,
			p2.y * p1.x,
			p2.y * p1.y,
			p2.y,
			p1.x,
			p1.y,
			1
		};
		
		cv::Mat(1, 9, CV_64F, &li).copyTo(A.row(i));
	}

	std::cout << "\n" << "[getFundamentalMatrix] A SIZE: " << A.size();
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing SVD";
	cv::SVD::compute(A, s, u, vt, cv::SVD::FULL_UV);
	std::cout << "\n" << "[getFundamentalMatrix] SVD Computed";
	
	std::cout << "\n" << "[getFundamentalMatrix] Reshaping F";
	
	F = vt.row(vt.rows - 1).reshape(1, 3);
	F = F / F.at<double>(2, 2);
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing Second SVD";
	cv::SVD::compute(F, s, u, vt, cv::SVD::FULL_UV);
	
	tmp.at<double>(0, 0) = s.at<double>(0, 0);
	tmp.at<double>(1, 1) = s.at<double>(1, 0);
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing Second F" << "\n";
	
	F = u * tmp * vt;
	F = F / F.at<double>(2, 2);
	
	return F;
}

cv::Mat getFundamentalMatrixNormalized(cv::Point* points1, cv::Point* points2) {
	std::cout << "\n" << "[getFundamentalMatrixNormalized] Start" << "\n";
	cv::Mat translation, t, F;
	double translationTmp[3][3] = {
		{1.0, 0, -image0.size().width / 2.0},
		{0, 1.0, -image0.size().height / 2.0},
		{0, 0, 1.0}
	};

	cv::Mat(3, 3, CV_64F, &translationTmp).copyTo(translation);

	t = translation;
	
	cv::Point points1T[numberOfPoints];
	cv::Point points2T[numberOfPoints];
	
	cv::Mat tmp = cv::Mat::zeros(3, 1, CV_64F);

	for (int i = 0; i < numberOfPoints; i++) {
		tmp.at<double>(2, 0) = 1;
		
		tmp.at<double>(0, 0) = points1[i].x;
		tmp.at<double>(1, 0) = points1[i].y;
		
		tmp = t * tmp;
		tmp = tmp / tmp.at<double>(2, 0);
		
		points1T[i] = cv::Point(tmp.at<double>(0, 0), tmp.at<double>(1, 0));
		
		tmp.at<double>(2, 0) = 1;
		
		tmp.at<double>(0, 0) = points2[i].x;
		tmp.at<double>(1, 0) = points2[i].y;
		
		tmp = t * tmp;
		tmp = tmp / tmp.at<double>(2, 0);
		
		points2T[i] = cv::Point(tmp.at<double>(0, 0), tmp.at<double>(1, 0));
	}
	
	F = getFundamentalMatrix(points1T, points2T);
	F = t.t() * F * t;
	F = F / F.at<double>(2, 2);
	return F;
}

cv::Point3d get3dPoint(cv::Mat F, cv::Mat x, cv::Mat p0, cv::Mat p1) {
	cv::Point3d result;
	
	cv::Point ptX = cv::Point(x.at<double>(0, 0), x.at<double>(1, 0));
	cv::Point pt1 = cv::Point(0, 0);
	cv::Point pt2 = cv::Point(0, 0);
	cv::Mat tmp;
	
	cv::Mat line = F * x;
	
	double cA = line.at<double>(0, 0), cB = line.at<double>(1, 0), cC = line.at<double>(2, 0);
	double y0 = -cC / cB;
	double yF = (-cC - cA * (image0.size().width - 1)) / cB;
	pt1.y = y0;
	
	pt2.x = image0.size().width - 1;
	pt2.y = yF;
	
	cv::LineIterator it(image1, pt1, pt2, 8);
	double bestValue = std::numeric_limits<double>::max();
	cv::Point bestMatch;

	for(int i = 0; i < it.count; i++, ++it) {
		if((it.pos().x > WINDOW_SIZE && it.pos().y > WINDOW_SIZE) &&
		   (it.pos().x < image0.size().width - WINDOW_SIZE && it.pos().y < image0.size().height - WINDOW_SIZE)) {
			double value = ssdValue(ptX, it.pos());
			if(value < bestValue) {
				bestValue = value;
				bestMatch = it.pos();
			}
		}
	}
	
	if(ENABLE_DEBUG) {
		std::cout << "\n" << "Coeficients " << cA << "x + " << cB << "y + " << cC << "\n";
		std::cout << "\n" << "Epipolar points " << pt1 << pt2 << "\n";
		std::cout << "\n" << ptX << " Is equivalent to " << bestMatch << "\n";
		cv::line(image1, pt1, pt2, CV_RGB(255, 0, 0));
		cv::namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
		cv::imshow("Gray image", image1);
		cv::waitKey(0);
	}
	
	cv::Mat A = cv::Mat::zeros(4, 4, CV_64FC1);
	cv::Mat s, u, vt, tmpARow1;
	
	tmpARow1 = (ptX.x * p0.row(2)) - (p0.row(0));
	tmpARow1.row(0).copyTo(A.row(0));

	tmpARow1 = (ptX.y * p0.row(2)) - (p0.row(1));
	tmpARow1.row(0).copyTo(A.row(1));
	
	tmpARow1 = (bestMatch.x * p1.row(2)) - (p1.row(0));
	tmpARow1.row(0).copyTo(A.row(2));
	
	tmpARow1 = (bestMatch.y * p1.row(2)) - (p1.row(1));
	tmpARow1.row(0).copyTo(A.row(3));
	
	cv::SVD::compute(A, u, s, vt, cv::SVD::FULL_UV);
	
	cv::Mat point3dTmp = vt.col(vt.cols - 1);
	
	point3dTmp = point3dTmp / point3dTmp.at<double>(3, 0);
	
	result.x = point3dTmp.at<double>(0,0);
	result.y = point3dTmp.at<double>(1,0);
	result.z = point3dTmp.at<double>(2,0);
	
	return result;
}

double avgDist(cv::Mat F, cv::Point* points1, cv::Point* points2) {
	double num = 0;
	double distance, a, b, c;
	
	cv::Mat x = cv::Mat::zeros(3, 1, CV_64F);
	x.at<double>(2, 0) = 1;
	
	for (int i = 0; i < numberOfPoints; i++) {
		cv::Point p = points1[i];
		cv::Point p_ = points2[i];
		
		x.at<double>(0, 0) = p.x;
		x.at<double>(1, 0) = p.y;
		
		cv::Mat line = F * x;
		
		a = line.at<double>(0, 0);
		b = line.at<double>(1, 0);
		c = line.at<double>(2, 0);
		
		distance = fabs(a * p_.x + b * p_.y + c) / sqrt((a * a) + (b * b));
		num += (distance * distance);
	}
	
	return sqrt(num / numberOfPoints);
}

int main() {
	std::cout << "\n" << "[main] Starting" << "\n";
	clock_t begin = clock();
	// ------------------------------------------------
	std::cout << "\n" << "[main] Initializing variables";
	cv::Mat p0, p1, f, fn, F, kP0, kP1, rtP1, rtP0;
	
	cvtColor(image0, image0, CV_BGR2Lab);
	cvtColor(image1, image1, CV_BGR2Lab);
	
	std::cout << "\n" << "[main] Calculating variables" << "\n";
	
	f = getFundamentalMatrix(image0Points, image1Points);
	fn = getFundamentalMatrixNormalized(image0Points, image1Points);
	std::cout << "\n" << "[main] F " << "\n" << f << "\n";
	std::cout << "\n" << "[main] Fn " << "\n" << fn << "\n";

	F = F_MODE == "n" ? fn : f;

	// Can't use R and T. Need to find
	cv::Mat(3, 3, CV_64F, &k0).copyTo(kP0);
	cv::Mat(3, 4, CV_64F, &rt0).copyTo(rtP0);
	p0 = kP0 * rtP0;
	std::cout << "\n" << "[main] P " << "\n" << p0 << "\n";
	
	cv::Mat(3, 3, CV_64F, &k1).copyTo(kP1);
	cv::Mat(3, 4, CV_64F, &rt1).copyTo(rtP1);
	p1 = kP1 * rtP1;
	std::cout << "\n" << "[main] P' " << "\n" << p1 << "\n";
	// Can't use R and T. Need to find
	
	std::ofstream outputFile;
	outputFile.open("cloud.obj", std::ofstream::out | std::ofstream::trunc);
	
	double average = avgDist(F, image0Points, image1Points);
	std::cout << "\n" << "[main] Quadratic Average " << average << "\n";
	
	std::cout << "\n" << "[main] Computing 3d points" << "\n";
	
	if(ENABLE_DEBUG) {
		cv::Mat testPoint = cv::Mat::zeros(3, 1, CV_64F);
		testPoint.at<double>(2, 0) = 1;
	
		testPoint.at<double>(0, 0) = 107;
		testPoint.at<double>(1, 0) = 461;
		cv::Point3d the3dPoint = get3dPoint(f, testPoint, p0, p1);
	
		testPoint.at<double>(0, 0) = 141;
		testPoint.at<double>(1, 0) = 148;
		the3dPoint = get3dPoint(f, testPoint, p0, p1);
	
		testPoint.at<double>(0, 0) = 197;
		testPoint.at<double>(1, 0) = 468;
		the3dPoint = get3dPoint(f, testPoint, p0, p1);

		testPoint.at<double>(0, 0) = 193;
		testPoint.at<double>(1, 0) = 243;
		the3dPoint = get3dPoint(f, testPoint, p0, p1);
	}
	
	for (int x = WINDOW_SIZE; x < image0.size().width - WINDOW_SIZE; x++) {
		std::cout << "\r" << ((x * 100) / image0.size().width) << "% ";
		std::cout.flush();
		for (int y = WINDOW_SIZE; y < image0.size().height - WINDOW_SIZE; y++) {
			cv::Mat p = cv::Mat::zeros(3, 1, CV_64F);
			p.at<double>(0, 0) = x;
			p.at<double>(1, 0) = y;
			p.at<double>(2, 0) = 1;
			
			// Skipping black pixels
			cv::Vec3b color0 = image0.at<cv::Vec3b>(y, x);
			cv::Vec3b color1 = image1.at<cv::Vec3b>(y, x);
			
			cv::Vec3b colorRGB = image0RGB.at<cv::Vec3b>(y, x);
			
			if((int)color0[0] > L_TRESHOLD && (int)color1[1] > L_TRESHOLD) {
				cv::Point3d the3dPoint = get3dPoint(F, p, p0, p1);
				
				double greyScale = ((int)color0[0]) / 255.00;
				double b = ((int)colorRGB[0]) / 255.00;
				double g = ((int)colorRGB[1]) / 255.00;
				double r = ((int)colorRGB[2]) / 255.00;
				
				// Removing NaN and Infinite values
				if(the3dPoint.x == the3dPoint.x && the3dPoint.y == the3dPoint.y && the3dPoint.z == the3dPoint.z && !std::isinf(the3dPoint.x)) {
					outputFile << "v " << std::fixed << the3dPoint.x << " " << the3dPoint.y << " " << the3dPoint.z << " " << r << " " << g << " " << b << "\n";
				}
			}
			
		}
	}
	
	std::cout << "\r" << "100% ";
	std::cout.flush();
	
	outputFile.close();
	
	std::cout << "\n" << "[main] Finish" << "\n";
	// ------------------------------------------------
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds\n";
	return 0;
}