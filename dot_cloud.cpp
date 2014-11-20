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

#define WINDOW_SIZE 7
#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/fm/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "/temple0102.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + "/temple0107.png", CV_LOAD_IMAGE_COLOR);

cv::Point image1Points[8] = {
	cv::Point(106, 460),
	cv::Point(113, 439),
	cv::Point(298, 394),
	cv::Point(149, 446),
	cv::Point(179, 448),
	cv::Point(274, 453),
	cv::Point(298, 183),
	cv::Point(317, 240)
};

cv::Point image0Points[8] = {
	cv::Point(109, 462),
	cv::Point(110, 440),
	cv::Point(288, 377),
	cv::Point(154, 449),
	cv::Point(192, 452),
	cv::Point(278, 442),
	cv::Point(288, 178),
	
	cv::Point(303, 228)
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
	{0.33395826580541360000, 0.83010730713039638000, -0.44653525655760135000},
	{0.78707906244147485000, 0.01507325297158268500, 0.61666793861129443000},
	{0.51863130079709752000, -0.55739990643484760000, -0.64832624359957369000}
};

double t0[3] = {-0.05157154578564796000, 0.00120001566069944090, 0.60066423290325455000};
double t1[3] = {-0.04592612096202596000, -0.00032656373638871063, 0.60909066773860632000};

cv::Mat addWindowFrames(cv::Mat image) {
	cv::Size s = image.size();
	
	std::cout << "[addWindowFrames] Image Height " << s.height << "\n";
	std::cout << "[addWindowFrames] Image Width " << s.width << "\n";
	
	cv::Mat output = cv::Mat::zeros(s.height + 2 * WINDOW_SIZE, s.width + 2 * WINDOW_SIZE, CV_8UC3);
	
	int column;
	int row;
	
	for(row = 0; row < s.height; row++) {
		for(column = 0; column < s.width; column++) {
			cv::Vec3b color = image.at<cv::Vec3b>(row, column);
			output.at<cv::Vec3b>(row + WINDOW_SIZE, column + WINDOW_SIZE) = color;
		}
	}
	
	return output;
}

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
			
			
			value += pow(distanceBetween(from, to), 2.0);
		}
	}
	
	return value;
}

// 8 points algorithm normalized
//cv::Mat getNormalizedFundamentalMatrix(cv::Point* points1, cv::Point* points2) {
//	std::cout << "\n" << "[getNormalizedFundamentalMatrix] Starting" << "\n";
//	cv::Mat s, u, vt, F, t, T, P;
//	cv::Mat A = cv::Mat::zeros(8, 9, CV_64F);
//	
//	cv::Size im0size = image0.size();
//	cv::Size im1size = image1.size();
//	
//	double tTmp[3][3] = {
//		{ im0size.width + im0size.height, 0, im0size.width / 2 },
//		{ 0, im0size.width + im0size.height, im0size.height / 2 },
//		{ 0, 0, 1 },
//	};
//	t = cv::Mat(3, 3, CV_64F, &tTmp);
//	
//	double TTmp[3][3] = {
//		{ im1size.width + im1size.height, 0, im1size.width / 2 },
//		{ 0, im1size.width + im1size.height, im1size.height / 2 },
//		{ 0, 0, 1 },
//	};
//	T = cv::Mat(3, 3, CV_64F, &TTmp);
//	
//	for (int i = 0; i < 8; i+=1) {
//		
//	}
//	
//	
//	return A;
//}

// 8 points algorithm
cv::Mat getFundamentalMatrix(cv::Point* points1, cv::Point* points2) {
	std::cout << "\n" << "[getFundamentalMatrix] Starting" << "\n";
	
	cv::Mat s, u, vt, F;
	cv::Mat A = cv::Mat::zeros(8, 9, CV_64F);
	cv::Mat tmp = cv::Mat::zeros(3, 3, CV_64F);
 
	std::cout << "\n" << "[getFundamentalMatrix] Computing Points";
	for (int i = 0; i < 8; i+=1) {
		cv::Point p1 = points1[i];
		cv::Point p2 = points2[i];
		 
		double li[9] = {p1.x * p2.x, p1.y * p2.x, p2.x, p2.y * p1.x, p2.y * p1.y, p2.y, p1.x, p1.y, 1};
		
		cv::Mat(1, 9, CV_64F, &li).copyTo(A.row(i));
	}

	std::cout << "\n" << "[getFundamentalMatrix] A SIZE: " << A.size();
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing SVD";
	cv::SVD::compute(A, u, s, vt, cv::SVD::FULL_UV);
	std::cout << "\n" << "[getFundamentalMatrix] SVD Computed";
	
	std::cout << "\n" << "[getFundamentalMatrix] Reshaping F";
	
	// First version of F
	F = vt.row(7).reshape(1, 3);
	
	std::cout << "\n" << "[getFundamentalMatrix] Computing Second SVD";
	cv::SVD::compute(F, u, s, vt, cv::SVD::FULL_UV);
	
	tmp.at<double>(0, 0) = u.at<double>(0, 0);
	tmp.at<double>(1, 1) = u.at<double>(1, 0);
	
	std::cout << "\n" << "[gnetFundamentalMatrix] Computing Second F" << "\n";
	
	// Second version of F
	F = s * tmp * vt;

//	std::cout << "\n" << F;
	return F;
}

cv::Mat getPMatrix(double k[3][3], double r[3][3],double t[3]) {
	std::cout << "\n" << "[getPMatrix] Starting" << "\n";
	cv::Mat k0,r0;
	std::cout << "\n" << "[getPMatrix] Initializing k0 and r0 with zeros";
	k0 = cv::Mat::zeros(3, 3, CV_64F);
	r0 = cv::Mat::zeros(3, 3, CV_64F);
	
	std::cout << "\n" << "[getPMatrix] Filling k0 and r0";
	for(int i = 0; i < 3; i++) {
		cv::Mat(1, 3, CV_64FC1, &k[i]).copyTo(k0.row(i));
		cv::Mat(1, 3, CV_64FC1, &r[i]).copyTo(r0.row(i));
	}

	
	std::cout << "\n" << "[getPMatrix] Initializing t0";
	cv::Mat t0 = cv::Mat::zeros(3, 4, CV_64FC1);
	cv::Mat p;

	std::cout << "\n" << "[getPMatrix] Filling t0";
	t0.at<double>(0,0) = 1;
	t0.at<double>(1,1) = 1;
	t0.at<double>(2,2) = 1;
	
	t0.at<double>(0,3) = t[0];
	t0.at<double>(1,3) = t[1];
	t0.at<double>(2,3) = t[2];
	
	std::cout << "\n" << "[getPMatrix] Calculating P" << "\n";
	p=k0*r0*t0;

	return p;
}

cv::Point3d get3dPoint(cv::Mat F, cv::Mat x) {
	cv::Point3d result;
	
	cv::Point ptX = cv::Point(x.at<double>(0, 0), x.at<double>(1, 0));
	cv::Point pt1 = cv::Point(0, 0);
	cv::Point pt2 = cv::Point(0, 0);
	cv::Mat tmp;
	bool hasP1 = false;
	cv::Mat lines[4];
	
	cv::Mat up = cv::Mat::zeros(3, 1, CV_64FC1);
	up.at<double>(1, 0) = 1;
	
	cv::Mat down = cv::Mat::zeros(3, 1, CV_64FC1);
	down.at<double>(1, 0) = 1;
	down.at<double>(2, 0) = -image0.size().height;
	
	cv::Mat left = cv::Mat::zeros(3, 1, CV_64FC1);
	left.at<double>(0, 0) = 1;
	
	cv::Mat right = cv::Mat::zeros(3, 1, CV_64FC1);
	left.at<double>(0, 0) = 1;
	left.at<double>(2, 0) = -image0.size().width;
	
	lines[0] = up;
	lines[1] = down;
	lines[2] = left;
	lines[3] = right;
	
	cv::Mat line = F * x;
	
	for (int i = 0; i < 4; i++) {
		tmp = lines[i].cross(line);
		tmp = tmp / tmp.at<double>(2, 0);
		if ((tmp.at<double>(0, 0) >= 0 && tmp.at<double>(0, 0) < image0.size().width) &&
			(tmp.at<double>(1, 0) >= 0 && tmp.at<double>(1, 0) < image0.size().height)) {
			if(hasP1) {
				pt2.x = tmp.at<double>(0, 0);
				pt2.y = tmp.at<double>(1, 0);
			}
			else {
				pt1.x = tmp.at<double>(0, 0);
				pt1.y = tmp.at<double>(1, 0);
				hasP1 = true;
			}
		}
	}
	
	cv::LineIterator it(image1, pt1, pt2, 8);
	
	double bestValue = std::numeric_limits<double>::max();;
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
	
//	std::cout << "\n" << ptX << " Is equivalent to " << bestMatch << "\n";
	
	cv::Mat A = cv::Mat::zeros(4, 4, CV_64FC1);
	
	
	return result;
}

int main() {
	std::cout << "\n" << "[main] Starting" << "\n";
	clock_t begin = clock();
	// ------------------------------------------------
	std::cout << "\n" << "[main] Initializing variables";
	cv::Mat p0,p1,f;
	
	cvtColor(image0, image0, CV_BGR2Lab);
	cvtColor(image1, image1, CV_BGR2Lab);
	
	image0 = addWindowFrames(image0);
	image1 = addWindowFrames(image1);
	
	std::cout << "\n" << "[main] Calculating variables" << "\n";
	f = getFundamentalMatrix(image0Points, image1Points);

	p0 = getPMatrix(k0, r0, t0);
	p1 = getPMatrix(k1, r1, t1);
	
	std::ofstream outputFile;
	outputFile.open("cloud.obj", std::ofstream::out | std::ofstream::trunc);
	
	std::cout << "\n" << "[main] Computing 3d points" << "\n";
	for (int x = WINDOW_SIZE; x < image0.size().width - WINDOW_SIZE; x++) {
		std::cout << "\r" << ((x * 100) / image0.size().width) << "% ";
		std::cout.flush();
		for (int y = WINDOW_SIZE; y < image0.size().height - WINDOW_SIZE; y++) {
			cv::Mat p = cv::Mat::zeros(3, 1, CV_64FC1);
			p.at<double>(0, 0) = x;
			p.at<double>(1, 0) = y;
			p.at<double>(2, 0) = 1;
			
			// Skipping black pixels
			// std::cout << "\nPIXEL COLOR: " << image0.at<cv::Vec3b>(y, x) << "\n";
			// Verify treshold
			if(image0.at<cv::Vec3b>(y, x)[0] > 20) {
				cv::Point3d the3dPoint = get3dPoint(f, p);
				outputFile << "v " << the3dPoint.x << " " << the3dPoint.y << " " << the3dPoint.z << " " << 1.0 << "\n";
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