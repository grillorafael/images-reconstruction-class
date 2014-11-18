#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <gsl/gsl_poly.h>

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/fm/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "/temple0102.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + "/temple0107.png", CV_LOAD_IMAGE_COLOR);

cv::Point image1Points[8] = {
	cv::Point(86, 407),
	cv::Point(170, 74),
	cv::Point(304, 433),
	cv::Point(220, 148),
	cv::Point(87, 349),
	cv::Point(184, 399),
	cv::Point(190, 282),
	cv::Point(124, 188)
};

cv::Point image0Points[8] = {
	cv::Point(60, 355),
	cv::Point(248, 71),
	cv::Point(220, 448),
	cv::Point(267, 154),
	cv::Point(58, 385),
	cv::Point(201, 410),
	cv::Point(164, 169),
	cv::Point(158, 166)
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

// 8 points algorithm normalized
cv::Mat getNormalizedFundamentalMatrix(cv::Point* points1, cv::Point* points2) {
	std::cout << "\n" << "[getNormalizedFundamentalMatrix] Starting" << "\n";
	cv::Mat s, u, vt, F, t, T, P;
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
	std::cout << "\n" << "[getPMatrix] Initializing k0 and r0 with zeroes";
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

cv::Point get3dPoint(cv::Mat F, cv::Point x, cv::Point X) {
	std::cout << "\n" << "[get3dPoint] Starting" << "\n";
	
	cv::Mat u, s, vt, e, E, r, R;
	double a, b, c, d, fe, fl;
	
	cv::Mat px = cv::Mat(3, 1, CV_64F);
	px.at<double>(0, 0) = x.x;
	px.at<double>(1, 0) = x.y;
	px.at<double>(2, 0) = 1;
	
	cv::Mat pX = cv::Mat(3, 1, CV_64F);
	px.at<double>(0, 0) = X.x;
	px.at<double>(1, 0) = X.y;
	px.at<double>(2, 0) = 1;
	
	double tTmp[3][3] = {
		{1, 0, -x.x},
		{0, 1, -x.y},
		{0, 0, 1},
	};
	cv::Mat t = cv::Mat(3, 3, CV_64F, &tTmp);
	
	double TTmp[3][3] = {
		{1, 0, -X.x},
		{0, 1, -X.y},
		{0, 0, 1},
	};
	cv::Mat T = cv::Mat(3, 3, CV_64F, &TTmp);
	
	px = t * px;
	px = px / px.at<double>(2);
	
	pX = T * pX;
	pX = pX / pX.at<double>(2);
	
	F = T.t().inv() * F * t.inv();
	cv::SVD::compute(F, u, s, vt, cv::SVD::FULL_UV);
	
	e = vt.col(2);
	E = u.row(2);

	e = e / sqrt((pow(vt.at<double>(1, 3), 2))+(pow(vt.at<double>(2, 3), 2)));
	E = E / sqrt((pow(u.at<double>(3, 1), 2))+(pow(u.at<double>(3, 2), 2)));
	
	double rTmp[3][3] = {
		{e.at<double>(0, 0), e.at<double>(1, 0), 0},
		{-e.at<double>(1, 0), e.at<double>(0, 0), 0},
		{0, 0, 1},
	};
	r = cv::Mat(3, 3, CV_64F, &rTmp);
	
	double RTmp[3][3] = {
		{E.at<double>(0, 0), E.at<double>(1, 0), 0},
		{-E.at<double>(1, 0), E.at<double>(0, 0), 0},
		{0, 0, 1},
	};
	R = cv::Mat(3, 3, CV_64F, &RTmp);
	
	F = R.t().inv() * F * r.inv();
	
	a = F.at<double>(2, 2);
	b = F.at<double>(2, 3);
	c = F.at<double>(3, 2);
	d = F.at<double>(3, 3);
	fe = e.at<double>(3, 1);
	fl = E.at<double>(3, 1);
	
	double coefs[7]
	
	return X;
}

int main() {
	std::cout << "\n" << "[main] Starting" << "\n";
	clock_t begin = clock();
	// ------------------------------------------------
	std::cout << "\n" << "[main] Initializing variables";
	cv::Mat p0,p1,f;
	
	std::cout << "\n" << "[main] Calculating variables" << "\n";
	f = getFundamentalMatrix(image0Points, image1Points);
//	f = getNormalizedFundamentalMatrix(image0Points, image1Points);

	p0 = getPMatrix(k0, r0, t0);
	p1 = getPMatrix(k1, r1, t1);
	
	
	for (int i = 0; i < 8; i++) {
		std::cout << "\n" << get3dPoint(f, image0Points[i], image1Points[i]) << "\n";
	}
	
	
	std::cout << "\n" << "[main] Finish" << "\n";
	// ------------------------------------------------
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds\n";
	return 0;
}