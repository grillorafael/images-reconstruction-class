#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <thread>

#define WINDOW_SIZE 3

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/stereo/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "im0mini.png", CV_LOAD_IMAGE_GRAYSCALE);
cv::Mat image1 = cv::imread(IMAGES_PATH + "im1mini.png", CV_LOAD_IMAGE_GRAYSCALE);

cv::Mat addWindowFrames(cv::Mat image);
cv::Point getBestMatch(cv::Point currentPosition);
float ssdValue(cv::Point position);
float getValue(std::string method, cv::Point position);
float distanceBetween(cv::Point p1, cv::Point p2);

cv::Mat addWindowFrames(cv::Mat image) {
	cv::Size s = image.size();

    std::cout << "[addWindowFrames] Image Height " << s.height << "\n";
    std::cout << "[addWindowFrames] Image Width " << s.width << "\n";

	cv::Mat output = cv::Mat::zeros(s.height + 2 * WINDOW_SIZE, s.width + 2 * WINDOW_SIZE, CV_8UC1);

	int column;
	int row;

	for(row = 0; row < s.height; row++) {
		for(column = 0; column < s.width; column++) {
			uchar color = image.at<uchar>(row, column);
			output.at<uchar>(row + WINDOW_SIZE, column + WINDOW_SIZE) = color;
		}
	}

	return output;
}

cv::Point getBestMatch(cv::Point currentPosition) {
	cv::Point bestMatch;
	float bestValue = -INFINITY;
	cv::Size imageSize = image0.size();
	int column;
	for(column = 0; column < imageSize.width; column++) {
		float value = getValue("ssd", cv::Point(column, currentPosition.y));
		if(value > bestValue) {
			bestValue = value;
			bestMatch = cv::Point(column, currentPosition.y);
		}
	}
	return bestMatch;
}

float ssdValue(cv::Point position) {
	float value = 0;
	int row, column;
	for(row = position.y; row < position.y + WINDOW_SIZE; row++) {
		for(column = position.x; column < position.x + WINDOW_SIZE; column++) {
			value += pow(image0.at<uchar>(row, column) - image1.at<uchar>(row, column), 2);
		}
	}
	return value;
}

float getValue(std::string method, cv::Point position) {
	if(method == "ssd") {
		return ssdValue(position);
	}
	return 0;
}

float distanceBetween(cv::Point p1, cv::Point p2) {
	int x1 = p1.x;
	int x2 = p2.x;

	int y1 = p1.y;
	int y2 = p2.y;

	return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

int main(int argc, char** argv) {
	clock_t begin = clock();

	image0 = addWindowFrames(image0);
	image1 = addWindowFrames(image1);

	cv::Size outputSize = image0.size();
	cv::Mat output = cv::Mat::zeros(outputSize.height, outputSize.width, CV_8UC1);
	int row, column;

    std::cout << "Calculating disparity map...\n";

	float biggest = 0;
	for(row = WINDOW_SIZE; row < outputSize.height - WINDOW_SIZE; row++) {
        std::cout << "\r" << ((row * 100) / outputSize.height) << "% ";
        std::cout.flush();
		for(column = WINDOW_SIZE; column < outputSize.width - WINDOW_SIZE; column++) {
			cv::Point currentPosition = cv::Point(column, row);
			cv::Point bestMatch = getBestMatch(currentPosition);
			float disparity = distanceBetween(currentPosition, bestMatch);
			output.at<uchar>(row + WINDOW_SIZE, column + WINDOW_SIZE) = disparity;
			if(disparity > biggest) {
				biggest = disparity;
			}
		}
	}

	// output = output / biggest;
	// output = output * 255;


	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::printf("%f Seconds", elapsed_secs);

	cv::namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray image", output);
	cv::waitKey(0);
	return 0;
}
