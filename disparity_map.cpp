#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>

#define WINDOW_SIZE 1
#define DISPARITY_INTERVAL 15

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/stereo/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "tsukuba0.png", CV_LOAD_IMAGE_GRAYSCALE);
cv::Mat image1 = cv::imread(IMAGES_PATH + "tsukuba1.png", CV_LOAD_IMAGE_GRAYSCALE);

cv::Mat addWindowFrames(cv::Mat image);
cv::Point getBestMatch(cv::Point currentPosition);
double ssdValue(cv::Point currentPosition, cv::Point position);
double getValue(std::string method, cv::Point currentPosition, cv::Point position);
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
			int color = image.at<uchar>(row, column);
			output.at<uchar>(row + WINDOW_SIZE, column + WINDOW_SIZE) = color;
		}
	}

	return output;
}

cv::Point getBestMatch(cv::Point currentPosition) {
	cv::Point bestMatch;
	double bestValue = std::numeric_limits<double>::max();

	cv::Size imageSize = image0.size();
	for(int column = WINDOW_SIZE; column < imageSize.width - WINDOW_SIZE; column++) {
//	for(int column = currentPosition.x; column <= currentPosition.x + DISPARITY_INTERVAL; column++) {
		double value = getValue("ssd", currentPosition, cv::Point(column, currentPosition.y));
		if(value < bestValue) {
			bestValue = value;
			bestMatch = cv::Point(column, currentPosition.y);
		}
	}

//	std::cout << "\nSSD for (" << currentPosition.y << ", " << currentPosition.x << ")" << " is " << bestValue;

	return bestMatch;
}

double ssdValue(cv::Point currentPosition, cv::Point position) {
	double value = 0;
	int row, column;
	int fromX = -WINDOW_SIZE;
	int fromY = -WINDOW_SIZE;

	for(row = -WINDOW_SIZE; row <= WINDOW_SIZE; row++) {
		for(column = -WINDOW_SIZE; column <= WINDOW_SIZE; column++) {
			int image0Value = image0.at<uchar>(currentPosition.y + row, currentPosition.x + column);
			int image1Value = image1.at<uchar>(position.y + row, position.x + column);

			value += pow(image0Value - image1Value, 2.0);
		}
	}

	return value;
}

double getValue(std::string method, cv::Point currentPosition, cv::Point position) {
	if(method == "ssd") {
		return ssdValue(currentPosition, position);
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

	int row;
	int column;
	int minValue = 0;
	int maxValue = 0;

	std::cout << "Calculating disparity map...\n";
	for(row = WINDOW_SIZE; row < outputSize.height - WINDOW_SIZE; row++) {
		std::cout << "\r" << ((row * 100) / outputSize.height) << "% ";
		std::cout.flush();

		for(column = WINDOW_SIZE; column < outputSize.width - WINDOW_SIZE; column++) {

			cv::Point currentPosition = cv::Point(column, row);
			cv::Point bestMatch = getBestMatch(currentPosition);
			float disparity = distanceBetween(currentPosition, bestMatch);
			output.at<uchar>(row, column) = disparity;
//			std::cout << "\nDisparity Found: " << disparity << " Image value: " << output.at<uchar>(row, column);

			if(disparity > maxValue) {
				maxValue = disparity;
			}
			if(disparity < minValue) {
				minValue = disparity;
			}
		}
	}

	std::cout << "\nImage min value: " << minValue << "\nImage max value: " << maxValue << "\n";


	// Normalizing image
	for (row = WINDOW_SIZE; row < (outputSize.height - WINDOW_SIZE); row++) {
		for (column = WINDOW_SIZE; column < (outputSize.width - WINDOW_SIZE); column++) {
			float value = ((((float)output.at<uchar>(row, column) - (float)minValue)) / (float) maxValue) * 255.0;
			output.at<uchar>(row, column) = (int)value;
		}
	}
	std::cout << "\n";
	// End image normalization

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds";

	cv::imwrite("out/disparity_map.png", output);

	cv::namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray image", output);
	cv::waitKey(0);
	return 0;
}
