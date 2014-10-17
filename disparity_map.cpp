#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <thread>

#define WINDOW_SIZE 3

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/stereo/";

cv::Mat image0 = cv::imread(IMAGES_PATH + "tsukuba0.png", CV_LOAD_IMAGE_GRAYSCALE);
cv::Mat image1 = cv::imread(IMAGES_PATH + "tsukuba1.png", CV_LOAD_IMAGE_GRAYSCALE);

cv::Mat addWindowFrames(cv::Mat image);
cv::Point getBestMatch(cv::Point currentPosition);
double ssdValue(cv::Point position);
double getValue(std::string method, cv::Point position);
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
			int color = image.at<int>(row, column);
			output.at<int>(row + WINDOW_SIZE, column + WINDOW_SIZE) = color;
		}
	}

	return output;
}

cv::Point getBestMatch(cv::Point currentPosition) {
	cv::Point bestMatch;
	double bestValue = std::numeric_limits<double>::max();

	cv::Size imageSize = image0.size();
	for(int column = WINDOW_SIZE; column < imageSize.width - WINDOW_SIZE; column++) {
		double value = getValue("ssd", cv::Point(column, currentPosition.y));
		if(value < bestValue) {
			bestValue = value;
			bestMatch = cv::Point(column, currentPosition.y);
		}
	}

	return bestMatch;
}

double ssdValue(cv::Point position) {
	double value = 0;
	int row, column;
	int fromX = position.x - (WINDOW_SIZE / 2);
	int fromY = position.y - (WINDOW_SIZE / 2);

	for(row = fromY; row < fromY + WINDOW_SIZE; row++) {
		for(column = fromX; column < position.x + WINDOW_SIZE; column++) {
			value += pow(image0.at<int>(row, column) - image1.at<int>(row, column), 2.0);
		}
	}

	return value;
}

double getValue(std::string method, cv::Point position) {
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
			output.at<int>(row, column) = disparity;
//			std::cout << "\nDisparity Found: " << disparity << " Image value: " << output.at<int>(row, column);

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
			float value = (((float)output.at<int>(row, column) - (float)minValue) / (float)maxValue) * 255.0;
//			std::cout << "\nImage old value: " << output.at<int>(row, column) << " Image new Value: " << value;
			output.at<int>(row, column) = (int)value;
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
