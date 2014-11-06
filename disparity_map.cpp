#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
// Best result SAD with Window = 7
#define WINDOW_SIZE 5
#define DISPARITY_INTERVAL 15
#define SET "tsukuba"
#define METHOD "ssd"
//TEDDY 59
//CONES 59
//VENUS 19
//TSUKUBA 15

std::string IMAGES_PATH =  "/Users/rafael/Projects/python-mosaic/stereo/";

cv::Mat image0 = cv::imread(IMAGES_PATH + SET + "/imR.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image1 = cv::imread(IMAGES_PATH + SET + "/imL.png", CV_LOAD_IMAGE_COLOR);

cv::Mat addWindowFrames(cv::Mat image);
cv::Point getBestMatch(cv::Point currentPosition);
double ssdValue(cv::Point currentPosition, cv::Point position);
double sadValue(cv::Point currentPosition, cv::Point position);
double getValue(cv::Point currentPosition, cv::Point position);
float distanceBetween(cv::Point p1, cv::Point p2);

cv::Mat removeWindowFrames(cv::Mat image) {
	cv::Size s = image.size();
	std::cout << "[removeWindowFrames] Image Height " << s.height << "\n";
	std::cout << "[removeWindowFrames] Image Width " << s.width << "\n";

	cv::Mat output = cv::Mat::zeros(s.height - 2 * WINDOW_SIZE, s.width - 2 * WINDOW_SIZE, CV_8UC1);

	int column;
	int row;

	for(row = 0; row < s.height - 2 * WINDOW_SIZE; row++) {
		for(column = 0; column < s.width - 2 * WINDOW_SIZE; column++) {
			int color = image.at<uchar>(row + WINDOW_SIZE, column + WINDOW_SIZE);
			output.at<uchar>(row, column) = color;
		}
	}

	return output;
}

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

cv::Point getBestMatch(cv::Point currentPosition) {
	cv::Point bestMatch;
	double bestValue = std::numeric_limits<double>::max();

	cv::Size imageSize = image0.size();
//	for(int column = WINDOW_SIZE; column < imageSize.width - WINDOW_SIZE; column++) {
	for(int column = currentPosition.x; column <= currentPosition.x + DISPARITY_INTERVAL; column++) {
		double value = getValue(currentPosition, cv::Point(column, currentPosition.y));
		if(value < bestValue) {
			bestValue = value;
			bestMatch = cv::Point(column, currentPosition.y);
		}
	}

	return bestMatch;
}

double ssdValue(cv::Point currentPosition, cv::Point position) {
	double value = 0;
	int row, column;
	int fromX = -WINDOW_SIZE;
	int fromY = -WINDOW_SIZE;
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

double sadValue(cv::Point currentPosition, cv::Point position) {
	double value = 0;
	int row, column;
	int fromX = -WINDOW_SIZE;
	int fromY = -WINDOW_SIZE;
	cv::Point from, to;

	for(row = -WINDOW_SIZE; row <= WINDOW_SIZE; row++) {
		for(column = -WINDOW_SIZE; column <= WINDOW_SIZE; column++) {
			cv::Vec3b image0Value = image0.at<cv::Vec3b>(currentPosition.y + row, currentPosition.x + column);
			cv::Vec3b image1Value = image1.at<cv::Vec3b>(position.y + row, position.x + column);

			from = cv::Point(image0Value[1], image0Value[2]);
			to = cv::Point(image1Value[1], image1Value[2]);

			value += std::abs(distanceBetween(from, to));
		}
	}

	return value;
}

double getValue(cv::Point currentPosition, cv::Point position) {
	if(METHOD == "ssd") {
		return ssdValue(currentPosition, position);
	}
	else if(METHOD == "sad") {
		return sadValue(currentPosition, position);
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

	cvtColor(image0, image0, CV_BGR2Lab);
	cvtColor(image1, image1, CV_BGR2Lab);

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
	for (row = 0; row < outputSize.height; row++) {
		for (column = 0; column < outputSize.width; column++) {
			float value = ((((float)output.at<uchar>(row, column) - (float)minValue)) / (float) maxValue) * 255.0;
			output.at<uchar>(row, column) = (int)value;
		}
	}
	std::cout << "\n";
	// End image normalization

	output = removeWindowFrames(output);

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "It took " << elapsed_secs << " seconds";
	std::stringstream ss;

	// Makes no difference
	//	cv::medianBlur(output, output, 7);

	ss << "out/disparity_map_" << SET << "_" << METHOD << "_" << WINDOW_SIZE << ".png";
	cv::imwrite(ss.str(), output);

	cv::namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray image", output);
	cv::waitKey(0);
	return 0;
}
