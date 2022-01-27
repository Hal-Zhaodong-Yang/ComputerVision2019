#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

void gradient(Mat former);
void CornerResponse(Mat A, Mat B, Mat C,double alpha);

double** Ixx;
double** Iyy;
double** Ixy;
double** lambda_max;
double** lambda_min;
double** R;

int row, col;


int main(int argc, char** argv)
{
	Mat primi_img = imread("component2.jpg");

	row = primi_img.rows;
	col = primi_img.cols;

	Mat grey_img;
	cvtColor(primi_img, grey_img, COLOR_RGB2GRAY);

	Mat norm_image;
	normalize(grey_img, norm_image, 1.0, 0, NORM_MINMAX);
	Mat result = norm_image * 255;
	Mat norm_img;
	result.convertTo(norm_img, CV_8UC1);
	GaussianBlur(norm_img, norm_img, Size(3, 3), 2);
	threshold(norm_img, norm_image, 150, 255, CV_THRESH_OTSU);
	bitwise_not(norm_image, norm_image);
	Mat erode_img;
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
	erode(norm_image, erode_img, element);


	gradient(erode_img);

	


	Mat A1(row, col, CV_32FC1);
	Mat B1(row, col, CV_32FC1);
	Mat C1(row, col, CV_32FC1);

	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			A1.at<float>(i, j) = Ixx[i][j];
			B1.at<float>(i, j) = Iyy[i][j];
			C1.at<float>(i, j) = Ixy[i][j];
		}
	}

	Mat A, B, C;

	GaussianBlur(A1, A, Size(3, 3), 1);
	GaussianBlur(B1, B, Size(3, 3), 1);
	GaussianBlur(C1, C, Size(3, 3), 1);

	CornerResponse(A, B, C, 0.04);

	double max2 = 0;

	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			if (R[i][j] > max2) {
				max2 = R[i][j];
			}
		}
	}

	double max = 0;
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			if (lambda_max[i][j] > max) {
				max = lambda_max[i][j];
			}
		}
	}

	double max1 = 0;
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			if (lambda_min[i][j] > max1) {
				max1 = lambda_min[i][j];
			}
		}
	}

	Mat R_img(row, col, CV_32FC1);
	Mat lambda_max_img(row, col, CV_32FC1);
	Mat lambda_min_img(row, col, CV_32FC1);
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			if (R[i][j]/max2 > 0.005) {
				R_img.at<float>(i, j) = R[i][j] / max2;
			}
			else {
				R_img.at<float>(i, j) = 0;
			}
			lambda_max_img.at<float>(i, j) = sqrt(abs(lambda_max[i][j]/max));
			lambda_min_img.at<float>(i, j) = sqrt(abs(lambda_min[i][j]/max1));
		}
	}
	Mat color_R(row, col, CV_32FC3);
	Mat show_R(row, col, CV_32FC1);
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
				show_R.at<float>(i, j) = pow(fabs(R[i][j] / max2) , 0.2);
			
		}
	}
	
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			if (show_R.at<float>(i, j) < 0.25) {
				color_R.at<Vec3f>(i, j)[0] = 0;
				color_R.at<Vec3f>(i, j)[1] = 1 - 4 * show_R.at<float>(i, j);
				color_R.at<Vec3f>(i, j)[2] = 1;

			}
			else {
				if (show_R.at<float>(i, j) < 0.5) {
					color_R.at<Vec3f>(i, j)[0] = 0;
					color_R.at<Vec3f>(i, j)[1] = 4 * show_R.at<float>(i, j) - 1;
					color_R.at<Vec3f>(i, j)[2] = 2 - 4 * show_R.at<float>(i, j);
				}
				else {
					if (show_R.at<float>(i, j) < 0.75) {
						color_R.at<Vec3f>(i, j)[0] = 4 * show_R.at<float>(i, j) - 2;
						color_R.at<Vec3f>(i, j)[1] = 1;
						color_R.at<Vec3f>(i, j)[2] = 0;
					}
					else {
						color_R.at<Vec3f>(i, j)[0] = 1;
						color_R.at<Vec3f>(i, j)[1] = 4 - 4 * show_R.at<float>(i, j);
						color_R.at<Vec3f>(i, j)[2] = 0;
					}
				}
			}
		}
	}

	Mat NMS_img(row, col, CV_32FC1);
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			NMS_img.at<float>(i, j) = 0;
		}
	}

	Point maxpoint;
	maxpoint.x = 1;
	maxpoint.y = 1;
	double maxp_value = 0;
	for (int i = 3;i < row - 3;i++) {
		for (int j = 3;j < col - 3;j++) {
			for (int m = -2;m <= 2;m++) {
				for (int n = -2;n <= 2;n++) {
					if (R_img.at<float>(i + m, j + n) > maxp_value) {
						maxp_value = R_img.at<float>(i + m, j + n);
						maxpoint.x = j + n;
						maxpoint.y = i + m;
					}
				}
			}
			maxp_value = 0;
			if (R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y - 1, maxpoint.x - 1) || R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y - 1, maxpoint.x) || R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y - 1, maxpoint.x + 1) || R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y, maxpoint.x - 1) ||
				R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y, maxpoint.x + 1) || R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y + 1, maxpoint.x - 1) ||
				R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y + 1, maxpoint.x) || R_img.at<float>(maxpoint.y, maxpoint.x) > R_img.at<float>(maxpoint.y + 1, maxpoint.x + 1)) {
				NMS_img.at<float>(maxpoint.y, maxpoint.x) = 1;
			}
		}
	}

	Point center;
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			if (NMS_img.at<float>(i, j) == 1) {
				center.x = j;
				center.y = i;
				circle(primi_img, center, 2, Scalar(0, 0, 255), 1);
			}
		}
	}



	cout << max <<" "<< max2;
	cout << endl << row << " " << col;

	imshow("primi", primi_img);
	imshow("R", color_R);
	imshow("max", lambda_max_img);
	imshow("min", lambda_min_img);
	imshow("NMS", NMS_img);
	imshow("norm", norm_image);
	imshow("erode", erode_img);

	imwrite("primi.jpg", primi_img);
	imwrite("R.jpg", color_R);
	imwrite("max.jpg", lambda_max_img);
	imwrite("min.jpg", lambda_min_img);
	imwrite("NMS.jpg", NMS_img);
	imwrite("norm.jpg", norm_image);
	imwrite("erode.jpg", erode_img);

	waitKey();



}

void gradient(Mat former)
{

	Mat extend;

	copyMakeBorder(former, extend, 1, 1, 1, 1, BORDER_REPLICATE);
	int row = former.rows;
	int col = former.cols;
	double Ix, Iy;

	Ixx = new double*[row];
	Iyy = new double*[row];
	Ixy = new double*[row];
	for (int i = 0;i < row;i++) {
		Ixx[i] = new double[col];
		Iyy[i] = new double[col];
		Ixy[i] = new double[col];
	}

	double kernelx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
	double kernely[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };

	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			Ix = kernelx[0][0] * extend.at<uchar>(i, j) + kernelx[0][1] * extend.at<uchar>(i, j + 1) + kernelx[0][2] * extend.at<uchar>(i, j + 2) + kernelx[1][0] * extend.at<uchar>(i + 1, j) + kernelx[1][1] * extend.at<uchar>(i + 1, j + 1) + kernelx[1][2] * extend.at<uchar>(i + 1, j + 2) + kernelx[2][0] * extend.at<uchar>(i + 2, j) + kernelx[2][1] * extend.at<uchar>(i + 2, j + 1) + kernelx[2][2] * extend.at<uchar>(i + 2, j + 2);
			Iy = kernely[0][0] * extend.at<uchar>(i, j) + kernely[0][1] * extend.at<uchar>(i, j + 1) + kernely[0][2] * extend.at<uchar>(i, j + 2) + kernely[1][0] * extend.at<uchar>(i + 1, j) + kernely[1][1] * extend.at<uchar>(i + 1, j + 1) + kernely[1][2] * extend.at<uchar>(i + 1, j + 2) + kernely[2][0] * extend.at<uchar>(i + 2, j) + kernely[2][1] * extend.at<uchar>(i + 2, j + 1) + kernely[2][2] * extend.at<uchar>(i + 2, j + 2);
			Ixx[i][j] = Ix * Ix;
			Iyy[i][j] = Iy * Iy;
			Ixy[i][j] = Ix * Iy;
		}
	}
}

void CornerResponse(Mat A, Mat B, Mat C,double alpha)
{
	Mat M(2,2,CV_32FC1);
	Mat value(2, 1, CV_32FC1);
	lambda_max = new double*[row];
	lambda_min = new double*[col];
	for (int i = 0;i < row;i++) {
		lambda_max[i] = new double[col];
		lambda_min[i] = new double[col];
	}
	R = new double*[row];
	for (int i = 0;i < row;i++) {
		R[i] = new double[col];
	}
	
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			M.at<float>(0, 0) = A.at<float>(i, j);
			M.at<float>(0, 1) = C.at<float>(i, j);
			M.at<float>(1, 0) = M.at<float>(0, 1);
			M.at<float>(1, 1) = B.at<float>(i, j);
			R[i][j] = determinant(M) - alpha * (M.at<float>(0, 0) + M.at<float>(1, 1)) * (M.at<float>(0, 0) + M.at<float>(1, 1));
			eigen(M, value);
			if (value.at<float>(0, 0) > value.at<float>(1, 0)) {
				lambda_max[i][j] = value.at<float>(0, 0);
				lambda_min[i][j] = value.at<float>(1, 0);
			}
			else {
				lambda_max[i][j] = value.at<float>(1, 0);
				lambda_min[i][j] = value.at<float>(0, 0);
			}
		}
	}
}

