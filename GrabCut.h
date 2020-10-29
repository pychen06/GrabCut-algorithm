#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "gcgraph.hpp"
#include <iostream>
using namespace cv;
using namespace std;
enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2,
};

class GMM
{
public:
	static const int n_component = 5;
	GMM(Mat& _model, int flag);
	void cal_invCov_det(int ci, double singularFix);
	void initGMM(Mat& img, Mat& mask);
	void initLearning();
	void addSample(int ci, Vec3d color);
	void cal_mean_cov();
	int whichComponent(Vec3d color);
	double dataTerm(Vec3d color);
	double get_sampleCountW(int k);
	
private:
	Mat model;
	int flag;  //0 for background, 1 for foreground
	double* a_weight;   //store the address in Mat model
	double* a_mean;
	double* a_cov;
	int totalSampleCount;
	double invCovs[n_component][3][3];
	double covDets[n_component];
	double mean_sum[n_component][3];
	double cov_prod[n_component][3][3];
	int sample_count[n_component];
};

class GrabCut2D
{
public:
	void init_mask_rect(Mat& mask, Size size, Rect rect);
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );  
	double cal_beta(Mat& img);
	//對每個像素分配GMM分量
	void assignGMM(Mat& img, Mat& mask, GMM& bGMM, GMM& fGMM, Mat& pIndex);
	void updateGMM(Mat& img, Mat& mask, Mat& pIndex, GMM& bGMM, GMM& fGMM);
	void constructGraph(Mat& img, Mat& mask, GMM& bGMM, GMM& fGMM, double lambda, Mat& leftW, Mat& topleftW, Mat& topW, Mat& toprightW, GCGraph<double>& graph, Mat& pIndex);
	void cal_edgeWeight(Mat& img, Mat& leftW, Mat& topleftW, Mat& topW, Mat& toprightW, double beta, double gamma);
	void maxFlow(GCGraph<double>& graph, Mat& mask);
	~GrabCut2D(void);
};

