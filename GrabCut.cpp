#include "GrabCut.h"

GMM::GMM(Mat& _model, int flag)
{
	int i;
	this->flag = flag;
	int modelSize = 13;
	if ( _model.empty() ) {
		_model.create(1, modelSize*n_component, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	a_weight = model.ptr<double>(0);  //紀錄GMM參數的記憶體位置
	a_mean = a_weight + n_component;
	a_cov = a_mean + 3 * n_component;

	for (i = 0; i < n_component; i++) {
		if (a_weight[i] > 0) {
			cal_invCov_det(i, 0.0);
		}
	}
	totalSampleCount = 0;
}

void GMM::cal_invCov_det(int ci, double singularFix)
{
	if (a_weight[ci] > 0) {
		double *c = a_cov + ci * 9;
		double det = c[0] * c[4] * c[8] + c[1] * c[5] * c[6] + c[2] * c[3] * c[7] - c[6] * c[4] * c[2] - c[7] * c[5] * c[0] - c[8] * c[3] * c[1];
		if (det < 1e-6 && singularFix>0) {
			c[0] += singularFix;
			c[4] += singularFix;
			c[8] += singularFix;
			det = c[0] * c[4] * c[8] + c[1] * c[5] * c[6] + c[2] * c[3] * c[7] - c[6] * c[4] * c[2] - c[7] * c[5] * c[0] - c[8] * c[3] * c[1];
		}
		covDets[ci] = det;
		double inv_det = 1.0 / det;
		invCovs[ci][0][0] =  (c[4] * c[8] - c[7] * c[5]) * inv_det;
		invCovs[ci][0][1] = -(c[3] * c[8] - c[6] * c[5]) * inv_det;
		invCovs[ci][0][2] =  (c[3] * c[7] - c[4] * c[6]) * inv_det;
		invCovs[ci][1][0] = -(c[1] * c[8] - c[7] * c[2]) * inv_det;
		invCovs[ci][1][1] =  (c[0] * c[8] - c[2] * c[6]) * inv_det;
		invCovs[ci][1][2] = -(c[0] * c[7] - c[1] * c[6]) * inv_det;
		invCovs[ci][2][0] =  (c[1] * c[5] - c[2] * c[4]) * inv_det;
		invCovs[ci][2][1] = -(c[0] * c[5] - c[2] * c[3]) * inv_det;
		invCovs[ci][2][2] =  (c[0] * c[4] - c[1] * c[3]) * inv_det;
	}
}

void GMM::initLearning()
{
	int i, j;
	for (i = 0; i < n_component; i++) {
		for (j = 0; j < 3; j++) {
			mean_sum[i][j] = 0;
			cov_prod[i][j][0] = cov_prod[i][j][1] = cov_prod[i][j][2] = 0;
		}
		sample_count[i] = 0;
	}
	totalSampleCount = 0;
}

void GMM::addSample(int ci, Vec3d color)
{
	int i, j;
	for (i = 0; i < 3; i++) {
		mean_sum[ci][i] += color[i];
		for (j = 0; j < 3; j++) {
			cov_prod[ci][i][j] += color[i] * color[j];
		}
	}
	sample_count[ci]++;
	totalSampleCount++;
}

void GMM::cal_mean_cov()
{
	int i, j, k, n;
	for (i = 0; i < n_component; i++) {
		n = sample_count[i];
		if (n == 0) {
			a_weight[i] = 0;
		}
		else {
			a_weight[i] = (double)n / totalSampleCount;
			double inv_n = 1.0 / n;
			double *m = a_mean + 3 * i;
			for (j = 0; j < 3; j++) {
				m[j] = mean_sum[i][j] * inv_n;
			}
			double *c = a_cov + 9 * i;
			for (j = 0; j < 3; j++) {
				for (k = 0; k < 3; k++) {
					c[3 * j + k] = cov_prod[i][j][k] * inv_n - m[j] * m[k];
				}
			}
			cal_invCov_det(i, 0.01);
		}
	}
}

void GMM::initGMM(Mat& img, Mat& mask)
{
	int n_kmeans_iter = 10;
	Mat label;
	vector<Vec3f> sample;
	int i, j;
	Point p;
	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			p.x = j;
			p.y = i;
			if (flag == 0) {  //background
				if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD) {
					sample.push_back((Vec3f)img.at<Vec3b>(p));
				}
			}
			else if (flag == 1) {  //foreground
				if (mask.at<uchar>(p) == GC_FGD || mask.at<uchar>(p) == GC_PR_FGD) {
					sample.push_back((Vec3f)img.at<Vec3b>(p));
				}
			}
		}
	}
	Mat _sample((int)sample.size(), 3, CV_32FC1);
	for (i = 0; i < (int)sample.size(); i++) {
		for (j = 0; j < 3; j++) {
			_sample.at<float>(i, j) = sample[i][j];
		}
	}
	cv::kmeans(_sample, n_component, label, TermCriteria(CV_TERMCRIT_ITER, n_kmeans_iter, 0.0), 0, KMEANS_PP_CENTERS);
	
	initLearning();
	for (i = 0; i < (int)sample.size(); i++) {
		addSample(label.at<int>(i, 0), sample[i]);
	}
	cal_mean_cov();
}

int GMM::whichComponent(const Vec3d color)
{
	double max_p = 0.0, p = 0.0;
	int i, j, k = 0, s;
	for (i = 0; i < n_component; i++) {
		if (a_weight[i] == 0) {
			p = 0;
		}
		else {
			double *m = a_mean + 3 * i;
			Vec3d d = color;
			d[0] -= m[0];
			d[1] -= m[1];
			d[2] -= m[2];
			double expTerm = 0.0, temp = 0.0;
			for (j = 0; j < 3; j++) {
				for (s = 0; s < 3; s++) {
					temp += d[s] * invCovs[i][s][j];
				}
				expTerm += temp * d[j];
				temp = 0.0;
			}
			p = 1.0 / sqrt(covDets[i])*exp(-0.5*expTerm);
		}
		if (p > max_p) {
			k = i;
			max_p = p;
		}
	}
	return k;
}

double GMM::dataTerm(const Vec3d color)
{
	int i, j, s;
	double smooth = 0.0, p;
	for (i = 0; i < n_component; i++) {
		if (a_weight[i] == 0) {
			p = 0;
		}
		else {
			double *m = a_mean + 3 * i;
			Vec3d d = color;
			d[0] -= m[0];
			d[1] -= m[1];
			d[2] -= m[2];
			double expTerm = 0.0, temp = 0.0;
			for (j = 0; j < 3; j++) {
				for (s = 0; s < 3; s++) {
					temp += d[s] * invCovs[i][s][j];
				}
				expTerm += temp * d[j];
				temp = 0.0;
			}
			p = 1.0 / sqrt(covDets[i])*exp(-0.5*expTerm);
		}
		smooth += a_weight[i] * p;
	}
	return smooth;
}

double GMM::get_sampleCountW(int k)
{
	return (double)sample_count[k] / (double)totalSampleCount;
}

GrabCut2D::~GrabCut2D(void)
{
}

void GrabCut2D::init_mask_rect(Mat& mask, Size size, Rect rect)
{
	mask.create(size, CV_8UC1);
	mask.setTo(GC_BGD);  //設為背景
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
 	rect.width = min(rect.width, size.width - rect.x);
	rect.height = min(rect.height, size.height - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));  //Tu可能的目標
}

double GrabCut2D::cal_beta(Mat& img)
{
	double beta = 0.0;
	int i, j;
	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			Vec3d color = img.at<Vec3b>(i, j);
			if (j > 0) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i, j - 1);
				beta += d.dot(d);
			}
			if (j > 0 && i > 0) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i - 1, j - 1);
				beta += d.dot(d);
			}
			if (i > 0) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i - 1, j);
				beta += d.dot(d);
			}
			if (i > 0 && j < img.cols - 1) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i - 1, j + 1);
				beta += d.dot(d);
			}
		}
	}
	beta = 1.0 / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2));
	return beta;
}

void GrabCut2D::assignGMM(Mat& img, Mat& mask, GMM& bGMM, GMM& fGMM, Mat& pIndex)
{
	int i, j;
	Point p;
	int countB[GMM::n_component];
	int countF[GMM::n_component];
	for (i = 0; i < GMM::n_component; i++) {
		countB[i] = 0;
		countF[i] = 0;
	}
	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			Vec3d color = img.at<Vec3b>(i, j);
			p.y = i;
			p.x = j;
			if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD) {
				pIndex.at<int>(p) = bGMM.whichComponent(color);
				countB[pIndex.at<int>(p)] += 1;
			}
			else {
				pIndex.at<int>(p) = fGMM.whichComponent(color);
				countF[pIndex.at<int>(p)] += 1;
			}
		}
	}
}

void GrabCut2D::updateGMM(Mat& img, Mat& mask, Mat& pIndex, GMM& bGMM, GMM& fGMM)
{
	bGMM.initLearning();
	fGMM.initLearning();
	int i, j, ci;
	Point p;
	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			p.y = i;
			p.x = j;
			ci = pIndex.at<int>(p);
			if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD) {
				bGMM.addSample(ci, img.at<Vec3b>(p));
			}
			else {
				fGMM.addSample(ci, img.at<Vec3b>(p));
			}
		}
	}
	bGMM.cal_mean_cov();
	fGMM.cal_mean_cov();
}

void GrabCut2D::constructGraph(Mat& img, Mat& mask, GMM& bGMM, GMM& fGMM, double lambda, Mat& leftW, Mat& topleftW, Mat& topW, Mat& toprightW, GCGraph<double>& graph, Mat& pIndex)
{
	int n_edge = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);
	graph.create(img.rows * img.cols, n_edge);
	int i, j, w;
	Point p;
	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			p.y = i;
			p.x = j;
			int vIndex = graph.addVtx();
			Vec3b color = img.at<Vec3b>(p);
			//t-link weight
			double source, sink;
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
				source = -log(bGMM.dataTerm(color));
				sink = -log(fGMM.dataTerm(color));
				if (mask.at<uchar>(p) == GC_PR_BGD) {
					source -= log(bGMM.get_sampleCountW(pIndex.at<int>(p)));
				}
				else {
					sink -= log(fGMM.get_sampleCountW(pIndex.at<int>(p)));
				}
			}
			else if (mask.at<uchar>(p) == GC_BGD) {
				source = 0;
				sink = lambda;
			}
			else {
				source = lambda;
				sink = 0;
			}
			graph.addTermWeights(vIndex, source, sink);
			//n_link weight
			if (j > 0) {
				w = leftW.at<double>(p);
				graph.addEdges(vIndex, vIndex - 1, w, w);
			}
			if (j > 0 && i > 0) {
				w = topleftW.at<double>(p);
				graph.addEdges(vIndex, vIndex - img.cols - 1, w, w);
			}
			if (i > 0) {
				w = topW.at<double>(p);
				graph.addEdges(vIndex, vIndex - img.cols, w, w);
			}
			if (i > 0 && j<img.cols - 1) {
				w = toprightW.at<double>(p);
				graph.addEdges(vIndex, vIndex - img.cols + 1, w, w);
			}
		}
	}
}

void GrabCut2D::cal_edgeWeight(Mat& img, Mat& leftW, Mat& topleftW, Mat& topW, Mat& toprightW, double beta, double gamma)
{
	leftW.create(img.size(), CV_64FC1);
	topleftW.create(img.size(), CV_64FC1);
	topW.create(img.size(), CV_64FC1);
	toprightW.create(img.size(), CV_64FC1);
	int i, j;
	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			Vec3d color = img.at<Vec3b>(i, j);
			if (j > 0) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i, j - 1);
				leftW.at<double>(i, j) = gamma * exp(-beta * d.dot(d));
			}
			else {
				leftW.at<double>(i, j) = 0;
			}
			if (j > 0 && i > 0) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i - 1, j - 1);
				topleftW.at<double>(i, j) = gamma / sqrt(2.0) * exp(-beta * d.dot(d));
			}
			else {
				topleftW.at<double>(i, j) = 0;
			}
			if (i > 0) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i - 1, j);
				topW.at<double>(i, j) = gamma * exp(-beta * d.dot(d));
			}
			else {
				topW.at<double>(i, j) = 0;
			}
			if (i > 0 && j<img.cols-1) {
				Vec3d d = color - (Vec3d)img.at<Vec3b>(i - 1, j + 1);
				toprightW.at<double>(i, j) = gamma / sqrt(2.0) * exp(-beta * d.dot(d));
			}
			else {
				toprightW.at<double>(i, j) = 0;
			}
		}
	}
}

void GrabCut2D::maxFlow(GCGraph<double>& graph, Mat& mask)
{
	graph.maxFlow();
	int i, j;
	Point p;
	for (i = 0; i < mask.rows; i++) {
		for (j = 0; j < mask.cols; j++) {
			p.y = i;
			p.x = j;
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
				if (graph.inSourceSegment(i*mask.cols + j)) {
					mask.at<uchar>(p) = GC_PR_FGD;
				}
				else
					mask.at<uchar>(p) = GC_PR_BGD;
			}
		}
	}
}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    //std::cout<<"Execute GrabCut Function: Please finish the code here!"<<std::endl;

	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	Mat pIndex(img.size(), CV_32SC1);

	GMM bgdGMM(bgdModel, 0);
	GMM fgdGMM(fgdModel, 1);

	double gamma = 50;
	double lambda = 9 * gamma;
	double beta = cal_beta(img);
	//cout << beta << endl;

	if (mode == GC_WITH_RECT || mode == GC_WITH_MASK) {
		if (mode == GC_WITH_RECT) {
			init_mask_rect(mask, img.size(), rect);
		}
		bgdGMM.initGMM(img, mask);
		fgdGMM.initGMM(img, mask);
	}
	if (iterCount <= 0) {
		return;
	}

	GCGraph<double> graph;
	//對每個像素分配GMM分量
	assignGMM(img, mask, bgdGMM, fgdGMM, pIndex);
	updateGMM(img, mask, pIndex, bgdGMM, fgdGMM);
	Mat leftW, topleftW, topW, toprightW;
	cal_edgeWeight(img, leftW, topleftW, topW, toprightW, beta, gamma);
	constructGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, topleftW, topW, toprightW, graph, pIndex);
	maxFlow(graph, mask);
	//cout << "done" << endl;
	//一.参数解释：
	//输入：
	//cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
	//cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
	//int iterCount :           :每次分割的迭代次数（类型-int)
	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)
	//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
}