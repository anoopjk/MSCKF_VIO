#ifndef MSCKF_H_
#define MSCKF_H_


#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <ctime>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#define MAXTRACK 30
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace Eigen;
using namespace std;



class MSCKF{


public:

	MSCKF(); // default constructor
	~MSCKF(); // default destructor

	//EKF propagation varaibles member variables
	VectorXd X;
	MatrixXd covariance;
	//accelerometer measurements
	Vector3d B_aPrev;
	//gyroscope measurements
	Vector3d B_wPrev;

	Vector3d g;
	double sigma_ac = 0.0034,
    sigma_gc = 0.0005,
    sigma_wac = 0.001,
    sigma_wgc = 0.0001;

	Vector3d bg;
	Vector3d ba;

	double dt;

	//EKF propagation member functions
	//VectorXd flattenMatrix(MatrixXd &M);
	//MatrixXd deFlattenMatrix(VectorXd &v);

	Matrix4d BigOmega( Vector3d v);
	Matrix3d skewMatrix( Vector3d v);

	MSCKF(VectorXd X0, MatrixXd P0, Vector3d B_wPrev, Vector3d B_aPrev, double dt);



	Quaterniond quatMultiplication(Quaterniond q1, Quaterniond q2);

	void propagateIMU(Vector3d B_wRaw, Vector3d B_aRaw);
	void processIMU( Vector3d linear_acceleration, Vector3d angular_velocity);


/////////////////////////////////////////////////////////////////////////////////////////
	
	Mat img;
	Matrix2d focal;
	Vector2d pp;
	Matrix3d C_R_B;
	Vector3d C_p_B;	
	// ORB
	Mat ORB_H_prev;
	vector<MatrixXd> features;
	vector<Vector2i > featuresIdx; // to be filled with current train indices
	vector<MatrixXd> lostfeatures;
    vector<int> lostfeaturesCamIdx;
 	int imageNum = 0;

	Ptr<cv::OrbFeatureDetector> ORB_detector;
	//cv::Ptr<cv::FeatureDetector> ORB_detector;
	vector<cv::KeyPoint> ORB_train_kpts;

	vector<Point2f> ORB_train_pts;

	Ptr<cv::DescriptorExtractor> ORB_descriptor;
	Mat ORB_train_desc;

	Ptr<cv::DescriptorMatcher> ORB_matcher;


	
	void augmentFilter();
		void augmentStateVector();
		void augmentCovariance();

    void runFeatureMatching(Mat &img, Mat &ORB_outputImg);
		void featuresInit();
		void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
			    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
			    std::vector<Point2f>& pts_query);
		void augmentFeatures(vector<KeyPoint> ORB_query_kpts);
		void augmentoldFeatures(vector<DMatch> matches, vector<KeyPoint> ORB_query_kpts);
		void featureMatching(Mat &img, Mat &ORB_outputImg);

	void marginalizefilter();
	    void marginalizeStateVector();
	    void marginalizeCovariance();

    Vector2d cameraProjection(Vector3d P);
	void cameraMeasurement(Vector3d& G_p_fi, Vector3d& G_p_B, Matrix3d& G_R_B, Vector2d& zi, Matrix<double, 2,9>& Hbi,  Matrix<double, 2,3>& Hfi, Vector2d& ri);

    void stackingResidualsFeature();


   	struct
   	{
   		double g;
   		double sigma_ac;
		double sigma_gc;
		double sigma_wac; 
		double sigma_wgc;

   	} IMUParams;

   	struct 
   	{
   		double sigma_img;
   		double fx;
   		double fy;
		double px;
		double py;

   	} CameraParams;


};

#endif
