#include "MSCKF.h"
using namespace cv;
using namespace Eigen;
using namespace std;




 /*   Mat img;    
    // ORB
    Mat ORB_H_prev;
    vector<MatrixXd> features;
    vector<vector<int> > featuresIdx; // to be filled with current train indices
    int imageNum = 0;

    Ptr<cv::OrbFeatureDetector> ORB_detector;
    //cv::Ptr<cv::FeatureDetector> ORB_detector;
    vector<cv::KeyPoint> ORB_train_kpts;

    vector<Point2f> ORB_train_pts;

    Ptr<cv::DescriptorExtractor> ORB_descriptor;
    Mat ORB_train_desc;

    Ptr<cv::DescriptorMatcher> ORB_matcher;

    void featureMatching(Mat &img, Mat &ORB_outputImg);

    void runFeatureMatching(Mat &img, Mat &outputImg);

    void augmentoldFeatures( 
        Point2f Point, int trainIdx, int queryIdx);
    void augmentnewFeatures( 
        Point2f Point, int Idx);

    void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
    std::vector<Point2f>& pts_query);*/



////////////////////////////////////////////////////////////////////////////////////////////////
//feature matching module

void drawMatchesRelative(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        std::vector<cv::DMatch>& matches, Mat& img, const vector<unsigned char>& mask = vector<
        unsigned char> ())
    {

        for (int i = 0; i < (int)matches.size(); i++)
        {
            if (mask.empty() || mask[i])
            {
                Point2f pt_new = query[matches[i].queryIdx].pt;
                Point2f pt_old = train[matches[i].trainIdx].pt;

                cv::line(img, pt_new, pt_old, Scalar(0, 0, 255), 2);
                cv::circle(img, pt_new, 2, Scalar(255, 0, 0), 1);

            }
        }
    }

//Takes a descriptor and turns it into an xy point
void keypoints2points(const vector<KeyPoint>& in, vector<Point2f>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(in[i].pt);
    }
}

//Takes an xy point and appends that to a keypoint structure
void points2keypoints(const vector<Point2f>& in, vector<KeyPoint>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(KeyPoint(in[i], 1));
    }
}

//Uses computed homography H to warp original input points to new planar position
void warpKeypoints(const Mat& H, const vector<KeyPoint>& in, vector<KeyPoint>& out)
{
    vector<Point2f> pts;
    keypoints2points(in, pts);
    vector<Point2f> pts_w(pts.size());
    Mat m_pts_w(pts_w);
    perspectiveTransform(Mat(pts), m_pts_w, H);
    points2keypoints(pts_w, out);
}

//Converts matching indices to xy points
void MSCKF::matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
    std::vector<Point2f>& pts_query)
{

    pts_train.clear();
    pts_query.clear();
    pts_train.reserve(matches.size());
    pts_query.reserve(matches.size());

    size_t i = 0;

    for (; i < matches.size(); i++)
    {

        const DMatch & dmatch = matches[i];

        pts_query.push_back(query[dmatch.queryIdx].pt);
        pts_train.push_back(train[dmatch.trainIdx].pt);

        augmentoldFeatures( pts_train[i], dmatch.trainIdx, dmatch.queryIdx);
       // cout << "trainIdx: " << dmatch.trainIdx << endl ;

    }

}

void MSCKF::augmentoldFeatures( 
Point2f Point, int trainIdx, int queryIdx)
{
    //cout << "entered augmentoldFeatures" << endl;

    if (!features.empty() && imageNum > 2 ) 
    {
        //cout << featuresIdx.size() << endl;
        for (unsigned int i=0; i< featuresIdx.size() ; ++i)
        {
           // cout << "entered if " << endl;
            cout << trainIdx << ',' << featuresIdx[i][0] << endl;

            if (trainIdx == featuresIdx[i][0])
            {

                int M = features[i].rows();
                int N = features[i].cols();
                //cout << M << ',' << N << endl;
                features[i].conservativeResize(M, N+1);
                features[i](0,N) = Point.x;
                features[i](1,N) = Point.y;

                featuresIdx[i][0] = queryIdx;

                break;

            }

        }

    }
    else 
    {
        augmentnewFeatures( Point, queryIdx);

    }

}

void MSCKF::augmentnewFeatures( 
Point2f Point, int Idx)
{
      //cout << "entered augmentnewFeatures" << endl;
      MatrixXd P = MatrixXd::Zero(2,1);
      P(0,0) = Point.x;
      P(1,0) = Point.y;
    
      features.reserve(features.size() + P.size());
      features.push_back(P);
      // augment the featuresIdx
      vector<int>v = {Idx, imageNum};
      featuresIdx.push_back(v);

}

void resetH(Mat&H)
{
    H = Mat::eye(3, 3, CV_32FC1);
}



//Note Prev => Train, Curr => Query
void MSCKF::featureMatching(Mat &img, Mat &ORB_outputImg)
{

			// ORB
	
    vector<cv::KeyPoint> ORB_query_kpts;
    vector<Point2f> ORB_query_pts;
	Mat  ORB_query_desc;

	vector<cv::DMatch> ORB_matches;


	cv::Mat imGray;
	if(img.channels() != 3)
		cvtColor(img, imGray, CV_RGB2GRAY);
	else
		img.copyTo(imGray);

	// ORB...
	img.copyTo(ORB_outputImg);
	ORB_detector->detect(imGray, ORB_query_kpts);
	ORB_descriptor->compute(imGray, ORB_query_kpts, ORB_query_desc);
	if(ORB_H_prev.empty())
		ORB_H_prev = Mat::eye(3,3,CV_32FC1);

	std::vector<unsigned char> ORB_match_mask;

	if(!ORB_train_kpts.empty())
	{
		std::vector<cv::KeyPoint> test_kpts;
		warpKeypoints(ORB_H_prev.inv(), ORB_query_kpts, test_kpts);
		cv::Mat ORB_mask = windowedMatchingMask(test_kpts, ORB_train_kpts, 25, 25);
		ORB_matcher->match(ORB_query_desc, ORB_train_desc, ORB_matches, ORB_mask);
		
		matches2points(ORB_train_kpts, ORB_query_kpts, ORB_matches, ORB_train_pts, ORB_query_pts);
		
		
		if(ORB_matches.size() > 5)
		{
			cv::Mat H = findHomography(ORB_train_pts, ORB_query_pts, RANSAC, 4, ORB_match_mask);
			if(countNonZero(Mat(ORB_match_mask)) > 15)
				ORB_H_prev = H;
			else
				ORB_H_prev = Mat::eye(3,3,CV_32FC1);

			drawMatchesRelative(ORB_train_kpts, ORB_query_kpts, ORB_matches, ORB_outputImg, ORB_match_mask);
		}
	}
	else
	{	
		ORB_H_prev = Mat::eye(3,3,CV_32FC1);

	}

	ORB_train_kpts = ORB_query_kpts;
	ORB_query_desc.copyTo(ORB_train_desc);	
	
	if(true)
		cout << ", ORB matches: " << ORB_matches.size() << endl;


}

void MSCKF::runFeatureMatching(Mat &inputImg, Mat &outputImg)
{
    //cout << "ran featureMatching" << endl;
	// ORB...
    this->imageNum++;
	ORB_detector = new cv::OrbFeatureDetector();
	ORB_descriptor = new cv::OrbDescriptorExtractor();
	ORB_matcher = DescriptorMatcher::create("BruteForce-Hamming");

    featureMatching(inputImg, outputImg);

    return ;

}




//////////////////////////////////////////////////////////////////////////////////////////////////////
//matches - temperory variable
//features and featuresIdx are class varaibles
//lostFeatures temporary variable
//newFeatures and newPoints are temporary varaibles
#include "MSCKF.h"
using namespace cv;
using namespace Eigen;
using namespace std;

/*void MSCKF::MSCKFfeatures()
{
   ORB_detector = new cv::OrbFeatureDetector();
    ORB_descriptor = new cv::OrbDescriptorExtractor();
    ORB_matcher = DescriptorMatcher::create("BruteForce-Hamming");

}*/




void MSCKF::featureMatching(Mat &img, Mat &ORB_outputImg)
{
    cout << "entered feature matching " << endl;

    //this->imageNum++;
    Mat imgGray;

    //convert to grayscale
    if(img.channels() == 3)
        cvtColor(img, imgGray, CV_RGB2GRAY);

    else
        img.copyTo(imgGray);

    cout << "image converted to grayscale " << endl;

    ////////////////////////////////////////////////////////////////////////////////

    vector<KeyPoint> ORB_query_kpts;
    vector<Point2f> ORB_query_pts;
    Mat  ORB_query_desc;

    vector<cv::DMatch> ORB_matches;
    vector<vector<DMatch> > all_matches;
    std::vector<unsigned char> ORB_match_mask;

    cout << "ORB_variables decalred" << endl;

    //cout << imgGray.channels() << endl;
    //cout << ORB_query_kpts.size() << endl;

    ORB_detector->detect(imgGray, ORB_query_kpts);

    cout <<  "detect" << endl;

    ORB_descriptor->compute(imgGray, ORB_query_kpts, ORB_query_desc);
    ORB_query_desc.convertTo(ORB_query_desc, CV_32F);
    cout << "compute " << endl;

    cout << "before knn match" << endl;

    if (! ORB_train_kpts.empty())
    {
        cout << ORB_query_desc.size() << endl;
        cout <<  ORB_train_desc.size() << endl;

        ORB_matcher->knnMatch(ORB_query_desc, ORB_train_desc, all_matches, 2);

        cout << "detect, compute, knnmatch" << endl;


        //////////////////////////////////////////////////////////////////////////////////////

        // Accept matches based on a ratio test
        for(int i = 0; i < ORB_query_desc.rows; i++)
        {
            // small / large
            if(all_matches[i][0].distance / all_matches[i][1].distance > 0.15)
                continue;

            ORB_matches.push_back(all_matches[i][0]);
        }

        cout << "ratio test done" << endl;

        //////////////////////////////////////////////////////////////////////////////////////
        // Remove inconsistent matches
    /*    for(int i = 0; i < ORB_matches.size(); i++)
        {

            bool inconsistent = false;
            for(int j = i+1; j < ORB_matches.size(); j++){

                if(ORB_matches[i].trainIdx == ORB_matches[j].trainIdx){
                    inconsistent = true;
                    ORB_matches.erase(ORB_matches.begin() + j);
                    j--;
                }
            }


            if(inconsistent)
                ORB_matches.erase(ORB_matches.begin() + i);
        }*/

       // cout << "inconsistent matches removed " << endl;
    }

    

    ///////////////////////////////////////////////////////////////////////////////////
    ORB_train_kpts = ORB_query_kpts;
    ORB_query_desc.copyTo(ORB_train_desc);

    cout << "good matches: " << ORB_matches.size() << endl;

    vector<Vector2d> fts(ORB_query_kpts.size());
    vector<int> m(ORB_query_kpts.size(), -1);

    cout << "matches refining part done " << endl;

    for(int i = 0; i < ORB_matches.size(); i++){
        m[ORB_matches[i].queryIdx] = ORB_matches[i].trainIdx;
    }
    for(int i = 0; i < ORB_query_kpts.size(); i++){
        fts[i] = Vector2d(ORB_query_kpts[i].pt.x, ORB_query_kpts[i].pt.y);
    }
    

    cout << "matches#: " << ORB_matches.size() << "idx size" << m.size() << endl;
    //////////////////////////////////////////////////////////////////////////////////////

   /* vector<int> trackmap;
    if(frames.size() > 0){
        trackmap.resize(frames.back().features.size()); // The track at which each of the latest features is located
        for(int i = 0; i < tracks.size(); i++){
            trackmap[tracks[i].indices.back()] = i;
        }
    }*/



}

void MSCKF::runFeatureMatching(Mat &inputImg, Mat &outputImg)
{
    //cout << "ran featureMatching" << endl;
    // ORB...
    this->imageNum++;
    ORB_detector = new cv::OrbFeatureDetector();
    ORB_descriptor = new cv::OrbDescriptorExtractor();
    ORB_matcher = DescriptorMatcher::create("FlannBased");

    featureMatching(inputImg, outputImg);

    return ;

}


        for (int i=0; i < matches.size() ; ++i)
        {
            //cout << "entered loop i: " << i << endl;
             DMatch dmatch = matches[i];
            //cout << "featuresIdx[j] " << featuresIdx[j][1] << endl;
            if (dmatch.trainIdx == featuresIdx[j](0)) // j
            {
                //cout << "match found" << endl;
                found = true;
                int M = features[dmatch.trainIdx].rows();
                int N = features[dmatch.trainIdx].cols();
                //cout << M << ',' << N << endl;
                features[dmatch.trainIdx].conservativeResize(M, N+1);
                features[dmatch.trainIdx](0,N) = ORB_query_kpts[dmatch.queryIdx].pt.x;
                features[dmatch.trainIdx](1,N) = ORB_query_kpts[dmatch.queryIdx].pt.y;
                featuresIdx[dmatch.trainIdx](0) = dmatch.queryIdx;
                featuresIdx[dmatch.trainIdx](1) = imageNum;

                break;

            }
        }


/*void MSCKF::augmentoldFeatures( 
Point2f Point, int trainIdx, int queryIdx)
{
    //cout << "entered augmentoldFeatures" << endl;

    if (!features.empty() && imageNum > 2 ) 
    {
        //cout << featuresIdx.size() << endl;
        for (unsigned int i=0; i< featuresIdx.size() ; ++i)
        {
           // cout << "entered if " << endl;
           // cout << trainIdx << ',' << featuresIdx[i][0] << endl;

            if (trainIdx == featuresIdx[i][0])
            {

                int M = features[i].rows();
                int N = features[i].cols();
                //cout << M << ',' << N << endl;
                features[i].conservativeResize(M, N+1);
                features[i](0,N) = Point.x;
                features[i](1,N) = Point.y;

                featuresIdx[i][0] = queryIdx;

                break;

            }

        }

    }
    else 
    {
        augmentnewFeatures( Point, queryIdx);

    }

}*/

/*void MSCKF::augmentnewFeatures( 
Point2f Point, int Idx)
{
      //cout << "entered augmentnewFeatures" << endl;
      MatrixXd P = MatrixXd::Zero(2,1);
      P(0,0) = Point.x;
      P(1,0) = Point.y;
    
      features.reserve(features.size() + P.size());
      features.push_back(P);
      // augment the featuresIdx
      vector<int>v = {Idx, imageNum};
      featuresIdx.push_back(v);

}*/