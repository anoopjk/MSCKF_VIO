#include "MSCKF.h"
using namespace cv;
using namespace Eigen;
using namespace std;


////////////////////////////////////////////////////////////////////////////////////////////////
//feature matching module
void MSCKF::featuresInit()
{

    focal  << CameraParams.fx, 0,
             0, CameraParams.fy;
    pp = Vector2d(CameraParams.px, CameraParams.py); // principal point 

    vector<MatrixXd> features;
    vector<Vector2i> featuresIdx;
    vector<MatrixXd> lostfeatures;
    vector<int> lostfeaturesCamIdx;
    ORB_detector = new cv::OrbFeatureDetector();
    ORB_descriptor = new cv::OrbDescriptorExtractor();
    ORB_matcher = DescriptorMatcher::create("BruteForce-Hamming");


}


void drawMatchesRelative(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        std::vector<cv::DMatch>& matches, Mat& img, const vector<unsigned char>& mask = vector<
        unsigned char> ())
    {
        int matchesCounter = 0;
        for (int i = 0; i < (int)matches.size(); i++)
        {
            if (mask.empty() || mask[i])
            {
                matchesCounter++;
                Point2f pt_new = query[matches[i].queryIdx].pt;
                Point2f pt_old = train[matches[i].trainIdx].pt;

                cv::line(img, pt_new, pt_old, Scalar(0, 0, 255), 2);
                cv::circle(img, pt_new, 2, Scalar(255, 0, 0), 1);

            }
        }
        cout << "matchesCounter: " << matchesCounter << endl;
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

        //augmentoldFeatures( pts_query[i], dmatch.trainIdx, dmatch.queryIdx);
       // cout << "trainIdx: " << dmatch.trainIdx << endl ;

    }

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
    if(img.channels() == 3)
        cvtColor(img, imGray, CV_RGB2GRAY);
    else
        img.copyTo(imGray);

    //cout << "cvtColor done" << endl;

    // ORB...
    img.copyTo(ORB_outputImg);
    ORB_detector->detect(imGray, ORB_query_kpts);
    ORB_descriptor->compute(imGray, ORB_query_kpts, ORB_query_desc);
    if(ORB_H_prev.empty())
        ORB_H_prev = Mat::eye(3,3,CV_32FC1);

    std::vector<unsigned char> ORB_match_mask;

    //cout << "feature detection done" << endl;
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

            augmentoldFeatures(ORB_matches, ORB_query_kpts);
        }
    }
    else
    {   
        ORB_H_prev = Mat::eye(3,3,CV_32FC1);
        ORB_train_kpts = ORB_query_kpts;
        ORB_query_desc.copyTo(ORB_train_desc);  

        augmentFeatures( ORB_query_kpts);

    }

    //ORB_train_kpts = ORB_query_kpts;
    //ORB_query_desc.copyTo(ORB_train_desc);  
    
    if(true)
        cout << ", ORB matches: " << ORB_matches.size() << endl;

}

void MSCKF::augmentoldFeatures(vector<DMatch> matches, vector<KeyPoint> ORB_query_kpts)
{

    //cout  << "entered augmentoldFeatures" << endl;

    //vector<MatrixXd> lostfeatures;
    //vector<int> lostfeaturesCamIdx;
    
    //cout << "features.size() before : " << features.size() << endl;

    int notFound =  0;
    for(int j=0; j< featuresIdx.size(); ++j)
    {  
       // cout << "entered loop j: " << j << endl;
        bool found = false;
        for (int i=0; i < matches.size() ; ++i)
        {
            //cout << "entered loop i: " << i << endl;
             DMatch dmatch = matches[i];
            //cout << "featuresIdx[j] " << featuresIdx[j][1] << endl;
            if (dmatch.trainIdx == featuresIdx[j](0)) // j
            {
                //cout << "match found" << endl;
                found = true;
                int M = features[j].rows();
                int N = features[j].cols();
                //cout << M << ',' << N << endl;
                features[j].conservativeResize(M, N+1);
                features[j](0,N) = ORB_query_kpts[dmatch.queryIdx].pt.x;
                features[j](1,N) = ORB_query_kpts[dmatch.queryIdx].pt.y;
                //featuresIdx[j](0) = dmatch.queryIdx;
                //featuresIdx[j](1) = imageNum;

                break;

            }
        }

         if(!found)
        {
            notFound++;
           // cout << j << " not found" << endl;
           // cout << "features [" << j << "] measurements " << features[j].cols() << endl;

            //cout << featuresIdx[j][1] << "-" << imageNum-1 << endl;

            
           //featuresIdx[j][0] = -1;
           lostfeatures.push_back(features[j]);
           lostfeaturesCamIdx.push_back(featuresIdx[j](1));
           cout << "no.of frames tracked: " << featuresIdx[j](1) << '-' << imageNum-1 << endl;
           //cout << "j: " << j << endl;
           features.erase(features.begin() + j);
           featuresIdx.erase(featuresIdx.begin() + j);
            j = j - 1;
          

        }
        //cout << "features.size() after: " << features.size() << endl;

    }

    cout << "lostfeatures.size: " << lostfeatures.size() << endl;
    if (notFound > 400)
            cout << "notFound counter " << notFound << " running low on features" << endl;

}


void MSCKF::augmentFeatures(vector<KeyPoint> ORB_query_kpts)
{
    //cout << "entered augmentFeatures" << endl;
    features = vector<MatrixXd> (ORB_query_kpts.size(), MatrixXd::Zero(2,1));
    featuresIdx = vector<Vector2i> (ORB_query_kpts.size(), Vector2i::Zero());
    //cout << featuresIdx[0][0] << endl;
    for(int i= 0; i< int(ORB_query_kpts.size()) ; ++i)
    {
        features[i](0,0) = ORB_query_kpts[i].pt.x;
        features[i](1,0) = ORB_query_kpts[i].pt.y;
        featuresIdx[i] = Vector2i(i,imageNum);
    }

}




//augment features
void MSCKF::augmentFilter()
{
    augmentCovariance();
    augmentStateVector();

}



void MSCKF::augmentCovariance()
{

    int N = (X.size() - 16) / 10; // no .of states augmented, this N should always be equal to imageNum
    cout << "no.of states augmented: " << N << endl;
    MatrixXd Jpi = MatrixXd::Zero(9, X.size()-(N+1));
    MatrixXd I9 = MatrixXd::Identity(9,9);
    Jpi.block<9,9>(0,0) = I9;
    cout << "Jpi: " << Jpi.rows() << ',' << Jpi.cols() << endl;
    int covSize = covariance.rows();  //covariance.cols();
    covariance.conservativeResize(covSize+9, covSize+9);
    
    cout << "matrix resize done" << endl;
    //Matrix.block(row,col, blocksize(rows,cols))
    cout << "before matrix product" << endl;       
    
    cout << "lhs: " <<   covSize << ',' << 9 << endl;
    cout << "rhs: " <<  covSize << ',' << covSize << " * " <<  Jpi.cols() << ',' << Jpi.rows() << endl;           
    covariance.block(0, covSize,covSize,9) = covariance.block(0,0,covSize,covSize)*Jpi.transpose();
    cout << "after matrix product" << endl;
    covariance.block(covSize,0,9,covSize) = covariance.block(0, covSize,covSize,9).transpose();
    covariance.block(covSize,covSize,9,9) = Jpi*covariance.block(0, covSize,covSize,9);

}


void MSCKF::augmentStateVector()
{
    //current body quaternion, position and velocity are added
    int Xsize = X.rows();
    X.conservativeResize(Xsize + 10);

    //appending the current body pose 
    X.segment<10>(Xsize) = X.head(10);

}


void MSCKF::runFeatureMatching(Mat &inputImg, Mat &outputImg)
{
    //cout << "ran featureMatching" << endl;
    // ORB...
    this->imageNum++;

    if (features.size() < 50)
    {
        ORB_train_kpts.clear();
        ORB_train_desc.release();
        features.clear();
        featuresIdx.clear();
        marginalizefilter();
    }
    
    lostfeatures.clear();
    lostfeaturesCamIdx.clear();
    
    featureMatching(inputImg, outputImg);

    return ;

}

void MSCKF::marginalizefilter()
{
    marginalizeStateVector();
    marginalizeCovariance();

}


void MSCKF::marginalizeStateVector()
{
    X.conservativeResize(16,1);
    
}

void MSCKF::marginalizeCovariance()
{

    covariance.conservativeResize(15,15);

}

Vector2d MSCKF::cameraProjection(Vector3d P)
{

    Vector2d uv(P(0)/P(2), P(1)/P(2));

    Vector2d CP = pp + focal*uv;

    return CP;


}

                            //G_pi, G_p_B, G_R_B, zij, Hbij, Hfij, rij
void MSCKF::cameraMeasurement(Vector3d& G_p_fi, Vector3d& G_p_B, Matrix3d& G_R_B, Vector2d& zi, Matrix<double, 2,9>& Hbi,  Matrix<double, 2,3>& Hfi, Vector2d& ri)
{
    Vector3d C_p_fi = C_R_B*G_R_B.transpose()*(G_p_fi- G_p_B) + C_p_B ;

    Vector2d zHati = cameraProjection(C_p_fi);
    
    double rho = 1/C_p_fi(2);
    double alpha = C_p_fi(0)*rho;
    double beta = C_p_fi(1)*rho;

    MatrixXd Jh = MatrixXd::Zero(2,3);

    Jh << rho, 0, -1*alpha,
          0, rho, -1*beta;

    Hfi << Jh*C_R_B*G_R_B.transpose();

    Matrix3d I3 = Matrix3d::Identity();
    Matrix3d Z3 = Matrix3d::Zero();

    MatrixXd Hfb; 
    Hfb << skewMatrix(G_p_fi - G_p_B), -I3, Z3;

    Hbi = Hfi*Hfb;
    ri = zi - zHati; //without Normalization

}


/*void stackingFeaturesPoses()
{


}*/

void nullSpace(MatrixXd &H, MatrixXd &A)
{
    //FULLPivLU<MatrixXd> lu(H);
    //A = lu.kernel();

    int n = H.rows() /2;

    JacobiSVD<MatrixXd> svd(H, ComputeFullU);
    A = svd.matrixU().rightcols(2*n-3).transpose();

}


void MSCKF::stackingResidualsFeature()
{
    //lostfeatures, lostfeaturesCamIdx,
        // single feature at a time
    


    //send the 2d measurements and associated camera poses to the traingulation function
    for (int i=0; i< int(lostfeatures.size()); ++i)
    {
        vector<Vector2d> Z;
        vector<Matrix3d> C_R_G;
        vector<Vector3d> C_p_G;
        int M = lostfeatures[i].cols(); // # of measurements
        for (int j=0; j < M; ++j)
        {
            Z.push_back(Vector2d(lostfeatures[i].col(j)));
            //C_R_B*G_R_B.transpose(); // C_p = C_R_G*(G_p_B) C_p_B
           // C_R_G.push_back(C_R_B*Quaterniond(X.segment<4>(16+4*j))).toRotationMatrix().transpose(); //
            MatrixXd C_R_Gj = C_R_B*((Quaterniond(X.segment<4>(16+10*j+4))).toRotationMatrix()).transpose();
            C_R_G.push_back(C_R_Gj); //

            Vector3d C_p_Gj = X.segment<3>(16+10*j+3) - C_R_Gj*C_p_B;
            C_p_G.push_back(C_p_Gj);

        }

        
       // VectorXd Poses = X.segment<10*M>(16);
        Vector3d  G_pi(1,1,1);            //GaussNewtonMinimization(Z,  C_R_G,  C_p_G);

        Matrix<double, 2,9> Hbij;
        MatrixXd Hxij = MatrixXd::Zero(2, 15+9*M);
        Matrix<double, 2,3> Hfij;
        Vector2d rij;
        MatrixXd Hxi = MatrixXd::Zero(2*M, 15+9*M);
        MatrixXd Hfi = MatrixXd::Zero(2*M, 3);
        VectorXd ri;
        VectorXd r0i;
        MatrixXd H0i;


        //compute Hx, Hf and stack them
        for (int j=0; j < M; ++j)
        {
            Vector3d G_p_B = X.segment<3>(16+10*j+3);
            MatrixXd G_R_B = Quaterniond(X.segment<4>(16+10*j+4)).toRotationMatrix();
            Vector2d zij = Z[i];
            cameraMeasurement(G_pi, G_p_B, G_R_B, zij, Hbij, Hfij, rij);
            Hxij.block<2,9>(0, 15+9*j) = Hbij;
            Hxi.block<2,Hxi.cols()>(2*j,0) = Hxij;
            Hfi.block<2,3>(2*j,0) = Hfij;
            ri.segment<2>(2*j) = rij;
        }

        MatrixXd Ai;
        nullSpace(Hfi, Ai);
    
        //applying nullspace transformation
        H0i = Ai.transpose()*Hxi;
        r0i =Ai.transpose()*ri; //size of r0i is 2*n-3, n: # of features

    }
       
}


/*bool ChiSqaureTest(MatrixXd &H, MatrixXd &Cov, MatrixXd &R, VectorXd r)
{
    int d = r.rows();
    MatrixXd gamma = r*(H*Cov*H.transpose()+ R).inverse()*r.transpose();

    bool result = 0;
    if (gamma(0) <= Chi2Inv(0.95, d))
    {
        //the feature passed the test
        result = 1;
    }
    return result;
}


void stackingResidualsFeatures(vector<MatrixXd> H, vector<VectorXd> r)
{
    vector<int> Hsizes, 

    int rowSizeAccumulator = 0;
    for (int i=0; i< H.size(); ++i)
    {
            
        int rowSizeAccmulator += H[i].rows();
        
    }

    Matrix<double, rowSizeAccumulator, 9> H0;
    VectorXd r0;
    
    for (int i= 0; i< H.size(); ++i)
    {
        //outlier rejection using chisquare test
        bool inlier = ChiSqaureTest(H0i, covariance, R, r0i);

        if (inlier)
        {
            //compute Hx, Hf and stack them
            int M = H[i].rows();
            int N = H[i].cols();
            H0.block<M,N>(M*i,0) = H[i];
            r0.segment<M>(M*i) = r[i];

        }

    }
    
}



void QRdecomposition(MatrixXd H0, VectorXd r0)
{
    HouseholderQR<MatrixXd> qr(H0);
    Q = qr.householderQ()*(MatrixXd::Identity(H0.rows(),H0.cols()));
    R = qr.matrixQR().block(0,0,H0.cols(),H0.cols()).triangularView<Upper>();   

    ColPivHouseholderQR<MatrixXd> cqr(H0);
    R = cqr.matrixR().topLeftCorner(rank(), rank()).template triangularView<Upper>() // rank deficient matrix
    R = cqr.matrixR().template triangularView<Upper>() // full rank matrix
    Q = cqr.householderQ().setLength(cqr.nonzeroPivots())
    
    int Qcols = Q.cols();

    
}


void MSCKFupdate(VectorXd &r0, MatrixXd  &R, MatrixXd &covariance, double sigma_im, int q)
{
    double sigma_im2 = sigma_im*sigma_im ;
    MatrixXd Iq = MatrixXd::Identity(q,q);
    MatrixXd Ibeta = MatrixXd::Identity(covariance.rows(),covariance.rows());
    K = covariance*R.transpose()*(R*covaraince*R.transpose() + sigma_im2*Iq).inverse();
    covariance = (Ibeta- K*R)*covariance*(Ibeta- K*R).transpose() + sigma_im2*K*Iq*K.transpose();
    
    VectorXd deltaX = K*rq;

    Quaterniond Q(X.segment<4>(0));
    Quaterniond dQ(0.5*deltaX(0), 0.5*deltaX(1), 0.5*deltaX(2), 1);
    X.segment<4>(0) = (Q*dq).coeffs();
    X.segment<3>(4) = deltaX.segment<3>(3);
    X.segment<3>(7) = deltaX.segment<3>(6);
    
    for (int i=16; i < X.size(); i +=10)
    {
        Map<Quaterniond> Q(X.segment<4>(i), 4)
        Quaterniond dQ(0.5*deltaX(i-1), 0.5*deltaX(i-1), 0.5*deltaX(i-1), 1);
        X.segment<4>(i) = Map<VectorXd> (quatMult(Q,dq), 4);
        X.segment<3>(i+4) = deltaX.segment<3>(i+3);
        X.segment<3>(i+7) = deltaX.segment<3>(i+6);
    }
} */