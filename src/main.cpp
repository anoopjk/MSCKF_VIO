/*main file for processing the data and running the EKF*/


// #include "features.h"
// #include "EKF.h"


#define MAX_FRAME 4000
#define MIN_NUM_FEAT 2000

#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <limits>

#include "MSCKF.h"
#include "../fast-cpp-csv-parser/csv.h"
#include <math.h>
//Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
//Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace cv;



int main( int argc, char** argv )
{

  Mat outputImg;

  ofstream output;
  output.open ("msckfOutput.txt");

  double scale = 1.00;
  io::CSVReader<7> imu_data("/home/anoop/Documents/robotics/EKF_mono_slam/mav0/imu0/data.csv");
  imu_data.read_header(io::ignore_extra_column, "#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]" , "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]");
  double imu_time, wx, wy, wz, ax, ay, az;
  imu_time = 1403636579758555392;

  io::CSVReader<2> camera_data("/home/anoop/Documents/robotics/EKF_mono_slam/mav0/cam0/data.csv"); 
  camera_data.read_header(io::ignore_extra_column, "#timestamp [ns]", "filename");
  double cam_time; std::string imgName;

  String folderpath = "/home/anoop/Documents/robotics/EKF_mono_slam/mav0/cam0/data/*.png";
  vector<String> filenames;
  cv::glob(folderpath, filenames);
  

  Mat K(3, 3, CV_64F);
  K.at<double>(0,0) = 458.654;  K.at<double>(0,1) = 0.; K.at<double>(0,2) = 367.215;
  K.at<double>(1,0) = 0.; K.at<double>(1,1) = 457.296; K.at<double>(1,2) = 248.375;
  K.at<double>(2,0) = 0.; K.at<double>(2,1) = 0.; K.at<double>(2,2) = 1.;

  Mat D(5, 1, CV_64F);
  D.at<double>(0,0) = -0.28340811;
  D.at<double>(0,1) = 0.07395907;
  D.at<double>(0,2) = 0.0;
  D.at<double>(0,3) = 0.0;
  D.at<double>(0,4) = 0.0;
  

  double focal = 458.654;
  cv::Point2d pp(371.50, 237.33);

  //camera to body transformation
  Matrix4d B_T_C ;
  
  B_T_C <<  0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0;
  Matrix3d C_R_B = B_T_C.inverse().block<3,3>(0,0);

  Vector3d C_p_B =  B_T_C.inverse().block<3,1>(0,3);
  



  namedWindow( "Camerafeed", WINDOW_AUTOSIZE );// Create a window for display.


    double sigma_ac = 2.0000e-3,
    sigma_gc = 1.6968e-04,
    sigma_wac =  3.0000e-3,
    sigma_wgc = 1.9393e-05,
    sigma_img = 20;

    double g = 9.803;

    double fx = 458.654;
    double fy = 457.296;
    double px= 367.215;
    double py = 248.375;
    

    //Iniatialize the kalman filter

    //state vector is 4+3+3+6  (16 state vector)

    // imu state vector order:    quaternion, position, velocity, bg, ba
    VectorXd X0 = VectorXd::Zero(16,1);
    MatrixXd P0 = MatrixXd::Zero(15,15);


    //time step
    double tPrev = imu_time;
    double dt = 0.0;

    Quaterniond orient = Quaterniond( 0.534108, -0.153029,  -0.827383,  -0.082152);

    cout << orient.coeffs().transpose() << endl;
    

    X0.segment<4>(0) = orient.coeffs();
    X0.segment<3>(4) = Vector3d(4.688319,  -1.786938,  0.783338);
    X0.segment<3>(10) = 1.9393e-05*Vector3d(1, 1, 1);
    X0.segment<3>(13) = 3.0000e-3*Vector3d(1, 1, 1);



    Vector3d B_wPrev = Vector3d(-0.099134701513277898,0.14730578886832138,0.02722713633111154);  
    Vector3d B_aPrev = Vector3d(8.1476917083333333,-0.37592158333333331,-2.4026292499999999);


    MSCKF filter( X0, P0, B_wPrev, B_aPrev, dt);
    filter.featuresInit();
    filter.C_R_B = C_R_B;
    filter.C_p_B = C_p_B;
    filter.IMUParams.sigma_ac = sigma_ac;
    filter.IMUParams.sigma_gc = sigma_gc ;
    filter.IMUParams.sigma_wgc = sigma_wgc;
    filter.IMUParams.sigma_wac = sigma_wac;
    filter.IMUParams.g = g;

    filter.CameraParams.sigma_img = sigma_img;

    filter.g = Vector3d(0,0,-g);

    filter.CameraParams.fx = fx;
    filter.CameraParams.fy = fy;
    filter.CameraParams.px = px;
    filter.CameraParams.py = py;
    filter.CameraParams.sigma_img = sigma_img;

    int framenum = 0;
    while((camera_data.read_row(cam_time, imgName) ))
    {
        cout << "Image#: " <<  framenum << endl;

        filter.img = imread(filenames[framenum]);

        if(! filter.img.data || filter.img.empty() )  {                            // Check for invalid input
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;  //break;
        }

        //cout << "imu time: " << imu_time << ' ' << "camera time: " <<  cam_time << endl;
        while(imu_time <= cam_time)
        {
            imu_data.read_row(imu_time, wx, wy, wz, ax, ay, az);
            //MSCKF propagate
            filter.dt = (imu_time - tPrev)/1000000000;

            //cout << filter.dt << endl;

            //cout << "filter g: " << filter.g.transpose() << endl;

            //filter.propagateIMU(Vector3d(wx, wy, wz), Vector3d(ax, ay, az));
            filter.processIMU( Vector3d(ax, ay, az), Vector3d(wx, wy, wz));

            tPrev = imu_time; // update the previous imu time           
        }

        filter.runFeatureMatching(filter.img, outputImg);
        cout << "covariance size: " <<  filter.covariance.rows() << ',' << filter.covariance.cols() << endl;
        filter.augmentFilter();
        filter.stackingResidualsFeature();
        

        

        cout << "imageNum: " << filter.imageNum << endl;

        //cout << filter.features[1].cols() << endl;
        //Idxfile << "features size" << filter.featuresIdx << '\n' << endl;



        imshow("Camerafeed", outputImg);
        waitKey(1);   

        VectorXd pose = filter.X.segment<7>(0);
        //output << pose.transpose() << endl;
        output << (pose(4)) << ' ' << (pose(5)) << ' ' << (pose(6)) << endl;  
        //output << t_f(0) << ' ' << t_f(1) << ' ' << t_f(2) << endl;    
        output.flush();  
            



    
    framenum++; // increment the image counter

    }                                  
   

 return 0;

}


