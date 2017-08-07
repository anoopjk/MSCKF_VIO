/*main file for processing the data and running the EKF*/


// #include "features.h"
// #include "EKF.h"

#define MIN_FEAT 500
#define MAXTRACK 30

#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <limits>

#include "MSCKF.h"

///////////////////////////////////////
//includes all the headers necessary to use the most
//common public pieces of the ROS system
//#include <ros/ros.h>
//use image_transport for publishing and subscribing to images in ROS
//#include <image_transport/image_transport.h>
//use cv_bridge to convert between ROS and Opencv Image formats
//#include <cv_bridge/cv_bridge.h>
//Include some useful constants for image encoding
//#include <sensor_msgs/image_encodings.h>
//Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
//Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>




using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace cv;



void  read_cam(std::ifstream& file, int& line_num, double& cam_time){
    file.seekg(std::ios::beg);

    for(int i=0; i < line_num - 1; ++i)  {
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    line_num += 1; // incerement the line number
    file >> cam_time;
    return ;
}




 void  read_imu(std::ifstream& file, int line_num, VectorXd& meas)  
 {
  
  //cout << "line_num: " << line_num << endl;
  std::string line;

  int i = 0;

  double item = 0.0;
   
  if (file.is_open())
  {
     bool success =  getline (file,line);
     //cout << line << endl;
    while ( success ) 
    {
        if (i==line_num) 
        {
              std::istringstream in(line);
              //cout << line << '\n';
              for (int j=0; j<7; j++)  {
                in >> item ;

                meas(j) = item;
                //cout << meas(j) << endl;
              }
              
              break;
        }
      i++;
     // cout << "i in imu_data: " << i << endl;
    }

    //line_num += 1;  // increase the line number for next iteration
    //imu_data.close();  // close the file after reading
  }

  else {
    cout << "Unable to open file";
    exit;
  }

  return  ;

}



int main( int argc, char** argv )
{

    // open the data files for both imu and camera
    ifstream camera_data("/home/anoop/Documents/robotics/EKF_mono_slam/VI_UPENN/visensor/bs/timestamps_cameras.txt"); 
    ifstream imu_data("/home/anoop/Documents/robotics/EKF_mono_slam/VI_UPENN/visensor/bs/imu.txt"); // imu
    ofstream output("/home/anoop/Documents/robotics/MSCKF_anoop/msckfOutput.txt");
    ofstream Idxfile("/home/anoop/Documents/robotics/MSCKF_anoop/featuresIdx.txt");
    Mat frame, outputImg;
    int framenum = 0;
    char filename[200];
    namedWindow("image", CV_WINDOW_NORMAL);


    double sigma_ac = 0.0034,
    sigma_gc = 0.0005,
    sigma_wac = 0.001,
    sigma_wgc = 0.0001,
    sigma_img = 120;

    double g = 9.803;

    double fx = 445.80;
    double fy = 445.15;
    double px= 371.50;
    double py = 237.33;

    Matrix4d viTb ;
    viTb << 0.9989551739, -0.0016800380, -0.0456698809, -0.0310,  
             0.0005817624, 0.9997105684, -0.0240508012, -0.1257,
             0.0456970689, 0.0239991032, 0.9986670221, -0.0171,
             0.0         , 0.0        , 0.0         , 1.0;

    MatrixXd bTvi = viTb.inverse();
    Matrix3d C_R_B;
     C_R_B << 0.9990911633, -0.0072415591, -0.0420048470, // Body to VI rightcamera
            0.0063429185, 0.9997489915, -0.0214877011,
            0.0421499079, 0.0212017389, 0.9988863156;
    Vector3d C_p_B;
    C_p_B << -0.0675, -0.1165, -0.0201;


   // VectorXd  imu_meas(7);
    //imu_meas << 0,0,0,0,0,0,0;

    VectorXd imu_meas = VectorXd::Zero(7,1);

    int imu_line = 0; // initialize the integer line number
    read_imu(imu_data, imu_line ,  imu_meas); // read the imu measurement vector for the first EKF propagation
    imu_line++;
    //////////////////////////////////////////////////////////////////////////////////////////////////
    double cam_time;
    int cam_line = 0;
    

    //Iniatialize the kalman filter

    //state vector is 4+3+3+6  (16 state vector)

    // imu state vector order:    quaternion, position, velocity, bg, ba
    VectorXd X0 = VectorXd::Zero(16,1);
    X0(3) =  1.0;
    MatrixXd P0 = MatrixXd::Zero(15,15);


    //time step
    double tPrev = imu_meas(0), imu_time = 0.0;
    double dt = 0.0;

    Vector4d B_wCurr, B_wPrev, B_aCurr, B_aPrev;

    Vector4d vigyr ;
    vigyr << imu_meas.segment<3>(4), 1.0 ;
    Vector4d viacc ;
    viacc << imu_meas.segment<3>(1), 1.0 ;

    B_wCurr = bTvi*vigyr;
    B_aCurr = bTvi*viacc;

    //Quaterniond q0 = Quaterniond::FromTwoVectors(Vector3d(0,0,1), Vector3d(B_aCurr.segment<3>(0)));

    //X0.segment<4>(0) = q0.coeffs();

    Matrix3d orientR;
    orientR << 0.568928208  ,  0.092530211  ,  0.817165133 ,
              -0.822016967  ,  0.093795894,  0.561685353  ,
               -0.024673870 ,  -0.991282246 ,   0.129424533 ;
                        
    
    Quaterniond orient = Quaterniond(orientR);   

    cout << orient.coeffs().transpose() << endl;
    
    

    X0.segment<4>(0) = orient.coeffs();
    X0.segment<3>(4) = Vector3d(44.16966 , 1.43745 , -0.03012);
    X0.segment<3>(10) = 0.001*Vector3d(1,1,1);
    X0.segment<3>(13) = 0.001*Vector3d(1,1,1);


    B_wPrev = B_wCurr; 
    B_aPrev = B_aCurr;


    MSCKF filter( X0, P0, B_wPrev.segment<3>(0), B_aPrev.segment<3>(0), dt);
    filter.featuresInit();
    filter.C_R_B = C_R_B;
    filter.C_p_B = C_p_B;
    filter.IMUParams.sigma_ac = sigma_ac;
    filter.IMUParams.sigma_gc = sigma_gc ;
    filter.IMUParams.sigma_wgc = sigma_wgc;
    filter.IMUParams.sigma_wac = sigma_wac;
    filter.IMUParams.g = g;

    filter.g = Vector3d(0,0,-g);

    filter.CameraParams.fx = fx;
    filter.CameraParams.fy = fy;
    filter.CameraParams.px = px;
    filter.CameraParams.py = py;
    filter.CameraParams.sigma_img = sigma_img;


    while(!imu_data.eof())
    {
        read_cam(camera_data, cam_line, cam_time);
      //  cout << "entered while loop" << endl;
        sprintf(filename, "/home/anoop/Documents/robotics/EKF_mono_slam/VI_UPENN/visensor/bs/right_cam_frames/frame%04d.jpg", framenum);
        filter.img = imread(filename);
        framenum++;

        if(! filter.img.data || filter.img.empty() )  {                            // Check for invalid input
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;  //break;
        }

        cout << "imu time: " << imu_time << ' ' << "camera time: " <<  cam_time << endl;
        while(imu_time <= cam_time)
        {
            //MSCKF propagate
            filter.dt = imu_time - tPrev;

            cout << filter.dt << endl;
            vigyr << imu_meas.segment<3>(4), 1.0 ;
            viacc << imu_meas.segment<3>(1), 1.0 ;

            B_wCurr = bTvi*vigyr;
            B_aCurr = bTvi*viacc;
            cout << "filter g: " << filter.g.transpose() << endl;
            cout << "Gyroscope" << B_wCurr.transpose() << endl;
            cout << "accelerometer" << B_aCurr.transpose() << endl;

            //filter.propagateIMU(B_wCurr.segment<3>(0), B_aCurr.segment<3>(0));
            filter.processIMU(  B_aCurr.segment<3>(0), B_wCurr.segment<3>(0));


           // cout << "IMU propagatation done" << endl;
            read_imu(imu_data, imu_line , imu_meas); // read the imu measurement vector for  EKF propagation
            imu_line++;
            //cout << "%%%%%%%%%%%%%%%%%%%%%%\n" ;
            //cout << "measurement vector" << meas << "\n";
            //cout << "%%%%%%%%%%%%%%%%%%%%%%\n" ;
            //cout << "reading imu line: " << imu_line << endl;

            tPrev = imu_time; // update the previous imu time

            imu_time = imu_meas(0); //current imu time             

            
        }


    //filter.runFeatureMatching(filter.img, outputImg);
    //cout << "covariance size: " <<  filter.covariance.rows() << ',' << filter.covariance.cols() << endl;
    //filter.augmentFilter();
    

    

    //cout << "imageNum: " << filter.imageNum << endl;

    //cout << filter.features[1].cols() << endl;
    //Idxfile << "features size" << filter.featuresIdx << '\n' << endl;

    imshow("image", filter.img);
    waitKey(1);   

    VectorXd pose = filter.X.segment<7>(0);
    //output << pose.transpose() << endl;
    output << (pose(4)) << ' ' << (pose(5)) << ' ' << (pose(6)) << endl;   

    output.flush();  
    Idxfile.flush();
    }                                  
   

 return 0;

}


