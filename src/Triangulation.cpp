#include "MSCKF.h"
using namespace cv;
using namespace Eigen;
using namespace std;


Vector3d inverseDepth2XYZ(double alpha, double beta, double rho, Matrix3d R, Vector3d t)
{
	Vector3d result;
	Vector3d abrho(alpha, beta, 1)
	result = (1/rho)*(R*abrho + rho*t);

	return result;

}

void MSCKF::normalizedImagePoints(const Vector2d &v , Vector2d &vNormal)
{
	/*Matrix3d K << fxy(0), 0, pp(0),
				  0, fxy(1), pp(1),
				  0, 0,	   , 1;*/

    vNormal(0) = (v(0) - pp(0)) / focal(0,0) ;	
    vNormal(1) = (v(1) - pp(1)) / focal(1,1) ;

}

// linear least squares
Vector3d MSCKF::leastSquares(Vector2d z1, Vector2d z2, Quaterniond C1_q_G, Quaterniond C2_q_G, 
		Vector3d G_p_C1, Vector3d G_p_C2)
{
	/*MatrixXd Z = MatrixXd::Ones(3,3);
	Z.block<2,1>(0,0) = z1;
	Z.block<2,1>(0,1) = z2; 

	vector<Vector2d>VNormal;*/
	Vector2d z1N, z2N;
	normalizedImagePoints(z1, z1N);
	normalizedImagePoints(z2, z2N);
	MatrixXd A = MatrixXd::Zero(4,3);

	Matrix3d C1_R_G = C1_q_G.toRotationMatrix();
	Matrix3d C2_R_G = C2_q_G.toRotationMatrix();

	Matrix3d rot = C2_R_G*C1_R_G.transpose();
	Vector3d t = C2_R_G*(G_p_C1 - G_p_C2);

	A << z2N(0)*rot(2,0)-rot(0,0),  z2N(0)*rot(2,1)-rot(0,1), z2N(0)*rot(2,2)-rot(0,2),
		 z2N(1)*rot(2,0)-rot(1,0),  z2N(1)*rot(2,1)-rot(1,1), z2N(1)*rot(2,2)-rot(1,2),
		 -1,						0,							z1N(0),
		  0,						-1,							z1N(1);

    VectorXd b << t(0) - z2N(0)*t(2), t(1) - z2N(1)*t(2), 0, 0;

    Vector3d C1pf;

    bool solved = A.lu().solve(b, &C1pf);  // Stable and fast. #include <Eigen/LU>
	
	Vector3d C1pfInvD(C1pf(0)/C1pf(2), C1pf(1)/C1pf(2), 1/C1pf(2)); // these are the inverse depth
	//parameters for the feature point interms of camera1 (the camera pose in which the feature was 
	//first recorded)

	return C1pfInvD;
}



//Non linear least squares
Vector3d MSCKF::GaussNewtonMinimization(vector<Vectro2d> &Z, vector<Matrix3d> &C_R_G, vector<Vector3d> &C_p_G)
{	
	// 3d point estimation of one feature at a time.
	Vector3d theta0 = leastSquares( Z[0], Z[Z.size()-1], C_R_G[0], C_R_G[C_R_G.size()-1], C_p_G[0], C_p_G[C_p_G.size()-1]);
	Vector3d thetai;

	for(int i = 0; i< 20; ++i)
	{

		MatrixXd dhdgi = MatrixXd::Zero(2,3);

		MatrixXd duvdg = MatrixXd::Zero(2,3);

		double u = x/z;
		double v = y/z;


		dhdgi = focal(0,0), 0, -focal(0,0)*theta0(0),
				0, focal(1,1), -focal(1,1)*theta0(1);

		dhdgi = (1/theta0(2))*dhdgi;

		Vector3d dgidtheta;
		Vector3d e1(1,0,0);
		Vector3d e2(0,1,0);
		dgidtheta(0) = Ci_R_C0*e1;
		dgidtheta(1) = Ci_R_C0*e2;
		dgidtheta(2) = Ci_p_C0;

		Jf = dhdg*dgidtheta;

		thetai = theta0 - (Jf.transpose()*Jf).inverse()*Jf.transpose()*f(theta0);

		
		theta0 = thetai;
	}

	Vector3d alphaBeta1(thetai(0), thetai(1), 1);
	Vector3d Gpf = (1/thetai(2))*G_R_C0*alphaBeta1 + G_p_C0 ;


	return Gpf;


}



//////////////////////////////////////////////////////////////////////////////////////////////////////

