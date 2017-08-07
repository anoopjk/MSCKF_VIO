#include "MSCKF.h"
using namespace Eigen;
using namespace std;


MSCKF::MSCKF()
{

}



MSCKF::MSCKF(VectorXd X, MatrixXd covariance, Vector3d B_wPrev, Vector3d B_aPrev, double dt)
{
		//initializing all the variables
	this->X= X;
	this->covariance = covariance;
    this->sigma_gc = sigma_gc;
    this->sigma_ac = sigma_ac;
    this->sigma_wac = sigma_wac;
    this->sigma_wgc = sigma_wgc;

	
	this->dt = dt;
	this->B_aPrev  = B_aPrev;
	this->B_wPrev = B_wPrev;

}

MSCKF::~MSCKF()
{



} 

Matrix3d MSCKF::skewMatrix( Vector3d v){

	Matrix3d skewV;
	skewV << 0, -v(2) , v(1),
		 	v(2), 0 , -v(0),
		 	-v(1), v(0),  0;

 	return skewV;

}

Matrix4d MSCKF::BigOmega( Vector3d w){
	//constructing big omega matrix
	 Matrix4d W= Matrix4d::Zero();
	 W.block<3,3>(0,0)  = -1*skewMatrix(w) ;
	 W.block<1,3>(3,0)  =  -w.transpose();
	 W.block<3,1>(0,3)  =  w;
	// W(3,3) = 0.0;

	 return W;

}




/*VectorXd MSCKF::flattenMatrix(MatrixXd &M)
{

	Map<VectorXd> v(M.data(), M.size());

	return VectorXd;

}

MatrixXd MSCKF::deFlattenMatrix(VectorXd v)
{

	Map<MatrixXd> M(&v[0], v.size());

	return MatrixXd;

}*/

	

Quaterniond MSCKF::quatMultiplication(Quaterniond q1, Quaterniond q2)
{
	Quaterniond result;
	result.setIdentity();
	
	result.w() = q1.w()*q2.w() - q1.vec().dot(q2.vec());
	result.vec() = q1.w()*q2.vec() + q2.w()*q1.vec() + q1.vec().cross(q2.vec());
	
	return result;

}	  

//std::pair<Quaterniond,Quaterniond>
void  MSCKF::propagateIMU(Vector3d B_wRaw, Vector3d B_aRaw)
 {

 	Quaterniond G_q_BPrev(this->X.segment<4>(0)); //Quaterniond G_q_BPrev(X(0),X(1),X(2),X(3));

 	Vector3d G_p_BPrev(this->X.segment<3>(4));

 	Vector3d G_v_BPrev(this->X.segment<3>(7));

 	Vector3d bg(this->X.segment<3>(10));

 	Vector3d ba(this->X.segment<3>(13));

 	Vector3d B_wCurr = (B_wRaw - bg) ;

 	//gyroscope propagation
	Vector4d q0(0,0,0,1);

	Vector3d B_wMid = (B_wCurr + B_wPrev);

	Vector4d k1 = 0.5*BigOmega(B_wPrev)*q0;
	Vector4d k2 = 0.5*BigOmega(B_wMid)*(q0 + 0.5*dt*k1);
	Vector4d k3 = 0.5*BigOmega(B_wMid)*(q0 + 0.5*dt*k2);
	Vector4d k4 = 0.5*BigOmega(B_wCurr)*(q0 + dt*k3);

	//cout << "Bigomega: " << BigOmega(B_wCurr) << endl;

	Vector4d B1_q_Bvec = q0 + (dt/6)*(k1 + 2.0*k2+ 2.0*k3 + k4); 

	//Quaterniond B1_q_B(B1_q_Bvec(0), B1_q_Bvec(1), B1_q_Bvec(2), B1_q_Bvec(3));
	//cout << B1_q_Bvec.transpose() << endl;
	Quaterniond B1_q_B = Quaterniond(B1_q_Bvec) ;                      //Map<Quaterniond> (B1_q_Bvec(0), B1_q_Bvec.size());
	//cout << B1_q_B.coeffs() << endl;

	B1_q_B.normalize();
	Quaterniond B1_q_G = B1_q_B*G_q_BPrev.conjugate() ;                 //quatMultiplication(B1_q_B,G_q_BPrev.conjugate());
	Quaterniond G_q_BCurr = B1_q_G.conjugate();

	//cout << "G_q_BCurr: " << G_q_BCurr.coeffs() <<',' << "G_q_BPrev: " << G_q_BPrev.coeffs() << endl;

	//accelerometer propagation


	Vector3d B_aCurr = (B_aRaw - ba);

	//cout << "B_aCurr: " << B_aCurr.transpose() << ',' <<  "B_aRaw: " << B_aRaw.transpose() << ',' 
	//<< "B_aPrev: " << B_aPrev.transpose() << endl;
	//cout << "B_aCurr: " << B_aCurr.transpose() << endl;
	//cout << "B_aPrev: " << B_aPrev.transpose() << endl;


	Vector3d s = (dt/2.0)*(B1_q_B.conjugate()._transformVector(B_aCurr) + B_aPrev);

	//cout << "s: " << s.transpose() << endl;
	Vector3d y = (dt/2.0)*s;
	//cout << "y: " << y.transpose() << endl;

	//cout << g.transpose() << endl;

	Vector3d G_v_BCurr = G_v_BPrev + G_q_BPrev._transformVector(s) + dt*g;

	//cout << "G_v_BCurr: " << G_v_BCurr.transpose() << ',' << "G_v_BPrev: " << G_v_BPrev.transpose() << endl;

	//cout << "dt*g: " << (dt*g).transpose() << endl;

	Vector3d G_p_BCurr = G_p_BPrev + dt*G_v_BPrev + G_q_BPrev._transformVector(y) + 0.5*dt*dt*g ;
	//cout << "G_p_BCurr: " << G_p_BCurr.transpose() << ',' << "G_p_BPrev: " << G_p_BPrev.transpose() << endl;
	

	this->B_aPrev = B_aCurr;
	this->B_wPrev = B_wCurr;
///////////////////////////////////////////////////////////////////////////////////////////////
//state_vectorX for global shutter cam & high precision Imu
	this->X.segment<4>(0) = G_q_BCurr.coeffs();
    this->X.segment<3>(4) = G_p_BCurr;
    this->X.segment<3>(7) = G_v_BCurr;
   // this->X.segment<3>(10)=  bg;
   //this->X.segment<3>(13)=  ba;
	   //X.segment<X.rows()-16>(16) = X.segment<X.rows()-16>(16);	
//////////////////////////////////////////////////////////////////////////////////////////////////////
	Vector3d G_aCurr = G_q_BPrev._transformVector(B_aCurr) + g;

	MatrixXd G_R_BMid = G_q_BCurr.toRotationMatrix() + G_q_BPrev.toRotationMatrix();
	Matrix3d G_R_BPrev = G_q_BPrev.toRotationMatrix();
	Matrix3d Phi_qq = Matrix3d::Identity();
	Matrix3d Phi_pq = -skewMatrix(G_R_BPrev*y);
	Matrix3d Phi_vq = -skewMatrix(G_R_BPrev*s);

	MatrixXd Phi_qbg = -1*(dt/2)*(G_R_BMid);
	MatrixXd Phi_vbg = (dt*dt/4)*(skewMatrix(G_aCurr- g))*(G_R_BMid);
	MatrixXd Phi_pbg = (dt/2)*Phi_vbg ;
	MatrixXd Phi_qba = (dt/2)*(G_R_BMid);
	MatrixXd Phi_vba = -1*(dt/2)*(G_R_BMid) + Phi_vbg;
	MatrixXd Phi_pba = (dt/2)*Phi_vba;

	long PhiSize = 15;
	MatrixXd Phi_B = MatrixXd::Identity(PhiSize, PhiSize);
	Matrix3d I3 = Matrix3d::Identity();
	Matrix3d Z3 = Matrix3d::Zero();
	
	/*Phi_B.block<3,3>(0,9) = Phi_qba;
	Phi_B.block<3,3>(0,12) = Phi_qba;
	Phi_B.block<3,3>(3,0) = Phi_pq;
	Phi_B.block<3,3>(3,6) = dt*I3;
	Phi_B.block<3,3>(3,9) = Phi_pbg;
	Phi_B.block<3,3>(3,12) = Phi_pba;
	Phi_B.block<3,3>(6,0) = Phi_vq;
	Phi_B.block<3,3>(6,9) = Phi_vbg;
	Phi_B.block<3,3>(6,12) = Phi_vba;*/

	Phi_B << I3,     Z3, Z3,    Phi_qbg, Phi_qba,
			 Phi_pq, I3, dt*I3, Phi_pbg, Phi_pba,
			 Phi_vq, Z3, I3, 	Phi_vbg, Phi_vba,
			 Z3,	 Z3, Z3,	I3,		 Z3,
			 Z3,	 Z3, Z3,	Z3,		 I3;


	/*double sigma_gc2 = sigma_gc*sigma_gc;
	double sigma_ac2 = sigma_ac*sigma_ac;
	double sigma_wgc2 = sigma_wgc*sigma_wgc;
	double sigma_wac2 = sigma_wac*sigma_wac;*/


	Matrix<double, 15, 15> Nc;
	Matrix<double, 15, 1> Nc_diag;
	Nc_diag <<
		sigma_gc*sigma_gc * Vector3d(1, 1, 1),Vector3d(0, 0, 0),
		sigma_ac*sigma_ac * Vector3d(1, 1, 1),
		sigma_wgc*sigma_wgc * Vector3d(1, 1, 1),
		sigma_wac*sigma_wac * Vector3d(1, 1, 1);
	
	Nc = Nc_diag.asDiagonal();

	Matrix<double, 15, 15> Qd = 0.5*dt*Phi_B*Nc*Phi_B.transpose() + Nc;

	/*covariance.block<15,15>(0,0) = Phi_B*covariance.block<15,15>(0,0)*Phi_B.transpose() + Qd;
	covariance.block<15, (int(covariance.cols())-15) >(0,15) = Phi_B*covariance.block<15, (int(covariance.cols())- 15)>(0,15);
	covariance.block<(int(covariance.rows())-15),15>(15,0)= covariance.block<15,(int(covariance.cols())-15)>(0,15).transpose();*/

	covariance.block<15,15>(0, 0) = Phi_B * covariance.block<15,15>(0, 0) * Phi_B.transpose() + Qd;
	covariance.block(0, 15, 15, covariance.cols() - 15) = Phi_B * covariance.block(0, 15, 15, covariance.cols() - 15);
	covariance.block(15, 0, covariance.rows() - 15, 15) = covariance.block(0, 15, 15, covariance.cols() - 15).transpose();
    

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
static Vector3d g(0.0f, 0.0f, -9.8f);
#include "math_tool.h"
void MSCKF::processIMU(Vector3d linear_acceleration, Vector3d angular_velocity)
{
	int ERROR_STATE_SIZE = 15;
    Vector4d small_rotation;
    Matrix3d d_R, prev_R, average_R, phi_vbg;
    Vector3d s_hat, y_hat;
    Vector3d tmp_vel, tmp_pos;
    // read nomial state to get variables
    Vector4d spatial_quaternion = X.segment(0, 4); //q_gb
    Vector3d spatial_position = X.segment(4, 3);
    Vector3d spatial_velocity = X.segment(7, 3);
    Vector3d gyro_bias = X.segment(10, 3);
    Vector3d acce_bias = X.segment(13, 3);
    
    Quaterniond spa(spatial_quaternion);
  
    Matrix3d spatial_rotation = spa.matrix();
    
    Vector3d curr_w = angular_velocity - gyro_bias;
    Vector3d curr_a = linear_acceleration - acce_bias;

    
    //calculate q_B{l+1}B{l}
    d_R = delta_quaternion(B_wPrev, curr_w, dt).matrix();
    // defined in paper P.49
    s_hat = 0.5f * dt * (d_R.transpose() * curr_a + B_aPrev);
    //s_hat = dt *curr_a;
    y_hat = 0.5f * dt * s_hat;

    /* update nominal state */
     prev_R = spatial_rotation;

    //spatial_quaternion = quaternion_correct(spatial_quaternion, curr_w * dt);
    spatial_rotation = spatial_rotation*d_R.transpose();
    spatial_quaternion = R_to_quaternion(spatial_rotation);

    //spatial_position += spatial_velocity * dt + spatial_rotation * curr_a * dt * dt / 2;
    //spatial_velocity += spatial_rotation * curr_a * dt + g * dt;

    tmp_pos = spatial_position 
                    + spatial_velocity * dt
                    + spatial_rotation * y_hat + 0.5 * g * dt * dt;
    tmp_vel = spatial_velocity + spatial_rotation * s_hat + g * dt;
    spatial_velocity = tmp_vel;
    spatial_position = tmp_pos;

    X.segment(0, 4) = spatial_quaternion; //q_gb
    X.segment(4, 3) = spatial_position;
    X.segment(7, 3) = spatial_velocity;

    // save prev
	this->B_wPrev =  curr_w;
    this->B_aPrev = curr_a;

    /* propogate error covariance */
    average_R = prev_R + spatial_rotation;
    MatrixXd phi = MatrixXd::Identity(ERROR_STATE_SIZE,ERROR_STATE_SIZE);
    //1. phi_pq
    phi.block<3,3>(3,0) = -skew_mtx(prev_R * y_hat);
    //2. phi_vq
    phi.block<3,3>(6,0) = -skew_mtx(prev_R * s_hat);
    //3. one bloack need to times dt;
    phi.block<3,3>(3,6) = Matrix3d::Identity() * dt;
    //4. phi_qbg
    phi.block<3,3>(0,9) = -0.5f * dt * average_R;
    //5. phi_vbg
    phi_vbg = 0.25f * dt * dt * (skew_mtx(spatial_rotation * curr_a) * average_R);
    phi.block<3,3>(6,9) = phi_vbg;
    //6. phi_pbg
    phi.block<3,3>(3,9) = 0.5f * dt * phi_vbg;
    
    //7. phi_vba
    phi.block<3,3>(6,12) = -0.5f * dt * average_R;
    //8. phi_pba
    phi.block<3,3>(3,12) = -0.25f * dt * dt * average_R;
    
    int errorStateLength =  15;//(int)covariance.rows();
    Matrix<double, 15, 15> Nc;
	Matrix<double, 15, 1> Nc_diag;
	Nc_diag <<
		sigma_gc*sigma_gc * Vector3d(1, 1, 1),Vector3d(0, 0, 0),
		sigma_ac*sigma_ac * Vector3d(1, 1, 1),
		sigma_wgc*sigma_wgc * Vector3d(1, 1, 1),
		sigma_wac*sigma_wac * Vector3d(1, 1, 1);
	
	Nc = Nc_diag.asDiagonal();
    MatrixXd Qd = phi * (0.5 * dt * Nc) * phi.transpose() + Nc;
  	covariance.block<15,15>(0, 0) = phi * covariance.block<15,15>(0, 0) * phi.transpose() + Qd;
	covariance.block(0, 15, 15, covariance.cols() - 15) = phi * covariance.block(0, 15, 15, covariance.cols() - 15);
	covariance.block(15, 0, covariance.rows() - 15, 15) = covariance.block(0, 15, 15, covariance.cols() - 15).transpose();  
   
    

    return;
}