// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;


bool is_gravity_set = false;
bool is_first_img = true;
bool is_init_imu = false;
Eigen::Vector3d acc_last;
Eigen::Vector3d gyr_last;
double imu_state_time = -1;
std::vector<sensor_msgs::Imu> imu_msg_buffer;

double param_imu_state[16] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_i_ekf(param_imu_state); // from current inertial to inertial world
Eigen::Map<Eigen::Vector3d> gyro_bias(param_imu_state + 4);
Eigen::Map<Eigen::Vector3d> velocity(param_imu_state + 7);
Eigen::Map<Eigen::Vector3d> acc_bias(param_imu_state + 10);
Eigen::Map<Eigen::Vector3d> t_w_i_ekf(param_imu_state + 13);
double param_ex[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_i_l(param_ex); // from lidar to inertial
Eigen::Map<Eigen::Vector3d> t_i_l(param_ex + 4);
Eigen::Matrix<double, 12, 12> continuous_noise_cov = Eigen::Matrix<double, 12, 12>::Zero();
Eigen::Matrix<double, 21, 21> state_cov = Eigen::Matrix<double, 21, 21>::Zero();
Eigen::Vector3d gravity;

Eigen::Quaterniond q_w_i_ekf_last(1, 0, 0, 0);
Eigen::Vector3d t_w_i_ekf_last(0, 0, 0);

const int iterNum = 2;
double LIDAR_STD;
double gyro_noise = 0.005, gyro_bias_noise = 4e-6, acc_noise = 0.01, acc_bias_noise = 0.0002;

void initializeGravityAndBias() {
	// Initialize gravity and gyro bias.
	Eigen::Vector3d sum_angular_vel = Eigen::Vector3d::Zero();
	Eigen::Vector3d sum_linear_acc = Eigen::Vector3d::Zero();

	for (const auto& imu_msg : imu_msg_buffer) {
	Eigen::Vector3d angular_vel = Eigen::Vector3d::Zero();
	Eigen::Vector3d linear_acc = Eigen::Vector3d::Zero();
	linear_acc << imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
				imu_msg.linear_acceleration.z;
	angular_vel << imu_msg.angular_velocity.x, imu_msg.angular_velocity.y,
					imu_msg.angular_velocity.z;

	sum_angular_vel += angular_vel;
	sum_linear_acc += linear_acc;
	}

	gyro_bias =
	sum_angular_vel / imu_msg_buffer.size();
	Eigen::Vector3d gravity_imu =
	sum_linear_acc / imu_msg_buffer.size();

	// Initialize the initial orientation, so that the estimation
	// is consistent with the inertial frame.
	double gravity_norm = gravity_imu.norm();
	gravity = Eigen::Vector3d(0.0, 0.0, -gravity_norm);

	Eigen::Quaterniond q0_i_w = Eigen::Quaterniond::FromTwoVectors(
	gravity_imu, -gravity);
	q_w_i_ekf = (q0_i_w.toRotationMatrix());
	std::cout << "INIT ROTATION" << std::endl << q_w_i_ekf.toRotationMatrix() << std::endl;
	std::cout << "INIT Bg" << std::endl << gyro_bias.transpose() << std::endl;

	const double gyro_bias_cov = 1e-4, acc_bias_cov = 1e-2, velocity_cov = 0.025;
	const double extrinsic_rotation_cov = 3.0462e-4, extrinsic_translation_cov = 1e-4;
	
	for (int i = 3; i < 6; ++i)
		state_cov(i, i) = gyro_bias_cov;
	for (int i = 6; i < 9; ++i)
		state_cov(i, i) = velocity_cov;
	for (int i = 9; i < 12; ++i)
		state_cov(i, i) = acc_bias_cov;
	for (int i = 15; i < 18; ++i)
		state_cov(i, i) = extrinsic_rotation_cov;
	for (int i = 18; i < 21; ++i)
		state_cov(i, i) = extrinsic_translation_cov;
	
	continuous_noise_cov.block<3, 3>(0, 0) =
    	Eigen::Matrix3d::Identity() * gyro_noise * gyro_noise;
	continuous_noise_cov.block<3, 3>(3, 3) =
		Eigen::Matrix3d::Identity() * gyro_bias_noise * gyro_bias_noise;
	continuous_noise_cov.block<3, 3>(6, 6) =
		Eigen::Matrix3d::Identity() * acc_noise * acc_noise;
	continuous_noise_cov.block<3, 3>(9, 9) =
		Eigen::Matrix3d::Identity() * acc_bias_noise * acc_bias_noise;
	return;
}

void imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {
	// IMU msgs are pushed backed into a buffer instead of
	// being processed immediately. The IMU msgs are processed
	// when the next image is available, in which way, we can
	// easily handle the transfer delay.
	imu_msg_buffer.push_back(*msg);
	if (!is_gravity_set) {
	if (imu_msg_buffer.size() < 200) return;
	//if (imu_msg_buffer.size() < 10) return;
	initializeGravityAndBias();
	is_gravity_set = true;
	}
	return;
}

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) {
	Eigen::Matrix3d w_hat;
	w_hat(0, 0) = 0;
	w_hat(0, 1) = -w(2);
	w_hat(0, 2) = w(1);
	w_hat(1, 0) = w(2);
	w_hat(1, 1) = 0;
	w_hat(1, 2) = -w(0);
	w_hat(2, 0) = -w(1); 
	w_hat(2, 1) = w(0);
	w_hat(2, 2) = 0;
	return w_hat;
}

static Eigen::Quaterniond axis2Quat(const Eigen::Vector3d &axis, double theta) {
  Eigen::Quaterniond q;

  if (theta < 1e-10) {
    q.w() = 1.0;
    q.x() = q.y() = q.z() = 0;
  }

  double magnitude = sin(theta / 2.0f);

  q.w() = cos(theta / 2.0f);
  q.x() = axis(0) * magnitude;
  q.y() = axis(1) * magnitude;
  q.z() = axis(2) * magnitude;

  return q;
}

static Eigen::Quaterniond axis2Quat(const Eigen::Vector3d &vec) {
  Eigen::Quaterniond q;
  double theta = vec.norm();

  if (theta < 1e-10) {
    q.w() = 1.0;
    q.x() = q.y() = q.z() = 0;
    return q;
  }

  Eigen::Vector3d tmp = vec / theta;
  return axis2Quat(tmp, theta);
}

void processModel(const double& time,
    const Eigen::Vector3d& m_gyro,
    const Eigen::Vector3d& m_acc) {
	// Remove the bias from the measured gyro and acceleration
	Eigen::Vector3d gyro = m_gyro - gyro_bias;
	Eigen::Vector3d acc = m_acc - acc_bias;
	double dtime = time - imu_state_time;
	// Compute discrete transition and noise covariance matrix
	Eigen::Matrix<double, 21, 21> F = Eigen::Matrix<double, 21, 21>::Zero();
	Eigen::Matrix<double, 21, 12> G = Eigen::Matrix<double, 21, 12>::Zero();

	// Note that the EigenQuaternionParameterization uses GLOBAL angular errors
	F.block<3, 3>(0, 3) = -q_w_i_ekf.toRotationMatrix();
	F.block<3, 3>(6, 0) = -skewSymmetric(q_w_i_ekf.toRotationMatrix() * acc);
	F.block<3, 3>(6, 9) = -q_w_i_ekf.toRotationMatrix();
	F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

	G.block<3, 3>(0, 0) = -q_w_i_ekf.toRotationMatrix();
	G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
	G.block<3, 3>(6, 6) = -q_w_i_ekf.toRotationMatrix();
	G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

	// Approximate matrix exponential to the 3rd order,
	// which can be considered to be accurate enough assuming
	// dtime is within 0.01s.
	Eigen::Matrix<double, 21, 21> Fdt = F * dtime;
	Eigen::Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
	Eigen::Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
	Eigen::Matrix<double, 21, 21> Phi = Eigen::Matrix<double, 21, 21>::Identity() +
	Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

	if (!is_init_imu) {
		is_init_imu = true;
		acc_last = m_acc;
		gyr_last = m_gyro;
	}

	// Average acceleration and angular rate
	Eigen::Vector3d un_acc_0 = q_w_i_ekf.toRotationMatrix() * (acc_last - acc_bias) + gravity;
	Eigen::Vector3d un_gyr = 0.5 * (gyr_last + m_gyro) - gyro_bias;
	Eigen::Quaterniond dq = axis2Quat(un_gyr * dtime);
	q_w_i_ekf = (q_w_i_ekf * dq).normalized();
	Eigen::Vector3d un_acc_1 = q_w_i_ekf.toRotationMatrix() * (m_acc - acc_bias) + gravity;
	Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

	// State integral
	t_w_i_ekf = t_w_i_ekf + dtime * velocity + 0.5 * dtime * dtime * un_acc;
	velocity = velocity + dtime * un_acc;

	acc_last = m_acc;
	gyr_last = m_gyro;

	// Propogate the state covariance matrix.
	Eigen::Matrix<double, 12, 12> discrete_noise_cov = Eigen::Matrix<double, 12, 12>::Zero();
	discrete_noise_cov.block<3, 3>(0, 0) =
    	continuous_noise_cov.block<3, 3>(0, 0) * dtime;
	discrete_noise_cov.block<3, 3>(3, 3) =
		continuous_noise_cov.block<3, 3>(3, 3) * dtime;
	discrete_noise_cov.block<3, 3>(6, 6) =
		continuous_noise_cov.block<3, 3>(6, 6) * dtime;
	discrete_noise_cov.block<3, 3>(9, 9) =
		continuous_noise_cov.block<3, 3>(9, 9) * dtime;
	Eigen::Matrix<double, 21, 21> Q = G * discrete_noise_cov * G.transpose();
	state_cov = Phi * state_cov * Phi.transpose() + Q;

	Eigen::MatrixXd state_cov_fixed = (state_cov + state_cov.transpose()) / 2.0;
	state_cov = state_cov_fixed;

	// Update the state info
	imu_state_time = time;
	return;
}

void batchImuProcessing(const double& time_bound) {
	// Counter how many IMU msgs in the buffer are used.
	int used_imu_msg_cntr = 0;

	for (const auto& imu_msg : imu_msg_buffer) {
	double imu_time = imu_msg.header.stamp.toSec();
	if (imu_time < imu_state_time) {
		++used_imu_msg_cntr;
		continue;
	}
	if (imu_time > time_bound) break;

	// Convert the msgs.
	Eigen::Vector3d m_gyro, m_acc;
	m_acc << imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
				imu_msg.linear_acceleration.z;
	m_gyro << imu_msg.angular_velocity.x, imu_msg.angular_velocity.y,
					imu_msg.angular_velocity.z;

	// Execute process model.
	processModel(imu_time, m_gyro, m_acc);
	++used_imu_msg_cntr;
	}

	// Remove all used IMU msgs.
	imu_msg_buffer.erase(imu_msg_buffer.begin(),
		imu_msg_buffer.begin()+used_imu_msg_cntr);
	return;
}

// set initial guess
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr_i = q_i_l * point_curr + t_i_l; // transform from current lidar to current imu
	Eigen::Vector3d point_w_i = q_w_i_ekf * point_curr_i + t_w_i_ekf; // transform from current imu to imu world frame
	Eigen::Vector3d point_w = q_i_l.inverse() * (point_w_i - t_i_l); // transform from imu world frame to lidar world frame
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void undisortPoint(PointType const *const pi, PointType *const po)
{
	constexpr double SCAN_PERIOD = 0.1;
    //interpolation ratio
    double s;
    if (0)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
	Eigen::Quaterniond q_w_cur_ekf = q_i_l.inverse() * q_w_i_ekf * q_i_l;
	Eigen::Vector3d t_w_cur_ekf = q_i_l.inverse() * (q_w_i_ekf * t_i_l + t_w_i_ekf - t_i_l);
	Eigen::Quaterniond q_w_cur_ekf_last = q_i_l.inverse() * q_w_i_ekf_last * q_i_l;
	Eigen::Vector3d t_w_cur_ekf_last = q_i_l.inverse() * (q_w_i_ekf_last * t_i_l + t_w_i_ekf_last - t_i_l);
	// q_i_l * (q_w_cur_ekf * p_l + t_w_cur_ekf) + t_i_l = q_w_i_ekf * (q_i_l * p_l + t_i_l) + t_w_i_ekf;
    Eigen::Quaterniond q_point_last = q_w_cur_ekf_last.slerp(s, q_w_cur_ekf);
    Eigen::Vector3d t_point_last = s * t_w_cur_ekf + (1.0 - s) * t_w_cur_ekf_last;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d global_point = q_point_last * point + t_point_last; // transform point from current lidar to global lidar
	Eigen::Vector3d un_point = q_w_cur_ekf.inverse() * (global_point - t_w_cur_ekf); // transform point from global lidar to lidar frame end

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void process()
{
	while(1)
	{
		if (!is_gravity_set) {
			std::chrono::milliseconds dura(2);
        	std::this_thread::sleep_for(dura);
			continue;
		}
		while (!cornerLastBuf.empty() && !surfLastBuf.empty())
		{
			mBuf.lock();

			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();

			if (timeLaserCloudCornerLast != timeLaserCloudSurfLast)
			{
				printf("time corner %f surf %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();

			/*
			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();
				printf("drop lidar frame in mapping for real time performance \n");
			}
			*/
			mBuf.unlock();

			TicToc t_whole;

			if (is_first_img) {
				imu_state_time = timeLaserCloudSurfLast;
				q_w_i_ekf_last = q_w_i_ekf;
				t_w_i_ekf_last = t_w_i_ekf;
			}
			batchImuProcessing(timeLaserCloudSurfLast);
			// ALTERNATIVE: define explicitly nominal state and error state, so the optimization needs no extra parameterization, ALL linear.
			
			TicToc t_shift;
			Eigen::Quaterniond q_w_cur_ekf = q_i_l.inverse() * q_w_i_ekf * q_i_l;
			Eigen::Vector3d t_w_cur_ekf = q_i_l.inverse() * (q_w_i_ekf * t_i_l + t_w_i_ekf - t_i_l);
			int centerCubeI = int((t_w_cur_ekf.x() + 25.0) / 50.0) + laserCloudCenWidth;
			int centerCubeJ = int((t_w_cur_ekf.y() + 25.0) / 50.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_cur_ekf.z() + 25.0) / 50.0) + laserCloudCenDepth;

			if (t_w_cur_ekf.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_cur_ekf.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_cur_ekf.z() + 25.0 < 0)
				centerCubeK--;

			while (centerCubeI < 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{ 
						int i = laserCloudWidth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i >= 1; i--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI++;
				laserCloudCenWidth++;
			}

			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}

			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}

			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;

			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
						{ 
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			laserCloudCornerFromMap->clear();
			laserCloudSurfFromMap->clear();
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

			//printf("map prepare time %f ms\n", t_shift.toc());
			//printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
			// std::cout << "before IMU position: " << t_w_i_ekf.transpose() << std::endl;
			if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50 && false == is_first_img)
			{
				TicToc t_opt;
				TicToc t_tree;
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
				kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				//printf("build tree time %f ms \n", t_tree.toc());
				Eigen::Quaterniond rotation = q_w_i_ekf;
				Eigen::Vector3d bg = gyro_bias;
				Eigen::Vector3d v = velocity;
				Eigen::Vector3d ba = acc_bias;
				Eigen::Vector3d position = t_w_i_ekf;
				Eigen::Quaterniond ex_rotation = q_i_l;
				Eigen::Vector3d ex_position = t_i_l;
				// std::cout << "Before Extraction Cov:" << std::endl << state_cov << std::endl;
				
				for (int iterCount = 0; iterCount < iterNum; iterCount++)
				{
					pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
					int laserCloudCornerLastNum = laserCloudCornerLast->size();
					for (int li = 0; li < laserCloudCornerLastNum; ++li) {
						undisortPoint(&(laserCloudCornerLast->points[li]), &(laserCloudCornerLast->points[li])); // transform point to frame end, through mapping to global and back
					}
					downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
					downSizeFilterCorner.filter(*laserCloudCornerStack);
					int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

					pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
					int laserCloudSurfLastNum = laserCloudSurfLast->size();
					for (int li = 0; li < laserCloudSurfLastNum; ++li) {
						undisortPoint(&(laserCloudSurfLast->points[li]), &(laserCloudSurfLast->points[li])); // transform point to frame end, through mapping to global and back
					}
					downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
					downSizeFilterSurf.filter(*laserCloudSurfStack);
					int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

					//ceres::LossFunction *loss_function = NULL;
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1 * (1.0 / LIDAR_STD) * (1.0 / LIDAR_STD));
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					problem.AddParameterBlock(param_imu_state, 4, q_parameterization); // q
					problem.AddParameterBlock(param_imu_state + 4, 3); // bg
					problem.AddParameterBlock(param_imu_state + 7, 3); // v
					problem.AddParameterBlock(param_imu_state + 10, 3); // ba
					problem.AddParameterBlock(param_imu_state + 13, 3); // p
					problem.AddParameterBlock(param_ex, 4, q_parameterization);
					problem.AddParameterBlock(param_ex + 4, 3);
					// problem.SetParameterBlockConstant(param_ex);
					// problem.SetParameterBlockConstant(param_ex + 4);
					ceres::CostFunction *prior_cost_function = PriorFactor::Create(rotation, bg, v, ba, position, ex_rotation, ex_position, state_cov);
					problem.AddResidualBlock(prior_cost_function, nullptr, param_imu_state, param_imu_state + 4, param_imu_state + 7, param_imu_state + 10, param_imu_state + 13, param_ex, param_ex + 4);

					TicToc t_data;
					int corner_num = 0;

					for (int i = 0; i < laserCloudCornerStackNum; i++)
					{
						pointOri = laserCloudCornerStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

						if (pointSearchSqDis[4] < 1.0)
						{ 
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								nearCorners.push_back(tmp);
							}
							center = center / 5.0;

							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							for (int j = 0; j < 5; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
							}

							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

							// if is indeed line feature
							// note Eigen library sort eigenvalues in increasing order
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
							{ 
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;
								point_a = 0.1 * unit_direction + point_on_line;
								point_b = -0.1 * unit_direction + point_on_line;

								ceres::CostFunction *cost_function = LidarEdgeFactorEx::Create(curr_point, point_a, point_b, 1.0, LIDAR_STD);
								problem.AddResidualBlock(cost_function, loss_function, param_imu_state, param_imu_state + 13, param_ex, param_ex + 4);
								corner_num++;	
							}							
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					int surf_num = 0;
					for (int i = 0; i < laserCloudSurfStackNum; i++)
					{
						pointOri = laserCloudSurfStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

						Eigen::Matrix<double, 5, 3> matA0;
						Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
						if (pointSearchSqDis[4] < 1.0)
						{
							
							for (int j = 0; j < 5; j++)
							{
								matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								//printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
							}
							// find the norm of plane
							Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
							double negative_OA_dot_norm = 1 / norm.norm();
							norm.normalize();

							// Here n(pa, pb, pc) is unit norm of plane
							bool planeValid = true;
							for (int j = 0; j < 5; j++)
							{
								// if OX * n > 0.2, then plane is not fit well
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									planeValid = false;
									break;
								}
							}
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							if (planeValid)
							{
								ceres::CostFunction *cost_function = LidarPlaneNormFactorEx::Create(curr_point, norm, negative_OA_dot_norm, LIDAR_STD);
								problem.AddResidualBlock(cost_function, loss_function, param_imu_state, param_imu_state + 13, param_ex, param_ex + 4);
								surf_num++;
							}
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
													laserCloudSurfFromMap->points[pointSearchInd[j]].y,
													laserCloudSurfFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					//printf("mapping data assosiation time %f ms \n", t_data.toc());

					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					if (iterCount == iterNum - 1) {
						// extract covariance
						ceres::Covariance::Options coptions;
						coptions.algorithm_type = ceres::DENSE_SVD;
						ceres::Covariance covariance(coptions);
						std::vector<std::pair<const double*, const double*> > covariance_blocks;
						std::vector<const double*> v_param;
						v_param.push_back(param_imu_state);
						v_param.push_back(param_imu_state + 4);
						v_param.push_back(param_imu_state + 7);
						v_param.push_back(param_imu_state + 10);
						v_param.push_back(param_imu_state + 13);
						v_param.push_back(param_ex);
						v_param.push_back(param_ex + 4);
						for (int i = 0; i < v_param.size(); ++i) {
							for (int j = 0; j <= i; ++j) {
								covariance_blocks.push_back(std::make_pair(v_param[i], v_param[j]));
							}
						}
						covariance.Compute(covariance_blocks, &problem);
						double covariance_recovered[7][7][3 * 3];
						for (int i = 0; i < v_param.size(); ++i) {
							for (int j = 0; j < v_param.size(); ++j) {
								covariance.GetCovarianceBlockInTangentSpace(v_param[i], v_param[j], covariance_recovered[i][j]);
							}
						}
						// the error definition of eskf is different from ceresquaternionparameterization (by a factor of 2)
						for (int j = 0; j < v_param.size(); ++j) {
							for (int r = 0; r < 3; ++r) {
								for (int c = 0; c < 3; ++ c) {
									covariance_recovered[0][j][r * 3 + c] *= 2.0;
									covariance_recovered[j][0][r * 3 + c] *= 2.0;
									covariance_recovered[5][j][r * 3 + c] *= 2.0;
									covariance_recovered[j][5][r * 3 + c] *= 2.0;
								}
							}
						}
						Eigen::Matrix<double, 21, 21> Pk_recovered = Eigen::Matrix<double, 21, 21>::Zero();
						for (int i = 0; i < v_param.size(); ++i) {
							for (int j = 0; j < v_param.size(); ++j) {
								int start_i = i * 3;
								int start_j = j * 3;
								for (int k = 0; k < 3; ++k) {
									for (int l = 0; l < 3; ++l) {
										Pk_recovered(start_i + k, start_j + l) = covariance_recovered[i][j][k * 3 + l];
									}
								}
							}
						}
						state_cov = Pk_recovered;
					}
					//printf("mapping solver time %f ms \n", t_solver.toc());

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				//printf("mapping optimization time %f \n", t_opt.toc());
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");
			}
			// std::cout << "after IMU position: " << t_w_i_ekf.transpose() << std::endl;
			std::cout << "After optimization, R_ex: " << std::endl << q_i_l.toRotationMatrix() << std::endl << "T_ex: " << t_i_l.transpose() << std::endl;	

			TicToc t_add;
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			int laserCloudCornerLastNum = laserCloudCornerLast->size();
			for (int li = 0; li < laserCloudCornerLastNum; ++li) {
				undisortPoint(&(laserCloudCornerLast->points[li]), &(laserCloudCornerLast->points[li]));
			}
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			int laserCloudSurfLastNum = laserCloudSurfLast->size();
			for (int li = 0; li < laserCloudSurfLastNum; ++li) {
				undisortPoint(&(laserCloudSurfLast->points[li]), &(laserCloudSurfLast->points[li]));
			}
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
			for (int i = 0; i < laserCloudCornerStackNum; i++)
			{
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}

			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			//printf("add points time %f ms\n", t_add.toc());

			
			TicToc t_filter;
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];

				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			//printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			//publish surround map for every 5 frame
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserCloudSurfLast);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserCloudSurfLast);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}
			//printf("mapping pub time %f ms \n", t_pub.toc());

			printf("whole mapping time %f ms +++++\n", t_whole.toc());

			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserCloudSurfLast);
			odomAftMapped.pose.pose.orientation.x = q_w_i_ekf.x();
			odomAftMapped.pose.pose.orientation.y = q_w_i_ekf.y();
			odomAftMapped.pose.pose.orientation.z = q_w_i_ekf.z();
			odomAftMapped.pose.pose.orientation.w = q_w_i_ekf.w();
			odomAftMapped.pose.pose.position.x = t_w_i_ekf.x();
			odomAftMapped.pose.pose.position.y = t_w_i_ekf.y();
			odomAftMapped.pose.pose.position.z = t_w_i_ekf.z();
			pubOdomAftMapped.publish(odomAftMapped);

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_i_ekf(0),
											t_w_i_ekf(1),
											t_w_i_ekf(2)));
			q.setW(q_w_i_ekf.w());
			q.setX(q_w_i_ekf.x());
			q.setY(q_w_i_ekf.y());
			q.setZ(q_w_i_ekf.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

			frameCount++;
			if (is_first_img) {
				is_first_img = false;
			} else {
				q_w_i_ekf_last = q_w_i_ekf;
				t_w_i_ekf_last = t_w_i_ekf;
			}
			std::ofstream foutC("/home/zhikaizhu/output/aloam_imu.csv", std::ios::app);
            foutC.setf(std::ios::fixed, std::ios::floatfield);
            foutC.precision(10);
            foutC << timeLaserCloudSurfLast << " ";
            foutC.precision(5);
            foutC << t_w_i_ekf.x() << " "
                    << t_w_i_ekf.y() << " "
                    << t_w_i_ekf.z() << " "
                    << q_w_i_ekf.w() << " "
                    << q_w_i_ekf.x() << " "
                    << q_w_i_ekf.y() << " "
                    << q_w_i_ekf.z() << std::endl;
            foutC.close();
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{

	std::ofstream foutC("/home/zhikaizhu/output/aloam_imu.csv", std::ios::trunc);
	foutC.close();
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	float lineRes = 0;
	float planeRes = 0;
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	nh.param<double>("lidar_std", LIDAR_STD, 0.1);
	nh.param<double>("gyro_noise", gyro_noise, 0.005);
	nh.param<double>("gyro_bias_noise", gyro_bias_noise, 4e-6);
	nh.param<double>("acc_noise", acc_noise, 0.01);
	nh.param<double>("acc_bias_noise", acc_bias_noise, 0.0002);

	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudCornerLastHandler);

	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudSurfLastHandler);

	//ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	//ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

	ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("/lips_sim/data_imu", 100, imuCallback);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	for (int i = 0; i < laserCloudNum; i++)
	{
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	std::thread mapping_process{process};

	ros::spin();

	return 0;
}