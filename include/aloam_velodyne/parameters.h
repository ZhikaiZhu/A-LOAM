// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/pcl_macros.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

typedef pcl::PointXYZI PointType;

typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::VectorXd VXD;
typedef Eigen::MatrixXd MXD;
typedef Eigen::Quaterniond Q4D;

namespace parameter {

const double G0 = 9.81;

// !@ENABLE_CALIBRATION
extern int CALIBARTE_IMU;

// !@LIDAR_PARAMETERS
extern int N_SCANS;
extern int SCAN_NUM;
extern double SCAN_PERIOD;
extern double MINIMUM_RANGE;
extern double EDGE_THRESHOLD;
extern double SURF_THRESHOLD;
extern double DISTANCE_SQ_THRESHOLD;

// !@TESTING
extern int ICP_FREQ;
extern int MAX_LIDAR_NUMS;
extern int NUM_ITER;
extern double LIDAR_SCALE;
extern double LIDAR_STD;

// !@SUB_TOPIC_NAME
extern std::string IMU_TOPIC;
extern std::string LIDAR_TOPIC;

extern std::string OUTPUT_FOLDER;
extern int SAVE_PCD_MAP;
extern int PURE_IMU;
extern int CALIB_EXTRINSIC;
//extern int EKF_UPDATE;

// !@KALMAN_FILTER
extern double ACC_N;
extern double ACC_W;
extern double GYR_N;
extern double GYR_W;
extern V3D INIT_POS_STD;
extern V3D INIT_VEL_STD;
extern V3D INIT_ATT_STD;
extern V3D INIT_ACC_STD;
extern V3D INIT_GYR_STD;

// !@INITIAL IMU BIASES
extern V3D INIT_BA;
extern V3D INIT_BW;

// !@EXTRINSIC_PARAMETERS
extern V3D INIT_TBL;
extern Q4D INIT_RBL;

extern double VOXEL_SIZE;
extern int USE_CERES;
extern double LOSS_THRESHOLD;
extern int CERES_MAX_ITER;

void readParameters(ros::NodeHandle& n);

void readV3D(cv::FileStorage* file, const std::string& name, V3D& vec_eigen);

void readQ4D(cv::FileStorage* file, const std::string& name, Q4D& quat_eigen);

enum StateOrder {
  O_R = 0,
  O_P = 3,
};

}  // namespace parameter

#endif  // INCLUDE_PARAMETERS_H_
