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
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <iomanip>
#include <signal.h>

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
#include <eigen3/Eigen/StdVector>
#include <ceres/ceres.h>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/map_viewer.hpp"
#include "aloam_velodyne/parameters.h"
#include "aloam_velodyne/pose_local_parameterization.hpp"
#include "aloam_velodyne/KeyFrame.h"

using namespace parameter;
using namespace utils;

int frameCount = 0;
int frame_drop_cnt = 0;
double opt_time = 0.0;
double whole_mapping_time = 0.0;

double timeLaserCloudCornerLast = 0.0;
double timeLaserCloudSurfLast = 0.0;
double timeLaserCloudFullRes = 0.0;
double timeLaserOdometry = 0.0;
double timeLastProcessing = -1.0;
const double mappingProcessInterval = 0.3;

// Result save
std::string RESULT_PATH, PCD_TRAJ, PCD_SURF, PCD_CORNER;

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyframes(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapKeyframes(new pcl::KdTreeFLANN<PointType>());

bool save_new_keyframe;
pcl::PointCloud<PointType>::Ptr surroundingKeyframes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surroundingKeyframesDS(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr globalMapKeyframes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalMapKeyframesDS(new pcl::PointCloud<PointType>());

std::vector<int> surroundingExistingKeyframesID;
std::vector<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyframes;
std::vector<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyframes;
std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyframes;
std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyframes;

pcl::PointCloud<PointType>::Ptr pose_keyframes_3d(new pcl::PointCloud<PointType>());
std::vector<std::pair<double, Pose6D>> pose_keyframes_6d;
aloam_velodyne::KeyFrame laser_keyframes_6d;
PointType pose_point_curr, pose_point_prev;
Eigen::Quaterniond q_ori_curr, q_ori_prev;

double para_pose[7];

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

Eigen::Quaterniond q_wmap_curr(1, 0, 0, 0);
Eigen::Vector3d t_wmap_curr(0, 0, 0);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;
std::mutex mProcess;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyframes;
pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyframes;
pcl::VoxelGrid<PointType> downSizeFilterSurfMap;
pcl::VoxelGrid<PointType> downSizeFilterCornerMap;
pcl::VoxelGrid<PointType> downSizeFilterGlobalMap;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
ros::Publisher pubKeyframes, pubKeyframes6D;

nav_msgs::Path laserAfterMappedPath;

pcl::PCDWriter pcd_writer;

void double2vector()
{
	q_wmap_curr = Eigen::Quaterniond(para_pose[6], para_pose[3], para_pose[4], para_pose[5]);
	t_wmap_curr = Eigen::Vector3d(para_pose[0], para_pose[1], para_pose[2]);
}

void vector2double()
{
	para_pose[0] = t_wmap_curr.x();
	para_pose[1] = t_wmap_curr.y();
	para_pose[2] = t_wmap_curr.z();
	para_pose[3] = q_wmap_curr.x();
	para_pose[4] = q_wmap_curr.y();
	para_pose[5] = q_wmap_curr.z();
	para_pose[6] = q_wmap_curr.w();
}

// set initial guess
void transformAssociateToMap()
{
	q_wmap_curr = q_wmap_wodom * q_wodom_curr;
	t_wmap_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
	q_wmap_wodom = q_wmap_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_wmap_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_wmap_curr * point_curr + t_wmap_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void transformPointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, 
						 pcl::PointCloud<PointType>::Ptr &cloudOut, const Pose6D &pose) 
{
	PointType tmp_p;
	int cloudSize = cloudIn->points.size();
	cloudOut->resize(cloudSize);
	for (int i = 0; i < cloudSize; ++i) {
		Eigen::Vector3d p_in(cloudIn->points[i].x, cloudIn->points[i].y,cloudIn->points[i].z);
		Eigen::Vector3d p_out = pose.q_ * p_in + pose.t_;
		tmp_p.x = p_out.x();
		tmp_p.y = p_out.y();
		tmp_p.z = p_out.z();
		tmp_p.intensity = cloudIn->points[i].intensity;
		cloudOut->points[i] = tmp_p;
	}
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_wmap_curr.inverse() * (point_w - t_wmap_curr);
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

void extractSurroundingKeyFrames()
{
    if (pose_keyframes_3d->points.empty()) return;
	/*if ((!laserCloudSurfFromMapDS->points.size() == 0) && (!laserCloudCornerFromMapDS->points.size() == 0)) {
		printf("not need to construct the map \n");
		return;
	}*/

	pose_point_curr.x = t_wmap_curr.x();
	pose_point_curr.y = t_wmap_curr.y();
	pose_point_curr.z = t_wmap_curr.z();

	/*surroundingKeyframes->clear();
	kdtreeSurroundingKeyframes->setInputCloud(pose_keyframes_3d);
	kdtreeSurroundingKeyframes->radiusSearch(pose_point_curr, SURROUNDING_KF_RADIUS, pointSearchInd, pointSearchSqDis, 0);
	for (size_t i = 0; i < pointSearchInd.size(); ++i) {
		surroundingKeyframes->push_back(pose_keyframes_3d->points[pointSearchInd[i]]);
	}
	
	// delete indexes in existing keyframes that are not in current surrounding keyframe
	for (size_t i = 0; i < surroundingExistingKeyframesID.size(); ++i) {
		bool existing_flag = false;
		for (size_t j = 0; j < surroundingKeyframes->points.size(); ++j) {			
			if (surroundingExistingKeyframesID[i] == (int)surroundingKeyframes->points[j].intensity) {
				existing_flag = true;
				break;
			}
		}
		if (!existing_flag) {
			surroundingExistingKeyframesID.erase(surroundingExistingKeyframesID.begin() + i);
			surroundingSurfCloudKeyframes.erase(surroundingSurfCloudKeyframes.begin() + i);
			surroundingCornerCloudKeyframes.erase(surroundingCornerCloudKeyframes.begin() + i);
			--i;
		}
	}

	// add points in current surrounding keyframes that are not in existing keyframes
	for (size_t i = 0; i < surroundingKeyframes->points.size(); ++i) {
		bool existing_flag = false;
		for (size_t j = 0; j < surroundingExistingKeyframesID.size(); ++j) {
			if (surroundingExistingKeyframesID[j] == (int)surroundingKeyframes->points[i].intensity) {
				existing_flag = true;
				break;
			}
		}
		if (existing_flag) {
			continue;
		}
		else {
			int key_ind = (int)surroundingKeyframes->points[i].intensity;
			surroundingExistingKeyframesID.push_back(key_ind);
			const Pose6D &curr_pose = pose_keyframes_6d[key_ind].second;

			pcl::PointCloud<PointType>::Ptr surf_trans(new pcl::PointCloud<PointType>());
			transformPointCloud(surfCloudKeyframes[key_ind], surf_trans, curr_pose);
			surroundingSurfCloudKeyframes.push_back(surf_trans);

			pcl::PointCloud<PointType>::Ptr corner_trans(new pcl::PointCloud<PointType>());
			transformPointCloud(cornerCloudKeyframes[key_ind], corner_trans, curr_pose);
			surroundingCornerCloudKeyframes.push_back(corner_trans);
		}
	}

	pcl::PointCloud<PointType>::Ptr surroundingExistingKeyframes(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr surroundingExistingKeyframesDS(new pcl::PointCloud<PointType>());
	for (size_t i = 0; i < surroundingExistingKeyframesID.size(); ++i) {
		int key_ind = surroundingExistingKeyframesID[i];
		PointType point = pose_keyframes_3d->points[key_ind];
		point.intensity = i;		// reorder
		surroundingExistingKeyframes->push_back(point);
	}
	downSizeFilterSurroundingKeyframes.setInputCloud(surroundingExistingKeyframes);
	downSizeFilterSurroundingKeyframes.filter(*surroundingExistingKeyframesDS);

	for (size_t i = 0; i < surroundingExistingKeyframesDS->points.size(); ++i) {
		int j = (int)surroundingExistingKeyframesDS->points[i].intensity;
		*laserCloudSurfFromMap += *surroundingSurfCloudKeyframes[j];
		*laserCloudCornerFromMap += *surroundingCornerCloudKeyframes[j];
	}*/

	surroundingKeyframes->clear();
	surroundingKeyframesDS->clear();
	kdtreeSurroundingKeyframes->setInputCloud(pose_keyframes_3d);
	kdtreeSurroundingKeyframes->radiusSearch(pose_point_curr, SURROUNDING_KF_RADIUS, pointSearchInd, pointSearchSqDis, 0);
	for (size_t i = 0; i < pointSearchInd.size(); ++i) {
		surroundingKeyframes->push_back(pose_keyframes_3d->points[pointSearchInd[i]]);
	}
	downSizeFilterSurroundingKeyframes.setInputCloud(surroundingKeyframes);
	downSizeFilterSurroundingKeyframes.filter(*surroundingKeyframesDS);
	
	// delete indexes in existing keyframes that are not in current surrounding keyframe
	for (size_t i = 0; i < surroundingExistingKeyframesID.size(); ++i) {
		bool existing_flag = false;
		for (size_t j = 0; j < surroundingKeyframesDS->points.size(); ++j) {			
			if (surroundingExistingKeyframesID[i] == (int)surroundingKeyframesDS->points[j].intensity) {
				existing_flag = true;
				break;
			}
		}
		if (!existing_flag) {
			surroundingExistingKeyframesID.erase(surroundingExistingKeyframesID.begin() + i);
			surroundingSurfCloudKeyframes.erase(surroundingSurfCloudKeyframes.begin() + i);
			surroundingCornerCloudKeyframes.erase(surroundingCornerCloudKeyframes.begin() + i);
			--i;
		}
	}

	// add points in current surrounding keyframes that are not in existing keyframes
	for (size_t i = 0; i < surroundingKeyframesDS->points.size(); ++i) {
		bool existing_flag = false;
		for (size_t j = 0; j < surroundingExistingKeyframesID.size(); ++j) {
			if (surroundingExistingKeyframesID[j] == (int)surroundingKeyframesDS->points[i].intensity) {
				existing_flag = true;
				break;
			}
		}
		if (existing_flag) {
			continue;
		}
		else {
			int key_ind = (int)surroundingKeyframesDS->points[i].intensity;
			surroundingExistingKeyframesID.push_back(key_ind);
			const Pose6D &curr_pose = pose_keyframes_6d[key_ind].second;

			pcl::PointCloud<PointType>::Ptr surf_trans(new pcl::PointCloud<PointType>());
			transformPointCloud(surfCloudKeyframes[key_ind], surf_trans, curr_pose);
			surroundingSurfCloudKeyframes.push_back(surf_trans);

			pcl::PointCloud<PointType>::Ptr corner_trans(new pcl::PointCloud<PointType>());
			transformPointCloud(cornerCloudKeyframes[key_ind], corner_trans, curr_pose);
			surroundingCornerCloudKeyframes.push_back(corner_trans);
		}
	}

	for (size_t i = 0; i < surroundingExistingKeyframesID.size(); ++i) {
		*laserCloudSurfFromMap += *surroundingSurfCloudKeyframes[i];
		*laserCloudCornerFromMap += *surroundingCornerCloudKeyframes[i];
	} 

	downSizeFilterSurfMap.setInputCloud(laserCloudSurfFromMap);
	downSizeFilterSurfMap.filter(*laserCloudSurfFromMapDS);
	downSizeFilterCornerMap.setInputCloud(laserCloudCornerFromMap);
	downSizeFilterCornerMap.filter(*laserCloudCornerFromMapDS);
}

void downsampleCurrentScan()
{
    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);

    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
}

void scan2MapOptimization()
{
	size_t laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
	size_t laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
	//printf("map surf num: %lu, corner num: %lu \n", laserCloudSurfFromMapDSNum, laserCloudCornerFromMapDSNum);
	if (laserCloudSurfFromMapDSNum > 50 && laserCloudCornerFromMapDSNum > 10) {
		PointType pointOri, pointSel;
		
		kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
		kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

		for (int iterCount = 0; iterCount < 2; iterCount++)
		{
			//ceres::LossFunction *loss_function = NULL;
			ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
			vector2double();

			ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
			ceres::Problem::Options problem_options;

			ceres::Problem problem(problem_options);
			problem.AddParameterBlock(para_pose, 7, local_parameterization);

			TicToc t_data;
			int corner_num = 0;

			for (size_t i = 0; i < laserCloudCornerLastDS->points.size(); i++)
			{
				pointOri = laserCloudCornerLastDS->points[i];
				pointAssociateToMap(&pointOri, &pointSel);
				kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

				if (pointSearchSqDis[4] < 1.0)
				{ 
					std::vector<Eigen::Vector3d> nearCorners;
					Eigen::Vector3d center(0, 0, 0);
					for (int j = 0; j < 5; j++)
					{
						Eigen::Vector3d tmp(laserCloudCornerFromMapDS->points[pointSearchInd[j]].x,
											laserCloudCornerFromMapDS->points[pointSearchInd[j]].y,
											laserCloudCornerFromMapDS->points[pointSearchInd[j]].z);
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

						//ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
						//problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						LidarMapEdgeFactor *cost_function = new LidarMapEdgeFactor(curr_point, point_a, point_b);
						problem.AddResidualBlock(cost_function, loss_function, para_pose);
						corner_num++;	
					}							
				}
			}

			int surf_num = 0;
			for (size_t i = 0; i < laserCloudSurfLastDS->points.size(); i++)
			{
				pointOri = laserCloudSurfLastDS->points[i];
				pointAssociateToMap(&pointOri, &pointSel);
				kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

				Eigen::Matrix<double, 5, 3> matA0;
				Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
				if (pointSearchSqDis[4] < 1.0)
				{
					
					for (int j = 0; j < 5; j++)
					{
						matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
						matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
						matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
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
						if (fabs(norm(0) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
								 norm(1) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
								 norm(2) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
						{
							planeValid = false;
							break;
						}
					}
					Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
					if (planeValid)
					{
						//ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
						//problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						LidarMapPlaneNormFactor *cost_function = new LidarMapPlaneNormFactor(curr_point, norm, negative_OA_dot_norm);
						problem.AddResidualBlock(cost_function, loss_function, para_pose);
						surf_num++;
					}
				}
			}
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
			//printf("mapping solver time %f ms \n", t_solver.toc());
			double2vector();

			//printf("time %f \n", timeLaserOdometry);
			//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
			//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
			//	   parameters[4], parameters[5], parameters[6]);
		}
	}
	else {
		printf("Map corner and surf num are not enough! \n");
	}
}

void saveKeyframe()
{
    pose_point_curr.x = t_wmap_curr.x();
    pose_point_curr.y = t_wmap_curr.y();
    pose_point_curr.z = t_wmap_curr.z();
    q_ori_curr = q_wmap_curr;

    save_new_keyframe = false;
    if (pose_keyframes_6d.size() == 0 || std::sqrt((pose_point_curr.x - pose_point_prev.x) * (pose_point_curr.x - pose_point_prev.x) + 
				  (pose_point_curr.y - pose_point_prev.y) * (pose_point_curr.y - pose_point_prev.y) + 
           		  (pose_point_curr.z - pose_point_prev.z) * (pose_point_curr.z - pose_point_prev.z)) > DISTANCE_KEYFRAMES ||
        q_ori_curr.angularDistance(q_ori_prev) / M_PI * 180 > ORIENTATION_KEYFRAMES)
    {
        save_new_keyframe = true;
    }

    if (!save_new_keyframe) return;
    pose_point_prev = pose_point_curr;
    q_ori_prev = q_ori_curr;

    PointType pose_3d;
    pose_3d.x = t_wmap_curr.x();
    pose_3d.y = t_wmap_curr.x();
    pose_3d.z = t_wmap_curr.x();
    pose_3d.intensity = pose_keyframes_3d->points.size();

    pose_keyframes_3d->push_back(pose_3d);
	Pose6D pose_wmap_curr(q_wmap_curr, t_wmap_curr);
    pose_keyframes_6d.push_back(std::make_pair(timeLaserOdometry, pose_wmap_curr));

    pcl::PointCloud<PointType>::Ptr this_surf_ds(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr this_corner_ds(new pcl::PointCloud<PointType>());

    pcl::copyPointCloud(*laserCloudSurfLastDS, *this_surf_ds);
    pcl::copyPointCloud(*laserCloudCornerLastDS, *this_corner_ds);

    surfCloudKeyframes.push_back(this_surf_ds);
    cornerCloudKeyframes.push_back(this_corner_ds);
	size_t keyframeNum = pose_keyframes_3d->points.size();
	printf("the size of keyframe: %lu \n", keyframeNum);
}

void pubPointCloud()
{
	int laserCloudFullResNum = laserCloudFullRes->points.size();
	for (int i = 0; i < laserCloudFullResNum; i++)
	{
		pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
	}

	sensor_msgs::PointCloud2 laserCloudFullRes3;
	pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
	laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
	laserCloudFullRes3.header.frame_id = "/camera_init";
	pubLaserCloudFullRes.publish(laserCloudFullRes3);
}

void pubOdometry()
{
    // publish odom
    nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
	odomAftMapped.pose.pose.orientation.x = q_wmap_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_wmap_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_wmap_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_wmap_curr.w();
	odomAftMapped.pose.pose.position.x = t_wmap_curr.x();
	odomAftMapped.pose.pose.position.y = t_wmap_curr.y();
	odomAftMapped.pose.pose.position.z = t_wmap_curr.z();
	pubOdomAftMapped.publish(odomAftMapped);

	std::ofstream loop_path_file(RESULT_PATH, std::ios::app);
	loop_path_file.setf(std::ios::fixed, std::ios::floatfield);
	loop_path_file.precision(10);
	loop_path_file << odomAftMapped.header.stamp.toSec() << " ";
	loop_path_file.precision(5);
	loop_path_file << odomAftMapped.pose.pose.position.x << " "
				   << odomAftMapped.pose.pose.position.y << " "
				   << odomAftMapped.pose.pose.position.z << " "
				   << odomAftMapped.pose.pose.orientation.w << " "
				   << odomAftMapped.pose.pose.orientation.x << " "
				   << odomAftMapped.pose.pose.orientation.y << " "
				   << odomAftMapped.pose.pose.orientation.z << std::endl;
	loop_path_file.close();

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
	transform.setOrigin(tf::Vector3(t_wmap_curr(0),
									t_wmap_curr(1),
									t_wmap_curr(2)));
	q.setW(q_wmap_curr.w());
	q.setX(q_wmap_curr.x());
	q.setY(q_wmap_curr.y());
	q.setZ(q_wmap_curr.z());
	transform.setRotation(q);
	br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

    // publish 3d keyframes
    if (pubKeyframes.getNumSubscribers() != 0 && save_new_keyframe)
    {
        sensor_msgs::PointCloud2 keyframes_msg;
        pcl::toROSMsg(*pose_keyframes_3d, keyframes_msg);
        keyframes_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        keyframes_msg.header.frame_id = "/camera_init";
        pubKeyframes.publish(keyframes_msg);
    }

    // publish 6d keyframes
    // if (pub_keyframes_6d.getNumSubscribers() != 0 && save_new_keyframe)
    if (save_new_keyframe)
    {
        const std::pair<double, Pose6D> &pkf = pose_keyframes_6d.back();
        geometry_msgs::PoseWithCovarianceStamped laser_keyframes_pose;
        laser_keyframes_pose.header.stamp = ros::Time().fromSec(pkf.first);
        laser_keyframes_pose.header.frame_id = "/camera_init";
        laser_keyframes_pose.pose.pose.position.x = pkf.second.t_.x();
        laser_keyframes_pose.pose.pose.position.y = pkf.second.t_.y();
        laser_keyframes_pose.pose.pose.position.z = pkf.second.t_.z();
        laser_keyframes_pose.pose.pose.orientation.x = pkf.second.q_.x();
        laser_keyframes_pose.pose.pose.orientation.y = pkf.second.q_.y();
        laser_keyframes_pose.pose.pose.orientation.z = pkf.second.q_.z();
        laser_keyframes_pose.pose.pose.orientation.w = pkf.second.q_.w();
        laser_keyframes_6d.poses.push_back(laser_keyframes_pose);
        laser_keyframes_6d.header = laser_keyframes_pose.header;
        pubKeyframes6D.publish(laser_keyframes_6d);
    }
}

void clearCloud()
{
	laserCloudSurfFromMap->clear();
	laserCloudCornerFromMap->clear();
	laserCloudSurfFromMapDS->clear();
	laserCloudCornerFromMapDS->clear();
}

void pubGlobalMap()
{
    ros::Rate rate(0.5);
    while (ros::ok())
    {
        rate.sleep();
        if (pubLaserCloudSurround.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laser_cloud_surround_msg;
            pcl::toROSMsg(*laserCloudSurfFromMapDS + *laserCloudCornerFromMapDS, laser_cloud_surround_msg);
            // pcl::toROSMsg(*laser_cloud_surf_from_map_cov_ds, laser_cloud_surround_msg);
            laser_cloud_surround_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laser_cloud_surround_msg.header.frame_id = "/camera_init";
            pubLaserCloudSurround.publish(laser_cloud_surround_msg);
        }

        if ((pubLaserCloudMap.getNumSubscribers() != 0) && (!pose_keyframes_3d->points.empty()))
        {
            globalMapKeyframes->clear();
            globalMapKeyframesDS->clear();

            std::vector<int> point_search_ind;
            std::vector<float> point_search_sq_dis;

            kdtreeGlobalMapKeyframes->setInputCloud(pose_keyframes_3d);            
            kdtreeGlobalMapKeyframes->radiusSearch(pose_point_curr, GLOBALMAP_KF_RADIUS, point_search_ind, point_search_sq_dis, 0);

            for (size_t i = 0; i < point_search_ind.size(); i++)
                globalMapKeyframes->points.push_back(pose_keyframes_3d->points[point_search_ind[i]]);
            downSizeFilterGlobalMapKeyframes.setInputCloud(globalMapKeyframes);
            downSizeFilterGlobalMapKeyframes.filter(*globalMapKeyframesDS);

            pcl::PointCloud<PointType>::Ptr laserCloudMap(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr laserCloudMapDS(new pcl::PointCloud<PointType>());
            for (size_t i = 0; i < globalMapKeyframesDS->points.size(); i++)
            {
                int key_ind = (int)globalMapKeyframesDS->points[i].intensity;
                pcl::PointCloud<PointType>::Ptr surf_trans(new pcl::PointCloud<PointType>());
                transformPointCloud(surfCloudKeyframes[key_ind], surf_trans, pose_keyframes_6d[key_ind].second);
                *laserCloudMap += *surf_trans;

                pcl::PointCloud<PointType>::Ptr corner_trans(new pcl::PointCloud<PointType>());
                transformPointCloud(cornerCloudKeyframes[key_ind], corner_trans, pose_keyframes_6d[key_ind].second);
                *laserCloudMap += *corner_trans;

            }

            downSizeFilterGlobalMap.setLeafSize(MAP_SURF_RES, MAP_SURF_RES, MAP_SURF_RES);
            downSizeFilterGlobalMap.setInputCloud(laserCloudMap);
            downSizeFilterGlobalMap.filter(*laserCloudMapDS);

            sensor_msgs::PointCloud2 laser_cloud_msg;
            pcl::toROSMsg(*laserCloudMapDS, laser_cloud_msg);
            laser_cloud_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laser_cloud_msg.header.frame_id = "/camera_init";
            pubLaserCloudMap.publish(laser_cloud_msg);
        }
    }
}

void saveGlobalMap()
{
    printf("\e[1;33m""Saving keyframe poses && map cloud (surf + corner) \n""\e[0m");
    pcd_writer.write(PCD_TRAJ, *pose_keyframes_3d);
    
    pcl::PointCloud<PointType>::Ptr laserCloudMap(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudSurfMap(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudSurfMapDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudCornerMap(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudCornerMapDS(new pcl::PointCloud<PointType>());

	globalMapKeyframes->clear();
    globalMapKeyframesDS->clear();

    printf("global keyframes num: %lu\n", pose_keyframes_3d->size());
    for (size_t i = 0; i < pose_keyframes_3d->points.size(); i++)
        globalMapKeyframes->points.push_back(pose_keyframes_3d->points[i]);
    downSizeFilterGlobalMapKeyframes.setInputCloud(globalMapKeyframes);
    downSizeFilterGlobalMapKeyframes.filter(*globalMapKeyframesDS);
    for (size_t i = 0; i < globalMapKeyframesDS->points.size(); i++)
    {
        int key_ind = (int)globalMapKeyframesDS->points[i].intensity;
        pcl::PointCloud<PointType>::Ptr surf_trans(new pcl::PointCloud<PointType>());
		transformPointCloud(surfCloudKeyframes[key_ind], surf_trans, pose_keyframes_6d[key_ind].second);
		*laserCloudSurfMap += *surf_trans;

		pcl::PointCloud<PointType>::Ptr corner_trans(new pcl::PointCloud<PointType>());
		transformPointCloud(cornerCloudKeyframes[key_ind], corner_trans, pose_keyframes_6d[key_ind].second);
		*laserCloudCornerMap += *corner_trans;
    }

    downSizeFilterGlobalMap.setLeafSize(MAP_SURF_RES * 2, MAP_SURF_RES * 2, MAP_SURF_RES * 2);
    downSizeFilterGlobalMap.setInputCloud(laserCloudSurfMap);
    downSizeFilterGlobalMap.filter(*laserCloudSurfMapDS);
    downSizeFilterGlobalMap.setInputCloud(laserCloudCornerMap);
    downSizeFilterGlobalMap.filter(*laserCloudCornerMapDS);

    
	pcd_writer.write(PCD_SURF, *laserCloudSurfMapDS);
	pcd_writer.write(PCD_CORNER, *laserCloudCornerMapDS);
}

void sigintHandler(int sig)
{
    printf("[lidarMapping] press ctrl-c\n");
   	printf("\e[1;33m""mapping drop frame: %u \n""\e[0m", frame_drop_cnt);
    saveGlobalMap();
    ros::shutdown();
}

void process()
{
	while(1)
	{
		if (!ros::ok()) break;
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
			!fullResBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

			if (std::abs(timeLaserCloudCornerLast - timeLaserOdometry) > 0.005 ||
				std::abs(timeLaserCloudSurfLast - timeLaserOdometry) > 0.005 ||
				std::abs(timeLaserCloudFullRes - timeLaserOdometry) > 0.005)
			{
				//printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
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

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();
				frame_drop_cnt++;
				printf("\e[1;32m""drop lidar frame in mapping for real time performance \n""\e[0m");
			}

			mBuf.unlock();

			std::lock_guard<std::mutex> lock(mProcess);

			TicToc t_whole;

			if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {
				timeLastProcessing = timeLaserOdometry;

				transformAssociateToMap();

				extractSurroundingKeyFrames();

				downsampleCurrentScan();

				TicToc t_opt;
				scan2MapOptimization();
				opt_time += t_opt.toc();
				//printf("mapping optimization time %f ms \n", opt_time / (frameCount + 1));

				transformUpdate();

				saveKeyframe();

				pubPointCloud();

				pubOdometry();

				/*if (save_new_keyframe) {
					clearCloud();
				}*/
				clearCloud();
			}

			whole_mapping_time += t_whole.toc();
			printf("whole mapping time %f ms +++++\n", whole_mapping_time / (frameCount + 1));
			frameCount++;
			
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "lidarMapping");
	ros::NodeHandle nh("~");

	parameter::readParameters(nh);

	printf("line resolution %f plane resolution %f \n", MAP_CORNER_RES, MAP_SURF_RES);
	downSizeFilterCorner.setLeafSize(MAP_CORNER_RES, MAP_CORNER_RES, MAP_CORNER_RES);
	downSizeFilterSurf.setLeafSize(MAP_SURF_RES, MAP_SURF_RES, MAP_SURF_RES);
	downSizeFilterCornerMap.setLeafSize(MAP_CORNER_RES, MAP_CORNER_RES, MAP_CORNER_RES);
	downSizeFilterSurfMap.setLeafSize(MAP_SURF_RES, MAP_SURF_RES, MAP_SURF_RES);
	downSizeFilterSurroundingKeyframes.setLeafSize(MAP_SURR_KF_RES, MAP_SURR_KF_RES, MAP_SURR_KF_RES);
	downSizeFilterGlobalMapKeyframes.setLeafSize(10.0, 10.0, 10.0);

	RESULT_PATH = OUTPUT_FOLDER + "/lio_mapped.csv";
	std::ofstream fout(RESULT_PATH, std::ios::out);
	fout.close();
	PCD_TRAJ = OUTPUT_FOLDER + "/mapping_keyframe.pcd";
	PCD_SURF = OUTPUT_FOLDER + "/mapping_surf_cloud.pcd";
	PCD_CORNER = OUTPUT_FOLDER + "/mapping_corner_cloud.pcd";

	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	pubKeyframes = nh.advertise<sensor_msgs::PointCloud2>("/laser_map_keyframes", 100);

	pubKeyframes6D = nh.advertise<aloam_velodyne::KeyFrame>("/laser_map_keyframes_6d", 100);

	pose_point_prev.x = 0.0;
	pose_point_prev.y = 0.0;
	pose_point_prev.z = 0.0;
	q_ori_prev.setIdentity();

	pose_point_curr.x = 0.0;
	pose_point_curr.y = 0.0;
	pose_point_curr.z = 0.0;
	q_ori_curr.setIdentity();

	pose_keyframes_3d->clear();
	pose_keyframes_6d.clear();
	laser_keyframes_6d.poses.clear();

	if (SAVE_PCD_MAP) {
		signal(SIGINT, sigintHandler);
	}

	std::thread mapping_process{process};
	std::thread pub_map_process{pubGlobalMap};

	ros::Rate loop_rate(100);
	while (ros::ok()) {
		ros::spinOnce();
		loop_rate.sleep();
	}

	pub_map_process.detach();
	mapping_process.join();

	return 0;
}