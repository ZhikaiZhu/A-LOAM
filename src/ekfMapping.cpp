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
#include <deque>
#include <thread>
#include <iostream>
#include <string>
#include <omp.h>

#include "lidarFactor.hpp"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/filter_state.hpp"
#include "aloam_velodyne/imu_process.hpp"
#include "aloam_velodyne/pose_local_parameterization.hpp"
#include "ikd-Tree/ikd_Tree.h"

using namespace filter;

int frameCount = 0;
double whole_map_time = 0;

//double timeLaserCloudFullRes = 0;

// input: from odom
pcl::PointCloud<PointType>::Ptr featsUndistort(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr featsDownBody(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr featsDownWorld(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr featsFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
//pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr featsArray(new pcl::PointCloud<PointType>());


std::deque<double> timeBuf;
std::deque<pcl::PointCloud<PointType>::Ptr> lidarBuf;
//std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::deque<sensor_msgs::ImuConstPtr> imuBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

std::string result_path, imu_topic;


bool is_first_scan = true, undistortion = false, lidar_pushed = false, EKF_init_flag = false;
int init_imu_num = 1, feats_num = 0, scan_num = 0;
double first_lidar_time = 0, last_imu_time = -1.0, last_lidar_time = 0, lidar_end_time = 0, lidar_mean_time = 0;
double loss_threshold = 0.1;
V3D lid_pos;
std::shared_ptr<ImuProcess> pImu_(new ImuProcess());
MeasureGroup Measures;
StatePredictor *filterEstimator_;
GlobalState linState_;
MXD Pk_;

double para_error_state[18];
double para_ex[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_i_l(para_ex); // from lidar to inertial
Eigen::Map<Eigen::Vector3d> t_i_l(para_ex + 4);

Eigen::Quaterniond q_w_i_last(1, 0, 0, 0);
Eigen::Vector3d t_w_i_last(0, 0, 0);

const int iterNum = 2;
double LIDAR_STD;
double gyr_noise = 0.005, gyr_bias_noise = 4e-6, acc_noise = 0.01, acc_bias_noise = 0.0002;


// ikdtree map 
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double cube_len = 0, plane_res = 0;;
int  kdtree_surf_delete_counter = 0, feats_down_size = 0;
std::vector<BoxPointType> cub_needrm;
KD_TREE ikdtreeSurf;
std::vector<PointVector> nearestPointsSurf;

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr_i = linState_.q_ex_ * point_curr + linState_.t_ex_; // transform from current lidar to current imu
	Eigen::Vector3d point_w_i = linState_.qbn_ * point_curr_i + linState_.rn_; // transform from current imu to imu world frame
	//Eigen::Vector3d point_w = q_i_l.inverse() * (point_w_i - t_i_l); // transform from imu world frame to lidar world frame
	po->x = point_w_i.x();
	po->y = point_w_i.y();
	po->z = point_w_i.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void undisortPoint(PointType const *const pi, PointType *const po)
{
	constexpr double SCAN_PERIOD = 0.1;
    //interpolation ratio
    double s;
    if (undistortion) {
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
	}
    else {
        s = 1.0;
	}

	Eigen::Quaterniond q_w_l = linState_.q_ex_.inverse() * linState_.qbn_ * linState_.q_ex_;
	Eigen::Vector3d t_w_l = linState_.q_ex_.inverse() * (linState_.qbn_ * linState_.t_ex_ + linState_.rn_ - linState_.t_ex_);
	Eigen::Quaterniond q_w_l_last = linState_.q_ex_.inverse() * q_w_i_last * linState_.q_ex_;
	Eigen::Vector3d t_w_l_last = linState_.q_ex_.inverse() * (q_w_i_last * linState_.t_ex_ + t_w_i_last - linState_.t_ex_);
	// q_i_l * (q_w_l * p_l + t_w_l) + t_i_l = q_w_i * (q_i_l * p_l + t_i_l) + t_w_i;
    Eigen::Quaterniond q_point_last = q_w_l_last.slerp(s, q_w_l);
    Eigen::Vector3d t_point_last = s * t_w_l + (1.0 - s) * t_w_l_last;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d global_point = q_point_last * point + t_point_last; // transform point from current lidar to global lidar
	Eigen::Vector3d un_point = q_w_l.inverse() * (global_point - t_w_l); // transform point from global lidar to lidar frame end

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void laserCloudFeatsHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
	mBuf.lock();
	if (laserCloudMsg->header.stamp.toSec() < last_lidar_time) {
		ROS_ERROR("lidar loop back, clear buffer");
		lidarBuf.clear();
	}
	pcl::PointCloud<PointType>::Ptr cloudTemp(new pcl::PointCloud<PointType>());
	pcl::fromROSMsg(*laserCloudMsg, *cloudTemp);
	lidarBuf.push_back(cloudTemp);
	timeBuf.push_back(laserCloudMsg->header.stamp.toSec());
	last_lidar_time = laserCloudMsg->header.stamp.toSec();
	mBuf.unlock();
}

/*void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}*/

void imuCallback(const sensor_msgs::ImuConstPtr& msg) 
{
	mBuf.lock();
	if (msg->header.stamp.toSec() < last_imu_time) {
		ROS_ERROR("imu loop back, clear buffer");
		imuBuf.clear();
	}
	imuBuf.push_back(msg);
	last_imu_time = msg->header.stamp.toSec();
	mBuf.unlock();
}

bool sync_messages(MeasureGroup &meas) 
{
	if (lidarBuf.empty() || imuBuf.empty()) {
		return false;
	}

	// Push a lidar scan
	if (!lidar_pushed) {
		meas.lidar = lidarBuf.front();
		meas.lidar_beg_time = timeBuf.front();
		if (meas.lidar->points.size() <= 1) {
			lidar_end_time = meas.lidar_beg_time + lidar_mean_time;
			ROS_WARN("too few input cloud! \n");
		}
		else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_time) {
			lidar_end_time = meas.lidar_beg_time + lidar_mean_time;
		}
		else {
			scan_num++;
			lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
			lidar_mean_time += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_time) / scan_num;
		}
		//lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
		meas.lidar_end_time = lidar_end_time;

		lidar_pushed = true;
	}

	if (last_imu_time < lidar_end_time) {
		return false;
	}

	// Push imu data to MeasureGroup, and pop from imuBuf
	mBuf.lock();
	meas.imu.clear();
	double imu_time = imuBuf.front()->header.stamp.toSec();
	while (!imuBuf.empty() && imu_time < lidar_end_time) {
		imu_time = imuBuf.front()->header.stamp.toSec();
		if (imu_time > lidar_end_time) break;
		meas.imu.push_back(imuBuf.front());
		imuBuf.pop_front();
	}
	mBuf.unlock();

	lidarBuf.pop_front();
	timeBuf.pop_front();
	lidar_pushed = false;

	return true;
}

void points_cache_collect()
{
	PointVector points_history;
	ikdtreeSurf.acquire_removed_points(points_history);
	for (size_t i = 0; i < points_history.size(); i++) {
		featsArray->push_back(points_history[i]);
	}
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
	kdtree_surf_delete_counter = 0;
    V3D pos_LiD = lid_pos;
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    //points_cache_collect();
	if (cub_needrm.size() > 0) {
		kdtree_surf_delete_counter = ikdtreeSurf.Delete_Point_Boxes(cub_needrm);
	}
}

void map_incremental() 
{
	PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointAssociateToMap(&(featsDownBody->points[i]), &(featsDownWorld->points[i]));
        /* decide if need add to map */
        if (!nearestPointsSurf[i].empty() && EKF_init_flag)
        {
            const PointVector &points_near = nearestPointsSurf[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(featsDownWorld->points[i].x / plane_res) * plane_res + 0.5 * plane_res;
            mid_point.y = floor(featsDownWorld->points[i].y / plane_res) * plane_res + 0.5 * plane_res;
            mid_point.z = floor(featsDownWorld->points[i].z / plane_res) * plane_res + 0.5 * plane_res;
            float dist  = calc_dist(featsDownWorld->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * plane_res && fabs(points_near[0].y - mid_point.y) > 0.5 * plane_res && fabs(points_near[0].z - mid_point.z) > 0.5 * plane_res) {
                PointNoNeedDownsample.push_back(featsDownWorld->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < 5; readd_i++)
            {
                if (points_near.size() < 5) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(featsDownWorld->points[i]);
        }
        else
        {
            PointToAdd.push_back(featsDownWorld->points[i]);
        }
    }

    ikdtreeSurf.Add_Points(PointToAdd, true);
    ikdtreeSurf.Add_Points(PointNoNeedDownsample, false); 
}

void performIESKF() 
{
	bool hasConverged = false;
    bool hasDiverged = false;

	Pk_ = filterEstimator_->covariance_;
	GlobalState nominalState = filterEstimator_->state_;
	for (size_t i = 0; i < 18; ++i) {
        para_error_state[i] = 0.0;
    }
	Eigen::Map<Eigen::Matrix<double, 18, 1>> errorState(para_error_state);
	ceres::Solver::Options options;
	//options.max_solver_time_in_seconds = 0.04;
	options.max_num_iterations = 3;
	options.linear_solver_type = ceres::DENSE_QR;
	options.num_threads = MP_PROC_NUM;
	options.minimizer_progress_to_stdout = false;
	options.check_gradients = false;
	options.gradient_check_relative_precision = 1e-4;
	Eigen::Matrix<double, 18, 1> last_errorState = errorState;
	Q4D ex_rotation = nominalState.q_ex_;
	V3D ex_trans = nominalState.t_ex_; 
	for (int iterCount = 0; iterCount < iterNum && !hasConverged && !hasDiverged; iterCount++) {
		ceres::LossFunction *loss_function = new ceres::HuberLoss(loss_threshold * (1.0 / LIDAR_STD) * (1.0 / LIDAR_STD));
		//ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        ceres::Problem problem;
		problem.AddParameterBlock(para_error_state, 18);
		//problem.AddParameterBlock(para_ex, 7, local_parameterization);
		PriorFactorSF *prior_factor = new PriorFactorSF(Pk_);
		problem.AddResidualBlock(prior_factor, nullptr, para_error_state); 

		// Find correlation between edge, surf features and map
		TicToc t_data;

		for (int i = 0; i < feats_down_size; i++)
		{
			PointType pointOri = featsDownBody->points[i];
			PointType pointSel;
			std::vector<float> pointSearchSqDis(5);
			auto &pointsNear = nearestPointsSurf[i];
			//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
			pointAssociateToMap(&pointOri, &pointSel);
			ikdtreeSurf.Nearest_Search(pointSel, 5, pointsNear, pointSearchSqDis);

			if (pointsNear.size() < 5) {
				continue;
			} 

			Eigen::Matrix<double, 5, 3> matA0;
			Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
			if (pointSearchSqDis[4] < 1.0)
			{
				
				for (int j = 0; j < 5; j++)
				{
					matA0(j, 0) = pointsNear[j].x;
					matA0(j, 1) = pointsNear[j].y;
					matA0(j, 2) = pointsNear[j].z;
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
					if (fabs(norm(0) * pointsNear[j].x +
							 norm(1) * pointsNear[j].y +
							 norm(2) * pointsNear[j].z + negative_OA_dot_norm) > 0.1)
					{
						planeValid = false;
						break;
					}
				}
				Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
				if (planeValid)
				{
					double pd = norm.x() * pointSel.x + norm.y() * pointSel.y + norm.z() * pointSel.z + negative_OA_dot_norm;
					double weight = 1 - 0.9 * fabs(pd) / 
										sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
					if (weight > 0.7) {
						LidarMapPlaneNormFactorSF *cost_function = new LidarMapPlaneNormFactorSF(curr_point, norm, negative_OA_dot_norm,
																		ex_trans, ex_rotation, nominalState.rn_, nominalState.qbn_, LIDAR_STD);
						problem.AddResidualBlock(cost_function, loss_function, para_error_state);
					}
				}
			}
		}

		//printf("mapping data assosiation time %f ms \n", t_data.toc());
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		double initial_cost = summary.initial_cost;
		double final_cost = summary.final_cost;
		// sanity check
		bool hasNaN = false;
		for (int i = 0; i < errorState.size(); i++) {
			if (isnan(errorState[i])) {
				errorState[i] = 0;
				hasNaN = true;
			}
		}
		if (hasNaN == true) {
			ROS_WARN("System diverges Because of NaN...");
			hasDiverged = true;
			break;
		}

		// convergence check
		if (final_cost > initial_cost * 10) {
			ROS_WARN("System diverges...");
			hasDiverged = true;
			break;
		}

		Eigen::Matrix<double, 18, 1> incremental_errorState = errorState - last_errorState;
		if (incremental_errorState.norm() <= 1e-2) {
			hasConverged = true;
		}

		//nominalState.boxPlus(errorState, linState_);
		nominalState.boxPlusInv(errorState, linState_);
		last_errorState = errorState;

		if (hasConverged || iterCount == iterNum - 1) {
			ceres::Covariance::Options cov_options;
			cov_options.algorithm_type = ceres::DENSE_SVD;
			cov_options.num_threads = MP_PROC_NUM;
			ceres::Covariance covariance(cov_options);
			std::vector<std::pair<const double*, const double*> > covariance_blocks;
			covariance_blocks.push_back(std::make_pair(para_error_state, para_error_state));
			covariance.Compute(covariance_blocks, &problem);
			Eigen::Matrix<double, 18, 18, Eigen::RowMajor> covariance_recovered;
			covariance.GetCovarianceBlock(para_error_state, para_error_state, covariance_recovered.data());
			Pk_ = covariance_recovered;
			enforceSymmetry(Pk_);
			linState_.gn_ = linState_.gn_ * G0 / linState_.gn_.norm();
			filterEstimator_->update(linState_, Pk_);
		}

		if (hasDiverged) {
			ROS_WARN("======System will suffer a large drift======");
		}
	}
}

void publish_odometry() 
{
	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
	odomAftMapped.pose.pose.orientation.x = linState_.qbn_.x();
	odomAftMapped.pose.pose.orientation.y = linState_.qbn_.y();
	odomAftMapped.pose.pose.orientation.z = linState_.qbn_.z();
	odomAftMapped.pose.pose.orientation.w = linState_.qbn_.w();
	odomAftMapped.pose.pose.position.x = linState_.rn_.x();
	odomAftMapped.pose.pose.position.y = linState_.rn_.y();
	odomAftMapped.pose.pose.position.z = linState_.rn_.z();
	pubOdomAftMapped.publish(odomAftMapped);

	std::ofstream lio_path_file(result_path, std::ios::app);
	lio_path_file.setf(std::ios::fixed, std::ios::floatfield);
	lio_path_file.precision(10);
	lio_path_file << odomAftMapped.header.stamp.toSec() << " ";
	lio_path_file.precision(5);
	lio_path_file << odomAftMapped.pose.pose.position.x << " "
				  << odomAftMapped.pose.pose.position.y << " "
				  << odomAftMapped.pose.pose.position.z << " "
			      << odomAftMapped.pose.pose.orientation.w << " "
				  << odomAftMapped.pose.pose.orientation.x << " "
				  << odomAftMapped.pose.pose.orientation.y << " "
				  << odomAftMapped.pose.pose.orientation.z << std::endl;
	lio_path_file.close();

	geometry_msgs::PoseStamped laserAfterMappedPose;
	laserAfterMappedPose.header = odomAftMapped.header;
	laserAfterMappedPose.pose = odomAftMapped.pose.pose;
	laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
	laserAfterMappedPath.header.frame_id = "camera_init";
	laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
	pubLaserAfterMappedPath.publish(laserAfterMappedPath);

	static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "/aft_mapped" ) );
}

void publish_topics() 
{
	if (pubLaserCloudMap.getNumSubscribers() != 0) {
		sensor_msgs::PointCloud2 laserCloudMsg;
		pcl::toROSMsg(*featsFromMap, laserCloudMsg);
		laserCloudMsg.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudMsg.header.frame_id = "camera_init";
		pubLaserCloudMap.publish(laserCloudMsg);
	}

	if (pubLaserCloudFullRes.getNumSubscribers() != 0) {		
		pcl::PointCloud<PointType>::Ptr laserCloudFullRes(featsUndistort);
		int laserCloudFullResNum = laserCloudFullRes->points.size();
		for (int i = 0; i < laserCloudFullResNum; ++i) {
			pointAssociateToMap(&(laserCloudFullRes->points[i]), &(laserCloudFullRes->points[i]));
		}

		sensor_msgs::PointCloud2 laserCloudMsg;
		pcl::toROSMsg(*laserCloudFullRes, laserCloudMsg);
		laserCloudMsg.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudMsg.header.frame_id = "camera_init";
		pubLaserCloudFullRes.publish(laserCloudMsg);
	}

	/*if (pubLaserCloudSurround.getNumSubscribers() != 0) {
		laserCloudSurround->clear();
		int laserCloudCornerLastNum = laserCloudCornerLast->size();
		for (int i = 0; i < laserCloudCornerLastNum; ++i) {
			undisortPoint(&(laserCloudCornerLast->points[i]), &(laserCloudCornerLast->points[i]));
			pointAssociateToMap(&(laserCloudCornerLast->points[i]), &(laserCloudCornerLast->points[i]));
		}
		*laserCloudSurround += *laserCloudCornerLast;
		
		int featsUndistortNum = featsUndistort->size();
		for (int i = 0; i < featsUndistortNum; ++i) {
			undisortPoint(&(featsUndistort->points[i]), &(featsUndistort->points[i]));
			pointAssociateToMap(&(featsUndistort->points[i]), &(featsUndistort->points[i]));
		}
		*laserCloudSurround += *featsUndistort;

		sensor_msgs::PointCloud2 laserCloudMsg;
		pcl::toROSMsg(*laserCloudSurround, laserCloudMsg);
		laserCloudMsg.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudMsg.header.frame_id = "camera_init";
		pubLaserCloudSurround.publish(laserCloudMsg);
	}*/
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	std::vector<double> t_ex(3, 0.0);
	std::vector<double> R_ex(9, 0.0);
	std::string traj_save_path;

	nh.param<double>("mapping_plane_res", plane_res, 0.8);
	nh.param<double>("lidar_std", LIDAR_STD, 0.1);
	nh.param<double>("cube_side_length", cube_len, 200);
	nh.param<std::string>("common/imu_topic", imu_topic, "/imu/data");	
	//nh.param<bool>("mapping/undistortion", undistortion, true);
	nh.param<bool>("mapping/undistortion", pImu_->is_undistort, true);
	nh.param<double>("mapping/loss_threshold", loss_threshold, 0.1);
	nh.param<double>("mapping/acc_noise", acc_noise, 0.01);
	nh.param<double>("mapping/gyr_noise", gyr_noise, 0.005);
	nh.param<double>("mapping/acc_bias_noise", acc_bias_noise, 0.0002);
	nh.param<double>("mapping/gyr_bias_noise", gyr_bias_noise, 4e-6);
	nh.param<vector<double>>("mapping/extrinsic_T", t_ex, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", R_ex, vector<double>());
	nh.param<std::string>("pcd_save/traj_save_path", traj_save_path, "/home/spc/output");

	result_path = traj_save_path + "/lio_mapped.csv";
	std::ofstream fout(result_path, std::ios::out);
	fout.close();

	downSizeFilterSurf.setLeafSize(plane_res, plane_res, plane_res);

	// Set extrinsic
	t_i_l << VEC_FROM_ARRAY(t_ex);
	M3D R_i_l;
	R_i_l << MAT_FROM_ARRAY(R_ex);
	q_i_l = R_i_l;
	pImu_->set_extrinsic(t_i_l, q_i_l);
	pImu_->set_acc_cov(acc_noise);
	pImu_->set_gyr_cov(gyr_noise);
	pImu_->set_acc_bias_cov(acc_bias_noise);
	pImu_->set_gyr_bias_cov(gyr_bias_noise);

	filterEstimator_ = new StatePredictor();
	linState_.setIdentity();
	Pk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
	Pk_.setZero();

	ros::Subscriber subLaserCloudFeats = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100, laserCloudFeatsHandler);

	//ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_full", 100, laserCloudFullResHandler);

	ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 1000, imuCallback);

	//pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 2);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 2);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 2);

	ROS_INFO("\033[1;32m---->\033[0m Mapping Started.");

	ros::Rate loop_rate(500);
	while (ros::ok()) {
		ros::spinOnce();
		if (sync_messages(Measures)) {
			if (is_first_scan) {
				first_lidar_time = Measures.lidar_beg_time;
				pImu_->first_lidar_time = first_lidar_time;
				is_first_scan = false;
				continue;
			}

			TicToc t_whole;

			pImu_->Process(Measures, *filterEstimator_, featsUndistort);
			linState_ = filterEstimator_->state_;
			lid_pos = linState_.qbn_ * linState_.t_ex_ + linState_.rn_;

			if (featsUndistort->empty() || featsUndistort == NULL) {
				ROS_WARN("No point, skip this scan!");
				continue;
			}

			EKF_init_flag = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
			/*** Segment the map in lidar FOV ***/
			lasermap_fov_segment();

			/*** downsample the feature points in a scan ***/
			downSizeFilterSurf.setInputCloud(featsUndistort);
			downSizeFilterSurf.filter(*featsDownBody);
			feats_down_size = featsDownBody->points.size();

			/*** initialize the map kdtree ***/
			if(ikdtreeSurf.Root_Node == nullptr)
			{
				if (feats_down_size > 5) {
					ikdtreeSurf.set_downsample_param(plane_res);
					featsDownWorld->resize(feats_down_size);
					for (int i = 0; i < feats_down_size; i++) {
						pointAssociateToMap(&(featsDownBody->points[i]), &(featsDownWorld->points[i]));
					}
					ikdtreeSurf.Build(featsDownWorld->points);
				}
				continue;
			}
			int featsFromMapNum = ikdtreeSurf.validnum();

			if (feats_down_size < 5) {
				ROS_WARN("No point, skip this scan!");
				continue;
			}

			featsDownWorld->resize(feats_down_size);

			if (0) {// If you need to see map point, change to "if(1)"
				if (frameCount % 20 == 0) {
					featsFromMap->clear();
					PointVector().swap(ikdtreeSurf.PCL_Storage);
					ikdtreeSurf.flatten(ikdtreeSurf.Root_Node, ikdtreeSurf.PCL_Storage, NOT_RECORD);
					featsFromMap->points = ikdtreeSurf.PCL_Storage;
				}			
			}

			nearestPointsSurf.resize(feats_down_size);
			/*** iterated state estimation ***/
			TicToc t_opt;
			performIESKF();
			//printf("mapping optimization time %f \n", t_opt.toc());

			/*** add the feature points to map kdtree ***/
			map_incremental();

			frameCount++;
			whole_map_time += t_whole.toc();
			feats_num = feats_num * (frameCount - 1) / frameCount + feats_down_size / frameCount;
			printf("average features number: %d \n", feats_num);
			printf("whole mapping time %f ms +++++\n", whole_map_time / frameCount);

			publish_odometry();

			publish_topics();

		}

		loop_rate.sleep();
	}

	return 0;
}