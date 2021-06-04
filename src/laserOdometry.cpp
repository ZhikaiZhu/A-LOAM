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

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
#include "aloam_velodyne/parameters.h"
#include "aloam_velodyne/filter_state.hpp"

using namespace parameter;
using namespace utils;
using namespace filter;

#define DISTORTION 0


constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum;
bool systemInited = false;
bool is_first_scan = true;
int frameCount = 0;
double opt_time = 0.0;
double whole_odom_time = 0.0;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

// Result save
std::string RESULT_PATH;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

double para_error_state[18];
double para_ex[7];
//double para_ex[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_b_l(para_ex);  // from lidar to inertial
Eigen::Map<Eigen::Vector3d> t_b_l(para_ex + 4);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::vector<sensor_msgs::ImuConstPtr> imuBuf;
std::mutex mBuf;

// State Estimator Variable
ros::Subscriber subCornerPointsSharp;
ros::Subscriber subCornerPointsLessSharp;
ros::Subscriber subSurfPointsFlat;
ros::Subscriber subSurfPointsLessFlat;
ros::Subscriber subLaserCloudFullRes;
ros::Subscriber subImu;
ros::Publisher pubLaserCloudCornerLast;
ros::Publisher pubLaserCloudSurfLast;
ros::Publisher pubLaserCloudFullRes;
ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserPath;
nav_msgs::Path laserPath;

int Scan::scan_counter_ = 0;
V3D ba_init_;
V3D bw_init_;
double scan_time_;
double last_imu_time_;

StatePredictor* filter_;
ScanPtr scan_new_;
ScanPtr scan_last_;

std::vector<double> pointSearchCornerInd1;
std::vector<double> pointSearchCornerInd2;
std::vector<double> pointSearchSurfInd1;
std::vector<double> pointSearchSurfInd2;
std::vector<double> pointSearchSurfInd3;

pcl::PointCloud<PointType>::Ptr keypoints_;
pcl::PointCloud<PointType>::Ptr jacobians_;
pcl::PointCloud<PointType>::Ptr keypointCorns_;
pcl::PointCloud<PointType>::Ptr keypointSurfs_;
pcl::PointCloud<PointType>::Ptr jacobianCoffCorns_;
pcl::PointCloud<PointType>::Ptr jacobianCoffSurfs_;

GlobalState globalState_;
GlobalState globalStateLidar_;
GlobalState linState_;
Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> difVecLinInv_;
Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> updateVec_;
double updateVecNorm_ = 0.0;

VXD residual_;
MXD Fk_, Gk_, Pk_, Qk_, Rk_, Hk_, Jk_, Kk_, IKH_, Py_, Pyinv_;

// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, linState_.qbn_);
    Eigen::Vector3d t_point_last = s * linState_.rn_;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_b_l.inverse() * (q_point_last * (q_b_l * point + t_b_l) + t_point_last - t_b_l);

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame

void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_b_l.inverse() * ((linState_.qbn_.inverse() * (q_b_l * un_point + t_b_l) - linState_.rn_) - t_b_l);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = pi->intensity;
}

void updatePointCloud() {
    int cornerPointsLessSharpNum = scan_new_->cornerPointsLessSharp_->points.size();
    for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        TransformToEnd(&scan_new_->cornerPointsLessSharp_->points[i], 
                       &scan_new_->cornerPointsLessSharp_->points[i]);
    }

    int surfPointsLessFlatNum = scan_new_->surfPointsLessFlat_->points.size();
    for (int i = 0; i < surfPointsLessFlatNum; i++) {
        TransformToEnd(&scan_new_->surfPointsLessFlat_->points[i], 
                       &scan_new_->surfPointsLessFlat_->points[i]);
    }

    int laserCloudFullResNum = scan_new_->laserCloudFullRes_->points.size();
    for (int i = 0; i < laserCloudFullResNum; i++) {
        TransformToEnd(&scan_new_->laserCloudFullRes_->points[i], 
                       &scan_new_->laserCloudFullRes_->points[i]);
    }

    /*if (cornerPointsLessSharpNum >= 5 && surfPointsLessFlatNum >= 20) {
        kdtreeCornerLast->setInputCloud(scan_new_->cornerPointsLessSharp_);
        kdtreeSurfLast->setInputCloud(scan_new_->surfPointsLessFlat_);
    } */
    kdtreeCornerLast->setInputCloud(scan_new_->cornerPointsLessSharp_);
    kdtreeSurfLast->setInputCloud(scan_new_->surfPointsLessFlat_);
}

void publishTopics() {
    TicToc t_pub;
    if (frameCount % skipFrameNum == 0) {

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*scan_last_->cornerPointsLessSharp_, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(scan_time_);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*scan_last_->surfPointsLessFlat_, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(scan_time_);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*scan_last_->laserCloudFullRes_, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(scan_time_);
        laserCloudFullRes3.header.frame_id = "/camera";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
    }

    // publish odometry
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "/camera_init";
    laserOdometry.child_frame_id = "/laser_odom";
    laserOdometry.header.stamp = ros::Time().fromSec(scan_time_);
    laserOdometry.pose.pose.orientation.x = globalStateLidar_.qbn_.x();
    laserOdometry.pose.pose.orientation.y = globalStateLidar_.qbn_.y();
    laserOdometry.pose.pose.orientation.z = globalStateLidar_.qbn_.z();
    laserOdometry.pose.pose.orientation.w = globalStateLidar_.qbn_.w();
    laserOdometry.pose.pose.position.x = globalStateLidar_.rn_[0];
    laserOdometry.pose.pose.position.y = globalStateLidar_.rn_[1];
    laserOdometry.pose.pose.position.z = globalStateLidar_.rn_[2];
    pubLaserOdometry.publish(laserOdometry);

    geometry_msgs::PoseStamped laserPose;
    laserPose.header = laserOdometry.header;
    laserPose.pose = laserOdometry.pose.pose;
    laserPath.header.stamp = laserOdometry.header.stamp;
    laserPath.poses.push_back(laserPose);
    laserPath.header.frame_id = "/camera_init";
    pubLaserPath.publish(laserPath);
    //printf("publication time %f ms \n", t_pub.toc());

    std::ofstream lio_path_file(RESULT_PATH, std::ios::app);
	lio_path_file.setf(std::ios::fixed, std::ios::floatfield);
	lio_path_file.precision(10);
	lio_path_file << laserOdometry.header.stamp.toSec() << " ";
	lio_path_file.precision(5);
	lio_path_file << laserOdometry.pose.pose.position.x << " "
				  << laserOdometry.pose.pose.position.y << " "
				  << laserOdometry.pose.pose.position.z << " "
				  << laserOdometry.pose.pose.orientation.w << " "
				  << laserOdometry.pose.pose.orientation.x << " "
				  << laserOdometry.pose.pose.orientation.y << " "
				  << laserOdometry.pose.pose.orientation.z << std::endl;
	lio_path_file.close();

}

void findCorrespondingCornerFeatures(ScanPtr lastScan, ScanPtr newScan,
                                     pcl::PointCloud<PointType>::Ptr keypoints,
                                     pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount,
                                     ceres::Problem *problem = nullptr,
                                     GlobalState *nominalState = nullptr) {
    int cornerPointsSharpNum = newScan->cornerPointsSharp_->points.size();

    for (int i = 0; i < cornerPointsSharpNum; ++i) {
        PointType pointSel;
        PointType coeff;

        TransformToStart(&newScan->cornerPointsSharp_->points[i], &pointSel);
        pcl::PointCloud<PointType>::Ptr laserCloudCornerLast = lastScan->cornerPointsLessSharp_;

        if (iterCount % ICP_FREQ == 0) {
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

            int closestPointInd = -1, minPointInd2 = -1;
            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
            {
                closestPointInd = pointSearchInd[0];
                int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                // search in the direction of increasing scan line
                for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                {
                    // if in the same scan line, continue
                    if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                        continue;

                    // if not in nearby scans, end the loop
                    if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                        break;

                    double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                            (laserCloudCornerLast->points[j].x - pointSel.x) +
                                        (laserCloudCornerLast->points[j].y - pointSel.y) *
                                            (laserCloudCornerLast->points[j].y - pointSel.y) +
                                        (laserCloudCornerLast->points[j].z - pointSel.z) *
                                            (laserCloudCornerLast->points[j].z - pointSel.z);

                    if (pointSqDis < minPointSqDis2)
                    {
                        // find nearer point
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }

                // search in the direction of decreasing scan line
                for (int j = closestPointInd - 1; j >= 0; --j)
                {
                    // if in the same scan line, continue
                    if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                        continue;

                    // if not in nearby scans, end the loop
                    if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                        break;

                    double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                            (laserCloudCornerLast->points[j].x - pointSel.x) +
                                        (laserCloudCornerLast->points[j].y - pointSel.y) *
                                            (laserCloudCornerLast->points[j].y - pointSel.y) +
                                        (laserCloudCornerLast->points[j].z - pointSel.z) *
                                            (laserCloudCornerLast->points[j].z - pointSel.z);

                    if (pointSqDis < minPointSqDis2)
                    {
                        // find nearer point
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }
            }
            pointSearchCornerInd1[i] = closestPointInd;
            pointSearchCornerInd2[i] = minPointInd2;
        }
        if (pointSearchCornerInd2[i] >= 0) // both closestPointInd and minPointInd2 is valid
        {
            V3D curr_point(newScan->cornerPointsSharp_->points[i].x,
                           newScan->cornerPointsSharp_->points[i].y,
                           newScan->cornerPointsSharp_->points[i].z);
            PointType last_point_a = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
            PointType last_point_b = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

            V3D P0(pointSel.x, pointSel.y, pointSel.z);
            V3D P1(last_point_a.x, last_point_a.y, last_point_a.z);
            V3D P2(last_point_b.x, last_point_b.y, last_point_b.z);

            V3D P = utils::skew(P0 - P1) * (P0 - P2);
            double r = P.norm();
            double d12 = (P1 - P2).norm();
            double res = r / d12;

            V3D jacxyz = P.transpose() * utils::skew(P2 - P1) / (d12 * r);

            double w = 1.0;
            if (iterCount >= ICP_FREQ) {
                w = 1 - 1.8 * fabs(res);
            }

            if (w > 0.1 && res != 0) {
                coeff.x = w * jacxyz(0);
                coeff.y = w * jacxyz(1);
                coeff.z = w * jacxyz(2);
                coeff.intensity = w * res;

                keypoints->push_back(newScan->cornerPointsSharp_->points[i]);
                jacobianCoff->push_back(coeff);
            }

            if (w > 0.1 && res != 0 && problem != nullptr) {
                double s;
                if (DISTORTION)
                    s = (newScan->cornerPointsSharp_->points[i].intensity - int(newScan->cornerPointsSharp_->points[i].intensity)) / SCAN_PERIOD;
                else
                    s = 1.0;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(LOSS_THRESHOLD);
                ceres::CostFunction *cost_function = LidarEdgeStateFactorEx::Create(curr_point, P1, P2, s, nominalState->rn_, 
                                                                                  nominalState->qbn_, LIDAR_STD / w);
                problem->AddResidualBlock(cost_function, loss_function, para_error_state, para_ex, para_ex + 4);
            }
        }
    }
}

void findCorrespondingSurfFeatures(ScanPtr lastScan, ScanPtr newScan,
                                   pcl::PointCloud<PointType>::Ptr keypoints,
                                   pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount,
                                   ceres::Problem *problem = nullptr,
                                   GlobalState *nominalState = nullptr) {
    int surfPointsFlatNum = newScan->surfPointsFlat_->points.size();

    for (int i = 0; i < surfPointsFlatNum; ++i) {
        PointType pointSel;
        PointType coeff;

        TransformToStart(&newScan->surfPointsFlat_->points[i], &pointSel);
        pcl::PointCloud<PointType>::Ptr laserCloudSurfLast = lastScan->surfPointsLessFlat_;

        if (iterCount % ICP_FREQ == 0) {
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

            int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                closestPointInd = pointSearchInd[0];

                // get closest point's scan ID
                int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                // search in the direction of increasing scan line
                for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                {
                    // if not in nearby scans, end the loop
                    if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                        break;

                    double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                            (laserCloudSurfLast->points[j].x - pointSel.x) +
                                        (laserCloudSurfLast->points[j].y - pointSel.y) *
                                            (laserCloudSurfLast->points[j].y - pointSel.y) +
                                        (laserCloudSurfLast->points[j].z - pointSel.z) *
                                            (laserCloudSurfLast->points[j].z - pointSel.z);

                    // if in the same or lower scan line
                    if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                    // if in the higher scan line
                    else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                    {
                        minPointSqDis3 = pointSqDis;
                        minPointInd3 = j;
                    }
                }

                // search in the direction of decreasing scan line
                for (int j = closestPointInd - 1; j >= 0; --j)
                {
                    // if not in nearby scans, end the loop
                    if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                        break;

                    double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                            (laserCloudSurfLast->points[j].x - pointSel.x) +
                                        (laserCloudSurfLast->points[j].y - pointSel.y) *
                                            (laserCloudSurfLast->points[j].y - pointSel.y) +
                                        (laserCloudSurfLast->points[j].z - pointSel.z) *
                                            (laserCloudSurfLast->points[j].z - pointSel.z);

                    // if in the same or higher scan line
                    if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                    else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                    {
                        // find nearer point
                        minPointSqDis3 = pointSqDis;
                        minPointInd3 = j;
                    }
                }
            }
            pointSearchSurfInd1[i] = closestPointInd;
            pointSearchSurfInd2[i] = minPointInd2;
            pointSearchSurfInd3[i] = minPointInd3;
        }

        if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0)
        {

            V3D curr_point(newScan->surfPointsFlat_->points[i].x,
                           newScan->surfPointsFlat_->points[i].y,
                           newScan->surfPointsFlat_->points[i].z);
            PointType last_point_a = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
            PointType last_point_b = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
            PointType last_point_c = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

            V3D P0(pointSel.x, pointSel.y, pointSel.z);
            V3D P1(last_point_a.x, last_point_a.y, last_point_a.z);
            V3D P2(last_point_b.x, last_point_b.y, last_point_b.z);
            V3D P3(last_point_c.x, last_point_c.y, last_point_c.z);

            V3D M = utils::skew(P1 - P2) * (P1 - P3);
            double r = (P0 - P1).transpose() * M;
            double m = M.norm();
            double res = r / m;

            V3D jacxyz = M.transpose() / (m);

            double w = 1.0;
            if (iterCount >= ICP_FREQ) {
                w = 1 - 1.8 * fabs(res) /
                              sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
            }

            if (w > 0.1 && res != 0) {
                coeff.x = w * jacxyz(0);
                coeff.y = w * jacxyz(1);
                coeff.z = w * jacxyz(2);
                coeff.intensity = w * res;

                keypoints->push_back(newScan->surfPointsFlat_->points[i]);
                jacobianCoff->push_back(coeff);
            }

            if (w > 0.1 && res != 0 && problem != nullptr) {
                double s;
                if (DISTORTION)
                    s = (newScan->surfPointsFlat_->points[i].intensity - int(newScan->surfPointsFlat_->points[i].intensity)) / SCAN_PERIOD;
                else
                    s = 1.0;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(LOSS_THRESHOLD);
                ceres::CostFunction *cost_function = LidarPlaneStateFactorEx::Create(curr_point, P1, P2, P3, s, nominalState->rn_,
                                                                                   nominalState->qbn_, LIDAR_STD / w);
                problem->AddResidualBlock(cost_function, loss_function, para_error_state, para_ex, para_ex + 4);
            }
        }
    }
}

bool calculateTransformation(ScanPtr lastScan, ScanPtr newScan,
                             pcl::PointCloud<PointType>::Ptr corners,
                             pcl::PointCloud<PointType>::Ptr jacoCornersCoff,
                             pcl::PointCloud<PointType>::Ptr surfs,
                             pcl::PointCloud<PointType>::Ptr jacoSurfsCoff, int iterCount) {
    keypoints_->clear();
    jacobians_->clear();
    (*keypoints_) += (*surfs);
    (*keypoints_) += (*corners);
    (*jacobians_) += (*jacoSurfsCoff);
    (*jacobians_) += (*jacoCornersCoff);

    const int stateNum = 6;
    const int pointNum = keypoints_->points.size();
    const int imuNum = 0;
    const int row = pointNum + imuNum;
    Eigen::Matrix<double, Eigen::Dynamic, stateNum> J(row, stateNum);
    Eigen::Matrix<double, stateNum, Eigen::Dynamic> JT(stateNum, row);
    Eigen::Matrix<double, stateNum, stateNum> JTJ;
    Eigen::VectorXd b(row);
    Eigen::Matrix<double, stateNum, 1> JTb;
    Eigen::Matrix<double, stateNum, 1> x;
    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    for (int i = 0; i < pointNum; ++i) {
        // Select keypoint i
        const PointType& keypoint = keypoints_->points[i];
        const PointType& coeff = jacobians_->points[i];

        V3D P2xyz(keypoint.x, keypoint.y, keypoint.z);
        V3D coff_xyz(coeff.x, coeff.y, coeff.z);

        double s;
        if (DISTORTION)
            s = (1.f / SCAN_PERIOD) * (keypoint.intensity - int(keypoint.intensity));
        else 
            s = 1.0;

        V3D phi = Quat2axis(linState_.qbn_);
        // Rotation matrix from frame2 (new) to frame1 (last)
        Q4D R21xyz = axis2Quat(s * phi);
        R21xyz.normalized();
        // Translation vector from frame1 to frame2 represented in frame1
        // V3D T112xyz = s * linState_.rn_;

        V3D jacobian1xyz =
            coff_xyz.transpose() *
            (-R21xyz.toRotationMatrix() * skew(P2xyz));  // rotation jacobian
        V3D jacobian2xyz =
            coff_xyz.transpose() * M3D::Identity();  // translation jacobian
        double residual = coeff.intensity;

        J.block<1, 3>(i, O_R) = jacobian1xyz;
        J.block<1, 3>(i, O_P) = jacobian2xyz;

        // Set the overall residual
        b(i) = -0.05 * residual;
    }

    // Solve x
    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;
    x = JTJ.colPivHouseholderQr().solve(JTb);

    // Determine whether x is degenerated
    bool isDegenerate = false;
    Eigen::Matrix<double, stateNum, stateNum> matP;
    if (iterCount == 0) {
        Eigen::Matrix<double, 1, stateNum> matE;
        Eigen::Matrix<double, stateNum, stateNum> matV;
        Eigen::Matrix<double, stateNum, stateNum> matV2;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, stateNum, stateNum> > esolver(JTJ);
        matE = esolver.eigenvalues().real();
        matV = esolver.eigenvectors().real();

        matV2 = matV;

        isDegenerate = false;
        std::vector<double> eignThre(stateNum, 10.);
        for (int i = 0; i < stateNum; i++) {
            // if eigenvalue is less than 10, set the corresponding eigenvector to 0
            // vector
            if (matE(0, i) < eignThre[i]) {
            for (int j = 0; j < stateNum; j++) {
                matV2(i, j) = 0;
            }
            isDegenerate = true;
            } else {
            break;
            }
        }
        matP = matV.inverse() * matV2;
    }

    if (isDegenerate) {
        cout << "System is Degenerate." << endl;
        Eigen::Matrix<double, stateNum, 1> matX2(x);
        x = matP * matX2;
    }

    // Update state linState_
    Q4D dq = rpy2Quat(x.segment<3>(O_R));
    linState_.qbn_ = (linState_.qbn_ * dq).normalized();
    linState_.rn_ += x.segment<3>(O_P);

    // Determine whether should it stop
    V3D rpy_rad = x.segment<3>(O_R);
    V3D rpy_deg = utils::rad2deg(rpy_rad);
    double deltaR = rpy_deg.norm();
    V3D trans = 100 * x.segment<3>(O_P);
    double deltaT = trans.norm();
    if (deltaR < 0.1 && deltaT < 0.1) {
        return true;
    }

    return false;
}

void estimateTransform(ScanPtr lastScan, ScanPtr newScan, V3D &t, Q4D &q) {
    linState_.rn_ = t;
    linState_.qbn_ = q;
    for (int iter = 0; iter < NUM_ITER; iter++) {
        keypointSurfs_->clear();
        jacobianCoffSurfs_->clear();
        keypointCorns_->clear();
        jacobianCoffCorns_->clear();

        findCorrespondingSurfFeatures(lastScan, newScan, keypointSurfs_,
                                        jacobianCoffSurfs_, iter);
        if (keypointSurfs_->points.size() < 10) {
            ROS_WARN("Insufficient matched surfs...");
            continue;
        }
        findCorrespondingCornerFeatures(lastScan, newScan, keypointCorns_,
                                        jacobianCoffCorns_, iter);
        if (keypointCorns_->points.size() < 5) {
            ROS_WARN("Insufficient matched corners...");
            continue;
        }

        if (calculateTransformation(lastScan, newScan, keypointCorns_,
                                    jacobianCoffCorns_, keypointSurfs_,
                                    jacobianCoffSurfs_, iter)) {
            ROS_INFO_STREAM("System Converges after " << iter << " iterations");
            break;
        }
    }

    t = linState_.rn_;
    q = linState_.qbn_;  // qbn_ is quaternion rotation from b-frame to n-frame
}

void integrateTransformation() {
    GlobalState filterState = filter_->state_;
    globalState_.rn_ = globalState_.qbn_ * filterState.rn_ + globalState_.rn_;
    globalState_.qbn_ = globalState_.qbn_ * filterState.qbn_;
    globalState_.vn_ = globalState_.qbn_ * filterState.qbn_.inverse() * filterState.vn_;
    globalState_.ba_ = filterState.ba_;
    globalState_.bw_ = filterState.bw_;
    globalState_.gn_ = globalState_.qbn_ * filterState.qbn_.inverse() * filterState.gn_;
    globalState_.q_ex_ = filterState.q_ex_;
    globalState_.t_ex_ = filterState.t_ex_;
}

void transformFromInertialToLidar(){
    globalStateLidar_.qbn_ = q_b_l.inverse() * globalState_.qbn_ * q_b_l;
    globalStateLidar_.rn_ = q_b_l.inverse() * (globalState_.qbn_ * t_b_l + globalState_.rn_ - t_b_l);
}

void calculateRPfromIMU(const V3D& acc, double& roll, double& pitch) {
    pitch = -sign(acc.z()) * asin(acc.x() / G0);
    //roll = sign(acc.z()) * asin(acc.y() / G0);
    roll = atan(acc.y() / acc.z());
}

void correctRollPitch(const double &roll, const double &pitch) {
    V3D rpy = utils::Q2rpy(globalState_.qbn_);
    Q4D quad = utils::rpy2Quat(V3D(roll, pitch, rpy[2]));
    globalState_.qbn_ = quad;
}

void performIESKF() {
    // Store current state and perform initialization
    Pk_ = filter_->covariance_;
    GlobalState filterState = filter_->state_;
    linState_ = filterState;

    double residualNorm = 1e6;
    bool hasConverged = false;
    bool hasDiverged = false;
    if (USE_CERES) {
        for (size_t i = 0; i < 18; ++i) {
            para_error_state[i] = 0.0;
        }
        GlobalState nominalState = filterState;
        Eigen::Map<Eigen::Matrix<double, 18, 1>> errorState(para_error_state);
        ceres::Solver::Options options;
        options.max_solver_time_in_seconds = 0.05;
        options.max_num_iterations = CERES_MAX_ITER;
        options.linear_solver_type = ceres::DENSE_QR;
        Eigen::Matrix<double, 18, 1> last_errorState = errorState;
        Q4D ex_rotation = q_b_l;
        V3D ex_translation = t_b_l;
        for (int iter = 0; iter <= NUM_ITER / CERES_MAX_ITER && !hasConverged && !hasDiverged; iter++) {
            keypointSurfs_->clear();
            jacobianCoffSurfs_->clear();
            keypointCorns_->clear();
            jacobianCoffCorns_->clear();

            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem problem;
            problem.AddParameterBlock(para_error_state, 18);
            problem.AddParameterBlock(para_ex, 4, q_parameterization);
            problem.AddParameterBlock(para_ex + 4, 3);
            if (!CALIB_EXTRINSIC) {
                problem.SetParameterBlockConstant(para_ex);
                problem.SetParameterBlockConstant(para_ex + 4);
            }
            ceres::CostFunction *prior_factor = PriorFactor::Create(Pk_, ex_rotation, ex_translation);
            problem.AddResidualBlock(prior_factor, nullptr, para_error_state, para_ex, para_ex + 4);
            
            if (!PURE_IMU) {
                findCorrespondingSurfFeatures(scan_last_, scan_new_, keypointSurfs_,
                                            jacobianCoffSurfs_, iter, &problem, &nominalState);
                if (keypointSurfs_->points.size() < 10) {
                    ROS_WARN("Insufficient matched surfs...");
                }
                findCorrespondingCornerFeatures(scan_last_, scan_new_, keypointCorns_,
                                                jacobianCoffCorns_, iter, &problem, &nominalState);
                if (keypointCorns_->points.size() < 5) {
                    ROS_WARN("Insufficient matched corners...");
                }
            }
                                            
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
        }

        if (hasDiverged == true) {
            ROS_WARN("======Using ICP Method======");
            // Reset extrinsic parameters
            para_ex[0] = INIT_RBL.x();
            para_ex[1] = INIT_RBL.y();
            para_ex[2] = INIT_RBL.z();
            para_ex[3] = INIT_RBL.w();
            para_ex[4] = INIT_TBL.x();
            para_ex[5] = INIT_TBL.y();
            para_ex[6] = INIT_TBL.z();

            V3D t = filterState.rn_;
            Q4D q = filterState.qbn_;
            estimateTransform(scan_last_, scan_new_, t, q);
            filterState.rn_ = t;
            filterState.qbn_ = q;
            filter_->update(filterState, Pk_);
        } 
        else {     
            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem problem;
            problem.AddParameterBlock(para_error_state, 18);
            problem.AddParameterBlock(para_ex, 4, q_parameterization);
            problem.AddParameterBlock(para_ex + 4, 3);
            if (!CALIB_EXTRINSIC) {
                problem.SetParameterBlockConstant(para_ex);
                problem.SetParameterBlockConstant(para_ex + 4);
            }
            ceres::CostFunction *prior_factor = PriorFactor::Create(Pk_, ex_rotation, ex_translation);
            problem.AddResidualBlock(prior_factor, nullptr, para_error_state, para_ex, para_ex + 4);
            
            if (!PURE_IMU) {
                findCorrespondingSurfFeatures(scan_last_, scan_new_, keypointSurfs_,
                                            jacobianCoffSurfs_, 10, &problem, &nominalState);
                findCorrespondingCornerFeatures(scan_last_, scan_new_, keypointCorns_,
                                                jacobianCoffCorns_, 10, &problem, &nominalState);
            }
                                            
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << "ex rotation: " << q_b_l.toRotationMatrix() << std::endl;
            std::cout << "ex translation: " << t_b_l.transpose() << std:: endl;
            ceres::Covariance::Options cov_options;
            cov_options.algorithm_type = ceres::DENSE_SVD;
            ceres::Covariance covariance(cov_options);
            std::vector<std::pair<const double*, const double*> > covariance_blocks;
            if (!CALIB_EXTRINSIC) {
                covariance_blocks.push_back(std::make_pair(para_error_state, para_error_state));
                covariance.Compute(covariance_blocks, &problem);
                Eigen::Matrix<double, 18, 18, Eigen::RowMajor> covariance_recovered;
                covariance.GetCovarianceBlockInTangentSpace(para_error_state, para_error_state, covariance_recovered.data());
                Pk_.block<18, 18>(0, 0) = covariance_recovered;
            }
            else {
                std::vector<const double*> v_param;
                v_param.push_back(para_error_state);
                v_param.push_back(para_ex);
                v_param.push_back(para_ex + 4);
                // covariance_blocks can't contain duplicates
                for (size_t i = 0; i < v_param.size(); ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        covariance_blocks.push_back(std::make_pair(v_param[i], v_param[j]));
                    }
                }
                covariance.Compute(covariance_blocks, &problem);

                Eigen::Matrix<double, 18, 18, Eigen::RowMajor> covariance_recovered_1;
                std::vector<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>> covariance_recovered_2;
                covariance_recovered_2.resize(v_param.size() - 1);
                std::vector<Eigen::Matrix<double, 3, 18, Eigen::RowMajor>> covariance_recovered_3;
                covariance_recovered_3.resize(v_param.size() - 1);
                std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> covariance_recovered_4;
                covariance_recovered_4.resize( (v_param.size() - 1) * (v_param.size() - 1) );
                int k = 0;
                for (size_t i = 0; i < v_param.size(); ++i) {
                    for (size_t j = 0; j < v_param.size(); ++j) {
                        if (i == 0 && j == 0) {
                            covariance.GetCovarianceBlockInTangentSpace(v_param[i], v_param[j], covariance_recovered_1.data());
                        }
                        else if (i == 0 && j != 0) {
                            covariance.GetCovarianceBlockInTangentSpace(v_param[i], v_param[j], covariance_recovered_2[j - 1].data());
                        }
                        else if (i != 0 && j == 0) {
                            covariance.GetCovarianceBlockInTangentSpace(v_param[i], v_param[j], covariance_recovered_3[i - 1].data());
                        }
                        else {
                            covariance.GetCovarianceBlockInTangentSpace(v_param[i], v_param[j], covariance_recovered_4[k].data());
                            k++;
                        }
                    }
                }
                for (size_t j = 0; j < v_param.size(); ++j) {
                    if (j == 0) {
                        covariance_recovered_2[j] *= 2.0;
                        covariance_recovered_3[j] *= 2.0;
                        covariance_recovered_4[j] *= 4.0;
                    }
                    else {
                        covariance_recovered_4[j] *= 2.0;
                    }
                }
                Pk_.setZero();
                Pk_.block<18, 18>(0, 0) = covariance_recovered_1;
                for (size_t i = 0; i < covariance_recovered_2.size(); ++i) {
                    Pk_.block<18, 3>(0, 18 + 3 * i) = covariance_recovered_2[i];
                    Pk_.block<3, 18>(18 + 3 * i, 0) = covariance_recovered_3[i];
                }
                k = 0;
                for (size_t i = 0; i < covariance_recovered_2.size(); ++i) {
                    for (size_t j = 0; j < covariance_recovered_2.size(); ++j){
                        Pk_.block<3, 3>(18 + 3 * i, 18 + 3 * j) = covariance_recovered_4[k];
                        k++;
                    }
                }
            }
            enforceSymmetry(Pk_);
            //nominalState.boxPlus(errorState, linState_);
            nominalState.boxPlusInv(errorState, linState_);
            filter_->update(linState_, Pk_);
            filter_->state_.q_ex_ = q_b_l;
            filter_->state_.t_ex_ = t_b_l;
        }
        return;
    }
    /*for (int iter = 0; iter < NUM_ITER && !hasConverged && !hasDiverged; iter++) {
        keypointSurfs_->clear();
        jacobianCoffSurfs_->clear();
        keypointCorns_->clear();
        jacobianCoffCorns_->clear();

        // Find corresponding features
        findCorrespondingSurfFeatures(scan_last_, scan_new_, keypointSurfs_,
                                        jacobianCoffSurfs_, iter);
        if (keypointSurfs_->points.size() < 10) {
            ROS_WARN("Insufficient matched surfs...");
        }
        findCorrespondingCornerFeatures(scan_last_, scan_new_, keypointCorns_,
                                        jacobianCoffCorns_, iter);
        if (keypointCorns_->points.size() < 5) {
            ROS_WARN("Insufficient matched corners...");
        }

        // Sum up jocobians and residuals
        keypoints_->clear();
        jacobians_->clear();
        (*keypoints_) += (*keypointSurfs_);
        (*keypoints_) += (*keypointCorns_);
        (*jacobians_) += (*jacobianCoffSurfs_);
        (*jacobians_) += (*jacobianCoffCorns_);

        // Memery allocation
        const int DIM_OF_MEAS = keypoints_->points.size();
        residual_.resize(DIM_OF_MEAS);
        Hk_.resize(DIM_OF_MEAS, DIM_OF_STATE);
        Rk_.resize(DIM_OF_MEAS, DIM_OF_MEAS);
        Kk_.resize(DIM_OF_STATE, DIM_OF_MEAS);
        Py_.resize(DIM_OF_MEAS, DIM_OF_MEAS);
        Pyinv_.resize(DIM_OF_MEAS, DIM_OF_MEAS);

        Hk_.setZero();
        V3D axis = Quat2axis(linState_.qbn_);
        for (int i = 0; i < DIM_OF_MEAS; ++i) {
            // Point represented in 2-frame (e.g., the end frame) in a
            // xyz-convention
            V3D P2xyz(keypoints_->points[i].x, keypoints_->points[i].y, keypoints_->points[i].z);
            V3D coff_xyz(jacobians_->points[i].x, jacobians_->points[i].y, jacobians_->points[i].z);
            residual_(i) = LIDAR_SCALE * jacobians_->points[i].intensity;

            Hk_.block<1, 3>(i, GlobalState::att_) =
                coff_xyz.transpose() * (-linState_.qbn_.toRotationMatrix() * skew(P2xyz)) * Rinvleft(-axis);
            Hk_.block<1, 3>(i, GlobalState::pos_) =
                coff_xyz.transpose() * M3D::Identity();
        }

        // Set the measurement covariance matrix
        VXD cov = VXD::Zero(DIM_OF_MEAS);
        for (int i = 0; i < DIM_OF_MEAS; ++i) {
            cov[i] = LIDAR_STD * LIDAR_STD;
        }
        Rk_ = cov.asDiagonal();

        // Kalman filter update. Details can be referred to ROVIO
        Py_ = Hk_ * Pk_ * Hk_.transpose() + Rk_;  // S = H * P * H.transpose() + R;
        Pyinv_.setIdentity();                   // solve Ax=B
        Py_.llt().solveInPlace(Pyinv_);
        Kk_ = Pk_ * Hk_.transpose() * Pyinv_;  // K = P*H.transpose()*S.inverse()

        filterState.boxMinus(linState_, difVecLinInv_);
        updateVec_ = -Kk_ * (residual_ + Hk_ * difVecLinInv_) + difVecLinInv_;

        // Divergence determination
        bool hasNaN = false;
        for (int i = 0; i < updateVec_.size(); i++) {
            if (isnan(updateVec_[i])) {
                updateVec_[i] = 0;
                hasNaN = true;
            }
        }
        if (hasNaN == true) {
            ROS_WARN("System diverges Because of NaN...");
            hasDiverged = true;
            break;
        }

        // Check whether the filter converges
        if (residual_.norm() > residualNorm * 10) {
            ROS_WARN("System diverges...");
            hasDiverged = true;
            break;
        }

        // Update the state
        linState_.boxPlus(updateVec_, linState_);
        //std::cout << linState_.rn_.transpose() << std::endl;

        updateVecNorm_ = updateVec_.norm();
        if (updateVecNorm_ <= 1e-2) {
            hasConverged = true;
        }

        residualNorm = residual_.norm();
    }

    // If diverges, swtich to traditional ICP method to get a rough relative
    // transformation. Otherwise, update the error-state covariance matrix
    if (hasDiverged == true) {
        ROS_WARN("======Using ICP Method======");
        V3D t = filterState.rn_;
        Q4D q = filterState.qbn_;
        estimateTransform(scan_last_, scan_new_, t, q);
        filterState.rn_ = t;
        filterState.qbn_ = q;
        filter_->update(filterState, Pk_);
    } 
    else {
        // Update only one time
        IKH_ = Eigen::Matrix<double, 24, 24>::Identity() - Kk_ * Hk_;
        Pk_ = IKH_ * Pk_ * IKH_.transpose() + Kk_ * Rk_ * Kk_.transpose();
        enforceSymmetry(Pk_);
        filter_->update(linState_, Pk_);
    } */
}

void initializeGravityAndBias() {
    // Initialize gravity and gyro bias
    V3D sum_angular_vel = Eigen::Vector3d::Zero();
    V3D sum_linear_acc = Eigen::Vector3d::Zero();
    for (const auto &imu_msg: imuBuf) {
        V3D angular_vel = Eigen::Vector3d::Zero();
        V3D linear_acc = Eigen::Vector3d::Zero();
        angular_vel << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
        linear_acc << imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z;

        sum_angular_vel += angular_vel;
        sum_linear_acc += linear_acc;
    }

    bw_init_ = sum_angular_vel / imuBuf.size();
    V3D gravity_imu = sum_linear_acc / imuBuf.size();

    G0 = gravity_imu.norm();
    V3D gravity = V3D(0.0, 0.0, -G0);
    globalState_.setIdentity();
    globalStateLidar_.setIdentity();
    linState_.setIdentity();
    Q4D q0 = Eigen::Quaterniond::FromTwoVectors(gravity_imu, -gravity);
    globalState_.qbn_ = q0;
    globalState_.bw_ = bw_init_;
    linState_.bw_ = bw_init_;
    std::cout<< "System Initialization Succeeded !!!" << std::endl;
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

void imuCallback(const sensor_msgs::ImuConstPtr &imuMsg) {
    mBuf.lock();
    imuBuf.push_back(imuMsg);
    mBuf.unlock();
    if (!systemInited) {
        if (imuBuf.size() < 200) return;
        initializeGravityAndBias();
        systemInited = true;
    }
}

void Initialization(ros::NodeHandle &nh) {
    subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    subImu = nh.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 100, imuCallback);

    pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    filter_ = new StatePredictor();
    scan_new_.reset(new Scan());
    scan_last_.reset(new Scan());

    keypoints_.reset(new pcl::PointCloud<PointType>());
    jacobians_.reset(new pcl::PointCloud<PointType>());
    keypointCorns_.reset(new pcl::PointCloud<PointType>());
    keypointSurfs_.reset(new pcl::PointCloud<PointType>());
    jacobianCoffCorns_.reset(new pcl::PointCloud<PointType>());
    jacobianCoffSurfs_.reset(new pcl::PointCloud<PointType>());

    pointSearchCornerInd1.resize(N_SCANS * SCAN_NUM);
    pointSearchCornerInd2.resize(N_SCANS * SCAN_NUM);
    pointSearchSurfInd1.resize(N_SCANS * SCAN_NUM);
    pointSearchSurfInd2.resize(N_SCANS * SCAN_NUM);
    pointSearchSurfInd3.resize(N_SCANS * SCAN_NUM);

    Fk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    Gk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);
    Pk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    Qk_.resize(GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_);
    IKH_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);

    Fk_.setIdentity();
    Gk_.setZero();
    Pk_.setZero();
    Qk_.setZero();

    para_ex[0] = INIT_RBL.x();
    para_ex[1] = INIT_RBL.y();
    para_ex[2] = INIT_RBL.z();
    para_ex[3] = INIT_RBL.w();
    para_ex[4] = INIT_TBL.x();
    para_ex[5] = INIT_TBL.y();
    para_ex[6] = INIT_TBL.z();
}

void swapScan() {
    pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
    cornerPointsLessSharp = scan_last_->cornerPointsLessSharp_;
    scan_last_->cornerPointsLessSharp_ = laserCloudTemp;

    laserCloudTemp = cornerPointsSharp;
    cornerPointsSharp = scan_last_->cornerPointsSharp_;
    scan_last_->cornerPointsSharp_ = laserCloudTemp;

    laserCloudTemp = surfPointsLessFlat;
    surfPointsLessFlat = scan_last_->surfPointsLessFlat_;
    scan_last_->surfPointsLessFlat_ = laserCloudTemp;

    laserCloudTemp = surfPointsFlat;
    surfPointsFlat = scan_last_->surfPointsFlat_;
    scan_last_->surfPointsFlat_ = laserCloudTemp;

    laserCloudTemp = laserCloudFullRes;
    laserCloudFullRes = scan_last_->laserCloudFullRes_;
    scan_last_->laserCloudFullRes_ = laserCloudTemp;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh("~");

    parameter::readParameters(nh);

    nh.param<int>("mapping_skip_frame", skipFrameNum, 1);
    RESULT_PATH = OUTPUT_FOLDER + "/lio.csv";
    std::ofstream fout(RESULT_PATH, std::ios::out);
    fout.close();

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    Initialization(nh);

    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!systemInited) {
            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);
            continue;
        }

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            if (is_first_scan) {
                scan_time_ = timeLaserCloudFullRes;
                scan_new_->setPointCloud(scan_time_, cornerPointsSharp, cornerPointsLessSharp, 
                                         surfPointsFlat, surfPointsLessFlat, laserCloudFullRes);
                //kdtreeCornerLast->setInputCloud(scan_new_->cornerPointsLessSharp_);
                //kdtreeSurfLast->setInputCloud(scan_new_->surfPointsLessFlat_);

                V3D p0, v0;
                p0.setZero(), v0.setZero(), ba_init_.setZero();
                Q4D q0;
                q0.setIdentity();
                filter_->initialization(scan_time_, p0, v0, q0, ba_init_, bw_init_);
                filter_->state_.q_ex_ = q_b_l;
                filter_->state_.t_ex_ = t_b_l;
                
                updatePointCloud();
                transformFromInertialToLidar();

                swapScan();
                scan_new_.reset(new Scan());

                publishTopics();

                is_first_scan = false;
                //ROS_BREAK();
                continue;
            }

            if (systemInited && !is_first_scan)
            {
                scan_time_ = timeLaserCloudFullRes;
                mBuf.lock();
                auto it = imuBuf.crbegin();
                mBuf.unlock();
                if ((*it)->header.stamp.toSec() < scan_time_) {
                    continue;
                }
                
                int used_imu_msg = 0;
                mBuf.lock();
                for (const auto &imu_msg: imuBuf) {
                    double imu_time = imu_msg->header.stamp.toSec();
                    last_imu_time_ = filter_->time_;
                    if (imu_time < last_imu_time_) {
                        ++used_imu_msg;
                        continue;
                    }
                    if (imu_time > scan_time_) {
                        V3D imu_acc, imu_gyr;
                        imu_gyr << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
                        imu_acc << imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z;
                        double dt = scan_time_ - last_imu_time_;
                        filter_->predict(dt, imu_acc, imu_gyr, true);
                        break;
                    }

                    V3D imu_acc, imu_gyr;
                    imu_gyr << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
                    imu_acc << imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z;
                    double dt = imu_time - last_imu_time_;
                    filter_->predict(dt, imu_acc, imu_gyr, true);
                    ++used_imu_msg;
                }

                imuBuf.erase(imuBuf.begin(), imuBuf.begin() + used_imu_msg);
                mBuf.unlock();

                scan_new_->setPointCloud(scan_time_, cornerPointsSharp, cornerPointsLessSharp, 
                                         surfPointsFlat, surfPointsLessFlat, laserCloudFullRes);

                TicToc t_opt;
                performIESKF();

                opt_time += t_opt.toc();
                printf("optimization time %f ms \n", opt_time / (frameCount + 1));

                integrateTransformation();
                filter_->reset(1);

                double roll, pitch;
                calculateRPfromIMU(filter_->state_.gn_, roll, pitch);
                correctRollPitch(roll, pitch);

                 // transform corner features and plane features to the scan end point
                updatePointCloud();
                transformFromInertialToLidar();

                swapScan();
                scan_new_.reset(new Scan());

                whole_odom_time += t_whole.toc();
                printf("whole odometry time %f ms +++++\n", whole_odom_time / (frameCount + 1));

                // publish odometry
                publishTopics();

                /*if(t_whole.toc() > 100)
                    ROS_WARN("odometry process over 100ms"); */

            }
            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}