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
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "aloam_velodyne/parameters.h"

using namespace parameter;
using std::atan2;
using std::cos;
using std::sin;


const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
float cloudCurvature[100000];
int cloudSortInd[100000];
int cloudNeighborPicked[100000];
int cloudLabel[100000];

pcl::PointCloud<PointType>::Ptr fullCloud;
pcl::PointCloud<PointType>::Ptr groundCloud;
pcl::PointCloud<PointType>::Ptr segmentedCloud;
pcl::PointCloud<PointType>::Ptr laserCloud;

cv::Mat rangeMat;
cv::Mat labelMat;
cv::Mat groundMat;
int labelCount;
std::vector<bool> segmentedCloudGroundFlag;
std::vector<int> scanStartInd;
std::vector<int> scanEndInd;

std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

uint16_t* allPushedIndX;
uint16_t* allPushedIndY;

uint16_t* queueIndX;
uint16_t* queueIndY;

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubGroundCloud;
ros::Publisher pubSegmentedCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

void initialization() {
    fullCloud.reset(new pcl::PointCloud<PointType>());
    groundCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloud.reset(new pcl::PointCloud<PointType>());
    laserCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCANS * SCAN_NUM);
    segmentedCloudGroundFlag.assign(N_SCANS * SCAN_NUM, false);
    scanStartInd.assign(N_SCANS, 0);
    scanEndInd.assign(N_SCANS, 0);

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = 1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = -1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);

    allPushedIndX = new uint16_t[N_SCANS * SCAN_NUM];
    allPushedIndY = new uint16_t[N_SCANS * SCAN_NUM];

    queueIndX = new uint16_t[N_SCANS * SCAN_NUM];
    queueIndY = new uint16_t[N_SCANS * SCAN_NUM];

}

void resetParameters() {
    groundCloud->clear();
    segmentedCloud->clear();
    laserCloud->clear();

    rangeMat = cv::Mat(N_SCANS, SCAN_NUM, CV_32F, cv::Scalar::all(FLT_MAX));
    groundMat = cv::Mat(N_SCANS, SCAN_NUM, CV_8S, cv::Scalar::all(0));
    labelMat = cv::Mat(N_SCANS, SCAN_NUM, CV_32S, cv::Scalar::all(0));
    labelCount = 1;

    PointType nanPoint;
    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;
    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
}

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres1, float thres2)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        float dis = std::sqrt(cloud_in.points[i].x * cloud_in.points[i].x + 
                              cloud_in.points[i].y * cloud_in.points[i].y + 
                              cloud_in.points[i].z + cloud_in.points[i].z);
        if ( dis < thres1 || dis > thres2)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void groundRemoval() {
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < SCAN_NUM; ++j) {
        for (size_t i = 0; i < groundScanInd; ++i) {
            lowerInd = j + i * SCAN_NUM;
            upperInd = j + (i + 1) * SCAN_NUM;

            if (fullCloud->points[lowerInd].intensity == -1 ||
                fullCloud->points[upperInd].intensity == -1) {
                groundMat.at<int8_t>(i, j) = -1;
                continue;
            }

            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

            if (abs(angle - sensorMountAngle) <= 10) {
                groundMat.at<int8_t>(i, j) = 1;
                groundMat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }

    for (size_t i = 0; i < N_SCANS; ++i) {
        for (size_t j = 0; j < SCAN_NUM; ++j) {
            if (groundMat.at<int8_t>(i, j) == 1 || rangeMat.at<float>(i, j) == FLT_MAX) {
                labelMat.at<int>(i, j) = -1;
            }
        }
    }
    if (pubGroundCloud.getNumSubscribers() != 0) {
        for (size_t i = 0; i <= groundScanInd; ++i) {
            for (size_t j = 0; j < SCAN_NUM; ++j) {
                if (groundMat.at<int8_t>(i, j) == 1)
                    groundCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
            }
        }
    }
}

void labelComponents(int row, int col) {
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY;
    bool lineCountFlag[N_SCANS] = {false};

    queueIndX[0] = row;
    queueIndY[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;

    while (queueSize > 0) {
        fromIndX = queueIndX[queueStartInd];
        fromIndY = queueIndY[queueStartInd];
        --queueSize;
        ++queueStartInd;
        labelMat.at<int>(fromIndX, fromIndY) = labelCount;

        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter) {
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;

            if (thisIndX < 0 || thisIndX >= N_SCANS) continue;

            if (thisIndY < 0) thisIndY = SCAN_NUM - 1;
            if (thisIndY >= SCAN_NUM) thisIndY = 0;

            if (labelMat.at<int>(thisIndX, thisIndY) != 0) continue;

            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                          rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                          rangeMat.at<float>(thisIndX, thisIndY));

            if ((*iter).first == 0)
                alpha = segmentAlphaX;
            else
                alpha = segmentAlphaY;

            angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

            if (angle > segmentTheta) {
                queueIndX[queueEndInd] = thisIndX;
                queueIndY[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;

                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;

                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;
            }
        }
    }

    bool feasibleSegment = false;
    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum) {
        int lineCount = 0;
        for (size_t i = 0; i < N_SCANS; ++i)
            if (lineCountFlag[i] == true) ++lineCount;
        if (lineCount >= segmentValidLineNum) 
            feasibleSegment = true;
    }

    if (feasibleSegment == true) {
        ++labelCount;
    } 
    else {
        for (size_t i = 0; i < allPushedIndSize; ++i) {
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
}

void cloudSegmentation() {
    for (size_t i = 0; i < N_SCANS; ++i)
        for (size_t j = 0; j < SCAN_NUM; ++j)
            if (labelMat.at<int>(i, j) == 0) labelComponents(i, j);

    int sizeOfSegCloud = 0;
    for (size_t i = 0; i < N_SCANS; ++i) {
        scanStartInd[i] = sizeOfSegCloud + 5;

        for (size_t j = 0; j < SCAN_NUM; ++j) {
            if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1) {
                if (labelMat.at<int>(i, j) == 999999) {
                    if (i > groundScanInd && j % 5 == 0) {
                        laserCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
                        ++sizeOfSegCloud;
                        continue;
                    } else {
                        continue;
                    }
                }
                if (groundMat.at<int8_t>(i, j) == 1) {
                    if (j % 5 != 0 && j > 5 && j < SCAN_NUM - 5) continue;
                }
                segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i, j) == 1);
                laserCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
                ++sizeOfSegCloud;
            }
        }

        scanEndInd[i] = sizeOfSegCloud - 1 - 5;
    }

    if (pubSegmentedCloud.getNumSubscribers() != 0) {
        for (size_t i = 0; i < N_SCANS; ++i) {
            for (size_t j = 0; j < SCAN_NUM; ++j) {
                if (labelMat.at<int>(i, j) > 0 && labelMat.at<int>(i, j) != 999999) {
                    segmentedCloud->push_back(fullCloud->points[j + i * SCAN_NUM]);
                    segmentedCloud->points.back().intensity = labelMat.at<int>(i, j);
                }
            }
        }
    }
}


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    //std::vector<int> scanStartInd(N_SCANS, 0);
    //std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
    std::vector<int> indices;

    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
    removeClosedPointCloud(*laserCloudIn, *laserCloudIn, MINIMUM_RANGE, MAXIMUM_RANGE);

    int cloudSize = laserCloudIn->points.size();
    float startOri = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
    float endOri = -atan2(laserCloudIn->points[cloudSize - 1].y,
                          laserCloudIn->points[cloudSize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn->points[i].x;
        point.y = laserCloudIn->points[i].y;
        point.z = laserCloudIn->points[i].z;

        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        float relTime = (ori - startOri) / (endOri - startOri);
        point.intensity = scanID + SCAN_PERIOD * relTime;
        laserCloudScans[scanID].push_back(point); 
    }
    
    cloudSize = count;
    //printf("points size %d \n", cloudSize);

    if (!OUT_DOOR) {
        for (int i = 0; i < N_SCANS; i++) { 
            scanStartInd[i] = laserCloud->size() + 5;
            *laserCloud += laserCloudScans[i];
            scanEndInd[i] = laserCloud->size() - 6;
        }
    }
    else {
        pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCANS; i++) { 
            *tmpCloud += laserCloudScans[i];
        }

        // project cloud to image
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index;
        PointType thisPoint;

        for (size_t i = 0; i < cloudSize; ++i) {
            thisPoint.x = tmpCloud->points[i].x;
            thisPoint.y = tmpCloud->points[i].y;
            thisPoint.z = tmpCloud->points[i].z;
            thisPoint.intensity = tmpCloud->points[i].intensity;

            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x +
                                                    thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            if (rowIdn < 0 || rowIdn >= N_SCANS) continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + SCAN_NUM / 2;
            if (columnIdn >= SCAN_NUM) columnIdn -= SCAN_NUM;

            if (columnIdn < 0 || columnIdn >= SCAN_NUM) continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y +
                        thisPoint.z * thisPoint.z);
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            //thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn + rowIdn * SCAN_NUM;
            fullCloud->points[index] = thisPoint;
        }

        groundRemoval();

        cloudSegmentation();
        cloudSize = laserCloud->points.size();
    }
    
    //printf("prepare time %f \n", t_prepare.toc());

    for (int i = 5; i < cloudSize - 5; i++)
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

        cloudCurvature[i] = diff;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;

        if (diff > 0.1) {
            float depth1 = std::sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                                     laserCloud->points[i].y * laserCloud->points[i].y + 
                                     laserCloud->points[i].z * laserCloud->points[i].z);
            float depth2 = std::sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y + 
                                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);
            if (depth1 > depth2) {
                diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
                diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
                diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

                if (std::sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
            }
            else {
                diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
                diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
                diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

                if (std::sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
        }
        float diffX1 = laserCloud->points[i].x - laserCloud->points[i + 1].x;
        float diffY1 = laserCloud->points[i].y - laserCloud->points[i + 1].y;
        float diffZ1 = laserCloud->points[i].z - laserCloud->points[i + 1].z;
        float diff1 = diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1;
        float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
        float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
        float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
        float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;
        float dis = laserCloud->points[i].x * laserCloud->points[i].x + 
                    laserCloud->points[i].y * laserCloud->points[i].y + 
                    laserCloud->points[i].z * laserCloud->points[i].z;
        
        if (diff1 > 0.0002 * dis && diff2 > 0.0002 * dis) {
            cloudNeighborPicked[i] = 1;
        }
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k]; 

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > EDGE_THRESHOLD &&
                    segmentedCloudGroundFlag[ind] == false)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 20)
                    {                        
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; 

                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < SURF_THRESHOLD)
                {

                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
   
    //printf("sort q time %f \n", t_q_sort);
    //printf("seperate points time %f \n", t_pts.toc());


    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    sensor_msgs::PointCloud2 laserCloudTemp;
    if (pubGroundCloud.getNumSubscribers() != 0) {
        pcl::toROSMsg(*groundCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = laserCloudMsg->header.stamp;
        laserCloudTemp.header.frame_id = "/camera_init";
        pubGroundCloud.publish(laserCloudTemp);
    }

    if (pubSegmentedCloud.getNumSubscribers() != 0) {
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = laserCloudMsg->header.stamp;
        laserCloudTemp.header.frame_id = "/camera_init";
        pubSegmentedCloud.publish(laserCloudTemp);
    }

    // pub each scam
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    resetParameters();

    //printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh("~");

    parameter::readParameters(nh);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    initialization();

    resetParameters();

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(LIDAR_TOPIC, 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 100);

    pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}