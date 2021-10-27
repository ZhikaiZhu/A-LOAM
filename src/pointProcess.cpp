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

class Pointprocess {
private:
    float cloudCurvature[400000];
    int cloudSortInd[400000];
    int cloudNeighborPicked[400000];
    int cloudLabel[400000];

    bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]);}

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subImu;

    ros::Publisher pubLaserCloud;
    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;

    std::vector<sensor_msgs::ImuConstPtr> imuBuf;
    size_t ind_imu = 0;
    double current_time_imu = -1;

    Eigen::Vector3d gyr_prev;
    Eigen::Quaterniond q_from_imu;
    Eigen::Vector3d r_from_imu;
    bool first_imu = false;

    std::deque<sensor_msgs::PointCloud2> pclBuf;
    sensor_msgs::PointCloud2 currentCloudMsg;
    double scan_time_next;

    Eigen::Vector3d t_b_l;
    Eigen::Quaterniond q_b_l;

    double whole_time = 0.0;
    int frameCount = 0;

public:
    Pointprocess(ros::NodeHandle &nh_): nh(nh_) {
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(LIDAR_TOPIC, 100, &Pointprocess::cloudHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(IMU_TOPIC, 200, &Pointprocess::imuHandler, this);

        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
        pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

        q_from_imu.setIdentity();
        r_from_imu.setZero();

        q_b_l = INIT_RBL;
        t_b_l = INIT_TBL;
    }

    ~Pointprocess() = default;

    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                pcl::PointCloud<PointT> &cloud_out, float thres1, float thres2) {
        if (&cloud_in != &cloud_out) {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i) {
            float dis = std::sqrt(cloud_in.points[i].x * cloud_in.points[i].x + 
                                  cloud_in.points[i].y * cloud_in.points[i].y + 
                                  cloud_in.points[i].z * cloud_in.points[i].z);
            if ( dis < thres1 || dis > thres2)
                continue;
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
        if (j != cloud_in.points.size()) {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    void undistortion(PointType const *const pi, PointType *const po) {
        double s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
        if (s >= 1.0) {
            s = 1.0;
        }

        Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
        Eigen::Quaterniond q_start = q0.slerp(s, q_from_imu);

        Eigen::Vector3d point(pi->x, pi->y, pi->z);
        //Eigen::Vector3d pt_start = q_b_l.inverse() * q_start * q_b_l * point; // transform to start of a scan

        // transform to end of a scan
        Eigen::Vector3d pt_end = q_b_l.inverse() * q_from_imu.inverse() * q_start * q_b_l * point; 

        po->x = pt_end.x();
        po->y = pt_end.y();
        po->z = pt_end.z();
        po->intensity = pi->intensity;
    }

    void solveRotation(double dt, Eigen::Vector3d gyr_curr)
    {
        Eigen::Vector3d gyr_mean = 0.5 * (gyr_prev + gyr_curr);
        q_from_imu *= utils::deltaQ(gyr_mean * dt);
        gyr_prev = gyr_curr;
    }

    void processIMU(double t_cur)
    {
        double rx = 0, ry = 0, rz = 0;
        size_t i = ind_imu;
        if(i >= imuBuf.size())
            i--;
        while(imuBuf[i]->header.stamp.toSec() < t_cur) {

            double t = imuBuf[i]->header.stamp.toSec();
            if (current_time_imu < 0)
                current_time_imu = t;
            double dt = t - current_time_imu;
            current_time_imu = imuBuf[i]->header.stamp.toSec();

            rx = imuBuf[i]->angular_velocity.x;
            ry = imuBuf[i]->angular_velocity.y;
            rz = imuBuf[i]->angular_velocity.z;
            solveRotation(dt, Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imuBuf.size())
                break;
        }

        if(i < imuBuf.size()) {
            double dt1 = t_cur - current_time_imu;
            double dt2 = imuBuf[i]->header.stamp.toSec() - t_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            rx = w1 * rx + w2 * imuBuf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imuBuf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imuBuf[i]->angular_velocity.z;
            solveRotation(dt1, Eigen::Vector3d(rx, ry, rz));
        }
        current_time_imu = t_cur;
        ind_imu = i;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& ImuIn)
    {
        imuBuf.push_back(ImuIn);

        if(imuBuf.size() > 600)
            imuBuf[imuBuf.size() - 601] = nullptr;

        if (current_time_imu < 0)
            current_time_imu = ImuIn->header.stamp.toSec();

        if (!first_imu)
        {
            first_imu = true;
            double rx = 0, ry = 0, rz = 0;
            rx = ImuIn->angular_velocity.x;
            ry = ImuIn->angular_velocity.y;
            rz = ImuIn->angular_velocity.z;
            Eigen::Vector3d gyr_cur(rx, ry, rz);
            gyr_prev = gyr_cur;
        }
    }


    void cloudHandler( const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // cache point cloud
        pclBuf.push_back(*laserCloudMsg);
        if (pclBuf.size() <= 2)
            return;
        else {
            currentCloudMsg = pclBuf.front();
            pclBuf.pop_front();

            scan_time_next = pclBuf.front().header.stamp.toSec();
        }

        if (IMU_DESKEW) {
            size_t tmpInd = 0;
            if(ind_imu > 0)
                tmpInd = ind_imu - 1;
            if (imuBuf.empty() || imuBuf[tmpInd]->header.stamp.toSec() > scan_time_next) {
                ROS_WARN("Waiting for IMU data ...");
                return;
            }
        }

        TicToc t_whole;

        std::vector<int> scanStartInd(N_SCANS, 0);
        std::vector<int> scanEndInd(N_SCANS, 0);

        pcl::PointCloud<PointType> laserCloudIn;
        pcl::fromROSMsg(currentCloudMsg, laserCloudIn);
        std::vector<int> indices;

        pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE, MAXIMUM_RANGE);


        int cloudSize = laserCloudIn.points.size();
        float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                laserCloudIn.points[cloudSize - 1].x) +
                2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
            endOri -= 2 * M_PI;

        else if (endOri - startOri < M_PI)
            endOri += 2 * M_PI;


        if (IMU_DESKEW) {
            if(first_imu)
                processIMU(scan_time_next);
            if(std::isnan(q_from_imu.w()) || std::isnan(q_from_imu.x()) || 
            std::isnan(q_from_imu.y()) || std::isnan(q_from_imu.z())) {
                q_from_imu = Eigen::Quaterniond::Identity();
            }
        }

        bool halfPassed = false;
        int count = cloudSize;
        PointType point;
        PointType point_undis;
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
        for (int i = 0; i < cloudSize; i++) {
            point.x = laserCloudIn.points[i].x;
            point.y = laserCloudIn.points[i].y;
            point.z = laserCloudIn.points[i].z;
            point.intensity = 0.1 * laserCloudIn.points[i].intensity;


            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (N_SCANS == 16) {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (N_SCANS - 1) || scanID < 0) {
                    count--;
                    continue;
                }
            }
            else if (N_SCANS == 32) {
                scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                if (scanID > (N_SCANS - 1) || scanID < 0) {
                    count--;
                    continue;
                }
            }
            else if (N_SCANS == 64) {
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                // use [0 50]  > 50 remove outlies
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
                    count--;
                    continue;
                }
            }
            else {
                printf("wrong scan number\n");
                ROS_BREAK();
            }

            float ori = -atan2(point.y, point.x);
            if (!halfPassed) {
                if (ori < startOri - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > startOri + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - startOri > M_PI)
                    halfPassed = true;
            }
            else {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > endOri + M_PI / 2)
                    ori -= 2 * M_PI;
            }

            float relTime = (ori - startOri) / (endOri - startOri);
            point.intensity = scanID + SCAN_PERIOD * relTime;

            if (IMU_DESKEW) {
                undistortion(&point, &point_undis);
                laserCloudScans[scanID].push_back(point_undis);
            }
            else {
                laserCloudScans[scanID].push_back(point); 
            }    
        }

        cloudSize = count;
        // printf("points size %d \n", cloudSize);

        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCANS; i++) {
            scanStartInd[i] = laserCloud->size() + 5;
            *laserCloud += laserCloudScans[i];
            scanEndInd[i] = laserCloud->size() - 6;
        }


        for (int i = 5; i < cloudSize - 5; i++) {
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudCurvature[i] = diff;
            cloudSortInd[i] = i;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            float diffX1 = laserCloud->points[i].x - laserCloud->points[i + 1].x;
            float diffY1 = laserCloud->points[i].y - laserCloud->points[i + 1].y;
            float diffZ1 = laserCloud->points[i].z - laserCloud->points[i + 1].z;
            float diff1 = diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1;
            if (!OUT_DOOR) {
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
        }

        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat;

        for (int i = 0; i < N_SCANS; i++) {
            if( scanEndInd[i] - scanStartInd[i] < 6 || i % DOWN_FILTER_SIZE != 0)
                continue;
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
            for (int j = 0; j < 6; j++) {
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                auto bound_comp = boost::bind(&Pointprocess::comp, this, _1, _2);
                std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, bound_comp);

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSortInd[k];

                    if (cloudNeighborPicked[ind] == 0 &&
                            cloudCurvature[ind] > EDGE_THRESHOLD) {

                        largestPickedNum++;
                        if (largestPickedNum <= 2) {
                            cloudLabel[ind] = 2;
                            cornerPointsSharp.push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else
                            break;

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSortInd[k];

                    if (cloudNeighborPicked[ind] == 0 &&
                            cloudCurvature[ind] < SURF_THRESHOLD) {

                        cloudLabel[ind] = -1;
                        surfPointsFlat.push_back(laserCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4)
                            break;

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0)
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }

            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
            downSizeFilter.filter(surfPointsLessFlatScanDS);

            surfPointsLessFlat += surfPointsLessFlatScanDS;
        }

        sensor_msgs::PointCloud2 laserCloudOutMsg;
        pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = currentCloudMsg.header.stamp;
        laserCloudOutMsg.header.frame_id = "/camera_init";
        pubLaserCloud.publish(laserCloudOutMsg);

        sensor_msgs::PointCloud2 cornerPointsSharpMsg;
        pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
        cornerPointsSharpMsg.header.stamp = currentCloudMsg.header.stamp;
        cornerPointsSharpMsg.header.frame_id = "/camera_init";
        pubCornerPointsSharp.publish(cornerPointsSharpMsg);

        sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
        pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
        cornerPointsLessSharpMsg.header.stamp = currentCloudMsg.header.stamp;
        cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
        pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

        sensor_msgs::PointCloud2 surfPointsFlat2;
        pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
        surfPointsFlat2.header.stamp = currentCloudMsg.header.stamp;
        surfPointsFlat2.header.frame_id = "/camera_init";
        pubSurfPointsFlat.publish(surfPointsFlat2);

        sensor_msgs::PointCloud2 surfPointsLessFlat2;
        pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
        surfPointsLessFlat2.header.stamp = currentCloudMsg.header.stamp;
        surfPointsLessFlat2.header.frame_id = "/camera_init";
        pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

        q_from_imu.setIdentity();
        r_from_imu.setZero();
        //t_pre.tic_toc();
        whole_time += t_whole.toc();
        ++frameCount;
        printf("scan registration average time %f ms *************\n", whole_time / frameCount);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointProcess");
    ros::NodeHandle nh("~");

    parameter::readParameters(nh);

    printf("scan line number %d \n", N_SCANS);

    if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    Pointprocess process(nh);
    ROS_INFO("\033[1;32m---->\033[0m Pointprocess Started.");

    ros::spin();
    return 0;
}
