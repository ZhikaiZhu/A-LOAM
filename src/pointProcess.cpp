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

using std::atan2;
using std::cos;
using std::sin;

std::string LIDAR_TOPIC;
double MINIMUM_RANGE = 0.5, SCAN_PERIOD = 0.1;
int N_SCANS = 16, POINT_FILTER_NUM = 1;

struct EIGEN_ALIGN16 PointXYZITR {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZITR,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

class Pointprocess {
private:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;

    ros::Publisher pubLaserCloud;
    ros::Publisher pubSurfPoints;

    bool given_offset_time;

    double whole_time = 0.0;
    int frameCount = 0;

public:
    Pointprocess(ros::NodeHandle &nh_): nh(nh_) {
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(LIDAR_TOPIC, 100, &Pointprocess::cloudHandler, this);

        pubSurfPoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100);
        //pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_full", 100);
    }

    ~Pointprocess() = default;

    template <typename PointT> 
    void removeNaNPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, std::vector<int> &index)
    {
        // If the clouds are not the same, prepare the output
        if (&cloud_in != &cloud_out) {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize (cloud_in.points.size ());
            cloud_out.sensor_origin_ = cloud_in.sensor_origin_;
            cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
        }
        // Reserve enough space for the indices
        index.resize(cloud_in.points.size());

        // If the data is dense, we don't need to check for NaN
        if (cloud_in.is_dense) {
            // Simply copy the data
            cloud_out = cloud_in;
            for (std::size_t j = 0; j < cloud_out.points.size(); ++j)
            index[j] = static_cast<int>(j);
        }
        else {
            std::size_t j = 0;
            for (std::size_t i = 0; i < cloud_in.points.size(); ++i) {
                if (!std::isfinite (cloud_in.points[i].x) ||
                    !std::isfinite (cloud_in.points[i].y) ||
                    !std::isfinite (cloud_in.points[i].z))
                    continue;
                cloud_out.points[j] = cloud_in.points[i];
                index[j] = static_cast<int>(i);
                j++;
            }
            if (j != cloud_in.points.size()) {
                // Resize to the correct size
                cloud_out.points.resize (j);
                index.resize (j);
            }

            cloud_out.height = 1;
            cloud_out.width  = static_cast<std::uint32_t>(j);

            // Removing bad points => dense (note: 'dense' doesn't mean 'organized')
            cloud_out.is_dense = true;
        }
    }

    void cloudHandler( const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        TicToc t_whole;

        pcl::PointCloud<PointXYZITR> laserCloudIn;
        pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
        std::vector<int> indices;

        removeNaNPointCloud(laserCloudIn, laserCloudIn, indices);

        int cloudSize = laserCloudIn.points.size();
        pcl::PointCloud<PointType> surfPoints;
        surfPoints.reserve(cloudSize);

        /*** These variables only works when no point timestamps given ***/
        double omega_l = 3.61;       // scan angular velocity
        std::vector<bool> is_first(N_SCANS, true);
        std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
        std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
        std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
        /*****************************************************************/

        if (laserCloudIn.points[cloudSize - 1].time > 0) {
            given_offset_time = true;
        }
        else {
            given_offset_time = false;
        }
        //given_offset_time = false;

        for (int i = 0; i < cloudSize; i++) {
            PointType point;
            
            point.normal_x = 0;
            point.normal_y = 0;
            point.normal_z = 0;
            point.x = laserCloudIn.points[i].x;
            point.y = laserCloudIn.points[i].y;
            point.z = laserCloudIn.points[i].z;
            point.intensity = laserCloudIn.points[i].intensity;
            point.curvature = laserCloudIn.points[i].time * 1000.0;  // curvature unit: ms

            if (!given_offset_time) {
                int layer = laserCloudIn.points[i].ring;
                double yaw_angle = atan2(point.y, point.x) * 57.2957;

                if (is_first[layer]) {
                    // printf("layer: %d; is first: %d", layer, is_first[layer]);
                    yaw_fp[layer] = yaw_angle;
                    is_first[layer] = false;
                    point.curvature = 0.0;
                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = point.curvature;
                    continue;
                }

                // compute offset time
                if (yaw_angle <= yaw_fp[layer]) {
                    point.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
                }
                else {
                    point.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
                }

                if (point.curvature < time_last[layer])  point.curvature += 360.0 / omega_l;

                yaw_last[layer] = yaw_angle;
                time_last[layer] = point.curvature;
                
            }

            if (i % POINT_FILTER_NUM == 0) {
                if (point.x * point.x + point.y * point.y + point.z * point.z > MINIMUM_RANGE * MINIMUM_RANGE) {
                    surfPoints.points.push_back(point);
                }
            }
        }

        // cloudSize = count;
        // printf("points size %d \n", cloudSize);

        sensor_msgs::PointCloud2 surfPointsMsg;
        pcl::toROSMsg(surfPoints, surfPointsMsg);
        surfPointsMsg.header.stamp = laserCloudMsg->header.stamp;
        surfPointsMsg.header.frame_id = "camera_init";
        pubSurfPoints.publish(surfPointsMsg);

        //t_pre.tic_toc();
        whole_time += t_whole.toc();
        ++frameCount;
        printf("scan registration average time %f ms *************\n", whole_time / frameCount);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointProcess");
    ros::NodeHandle nh;

    nh.param<int>("point_filter_num", POINT_FILTER_NUM, 2);
    nh.param<std::string>("common/lid_topic", LIDAR_TOPIC, "/velodyne_points");
    nh.param<int>("preprocess/n_scans", N_SCANS, 16);
    nh.param<double>("preprocess/minimum_range", MINIMUM_RANGE, 0.5);
    nh.param<double>("preprocess/scan_period", SCAN_PERIOD, 0.1);

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
