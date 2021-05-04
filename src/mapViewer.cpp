#include "aloam_velodyne/map_viewer.hpp"
#include <ros/ros.h>


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "mapViewer");
    ros::NodeHandle nh;

    double background_color;
    bool is_seed_origin;
    double intensity_scale;
    int used_color;
    double laser_visualization_size;
    int refreshing_time_ms;
    int sleep_micro_second;
    std::string output_path;
    int window_height, window_width;
    bool show_gt;
    nh.param<double>("background_color", background_color, 0.0);
    nh.param<bool>("is_seed_origin", is_seed_origin, true);
    nh.param<double>("intensity_scale", intensity_scale, 255.0);
    nh.param<int>("used_color", used_color, 1);
    nh.param<double>("laser_visualization_size", laser_visualization_size, 0.5);
    nh.param<int>("refreshing_time_ms", refreshing_time_ms, 100);
    nh.param<int>("sleep_micro_second", sleep_micro_second, 100);
    nh.param<std::string>("output_path", output_path, "~/output");
    nh.param<int>("window_height", window_height, 640);
    nh.param<int>("window_width", window_width, 480);
    nh.param<bool>("show_gt", show_gt, false);

    boost::shared_ptr<MapViewer> Viewer(new MapViewer(background_color, is_seed_origin, intensity_scale, color_type(used_color), 
                                                      laser_visualization_size, refreshing_time_ms, sleep_micro_second));
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Map View"));
    pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
    std::string map_pcd_path = output_path + "/map.pcd";  
    std::string lo_pose_path = output_path + "/lio_mapped.csv";
    std::string gt_pose_path = output_path + "/ss_01_70s_gt.csv";
    read_pcd_file(map_pcd_path, map_cloud);

    ComputeSim3<double> computeSim3(0.01);
    ComputeSim3<double>::Traj traj_1, traj_2;
    computeSim3.LoadTraj(lo_pose_path, gt_pose_path, show_gt);
    computeSim3.getTraj(traj_1, traj_2, show_gt);
    if (show_gt) {
        ComputeSim3<double>::Traj aligned_traj_1;
        aligned_traj_1 = traj_1;

        computeSim3.syncTraj();
        Eigen::Matrix4d sim3 = computeSim3.GetSim3();
        for (size_t i = 0; i < traj_1.size(); ++i) {
            aligned_traj_1[i].translation = sim3.block<3, 3>(0, 0) * traj_1[i].translation + sim3.block<3, 1>(0, 3);
        }

        pcl::PointCloud<PointType>::Ptr trans_map_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*map_cloud, *trans_map_cloud, sim3);
        Viewer->set_interactive_events(viewer, window_height, window_width);
        Viewer->display_map_cloud(viewer, trans_map_cloud);
        Viewer->display_trajectory(viewer, aligned_traj_1, traj_2, show_gt);
    }
    else {
        Viewer->set_interactive_events(viewer, window_height, window_width);
        Viewer->display_map_cloud(viewer, map_cloud);
        Viewer->display_trajectory(viewer, traj_1, traj_2, show_gt);
    }
       
}