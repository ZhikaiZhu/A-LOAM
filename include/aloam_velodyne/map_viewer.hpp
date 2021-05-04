
#ifndef _INCLUDE_MAP_VIEWER_HPP
#define _INCLUDE_MAP_VIEWER_HPP

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/console/parse.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>

// VTK include needed for drawing graph lines
#include <vtkPolyLine.h>
#include <vtkLine.h>

// boost
#include <boost/filesystem.hpp>
#include <boost/function.hpp>

#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/compute_sim3.hpp"

typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Matrix4ds;

enum color_type {
    FRAME = 0,
    HEIGHT = 1,
    INTENSITY1 = 2,
    INTENSITY2 = 3
};

std::string trim_str(std::string &str) {
    str.erase(0, str.find_first_not_of(" \t\r\n"));
    str.erase(str.find_last_not_of(" \t\r\n") + 1);
    return str;
}

bool write_pcd_file(const std::string &fileName, const pcl::PointCloud<PointType>::Ptr &pointCloud, bool as_binary) {
    pointCloud->width = 1;
    pointCloud->height = pointCloud->points.size();

    if (as_binary) {
        if (pcl::io::savePCDFileBinary(fileName, *pointCloud) == -1) {
            PCL_ERROR("Cloudn't write file\n");
            return false;
        }
    }
    else {
        if (pcl::io::savePCDFile(fileName, *pointCloud) == -1) {
            PCL_ERROR("Cloudn't write file\n");
            return false;
        }
    }
    return true;
}

bool read_pcd_file(const std::string &fileName, pcl::PointCloud<PointType>::Ptr &pointCloud) {
    if (pcl::io::loadPCDFile<PointType>(fileName, *pointCloud) == -1) {
        PCL_ERROR("Cloudn't read file\n");
        return false;
    }
    return true;
} 

template <typename Type>
bool write_pose_point_cloud(const std::string &fileName, const Type &poses) {
    pcl::PointCloud<PointType>::Ptr traj(new pcl::PointCloud<PointType>());

    for (size_t i = 0; i < poses.size(); i++) {
        PointType pt_tmp;
        pt_tmp.x = poses[i].translation(0);
        pt_tmp.y = poses[i].translation(1);
        pt_tmp.z = poses[i].translation(2);
        pt_tmp.intensity = i;
        traj->points.push_back(pt_tmp);
    }

    write_pcd_file(fileName, traj, false);
    return true;
}

class MapViewer {
    public:
        MapViewer() = default;

        MapViewer(double initial_background_color, bool is_seed_origin, double intensity_scale, 
                  color_type used_color_style, double laser_visualization_size, int refreshing_time_ms, 
                  int sleep_micro_second): background_color_(initial_background_color), 
        is_seed_origin_(is_seed_origin), intensity_scale_(intensity_scale), 
        color_rendering_type_(used_color_style), laser_visualization_size_(laser_visualization_size),
        refreshing_time_ms_(refreshing_time_ms), sleep_micro_second_(sleep_micro_second) {
            lo_pose_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
            gt_pose_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
        }

        ~MapViewer() {}

        static void keyboard_event_occurred(const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
            if (event.getKeySym() == "s" && event.keyDown()) {
                viewer->saveScreenshot("/home/spc/output/map.png");
            }
        }

        void set_interactive_events(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, int height, int width) {
            viewer->setBackgroundColor(background_color_, background_color_, background_color_);
            viewer->registerKeyboardCallback(keyboard_event_occurred, (void *)&viewer);
            viewer->setSize(height, width);
        }

        void get_random_color(double &r, double &g, double &b, double range_max) {
            r = range_max * (rand() / (1.0 + RAND_MAX));
            g = range_max * (rand() / (1.0 + RAND_MAX));
            b = range_max * (rand() / (1.0 + RAND_MAX));
        }

        double simple_look_up_table(double x, double x_a, double x_b, double y_a, double y_b) {
            double y;
            double k1 = y_a / x_a;
            double k2 = (y_b - y_a) / (x_b - x_a);
            double k3 = (1.0 - y_b) / (1.0 - x_b);

            if (x < x_a)
                y = k1 * x;
            else if (x <= x_b && x > x_a)
                y = y_a + k2 * (x - x_a);
            else if (x <= 1.0 && x > x_b)
                y = y_b + k3 * (x - x_b);
            else
                y = 0.0;

            return y;
        }

        void display_map_cloud(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, 
                               pcl::PointCloud<PointType>::Ptr &map_cloud) {
            std::string map_cloud_name = "map_cloud";

            switch(color_rendering_type_) {
                case FRAME: {   // random color
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_map_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                    double color_r, color_g, color_b;
                    get_random_color(color_r, color_g, color_b, 255);
                    for (size_t i = 0; i < map_cloud->points.size(); ++i) {
                        pcl::PointXYZRGB pt;
                        pt.x = map_cloud->points[i].x;
                        pt.y = map_cloud->points[i].y;
                        pt.z = map_cloud->points[i].z;
                        pt.r = color_r;
                        pt.g = color_g;
                        pt.b = color_b;
                        color_map_cloud->points.push_back(pt);
                    }
                    viewer->addPointCloud(color_map_cloud, map_cloud_name);
                    break;
                }
                case HEIGHT: {  // height color scalar
                    pcl::visualization::PointCloudColorHandlerGenericField<PointType> rgb_z(map_cloud, "z");
                    viewer->addPointCloud(map_cloud, rgb_z, map_cloud_name);
                    break;
                }
                case INTENSITY1: { // intensity color scalar
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_map_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                    double intensity_color;
                    for (size_t i = 0; i < map_cloud->points.size(); ++i) {
                        pcl::PointXYZRGB pt;
                        pt.x = map_cloud->points[i].x;
                        pt.y = map_cloud->points[i].y;
                        pt.z = map_cloud->points[i].z;
                        intensity_color = 0.2 + std::min(0.8, 1.0 / intensity_scale_ * map_cloud->points[i].intensity);
                        pt.r = 255 * intensity_color;
                        pt.g = 255 * intensity_color;
                        pt.b = 255 * intensity_color;
                        color_map_cloud->points.push_back(pt);
                    }
                    viewer->addPointCloud(color_map_cloud, map_cloud_name);
                    break;
                }
                case INTENSITY2: {
                     for (size_t i = 0; i < map_cloud->points.size(); ++i) {
                         map_cloud->points[i].intensity = intensity_scale_ * simple_look_up_table((intensity_scale_ - map_cloud->points[i].intensity) / intensity_scale_,
                                                                                                   intensity_lut_x_a, intensity_lut_x_b, intensity_lut_y_a, intensity_lut_y_b);
                     }
                     pcl::visualization::PointCloudColorHandlerGenericField<PointType> rgb_i(map_cloud, "intensity");
                     viewer->addPointCloud(map_cloud, rgb_i, map_cloud_name);
                }
                default:
                    break;
            }
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, map_cloud_name);

        }

        void add_lines_to_viewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, 
                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr &pose) {
            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
            vtkSmartPointer<vtkLine> line;
            vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
            vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();

            std::string lines_name = "traj";
            int line_num = pose->points.size() - 1;
            for (int i = 0; i <= line_num; ++i) {
                points->InsertNextPoint(pose->points[i].data);
            }

            // Add the points to the dataset
            polyData->SetPoints(points);

            colors->SetNumberOfComponents(3);

            unsigned char rgb[3];

            for (int i = 0; i < line_num; i++) {
                line = vtkSmartPointer<vtkLine>::New();
                line->GetPointIds()->SetNumberOfIds(2);
                line->GetPointIds()->SetId(0, i);
                line->GetPointIds()->SetId(1, i + 1);
                cells->InsertNextCell(line);
                rgb[0] = 255;
                rgb[1] = 255;
                rgb[2] = 0;
#if VTK_MAJOR_VERSION < 7
                colors->InsertNextTupleValue(rgb);
#else
                colors->InsertNextTypedTuple(rgb);
#endif
            }

            // Add the lines to the dataset
            polyData->SetLines(cells);
            // Add the color
            polyData->GetCellData()->SetScalars(colors);

            viewer->addModelFromPolyData(polyData, lines_name);
            
        }

        template <typename Type>
        void display_trajectory(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, 
                                const Type &pose_vec1, const Type &pose_vec2, bool show_gt) {
            std::string lo_pose_name = "lo_pose";
            double sphere_size = laser_visualization_size_;
            double font_color = 1.0 - background_color_;

            pcl::PointXYZ lo_origin(pose_vec1[0].translation(0), pose_vec1[0].translation(1), pose_vec1[0].translation(2));
            
            if (is_seed_origin_) {
                viewer->addSphere(lo_origin, sphere_size, font_color, font_color, font_color, "lo_origin");
            }

            for (size_t i = 1; i < pose_vec1.size(); ++i) {
                pcl::PointXYZ pt_tmp;
                pt_tmp.x = pose_vec1.at(i).translation(0);
                pt_tmp.y = pose_vec1.at(i).translation(1);
                pt_tmp.z = pose_vec1.at(i).translation(2);
                lo_pose_->points.push_back(pt_tmp);
            }

            /* char str[20];
            for (int j = 0; j < pose_vec1.size() - 2; ++j) {
                sprintf(str, "%d", j);
                viewer->addLine(lo_pose_->points[j], lo_pose_->points[j + 1], str);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, str);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 255, 0, str);
            } */

            add_lines_to_viewer(viewer, lo_pose_);

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> lo_pose_color(lo_pose_, 255, 255, 0); 
            viewer->addPointCloud(lo_pose_, lo_pose_color, lo_pose_name);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, lo_pose_name);

            if (show_gt) {
                std::string gt_pose_name = "gt_pose";
                pcl::PointXYZ gt_origin(pose_vec2[0].translation(0), pose_vec2[0].translation(1), pose_vec2[0].translation(2));

                if (is_seed_origin_) {
                    viewer->addSphere(gt_origin, sphere_size, 0.0, 255.0, 0.0, "gt_origin");
                }

                for (size_t i = 1; i < pose_vec2.size(); ++i) {
                    pcl::PointXYZ pt_tmp;
                    pt_tmp.x = pose_vec2.at(i).translation(0);
                    pt_tmp.y = pose_vec2.at(i).translation(1);
                    pt_tmp.z = pose_vec2.at(i).translation(2);
                    gt_pose_->points.push_back(pt_tmp);
                }

                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> gt_pose_color(gt_pose_, 0, 255, 0);
                viewer->addPointCloud(gt_pose_, gt_pose_color, gt_pose_name);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, gt_pose_name);
            }

            while (!viewer->wasStopped()) {
                viewer->spinOnce(refreshing_time_ms_);
                boost::this_thread::sleep(boost::posix_time::microseconds(sleep_micro_second_));
            }
            //viewer->spin();
        }

    private:
        double background_color_;
        bool is_seed_origin_;
        double intensity_scale_;

        color_type color_rendering_type_;

        double laser_visualization_size_;
        int refreshing_time_ms_;
        int sleep_micro_second_;

        double intensity_lut_x_a = 0.65;
        double intensity_lut_x_b = 0.97;
        double intensity_lut_y_a = 0.05;
        double intensity_lut_y_b = 0.6;

        typename pcl::PointCloud<pcl::PointXYZ>::Ptr lo_pose_;
        typename pcl::PointCloud<pcl::PointXYZ>::Ptr gt_pose_;
};

#endif // _INCLUDE_MAP_VIEWER_HPP