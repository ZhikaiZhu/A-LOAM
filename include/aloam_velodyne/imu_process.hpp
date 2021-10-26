#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/filter_state.hpp"

using namespace filter;

/// *************Preconfiguration

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();
    
    void Reset();
    void set_extrinsic(const V3D &transl, const Q4D &rot);
    void set_gyr_cov(const double &gyr_n);
    void set_acc_cov(const double &acc_n);
    void set_gyr_bias_cov(const double &gyr_w);
    void set_acc_bias_cov(const double &acc_w);
    Eigen::Matrix<double, 12, 12> Q;
    void Process(const MeasureGroup &meas,  StatePredictor &filterEstimator, pcl::PointCloud<PointType>::Ptr pcl_un_);

    double acc_noise;
    double gyr_noise;
    double acc_bias_noise;
    double gyr_bias_noise;
    double first_lidar_time;
    bool is_undistort;

private:
    void IMU_init(const MeasureGroup &meas, StatePredictor &filterEstimator, int &N);
    void UndistortPcl(const MeasureGroup &meas, StatePredictor &filterEstimator, pcl::PointCloud<PointType> &pcl_in_out);

    pcl::PointCloud<PointType>::Ptr cur_pcl_un_;
    sensor_msgs::ImuConstPtr last_imu_;
    std::deque<sensor_msgs::ImuConstPtr> v_imu_;
    std::vector<Pose6D> IMUpose;
    Q4D q_b_l;
    V3D t_b_l;
    V3D mean_acc;
    V3D mean_gyr;
    V3D angvel_last;
    V3D accvel_last;
    double last_lidar_end_time_;
    int    init_iter_num = 1;
    bool   b_first_frame_ = true;
    bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess(): b_first_frame_(true), imu_need_init_(true)
{
    init_iter_num  = 1;
    acc_noise      = 0.1;
    gyr_noise      = 0.1;
    acc_bias_noise = 0.0001;
    gyr_bias_noise = 0.0001;
    mean_acc       = V3D(0, 0, -1.0);
    mean_gyr       = V3D(0, 0, 0);
    angvel_last.setZero();
    t_b_l.setZero();
    q_b_l.setIdentity();
    last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() = default;

void ImuProcess::Reset() 
{
    // ROS_WARN("Reset ImuProcess");
    imu_need_init_ = true;
    init_iter_num  = 1;
    mean_acc       = V3D(0, 0, -1.0);
    mean_gyr       = V3D(0, 0, 0);
    angvel_last.setZero();
    v_imu_.clear();
    IMUpose.clear();
    last_imu_.reset(new sensor_msgs::Imu());
    cur_pcl_un_.reset(new pcl::PointCloud<PointType>());
}

void ImuProcess::set_extrinsic(const V3D &transl, const Q4D &rot)
{
    t_b_l = transl;
    q_b_l = rot;
}

void ImuProcess::set_gyr_cov(const double &gyr_n)
{
    gyr_noise = gyr_n;
}

void ImuProcess::set_acc_cov(const double &acc_n)
{
    acc_noise = acc_n;
}

void ImuProcess::set_gyr_bias_cov(const double &gyr_w)
{
    gyr_bias_noise = gyr_w;
}

void ImuProcess::set_acc_bias_cov(const double &acc_w)
{
    acc_bias_noise = acc_w;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatePredictor &filterEstimator, int &N)
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/
    
    V3D cur_acc, cur_gyr;
    
    if (b_first_frame_) {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        first_lidar_time = meas.lidar_beg_time;
    }

    for (const auto &imu : meas.imu) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        //cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
        //cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

        N++;
    }
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, StatePredictor &filterEstimator, pcl::PointCloud<PointType> &pcl_out)
{
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;
    const double &pcl_end_time = meas.lidar_end_time;
    
    /*** sort point clouds by offset time ***/
    pcl_out = *(meas.lidar);
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

    /*** Initialize IMU pose ***/
    GlobalState imu_state = filterEstimator.state_;
    IMUpose.clear();
    IMUpose.push_back(setPose6D(0.0, imu_state.rn_, imu_state.vn_, imu_state.qbn_.toRotationMatrix(), accvel_last, angvel_last));

    /*** forward propagation at each imu point ***/
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;
    V3D cur_acc, cur_gyr;

    double dt = 0;

    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);
        
        if (tail->header.stamp.toSec() < last_lidar_end_time_) {
            continue;
        }

        cur_gyr << tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z;
        cur_acc << tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z;


        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                      0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                      0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr    << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                      0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                      0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
        
        acc_avr = acc_avr * G0 / mean_acc.norm();

        if(head->header.stamp.toSec() < last_lidar_end_time_) {
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            // dt = tail->header.stamp.toSec() - pcl_beg_time;
        }
        else {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }
        
        filterEstimator.predict(dt, cur_acc, cur_gyr, true);
        //filterEstimator.predict(dt, acc_avr, angvel_avr, true);

        /* save the poses at each IMU measurements */
        imu_state = filterEstimator.state_;
        angvel_last = angvel_avr - imu_state.bw_;
        accvel_last = imu_state.qbn_ * (acc_avr - imu_state.ba_) + imu_state.gn_;

        double offs_t = head->header.stamp.toSec() - pcl_beg_time;
        IMUpose.push_back(setPose6D(offs_t, imu_state.rn_, imu_state.vn_, imu_state.qbn_.toRotationMatrix(), accvel_last, angvel_last));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    if (pcl_end_time > imu_end_time) {
        dt = pcl_end_time - imu_end_time;
    }
    else {
        dt = imu_end_time - pcl_end_time;
    }
    filterEstimator.predict(dt, cur_acc, cur_gyr, true);
    //filterEstimator.predict(dt, acc_avr, angvel_avr, true);
    
    imu_state = filterEstimator.state_;
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    if (is_undistort) {
        auto it_pcl = pcl_out.points.end() - 1;
        for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
            auto head = it_kp - 1;
            auto tail = it_kp;
            R_imu = head->rot;
            vel_imu = head->vel;
            pos_imu = head->pos;
            acc_imu = tail->acc;
            angvel_avr = tail->gyr;

            for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
                dt = it_pcl->curvature / double(1000) - head->offset_time;

                /* Transform to the 'end' frame, using only the rotation
                * Note: Compensation direction is INVERSE of Frame's moving direction
                * So if we want to compensate a point at timestamp-i to the frame-e
                * P_compensate = R_imu_e ^ T * ((R_i * P_i + T_i) - T_e) where T_ei is represented in global frame */
                M3D R_i(Exp(angvel_avr, dt) * R_imu);
                
                V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                V3D T_i(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt);
                V3D P_compensate = imu_state.q_ex_.conjugate() * (imu_state.qbn_.conjugate() * (R_i * (imu_state.q_ex_ * P_i + imu_state.t_ex_) + T_i - imu_state.rn_) - imu_state.t_ex_);// not accurate!
                
                // save Undistorted points and their rotation
                it_pcl->x = P_compensate(0);
                it_pcl->y = P_compensate(1);
                it_pcl->z = P_compensate(2);

                if (it_pcl == pcl_out.points.begin()) break;
            }
        }
    }
}

void ImuProcess::Process(const MeasureGroup &meas,  StatePredictor &filterEstimator, pcl::PointCloud<PointType>::Ptr cur_pcl_un_)
{
    if (meas.imu.empty()) {
        return;
    }
    
    ROS_ASSERT(meas.lidar != nullptr);

    if (imu_need_init_) {
        /// The very first lidar frame
        IMU_init(meas, filterEstimator, init_iter_num);

        //imu_need_init_ = true;
        
        if (init_iter_num > MAX_INI_CNT) {
            // Initialize filter state
            V3D p0, v0, ba;
            p0.setZero();
            v0.setZero();
            ba.setZero();
            Q4D q0;
            q0.setIdentity();
            const double pcl_end_time = meas.lidar_end_time;
            filterEstimator.initialization(pcl_end_time, p0, v0, q0, ba, mean_gyr);
            filterEstimator.state_.gn_ = -mean_acc / mean_acc.norm() * G0;
            filterEstimator.state_.setExtrinsic(q_b_l, t_b_l);
            // Initialize filter covariance
            Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_> init_P;
            init_P.setZero();
            V3D cov_acc(1e-3, 1e-3, 1e-4);
            V3D cov_gyr(1e-4, 1e-4, 1e-4);
            V3D cov_grav(1e-5, 1e-5, 1e-12);
            init_P.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = cov_acc.asDiagonal();
            init_P.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = cov_gyr.asDiagonal();
            init_P.block<3, 3>(GlobalState::gra_, GlobalState::gra_) = cov_grav.asDiagonal();
            filterEstimator.setStateCov(init_P);
            filterEstimator.setNoiseCov(acc_noise, gyr_noise, acc_bias_noise, gyr_bias_noise);

            last_lidar_end_time_ = pcl_end_time;
            last_imu_ = meas.imu.back();
            imu_need_init_ = false;

            ROS_INFO("System Initialization Succeeded !!!");
        }
        return;
    }

    UndistortPcl(meas, filterEstimator, *cur_pcl_un_);
}
