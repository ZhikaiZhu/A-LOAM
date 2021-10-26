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

#pragma once

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <cstdlib>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <deque>

#define G0 (9.81)     // gravity
#define MAX_INI_CNT (20)
#define INIT_TIME (0.1)
#define VEC_FROM_ARRAY(v) v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v) v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]

typedef pcl::PointXYZINormal PointType;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::VectorXd VXD;
typedef Eigen::MatrixXd MXD;
typedef Eigen::Quaterniond Q4D;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

class MeasureGroup {    // Lidar data and imu dates for the curent process
public:
    MeasureGroup() {
        lidar_beg_time = 0.0;
        this->lidar.reset(new pcl::PointCloud<PointType>());
    };

    ~MeasureGroup() = default;

    double lidar_beg_time;
    double lidar_end_time;
    pcl::PointCloud<PointType>::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
};

class Pose6D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double offset_time;
    Eigen::Matrix<double, 3, 1> pos;
    Eigen::Matrix<double, 3, 1> vel;
    Eigen::Matrix3d rot;
    Eigen::Matrix<double, 3, 1> acc;
    Eigen::Matrix<double, 3, 1> gyr;

    Pose6D() { setIdentity(); }
    ~Pose6D() = default;

    void setIdentity() {
        offset_time = 0;
        pos.setZero();
        vel.setZero();
        rot.setIdentity();
        acc.setZero();
        gyr.setZero();
    }
};

template <typename T>
auto setPose6D(const double t, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 1> &v,
               const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &a, 
               const Eigen::Matrix<T, 3, 1> &g) {
    Pose6D kp;
    kp.offset_time = t;
    kp.pos = p;   
    kp.vel = v;
    kp.rot = R;
    kp.acc = a;
    kp.gyr = g;

    return std::move(kp);
}

float calc_dist(PointType p1, PointType p2) {
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <class T>
static T rad2deg(const T &radians) {
    return radians * 180.0 / M_PI;
}

template <class T>
static T deg2rad(const T &degrees) {
    return degrees * M_PI / 180.0;
}

template <typename Type>
static Type wrap_pi(Type x) {
    while (x >= Type(M_PI)) {
      x -= Type(2.0 * M_PI);
    }

    while (x < Type(-M_PI)) {
      x += Type(2.0 * M_PI);
    }
    return x;
}

template <typename Derived>
static inline Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived> &q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1), q(2),
           typename Derived::Scalar(0), -q(0), -q(1), q(0),
           typename Derived::Scalar(0);
    return ans;
}

static inline Eigen::Vector3d vee(const Eigen::Matrix3d &w_hat) {
    const double EPS = 1e-10;
    assert(fabs(w_hat(2, 1) + w_hat(1, 2)) < EPS);
    assert(fabs(w_hat(0, 2) + w_hat(2, 0)) < EPS);
    assert(fabs(w_hat(1, 0) + w_hat(0, 1)) < EPS);
    return Eigen::Vector3d(w_hat(2, 1), w_hat(0, 2), w_hat(1, 0));
}

template<typename T, typename Ts>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt) {
    T ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_vel_norm > 0.0000001) {
        Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
        Eigen::Matrix<T, 3, 3> K;

        K = skew(r_axis);

        T r_ang = ang_vel_norm * dt;

        /// Roderigous Tranformation
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }
    else {
        return Eye3;
    }
}

template<typename T>
Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3> &R) {
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

static inline void enforceSymmetry(Eigen::MatrixXd &mat) {
    mat = 0.5 * (mat + mat.transpose()).eval();
}

static inline Eigen::Quaterniond axis2Quat(const Eigen::Vector3d &axis, double theta) {
    Eigen::Quaterniond q;

    if (theta < 1e-10) {
      q.w() = 1.0;
      q.x() = q.y() = q.z() = 0;
    }

    double magnitude = sin(theta / 2.0f);

    q.w() = cos(theta / 2.0f);
    q.x() = axis(0) * magnitude;
    q.y() = axis(1) * magnitude;
    q.z() = axis(2) * magnitude;

    return q;
}

static Eigen::Quaterniond axis2Quat(const Eigen::Vector3d &vec) {
    Eigen::Quaterniond q;
    double theta = vec.norm();

    if (theta < 1e-10) {
      q.w() = 1.0;
      q.x() = q.y() = q.z() = 0;
      return q;
    }

    Eigen::Vector3d tmp = vec / theta;
    return axis2Quat(tmp, theta);
}

static Eigen::Vector3d Quat2axis(const Eigen::Quaterniond &q) {
    double axis_magnitude = sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    Eigen::Vector3d vec;
    vec(0) = q.x();
    vec(1) = q.y();
    vec(2) = q.z();

    if (axis_magnitude >= 1e-10) {
      vec = vec / axis_magnitude;
      vec = vec * wrap_pi(2.0 * atan2(axis_magnitude, q.w()));
    }

    return vec;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q) {
    // printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
    //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
    // printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
    //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
    return q;
}

template <typename Derived>
static inline Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta) {
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

template <typename Derived>
static inline Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q) {
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(
        1, 0) = qq.vec(),
          ans.template block<3, 3>(1, 1) =
              qq.w() *
                  Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
              skew(qq.vec());
    return ans;
}

template <typename Derived>
static inline Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p) {
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(
        1, 0) = pp.vec(),
          ans.template block<3, 3>(1, 1) =
              pp.w() *
                  Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
              skew(pp.vec());
    return ans;
}

static inline Eigen::Matrix<double , 3, 3> Jleft(const Eigen::Matrix<double, 3, 1> axis) {
    Eigen::Matrix<double, 3, 3> ans;
    double theta = axis.norm();
    if (theta < 1e-10) {
        ans.setIdentity();
        return ans;
    }
    Eigen::Matrix<double, 3, 1> a = axis / theta;
    double s = sin(theta) / theta;
    double c = (1 - cos(theta)) / theta;
    ans = s * Eigen::Matrix<double, 3, 3>::Identity() +
          (1 - s) * a * a.transpose() + c * skew(a);

    return ans;
}

static inline Eigen::Matrix<double, 3, 3> Jinvleft(const Eigen::Matrix<double, 3, 1> axis) {
    Eigen::Matrix<double, 3, 3> ans;
    double theta = axis.norm();

    if (theta < 1e-10) {
        ans.setIdentity();
        return ans;
    }

    double half_theta = theta / 2.0;
    Eigen::Matrix<double, 3, 1> a = axis / axis.norm();
    double cot_half_theta = cos(half_theta) / sin(half_theta);
    double s = half_theta * cot_half_theta;
    ans = s * Eigen::Matrix<double, 3, 3>::Identity() +
          (1.0 - s) * a * a.transpose() - half_theta * skew(a);
    return ans;
}