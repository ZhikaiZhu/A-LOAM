// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#ifndef INCLUDE_FILTER_STATE_HPP_
#define INCLUDE_FILTER_STATE_HPP_

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/parameters.h"

#include <iostream>

using namespace std;
using namespace utils;
using namespace parameter;

namespace filter {

class Scan {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Scan(): id_(scan_counter_++) {
      cornerPointsSharp_.reset(new pcl::PointCloud<PointType>());
      cornerPointsLessSharp_.reset(new pcl::PointCloud<PointType>());
      surfPointsFlat_.reset(new pcl::PointCloud<PointType>());
      surfPointsLessFlat_.reset(new pcl::PointCloud<PointType>());
      laserCloudFullRes_.reset(new pcl::PointCloud<PointType>());

      reset();
    }  

    ~Scan() {
      cornerPointsSharp_.reset();
      cornerPointsLessSharp_.reset();
      surfPointsFlat_.reset();
      surfPointsLessFlat_.reset();
      laserCloudFullRes_.reset();
    }

    void reset() {
      time_ = 0.0;
      
      cornerPointsSharp_->clear();
      cornerPointsLessSharp_->clear();
      surfPointsFlat_->clear();
      surfPointsLessFlat_->clear();
      laserCloudFullRes_->clear();
    } 

    void setPointCloud(double time, 
                       pcl::PointCloud<PointType>::Ptr cornerPointsSharp,
                       pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp,
                       pcl::PointCloud<PointType>::Ptr surfPointsFlat,
                       pcl::PointCloud<PointType>::Ptr surfPointsLessFlat,
                       pcl::PointCloud<PointType>::Ptr laserCloudFullRes) {
      cornerPointsSharp_ = cornerPointsSharp;
      cornerPointsLessSharp_ = cornerPointsLessSharp;
      surfPointsFlat_ = surfPointsFlat;
      surfPointsLessFlat_ = surfPointsLessFlat;
      laserCloudFullRes_ = laserCloudFullRes;
      time_ = time;
    }

  public:
    static int scan_counter_;
    int id_;
    double time_;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp_;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp_;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat_;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat_;
    pcl::PointCloud<PointType>::Ptr laserCloudFullRes_;
};
typedef std::shared_ptr<Scan> ScanPtr;

// GlobalState Class contains state variables including position, velocity,
// attitude, acceleration bias, gyroscope bias, and gravity
class GlobalState {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr unsigned int DIM_OF_STATE_ = 18;
  static constexpr unsigned int DIM_OF_NOISE_ = 12;
  static constexpr unsigned int pos_ = 0;
  static constexpr unsigned int vel_ = 3;
  static constexpr unsigned int att_ = 6;
  static constexpr unsigned int acc_ = 9;
  static constexpr unsigned int gyr_ = 12;
  static constexpr unsigned int gra_ = 15;

  GlobalState() { setIdentity(); }

  GlobalState(const V3D& rn, const V3D& vn, const Q4D& qbn, const V3D& ba,
              const V3D& bw) {
    setIdentity();
    rn_ = rn;
    vn_ = vn;
    qbn_ = qbn;
    ba_ = ba;
    bw_ = bw;
  }

  ~GlobalState() {}

  void setIdentity() {
    rn_.setZero();
    vn_.setZero();
    qbn_.setIdentity();
    ba_.setZero();
    bw_.setZero();
    gn_ << 0.0, 0.0, -G0;
  }

  // boxPlus operator
  void boxPlus(const Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk,
               GlobalState& stateOut) {
    stateOut.rn_ = rn_ + xk.template segment<3>(pos_);
    stateOut.vn_ = vn_ + xk.template segment<3>(vel_);
    stateOut.ba_ = ba_ + xk.template segment<3>(acc_);
    stateOut.bw_ = bw_ + xk.template segment<3>(gyr_);
    Q4D dq = axis2Quat(xk.template segment<3>(att_));
    stateOut.qbn_ = (qbn_ * dq).normalized();

    stateOut.gn_ = gn_ + xk.template segment<3>(gra_);
  }

  // boxPlus operator for InEKF
  void boxPlusInv(const Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk,
               GlobalState& stateOut) {
    Q4D dq = axis2Quat(xk.template segment<3>(att_));
    stateOut.qbn_ = (dq * qbn_).normalized();
    stateOut.rn_ = dq * rn_ + xk.template segment<3>(pos_);
    stateOut.vn_ = dq * vn_ + xk.template segment<3>(vel_);
    stateOut.ba_ = ba_ + xk.template segment<3>(acc_);
    stateOut.bw_ = bw_ + xk.template segment<3>(gyr_);
    stateOut.gn_ = gn_ + xk.template segment<3>(gra_);
  }

  // boxMinus operator
  void boxMinus(const GlobalState& stateIn,
                Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk) {
    xk.template segment<3>(pos_) = rn_ - stateIn.rn_;
    xk.template segment<3>(vel_) = vn_ - stateIn.vn_;
    xk.template segment<3>(acc_) = ba_ - stateIn.ba_;
    xk.template segment<3>(gyr_) = bw_ - stateIn.bw_;
    V3D da = Quat2axis(stateIn.qbn_.inverse() * qbn_);
    xk.template segment<3>(att_) = da;

    xk.template segment<3>(gra_) = gn_ - stateIn.gn_;
  }

  GlobalState& operator=(const GlobalState& other) {
    if (this == &other) return *this;

    this->rn_ = other.rn_;
    this->vn_ = other.vn_;
    this->qbn_ = other.qbn_;
    this->ba_ = other.ba_;
    this->bw_ = other.bw_;
    this->gn_ = other.gn_;

    return *this;
  }

  // !@State
  V3D rn_;   // position in n-frame
  V3D vn_;   // velocity in n-frame
  Q4D qbn_;  // rotation from b-frame to n-frame
  V3D ba_;   // acceleartion bias
  V3D bw_;   // gyroscope bias
  V3D gn_;   // gravity
};

class StatePredictor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StatePredictor() { reset(); }

  ~StatePredictor() {}

  bool predict(double dt, const V3D& acc, const V3D& gyr,
               bool update_jacobian_ = true) {
    if (!isInitialized()) return false;

    if (!flag_init_imu_) {
      flag_init_imu_ = true;
      acc_last = acc;
      gyr_last = gyr;
    }

    // Average acceleration and angular rate
    GlobalState state_tmp = state_;
    V3D un_acc_0 = state_tmp.qbn_ * (acc_last - state_tmp.ba_) + state_tmp.gn_;
    V3D un_gyr = 0.5 * (gyr_last + gyr) - state_tmp.bw_;
    Q4D dq = axis2Quat(un_gyr * dt);
    state_tmp.qbn_ = (state_tmp.qbn_ * dq).normalized();
    V3D un_acc_1 = state_tmp.qbn_ * (acc - state_tmp.ba_) + state_tmp.gn_;
    V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // State integral
    state_tmp.rn_ = state_tmp.rn_ + dt * state_tmp.vn_ + 0.5 * dt * dt * un_acc;
    state_tmp.vn_ = state_tmp.vn_ + dt * un_acc;

    if (update_jacobian_) {
      /*
      MXD Ft =
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);      
      */

      // Calculate F and G of InEKF
      MXD Ft =
            MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
      MXD Gt =
            MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);
      if (USE_CERES) {
        Ft.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = M3D::Identity();
        Ft.block<3, 3>(GlobalState::pos_, GlobalState::gyr_) = -skew(state_tmp.rn_) * state_tmp.qbn_.toRotationMatrix();
        Ft.block<3, 3>(GlobalState::vel_, GlobalState::att_) = skew(state_tmp.gn_);
        Ft.block<3, 3>(GlobalState::vel_, GlobalState::acc_) = -state_tmp.qbn_.toRotationMatrix();
        Ft.block<3, 3>(GlobalState::vel_, GlobalState::gyr_) = -skew(state_tmp.vn_) * state_tmp.qbn_.toRotationMatrix();
        Ft.block<3, 3>(GlobalState::vel_, GlobalState::gra_) = M3D::Identity();
        Ft.block<3, 3>(GlobalState::att_, GlobalState::gyr_) = -state_tmp.qbn_.toRotationMatrix();
        F_inekf = Ft;     // Used in imu motion model test

        Gt.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = -skew(state_tmp.rn_) * state_tmp.qbn_.toRotationMatrix();
        Gt.block<3, 3>(GlobalState::vel_, GlobalState::pos_) = -state_tmp.qbn_.toRotationMatrix();
        Gt.block<3, 3>(GlobalState::vel_, GlobalState::vel_) = -skew(state_tmp.vn_) * state_tmp.qbn_.toRotationMatrix();
        Gt.block<3, 3>(GlobalState::att_, GlobalState::vel_) = -state_tmp.qbn_.toRotationMatrix();
        Gt.block<3, 3>(GlobalState::acc_, GlobalState::att_) = M3D::Identity();
        Gt.block<3, 3>(GlobalState::gyr_, GlobalState::acc_) = M3D::Identity();
        G_inekf = Gt;     // Used in imu motion model test
        //Gt = Gt * dt;
      } else {
        Ft.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = M3D::Identity();

        Ft.block<3, 3>(GlobalState::vel_, GlobalState::att_) =
            -state_tmp.qbn_.toRotationMatrix() * skew(acc - state_tmp.ba_);
        Ft.block<3, 3>(GlobalState::vel_, GlobalState::acc_) =
            -state_tmp.qbn_.toRotationMatrix();
        Ft.block<3, 3>(GlobalState::vel_, GlobalState::gra_) = M3D::Identity();

        Ft.block<3, 3>(GlobalState::att_, GlobalState::att_) =
            - skew(gyr - state_tmp.bw_);
        Ft.block<3, 3>(GlobalState::att_, GlobalState::gyr_) = -M3D::Identity();

        Gt.block<3, 3>(GlobalState::vel_, 0) = -state_tmp.qbn_.toRotationMatrix();
        Gt.block<3, 3>(GlobalState::att_, 3) = -M3D::Identity();
        Gt.block<3, 3>(GlobalState::acc_, 6) = M3D::Identity();
        Gt.block<3, 3>(GlobalState::gyr_, 9) = M3D::Identity();
        //Gt = Gt * dt;
      }

      const MXD I =
          MXD::Identity(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
      F_ = I + Ft * dt + 0.5 * Ft * Ft * dt * dt;

      // jacobian_ = F * jacobian_;
      covariance_ =
          F_ * covariance_ * F_.transpose() + dt * Gt * noise_ * Gt.transpose();
      covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
    }

    state_ = state_tmp;
    time_ += dt;
    acc_last = acc;
    gyr_last = gyr;
    return true;
  }

  void set(const GlobalState& state) { state_ = state; }

  void update(const GlobalState& state,
              const Eigen::Matrix<double, GlobalState::DIM_OF_STATE_,
                                  GlobalState::DIM_OF_STATE_>& covariance) {
    state_ = state;
    covariance_ = covariance;
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw, const V3D& acc,
                      const V3D& gyr) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, double roll = 0.0, double pitch = 0.0,
                      double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, const V3D& acc, const V3D& gyr,
                      double roll = 0.0, double pitch = 0.0, double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initializeCovariance(int type = 0) {
    double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
    double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
    double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

    V3D covPos = INIT_POS_STD.array().square();
    V3D covVel = INIT_VEL_STD.array().square();
    V3D covAcc = INIT_ACC_STD.array().square();
    V3D covGyr = INIT_GYR_STD.array().square();

    double peba = pow(ACC_N, 2);
    double pebg = pow(GYR_N, 2);
    double pweba = pow(ACC_W, 2);
    double pwebg = pow(GYR_W, 2);
    V3D gra_cov(0.01, 0.01, 0.01);

    if (type == 0) {
      // Initialize using offline parameters
      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          covVel.asDiagonal();  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) =
          covAcc.asDiagonal();  // ba
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) =
          covGyr.asDiagonal();  // bg
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          gra_cov.asDiagonal();  // gravity
    } else if (type == 1) {
      // Inheritage previous covariance
      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          vel_cov;  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) = gra_cov;
    }

    noise_.setZero();
    noise_.block<3, 3>(0, 0) = V3D(peba, peba, peba).asDiagonal();
    noise_.block<3, 3>(3, 3) = V3D(pebg, pebg, pebg).asDiagonal();
    noise_.block<3, 3>(6, 6) = V3D(pweba, pweba, pweba).asDiagonal();
    noise_.block<3, 3>(9, 9) = V3D(pwebg, pwebg, pwebg).asDiagonal();
  }

  void reset(int type = 0) {
    if (type == 0) {
      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.qbn_.setIdentity();
      initializeCovariance();
    } else if (type == 1) {
      V3D covPos = INIT_POS_STD.array().square();
      double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
      double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
      double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          state_.qbn_.inverse() * vel_cov * state_.qbn_;  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          state_.qbn_.inverse() * gra_cov * state_.qbn_;

      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.gn_ = state_.qbn_.inverse() * state_.gn_;
      state_.qbn_.setIdentity();
      state_.gn_ = state_.gn_ * G0 / state_.gn_.norm();
      // initializeCovariance(1);
    }
  }

  void reset(V3D vn, V3D ba, V3D bw) {
    state_.setIdentity();
    state_.vn_ = vn;
    state_.ba_ = ba;
    state_.bw_ = bw;
    initializeCovariance();
  }

  inline bool isInitialized() { return flag_init_state_; }

  GlobalState state_;
  double time_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
      F_, F_inekf;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
      jacobian_, covariance_;
  Eigen::Matrix<double, GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_>
      noise_;

  V3D acc_last;  // last acceleration measurement
  V3D gyr_last;  // last gyroscope measurement

  bool flag_init_state_;
  bool flag_init_imu_;

  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_>
      G_inekf;
};

};  // namespace filter

#endif  // INCLUDE_FILTER_STATE_HPP_