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
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>

using namespace std;

#define IS_CALIB_EX 0

namespace filter {
// GlobalState Class contains state variables including position, velocity,
// attitude, acceleration bias, gyroscope bias, and gravity
class GlobalState {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr unsigned int DIM_OF_STATE_ = IS_CALIB_EX ? 24 : 18;
    static constexpr unsigned int DIM_OF_NOISE_ = 12;
    static constexpr unsigned int pos_ = 0;
    static constexpr unsigned int vel_ = 3;
    static constexpr unsigned int att_ = 6;
    static constexpr unsigned int acc_ = 9;
    static constexpr unsigned int gyr_ = 12;
    static constexpr unsigned int gra_ = 15;
    static constexpr unsigned int ex_pos_ = 18;
    static constexpr unsigned int ex_att_ = 21;

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
    void boxPlusInv(const Eigen::Matrix<double, 18, 1>& xk,
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
        this->q_ex_ = other.q_ex_;
        this->t_ex_ = other.t_ex_;

        return *this;
    }

    void setExtrinsic(const Q4D &q_ex, const V3D &t_ex) {
        q_ex_ = q_ex;
        t_ex_ = t_ex;
    }

    // !@State
    V3D rn_;   // position in w-frame
    V3D vn_;   // velocity in w-frame
    Q4D qbn_;  // rotation from b-frame to w-frame
    V3D ba_;   // acceleartion bias
    V3D bw_;   // gyroscope bias
    V3D gn_;   // gravity
    Q4D q_ex_;
    V3D t_ex_;
};

class StatePredictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StatePredictor() = default;

    ~StatePredictor() {}

    bool predict(double dt, const V3D& acc, const V3D& gyr, bool update_jacobian_ = true) {
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
        state_tmp.qbn_ = (dq * state_tmp.qbn_).normalized();
        V3D un_acc_1 = state_tmp.qbn_ * (acc - state_tmp.ba_) + state_tmp.gn_;
        V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);

        /*V3D un_gyr = gyr - state_tmp.bw_;
        state_tmp.qbn_ = Exp(un_gyr, dt) * state_tmp.qbn_.toRotationMatrix();
        V3D un_acc = state_tmp.qbn_ * (acc - state_tmp.ba_) + state_tmp.gn_;*/


        // State integral
        state_tmp.rn_ = state_tmp.rn_ + dt * state_tmp.vn_ + 0.5 * dt * dt * un_acc;
        state_tmp.vn_ = state_tmp.vn_ + dt * un_acc;

        if (update_jacobian_) {

            // Calculate F and G of InEKF
            MXD Ft =
                  MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
            MXD Gt =
                  MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);

            Ft.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = M3D::Identity();
            Ft.block<3, 3>(GlobalState::pos_, GlobalState::gyr_) = -skew(state_tmp.rn_) * state_tmp.qbn_.toRotationMatrix();
            Ft.block<3, 3>(GlobalState::vel_, GlobalState::att_) = skew(state_tmp.gn_);
            Ft.block<3, 3>(GlobalState::vel_, GlobalState::acc_) = -state_tmp.qbn_.toRotationMatrix();
            Ft.block<3, 3>(GlobalState::vel_, GlobalState::gyr_) = -skew(state_tmp.vn_) * state_tmp.qbn_.toRotationMatrix();
            Ft.block<3, 3>(GlobalState::vel_, GlobalState::gra_) = M3D::Identity();
            Ft.block<3, 3>(GlobalState::att_, GlobalState::gyr_) = -state_tmp.qbn_.toRotationMatrix();

            Gt.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = -skew(state_tmp.rn_) * state_tmp.qbn_.toRotationMatrix();
            Gt.block<3, 3>(GlobalState::vel_, GlobalState::pos_) = -state_tmp.qbn_.toRotationMatrix();
            Gt.block<3, 3>(GlobalState::vel_, GlobalState::vel_) = -skew(state_tmp.vn_) * state_tmp.qbn_.toRotationMatrix();
            Gt.block<3, 3>(GlobalState::att_, GlobalState::vel_) = -state_tmp.qbn_.toRotationMatrix();
            Gt.block<3, 3>(GlobalState::acc_, GlobalState::att_) = M3D::Identity();
            Gt.block<3, 3>(GlobalState::gyr_, GlobalState::acc_) = M3D::Identity();
            //Gt = Gt * dt;

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

    void setState(const GlobalState& state) { state_ = state; }

    void setStateCov(const Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_> &Pk) {
        covariance_.setZero();
        covariance_ = Pk;
    }

    void setNoiseCov(double acc_n, double gyr_n, double acc_w, double gyr_w) {
        double acc_n_2, gyr_n_2, acc_w_2, gyr_w_2;
        acc_n_2 = acc_n * acc_n;
        gyr_n_2 = gyr_n * gyr_n;
        acc_w_2 = acc_w * acc_w;
        gyr_w_2 = gyr_w * gyr_w;
        acc_cov_ = V3D(acc_n_2, acc_n_2, acc_n_2);
        gyr_cov_ = V3D(gyr_n_2, gyr_n_2, gyr_n_2);
        acc_bias_cov_ = V3D(acc_w_2, acc_w_2, acc_w_2);
        gyr_bias_cov_ = V3D(gyr_w_2, gyr_w_2, gyr_w_2);
        noise_.setZero();
        noise_.block<3, 3>(0, 0) = acc_cov_.asDiagonal();
        noise_.block<3, 3>(3, 3) = gyr_cov_.asDiagonal();
        noise_.block<3, 3>(6, 6) = acc_bias_cov_.asDiagonal();
        noise_.block<3, 3>(9, 9) = gyr_bias_cov_.asDiagonal();
    }

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
    }

    void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                        const V3D& ba, const V3D& bw, const V3D& acc, const V3D& gyr) {
        state_ = GlobalState(rn, vn, qbn, ba, bw);
        time_ = time;
        acc_last = acc;
        gyr_last = gyr;
        flag_init_imu_ = true;
        flag_init_state_ = true;
    }

    inline bool isInitialized() { return flag_init_state_; }

    GlobalState state_;
    double time_;
    Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
        F_;
    Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_>
        jacobian_, covariance_;
    Eigen::Matrix<double, GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_>
        noise_;

    V3D acc_last;  // last acceleration measurement
    V3D gyr_last;  // last gyroscope measurement

    V3D acc_cov_;
    V3D gyr_cov_;
    V3D acc_bias_cov_;
    V3D gyr_bias_cov_;

    bool flag_init_state_;
    bool flag_init_imu_;
};

};  // namespace filter

#endif  // INCLUDE_FILTER_STATE_HPP_