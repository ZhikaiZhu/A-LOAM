
#ifndef _INCLUDE_COMPUTE_SIM3_HPP
#define _INCLUDE_COMPUTE_SIM3_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

template <typename T>
struct TrajPoint {
    T timestamp;
    Eigen::Matrix<T, 3, 1> translation;
    Eigen::Quaternion<T> q;
};

template <typename T>
class ComputeSim3 {
    public:
        using Traj = std::vector<TrajPoint<T>>;
        using TrajPoints = std::vector<Eigen::Matrix<T, 3, 1>>;

        ComputeSim3(T syncThreshold = 0.01): syncThreshold_(syncThreshold) {}
        ~ComputeSim3() {}

        void LoadTraj(const std::string &strTraj1, const std::string &strTraj2, bool show_gt);

        void syncTraj();

        Eigen::Matrix<T, 4, 4> GetSim3();

        void getTraj(Traj &traj1, Traj &traj2, bool show_gt) const;

        void GetSyncedTrajPoints(TrajPoints &points1, TrajPoints &points2) const;

        void GetSyncedTraj(Traj &syncedTraj1, Traj &syncedTraj2) const;

    private:
        Traj traj1_;
        Traj traj2_;
        Traj syncedTraj1_;
        Traj syncedTraj2_;
        T syncThreshold_;
};

template <typename T>
void ComputeSim3<T>::LoadTraj(const std::string &strTraj1, const std::string &strTraj2, bool show_gt) {
    std::ifstream fin1(strTraj1);
    if (!fin1.is_open()) {
        std::cerr << "Fail to open trajectory file1 !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line1;

    while(getline(fin1, line1)) {     // The entire line is read
        if (!line1.empty()) {
            std::stringstream ss;
            ss.setf(std::ios::fixed);
            ss << line1;
            TrajPoint<T> trajPoint;
            ss >> trajPoint.timestamp;
            ss >> trajPoint.translation(0) >> trajPoint.translation(1) >> trajPoint.translation(2);
            ss >> trajPoint.q.w() >> trajPoint.q.x() >> trajPoint.q.y() >> trajPoint.q.z();
            traj1_.emplace_back(trajPoint);
        }
    }
    fin1.close();

    if (show_gt) {
        std::ifstream fin2(strTraj2);
        if (!fin2.is_open()) {
            std::cerr << "Fail to open trajectory file2 !!!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line2;

        while(getline(fin2, line2)) {     // The entire line is read
            if (!line2.empty()) {
                std::stringstream ss;
                ss.setf(std::ios::fixed);
                ss << line2;
                TrajPoint<T> trajPoint;
                ss >> trajPoint.timestamp;
                ss >> trajPoint.translation(0) >> trajPoint.translation(1) >> trajPoint.translation(2);
                ss >> trajPoint.q.w() >> trajPoint.q.x() >> trajPoint.q.y() >> trajPoint.q.z();
                traj2_.emplace_back(trajPoint);
            }
        }
        fin2.close();
    }  
}

template <typename T>
void ComputeSim3<T>::syncTraj() {
    size_t m = traj1_.size();
    size_t n = traj2_.size();
    size_t curFirst = 0;

    if (m > n) {
        syncedTraj1_.reserve(n);
        syncedTraj2_.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = curFirst; j < m; ++j) {
                if (std::abs(traj1_[j].timestamp - traj2_[i].timestamp) > syncThreshold_ && 
                             traj1_[j].timestamp > traj2_[i].timestamp) {
                    break;
                }

                if (std::abs(traj1_[j].timestamp - traj2_[i].timestamp) <= syncThreshold_) {
                    syncedTraj1_.push_back(traj1_[j]);
                    syncedTraj2_.push_back(traj2_[i]);
                    curFirst++;
                    break;
                }
            }
        }
    }
    else { // m <= n
        syncedTraj1_.reserve(m);
        syncedTraj2_.reserve(m);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = curFirst; j < n; ++j) {
                if (std::abs(traj1_[i].timestamp - traj2_[j].timestamp) > syncThreshold_ && 
                             traj2_[j].timestamp > traj1_[i].timestamp) {
                    break;
                }

                if (std::abs(traj1_[i].timestamp - traj2_[j].timestamp) <= syncThreshold_) {
                    syncedTraj1_.push_back(traj1_[i]);
                    syncedTraj2_.push_back(traj2_[j]);
                    curFirst++;
                    break;
                }
            }
        }
    }

}

template <typename T>
Eigen::Matrix<T, 4, 4> ComputeSim3<T>::GetSim3() {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> trajPoints_1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> trajPoints_2;

    const size_t n = syncedTraj1_.size();
    trajPoints_1.resize(3, n);
    trajPoints_2.resize(3, n);

    for (size_t i = 0 ; i < n; ++i) {
        trajPoints_1.template block<3, 1>(0, i) = syncedTraj1_[i].translation;
        trajPoints_2.template block<3, 1>(0, i) = syncedTraj2_[i].translation;
    }

    Eigen::Matrix<T, 3, 1> mean_1, mean_2;
    T one_over_n = 1 / static_cast<T>(n);
    mean_1 = trajPoints_1.rowwise().sum() * one_over_n;
    mean_2 = trajPoints_2.rowwise().sum() * one_over_n;

    Eigen::Matrix<T, 3, Eigen::Dynamic> meas_1, meas_2;
    meas_1 = trajPoints_1.colwise() - mean_1;
    meas_2 = trajPoints_2.colwise() - mean_2;

    Eigen::Matrix<T, 3, 3> Sigma = meas_2 * meas_1.transpose() * one_over_n;
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T, 3, 1> S = Eigen::Matrix<T, 3, 1>::Ones();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) {
        S(2) = static_cast<T>(-1);
    }
    T Var_1 = meas_1.rowwise().squaredNorm().sum() * one_over_n;
    T c = svd.singularValues().dot(S) / Var_1;

    Eigen::Matrix<T, 4, 4> sim3 = Eigen::Matrix<T, 4, 4>::Identity();
    sim3.template block<3, 3>(0, 0).noalias() = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose() * c;
    sim3.template block<3, 1>(0, 3) = mean_2 - sim3.template block<3, 3>(0, 0) * mean_1;

    return sim3;
}

template <typename T>
 void ComputeSim3<T>::getTraj(Traj &traj1, Traj &traj2, bool show_gt) const {
     traj1 = traj1_;
     if (show_gt) {
         traj2 = traj2_;
     } 
 }

 template <typename T>
 void ComputeSim3<T>::GetSyncedTraj(Traj &syncedTraj1, Traj &syncedTraj2) const {
     syncedTraj1 = syncedTraj1_;
     syncedTraj2 = syncedTraj2_;
 }

 template <typename T>
 void ComputeSim3<T>::GetSyncedTrajPoints(TrajPoints &points1, TrajPoints &points2) const {
     const size_t n = syncedTraj1_.size();
     points1.reserve(n);
     points2.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        points1.push_back(syncedTraj1_[i].translation);
        points2.push_back(syncedTraj2_[i].translation);
    }
 }


#endif // _INCLUDE_COMPUTE_SIM3_HPP