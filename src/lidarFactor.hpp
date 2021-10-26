// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen3/Eigen/Cholesky>
#include "aloam_velodyne/common.h"

struct PriorFactor
{
	PriorFactor(Eigen::Quaterniond rotation_, Eigen::Vector3d gyro_bias_, 
				Eigen::Vector3d velocity_, Eigen::Vector3d acc_bias_, 
				Eigen::Vector3d position_, Eigen::Quaterniond ex_rotation_,
				Eigen::Vector3d ex_position_, Eigen::Matrix<double, 21, 21> Pk_)
		: rotation(rotation_), gyro_bias(gyro_bias_), velocity(velocity_), acc_bias(acc_bias_), 
		position(position_), ex_rotation(ex_rotation_), ex_position(ex_position_) {
			sqrt_Pk_inv = Eigen::LLT<Eigen::Matrix<double, 21, 21>>(Pk_.inverse()).matrixL().transpose();
		}

	template <typename T>
	bool operator()(const T *q, const T *bg, const T *v, const T *ba, const T *t, const T *q_ex, const T *t_ex, T *residual) const
	{
		Eigen::Matrix<T, 21, 1> delta_x;
		Eigen::Quaternion<T> q_t{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> rotation_t{(T)rotation.w(), (T)rotation.x(), (T)rotation.y(), (T)rotation.z()};
		Eigen::Quaternion<T> q_diff = q_t * rotation_t.inverse();
		Eigen::Quaternion<T> q_t_ex{q_ex[3], q_ex[0], q_ex[1], q_ex[2]};
		Eigen::Quaternion<T> ex_rotation_t{(T)ex_rotation.w(), (T)ex_rotation.x(), (T)ex_rotation.y(), (T)ex_rotation.z()};
		Eigen::Quaternion<T> q_diff_ex = q_t_ex * ex_rotation_t.inverse();
		Eigen::Matrix<T, 3, 1> vec = q_diff.vec();
		for (size_t i = 0; i < 3; ++i) {
            delta_x(i, 0) = T(2) * vec(i, 0);
        }
		vec = q_diff_ex.vec();
		for (size_t i = 0; i < 3; ++i) {
            delta_x(i + 15, 0) = T(2) * vec(i, 0);
        }
        for (size_t i = 3; i < 6; ++i) {
            delta_x(i, 0) = bg[i - 3] - (T)gyro_bias(i - 3);
        }
		for (size_t i = 6; i < 9; ++i) {
            delta_x(i, 0) = v[i - 6] - (T)velocity(i - 6);
        }
		for (size_t i = 9; i < 12; ++i) {
            delta_x(i, 0) = ba[i - 9] - (T)acc_bias(i - 9);
        }
		for (size_t i = 12; i < 15; ++i) {
            delta_x(i, 0) = t[i - 12] - (T)position(i - 12);
        }
		for (size_t i = 18; i < 21; ++i) {
            delta_x(i, 0) = t_ex[i - 18] - (T)ex_position(i - 18);
        }
		Eigen::Matrix<T, 21, 21> sqrt_info = sqrt_Pk_inv.cast<T>();
        Eigen::Matrix<T, 21, 1> r = sqrt_info * delta_x;
		for (size_t i = 0; i < 21; ++i) {
			residual[i] = r(i, 0);
		}
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Quaterniond rotation_, Eigen::Vector3d gyro_bias_, 
										const Eigen::Vector3d velocity_, const Eigen::Vector3d acc_bias_, 
										const Eigen::Vector3d position_, const Eigen::Quaterniond ex_rotation_,
										const Eigen::Vector3d ex_position_, const Eigen::Matrix<double, 21, 21> Pk_)
	{
		return (new ceres::AutoDiffCostFunction<
				PriorFactor, 21, 4, 3, 3, 3, 3, 4, 3>(
				new PriorFactor(rotation_, gyro_bias_, velocity_, acc_bias_, position_, ex_rotation_, ex_position_, Pk_)));
	}

	Eigen::Quaterniond rotation;
	Eigen::Vector3d gyro_bias;
	Eigen::Vector3d velocity;
	Eigen::Vector3d acc_bias;
	Eigen::Vector3d position;
	Eigen::Quaterniond ex_rotation;
	Eigen::Vector3d ex_position;
	Eigen::Matrix<double, 21, 21> sqrt_Pk_inv;
};

struct LidarEdgeFactorEx
{
	LidarEdgeFactorEx(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_, double lidar_std)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {
			sqrt_info = (1.0 / lidar_std);
		}

	template <typename T>
	bool operator()(const T *q, const T *t, const T *q_ex, const T *t_ex, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};
		Eigen::Quaternion<T> q_w_i{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_i{T(1) * t[0], T(1) * t[1], T(1) * t[2]};

		Eigen::Quaternion<T> q_i_l{q_ex[3], q_ex[0], q_ex[1], q_ex[2]};
		Eigen::Matrix<T, 3, 1> t_i_l{T(1) * t_ex[0], T(1) * t_ex[1], T(1) * t_ex[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_i_l.inverse() * ((q_w_i * (q_i_l * cp + t_i_l) + t_w_i) - t_i_l);

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = (T)sqrt_info * nu.x() / de.norm();
		residual[1] = (T)sqrt_info * nu.y() / de.norm();
		residual[2] = (T)sqrt_info * nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_, const double lidar_std)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactorEx, 3, 4, 3, 4, 3>(
			new LidarEdgeFactorEx(curr_point_, last_point_a_, last_point_b_, s_, lidar_std)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
	double sqrt_info;
};

struct LidarPlaneNormFactorEx
{

	LidarPlaneNormFactorEx(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_, double lidar_std) : curr_point(curr_point_), 
						 plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_) {
		sqrt_info = (1.0 / lidar_std);
	}

	template <typename T>
	bool operator()(const T *q, const T *t, const T *q_ex, const T *t_ex, T *residual) const
	{
		Eigen::Quaternion<T> q_w_i{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_i{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Quaternion<T> q_i_l{q_ex[3], q_ex[0], q_ex[1], q_ex[2]};
		Eigen::Matrix<T, 3, 1> t_i_l{T(1) * t_ex[0], T(1) * t_ex[1], T(1) * t_ex[2]};

		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_i_l.inverse() * ((q_w_i * (q_i_l * cp + t_i_l) + t_w_i) - t_i_l);

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = (T)sqrt_info * (norm.dot(point_w) + T(negative_OA_dot_norm));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_, const double lidar_std)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactorEx, 1, 4, 3, 4, 3>(
			new LidarPlaneNormFactorEx(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, lidar_std)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
	double sqrt_info;
};

struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};


class PriorFactorSF: public ceres::SizedCostFunction<18, 18>
{
public:
    PriorFactorSF(const Eigen::Matrix<double, 18, 18> &covariance_): covariance(covariance_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_V{parameters[0][3], parameters[0][4], parameters[0][5]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d delta_Ba{parameters[0][9], parameters[0][10], parameters[0][11]};
        Eigen::Vector3d delta_Bg{parameters[0][12], parameters[0][13], parameters[0][14]};
        Eigen::Vector3d delta_g{parameters[0][15], parameters[0][16], parameters[0][17]};

        Eigen::Map<Eigen::Matrix<double, 18, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = delta_P;
        residual.block<3, 1>(3, 0) = delta_V;
        residual.block<3, 1>(6, 0) = delta_theta;
        residual.block<3, 1>(9, 0) = delta_Ba;
        residual.block<3, 1>(12, 0) = delta_Bg;
        residual.block<3, 1>(15, 0) = delta_g;
        Eigen::Matrix<double, 18, 18> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 18, 18>>(covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 18, 18, Eigen::RowMajor>> jacobian_error_state(jacobians[0]);
                jacobian_error_state = Eigen::MatrixXd::Identity(18, 18);
                jacobian_error_state = sqrt_info * jacobian_error_state;
            }
        }
        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[18];
        double **jaco = new double *[1];
        jaco[0] = new double[18 * 18];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 18, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 18, 18, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;

        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_V{parameters[0][3], parameters[0][4], parameters[0][5]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d delta_Ba{parameters[0][9], parameters[0][10], parameters[0][11]};
        Eigen::Vector3d delta_Bg{parameters[0][12], parameters[0][13], parameters[0][14]};
        Eigen::Vector3d delta_g{parameters[0][15], parameters[0][16], parameters[0][17]};

        Eigen::Matrix<double, 18, 1> residual;
        residual.block<3, 1>(0, 0) = delta_P;
        residual.block<3, 1>(3, 0) = delta_V;
        residual.block<3, 1>(6, 0) = delta_theta;
        residual.block<3, 1>(9, 0) = delta_Ba;
        residual.block<3, 1>(12, 0) = delta_Bg;
        residual.block<3, 1>(15, 0) = delta_g;
        Eigen::MatrixXd sqrt_info = Eigen::LLT<Eigen::Matrix<double, 18, 18>>(covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;

        puts("num");
        std::cout << residual.transpose() << std::endl << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 18, 18> num_jacobian;
        for (int k = 0; k < 18; k++) {
            Eigen::Vector3d delta_P_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Vector3d delta_V_new{parameters[0][3], parameters[0][4], parameters[0][5]};
            Eigen::Vector3d delta_theta_new{parameters[0][6], parameters[0][7], parameters[0][8]};
            Eigen::Vector3d delta_Ba_new{parameters[0][9], parameters[0][10], parameters[0][11]};
            Eigen::Vector3d delta_Bg_new{parameters[0][12], parameters[0][13], parameters[0][14]};
            Eigen::Vector3d delta_g_new{parameters[0][15], parameters[0][16], parameters[0][17]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                delta_P_new += delta;
            }
            else if (a == 1) {
                delta_V_new += delta;
            }
            else if (a == 2) {
                delta_theta_new += delta;
            }
            else if (a == 3) {
                delta_Ba_new += delta;
            }
            else if (a == 4) {
                delta_Bg_new += delta;
            }
            else {
                delta_g_new += delta;
            }

            Eigen::Matrix<double, 18, 1> tmp_residual;
            tmp_residual.block<3, 1>(0, 0) = delta_P_new;
            tmp_residual.block<3, 1>(3, 0) = delta_V_new;
            tmp_residual.block<3, 1>(6, 0) = delta_theta_new;
            tmp_residual.block<3, 1>(9, 0) = delta_Ba_new;
            tmp_residual.block<3, 1>(12, 0) = delta_Bg_new;
            tmp_residual.block<3, 1>(15, 0) = delta_g_new;
            tmp_residual = sqrt_info * tmp_residual;

            num_jacobian.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian << std::endl << std::endl;
    }

    Eigen::Matrix<double, 18, 18> covariance;
};

class LidarMapEdgeFactorSF: public ceres::SizedCostFunction<1, 18>
{
public:
    LidarMapEdgeFactorSF(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_a_,
                         const Eigen::Vector3d &last_point_b_,const Eigen::Vector3d &t_b_l_,
                         const Eigen::Quaterniond &q_b_l_, const Eigen::Vector3d &rn_,
                         const Eigen::Quaterniond &qbn_, double cov_): curr_point(curr_point_),
        last_point_a(last_point_a_), last_point_b(last_point_b_), t_b_l(t_b_l_), q_b_l(q_b_l_),
        rn(rn_), qbn(qbn_), cov(cov_){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        }

        Eigen::Quaterniond q_w_curr = (dq * nominal_qbn).normalized();
        Eigen::Vector3d t_w_curr = dq * nominal_rn + delta_P;

        Eigen::Matrix<double, 3, 1> lp = q_w_curr * (q_b_l * cp + t_b_l) + t_w_curr;

        Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<double, 3, 1> de = lpa - lpb;

        residuals[0] = nu.norm() / (de.norm() * cov);

        Eigen::Vector3d coeff = nu.transpose() * skew(lpb - lpa) / (nu.norm() * de.norm());

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>> jacobian_error_state(jacobians[0]);
                jacobian_error_state.setZero();
                Eigen::Matrix3d J_l = Jleft(delta_theta);
                jacobian_error_state.block<1, 3>(0, 0) = coeff.transpose();
                jacobian_error_state.block<1, 3>(0, 6) = -coeff.transpose() *
                        (skew(q_w_curr * (q_b_l * cp + t_b_l)) * J_l + skew(dq * nominal_rn) * J_l);
                jacobian_error_state = jacobian_error_state / cov;
            }
        }

        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 18];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;

        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        }

        Eigen::Quaterniond q_w_curr = (dq * nominal_qbn).normalized();
        Eigen::Vector3d t_w_curr = dq * nominal_rn + delta_P;
        Eigen::Matrix<double, 3, 1> lp = q_w_curr * (q_b_l * cp + t_b_l) + t_w_curr;

        Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<double, 3, 1> de = lpa - lpb;
        Eigen::Matrix<double, 1, 1> residual;
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / (de.norm() * cov) );

        puts("num");
        std::cout << residual.transpose() << std::endl <<  std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 18> num_jacobian_error_state;
        for (int k = 0; k < 18; k++) {
            Eigen::Vector3d delta_P_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Vector3d delta_theta_new{parameters[0][6], parameters[0][7], parameters[0][8]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                delta_P_new += delta;
            }
            else if (a == 2) {
                delta_theta_new += delta;
            }

            Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
            if (delta_theta_new.norm() > 1e-10) {
                Eigen::AngleAxisd axis_dq(delta_theta_new.norm(), delta_theta_new / delta_theta_new.norm());
                dq = axis_dq;
            }

            Eigen::Quaterniond q_w_curr = (dq * nominal_qbn).normalized();
            Eigen::Vector3d t_w_curr = dq * nominal_rn + delta_P_new;

            Eigen::Matrix<double, 3, 1> lp = q_w_curr * (q_b_l * cp + t_b_l) + t_w_curr;

            Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
            Eigen::Matrix<double, 3, 1> de = lpa - lpb;

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / (de.norm() * cov));

            num_jacobian_error_state.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_error_state << std::endl << std::endl;
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    Eigen::Vector3d t_b_l;
    Eigen::Quaterniond q_b_l;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};

class LidarMapPlaneNormFactorSF: public ceres::SizedCostFunction<1, 18>
{
public:
    LidarMapPlaneNormFactorSF(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &plane_unit_norm_,
                              double negative_OA_dot_norm_, const Eigen::Vector3d &t_b_l_,
                              const Eigen::Quaterniond &q_b_l_, const Eigen::Vector3d &rn_,
                              const Eigen::Quaterniond &qbn_, double cov_) : curr_point(curr_point_),
        plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_),
        t_b_l(t_b_l_), q_b_l(q_b_l_), rn(rn_), qbn(qbn_), cov(cov_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        }

        Eigen::Quaterniond q_w_curr = (dq * nominal_qbn).normalized();
        Eigen::Vector3d t_w_curr = dq * nominal_rn + delta_P;

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d norm = plane_unit_norm;
        Eigen::Vector3d point_w = q_w_curr * (q_b_l * cp + t_b_l) + t_w_curr;

        residuals[0] = (norm.dot(point_w) + negative_OA_dot_norm) / cov;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>> jacobian_error_state(jacobians[0]);
                jacobian_error_state.setZero();
                Eigen::Matrix3d J_l = Jleft(delta_theta);
                jacobian_error_state.block<1, 3>(0, 0) = norm.transpose();
                jacobian_error_state.block<1, 3>(0, 6) = -norm.transpose() *
                        (skew(q_w_curr * (q_b_l * cp + t_b_l)) * J_l + skew(dq * nominal_rn) * J_l);
                jacobian_error_state = jacobian_error_state / cov;
            }
        }
        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 18];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;

        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        }

        Eigen::Quaterniond q_w_curr = (dq * nominal_qbn).normalized();
        Eigen::Vector3d t_w_curr = dq * nominal_rn + delta_P;

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d norm = plane_unit_norm;
        Eigen::Vector3d point_w = q_w_curr * (q_b_l * cp + t_b_l) + t_w_curr;

        Eigen::Matrix<double, 1, 1> residual;
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( (norm.dot(point_w) + negative_OA_dot_norm) / cov );

        puts("num");
        std::cout << residual.transpose() << std::endl <<  std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 18> num_jacobian_error_state;
        for (int k = 0; k < 18; k++) {
            Eigen::Vector3d delta_P_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Vector3d delta_theta_new{parameters[0][6], parameters[0][7], parameters[0][8]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                delta_P_new += delta;
            }
            else if (a == 2) {
                delta_theta_new += delta;
            }

            Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
            if (delta_theta_new.norm() > 1e-10) {
                Eigen::AngleAxisd axis_dq(delta_theta_new.norm(), delta_theta_new / delta_theta_new.norm());
                dq = axis_dq;
            }

            Eigen::Quaterniond q_w_curr = (dq * nominal_qbn).normalized();
            Eigen::Vector3d t_w_curr = dq * nominal_rn + delta_P_new;

            Eigen::Vector3d point_w = q_w_curr * (q_b_l * cp + t_b_l) + t_w_curr;

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( (norm.dot(point_w) + negative_OA_dot_norm) / cov );

            num_jacobian_error_state.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout <<  num_jacobian_error_state << std::endl << std::endl;
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
    Eigen::Vector3d t_b_l;
    Eigen::Quaterniond q_b_l;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};