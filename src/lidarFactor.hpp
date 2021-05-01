// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include<eigen3/Eigen/Cholesky>

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