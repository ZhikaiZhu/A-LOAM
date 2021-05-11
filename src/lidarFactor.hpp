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
#include "aloam_velodyne/parameters.h"

using namespace parameter;

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

struct PriorFactor
{
    PriorFactor(Eigen::Matrix<double, 18, 18> covariance) : covariance_(covariance) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }
		Eigen::Matrix<double, 18, 18> tmp = Eigen::LLT<Eigen::Matrix<double, 18, 18>>(covariance_.inverse()).matrixL().transpose();
		Eigen::Matrix<T, 18, 18> sqrt_info = tmp.cast<T>();
        Eigen::Matrix<T, 18, 1> r = sqrt_info * delta_x;
		
		for (size_t i = 0; i < 18; ++i) {
			residual[i] = r(i, 0);
		}

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Matrix<double, 18, 18> covariance)
	{
		return (new ceres::AutoDiffCostFunction<
				PriorFactor, 18, 18>(new PriorFactor(covariance)));
	}

    Eigen::Matrix<double, 18, 18> covariance_;
    
};

struct LidarEdgeStateFactor
{
	LidarEdgeStateFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
						 Eigen::Vector3d last_point_b_, double s_, Eigen::Vector3d rn_, Eigen::Quaterniond qbn_, double cov_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        Eigen::Matrix<T, 3, 1> Tbl{T(INIT_TBL.x()), T(INIT_TBL.y()), T(INIT_TBL.z())};
        Eigen::Quaternion<T> Rbl = INIT_RBL.cast<T>();

        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }

        Eigen::Matrix<T, 3, 1> nominal_rn{T(rn[0]), T(rn[1]), T(rn[2])};
        Eigen::Matrix<T, 3, 1> delta_rn{T(dx[0]), T(dx[1]), T(dx[2])};
        Eigen::Quaternion<T> nominal_qbn{T(qbn.w()), T(qbn.x()), T(qbn.y()), T(qbn.z())};
        Eigen::Matrix<T, 3, 1> d_theta{T(dx[6]), T(dx[7]), T(dx[8])};

        Eigen::Quaternion<T> dq{T(1), T(0), T(0), T(0)};
        if (d_theta.norm() > 1e-10) {
            Eigen::AngleAxis<T> axis_dq(d_theta.norm(), d_theta / d_theta.norm());
            dq = axis_dq;
        }
        
        //Eigen::Quaternion<T> dq = axis2Quat(delta_x.template segment<3>(6));
        Eigen::Quaternion<T> q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);

        nominal_rn = dq * nominal_rn + delta_rn;
        /*
        Eigen::Matrix<T, 3, 1> phi = Quat2axis(nominal_qbn);
        Eigen::Quaternion<T> q_last_curr = axis2Quat(s * phi);
        q_last_curr.normalized();
        */

        Eigen::Matrix<T, 3, 1> t_last_curr = T(s) * nominal_rn;
        Eigen::Matrix<T, 3, 1> lp = Rbl.inverse() * (q_last_curr * (Rbl * cp + Tbl) + t_last_curr - Tbl);

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        residual[0] = nu.norm() / (de.norm() * T(cov));

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_, 
                                       const Eigen::Vector3d rn_, const Eigen::Quaterniond qbn_, const double cov_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeStateFactor, 1, 18>(
			new LidarEdgeStateFactor(curr_point_, last_point_a_, last_point_b_, s_, rn_, qbn_, cov_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};

struct LidarPlaneStateFactor
{
	LidarPlaneStateFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, 
                    	  Eigen::Vector3d last_point_b_, Eigen::Vector3d last_point_c_, 
                    	  double s_, Eigen::Vector3d rn_, Eigen::Quaterniond qbn_, double cov_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), 
        last_point_c(last_point_c_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};
		Eigen::Matrix<T, 3, 1> lpc{T(last_point_c.x()), T(last_point_c.y()), T(last_point_c.z())};

        Eigen::Matrix<T, 3, 1> Tbl{T(INIT_TBL.x()), T(INIT_TBL.y()), T(INIT_TBL.z())};
        Eigen::Quaternion<T> Rbl = INIT_RBL.cast<T>();

        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }

        Eigen::Matrix<T, 3, 1> nominal_rn{T(rn[0]), T(rn[1]), T(rn[2])};
        Eigen::Matrix<T, 3, 1> delta_rn{T(dx[0]), T(dx[1]), T(dx[2])};
        Eigen::Quaternion<T> nominal_qbn{T(qbn.w()), T(qbn.x()), T(qbn.y()), T(qbn.z())};
        Eigen::Matrix<T, 3, 1> d_theta{T(dx[6]), T(dx[7]), T(dx[8])};

        Eigen::Quaternion<T> dq{T(1), T(0), T(0), T(0)};
        if (d_theta.norm() > 1e-10) {
            Eigen::AngleAxis<T> axis_dq(d_theta.norm(), d_theta / d_theta.norm());
            dq = axis_dq;
        }

        //Eigen::Quaternion<T> dq = axis2Quat(delta_x.template segment<3>(6));
        Eigen::Quaternion<T> q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);

        nominal_rn = dq * nominal_rn + delta_rn;
        /*
        Eigen::Matrix<T, 3, 1> phi = Quat2axis(nominal_qbn);
        Eigen::Quaternion<T> q_last_curr = axis2Quat(s * phi);
        q_last_curr.normalized();
        */

        Eigen::Matrix<T, 3, 1> t_last_curr = T(s) * nominal_rn;
        Eigen::Matrix<T, 3, 1> lp = Rbl.inverse() * (q_last_curr * (Rbl * cp + Tbl) + t_last_curr - Tbl);
        Eigen::Matrix<T, 3, 1> M = (lpa - lpb).cross(lpa - lpc);
        residual[0] = ((lp - lpa).transpose() * M).norm() / (M.norm() * T(cov));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const Eigen::Vector3d last_point_c_,
                                       const double s_, const Eigen::Vector3d rn_, const Eigen::Quaterniond qbn_, const double cov_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneStateFactor, 1, 18>(
			new LidarPlaneStateFactor(curr_point_, last_point_a_, last_point_b_, last_point_c_, s_, rn_, qbn_, cov_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b, last_point_c;
	double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};