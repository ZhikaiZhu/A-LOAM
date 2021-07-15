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
    PriorFactor(Eigen::Matrix<double, 18, 18> covariance_) : covariance(covariance_) {}

	template <typename T>
	bool operator()(const T *dx, T *residual) const
	{
        Eigen::Matrix<T, 18, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }
		Eigen::Matrix<double, 18, 18> tmp = Eigen::LLT<Eigen::Matrix<double, 18, 18>>(covariance.inverse()).matrixL().transpose();
		Eigen::Matrix<T, 18, 18> sqrt_info = tmp.cast<T>();
        Eigen::Matrix<T, 18, 1> r = sqrt_info * delta_x;
		
		for (size_t i = 0; i < 18; ++i) {
			residual[i] = r(i, 0);
		}

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Matrix<double, 18, 18> covariance_)
	{
		return (new ceres::AutoDiffCostFunction<
				PriorFactor, 18, 18>(new PriorFactor(covariance_)));
	}

    Eigen::Matrix<double, 18, 18> covariance;
    
};

struct PriorFactorEx
{
    PriorFactorEx(Eigen::Matrix<double, 24, 24> covariance_, Eigen::Quaterniond ex_rotation_, Eigen::Vector3d ex_translation_) 
		: covariance(covariance_), ex_rotation(ex_rotation_), ex_translation(ex_translation_) {}

	template <typename T>
	bool operator()(const T *dx, const T *ex, T *residual) const
	{
        Eigen::Quaternion<T> q_t_ex{ex[6], ex[3], ex[4], ex[5]};
		Eigen::Quaternion<T> ex_rotation_t{T(ex_rotation.w()), T(ex_rotation.x()), T(ex_rotation.y()), T(ex_rotation.z())};
		Eigen::Quaternion<T> q_diff_ex = q_t_ex * ex_rotation_t.inverse();
		Eigen::Matrix<T, 3, 1> vec = q_diff_ex.vec();
		
		Eigen::Matrix<T, 24, 1> delta_x;
        for (size_t i = 0; i < 18; ++i) {
            delta_x(i, 0) = dx[i];
        }

		for (size_t i = 18; i < 21; i++) {
            delta_x(i , 0) = ex[i - 18] - T(ex_translation(i - 18));
		}

		for (size_t i = 21; i < 24; i++) {
            delta_x(i, 0) = T(2) * vec(i - 21, 0);
		}
		Eigen::Matrix<double, 24, 24> tmp = Eigen::LLT<Eigen::Matrix<double, 24, 24>>(covariance.inverse()).matrixL().transpose();
		Eigen::Matrix<T, 24, 24> sqrt_info = tmp.cast<T>();
        Eigen::Matrix<T, 24, 1> r = sqrt_info * delta_x;
		
		for (size_t i = 0; i < 24; ++i) {
			residual[i] = r(i, 0);
		}

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Matrix<double, 24, 24> covariance_, const Eigen::Quaterniond ex_rotation_,
									   const Eigen::Vector3d ex_translation_)
	{
		return (new ceres::AutoDiffCostFunction<
				PriorFactorEx, 24, 18, 7>(new PriorFactorEx(covariance_, ex_rotation_, ex_translation_)));
	}

    Eigen::Matrix<double, 24, 24> covariance;
    Eigen::Quaterniond ex_rotation;
	Eigen::Vector3d ex_translation;
};

struct LidarEdgeStateFactorEx
{
	LidarEdgeStateFactorEx(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
						   Eigen::Vector3d last_point_b_, double s_, Eigen::Vector3d rn_, Eigen::Quaterniond qbn_, double cov_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

	template <typename T>
	bool operator()(const T *dx, const T *ex, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

       	Eigen::Matrix<T, 3, 1> t_b_l{ex[0], ex[1], ex[2]};
        Eigen::Quaternion<T> q_b_l{ex[6], ex[3], ex[4], ex[5]};

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
        Eigen::Matrix<T, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

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
				LidarEdgeStateFactorEx, 1, 18, 7>(
			new LidarEdgeStateFactorEx(curr_point_, last_point_a_, last_point_b_, s_, rn_, qbn_, cov_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};

struct LidarPlaneStateFactorEx
{
	LidarPlaneStateFactorEx(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, 
                    	    Eigen::Vector3d last_point_b_, Eigen::Vector3d last_point_c_, 
                    	    double s_, Eigen::Vector3d rn_, Eigen::Quaterniond qbn_, double cov_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), 
        last_point_c(last_point_c_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

	template <typename T>
	bool operator()(const T *dx, const T *ex, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};
		Eigen::Matrix<T, 3, 1> lpc{T(last_point_c.x()), T(last_point_c.y()), T(last_point_c.z())};

        Eigen::Matrix<T, 3, 1> t_b_l{ex[0], ex[1], ex[2]};
        Eigen::Quaternion<T> q_b_l{ex[6], ex[3], ex[4], ex[5]};

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
        Eigen::Matrix<T, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);
        Eigen::Matrix<T, 3, 1> M = (lpa - lpb).cross(lpa - lpc);
        residual[0] = ((lp - lpa).transpose() * M).norm() / (M.norm() * T(cov));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const Eigen::Vector3d last_point_c_,
                                       const double s_, const Eigen::Vector3d rn_, const Eigen::Quaterniond qbn_, const double cov_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneStateFactorEx, 1, 18, 7>(
			new LidarPlaneStateFactorEx(curr_point_, last_point_a_, last_point_b_, last_point_c_, s_, rn_, qbn_, cov_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b, last_point_c;
	double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
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

class PriorFactorExSF: public ceres::SizedCostFunction<24, 18, 7>
{
public:
    PriorFactorExSF(const Eigen::Matrix<double, 24, 24> &covariance_, const Eigen::Quaterniond &ex_rotation_,
                    const Eigen::Vector3d &ex_translation_): covariance(covariance_),
        ex_rotation(ex_rotation_), ex_translation(ex_translation_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_V{parameters[0][3], parameters[0][4], parameters[0][5]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d delta_Ba{parameters[0][9], parameters[0][10], parameters[0][11]};
        Eigen::Vector3d delta_Bg{parameters[0][12], parameters[0][13], parameters[0][14]};
        Eigen::Vector3d delta_g{parameters[0][15], parameters[0][16], parameters[0][17]};

        Eigen::Quaterniond q_ex{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
        Eigen::Vector3d t_ex{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Map<Eigen::Matrix<double, 24, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = delta_P;
        residual.block<3, 1>(3, 0) = delta_V;
        residual.block<3, 1>(6, 0) = delta_theta;
        residual.block<3, 1>(9, 0) = delta_Ba;
        residual.block<3, 1>(12, 0) = delta_Bg;
        residual.block<3, 1>(15, 0) = delta_g;
        residual.block<3, 1>(18, 0) = t_ex - ex_translation;
        residual.block<3, 1>(21, 0) = 2 * (q_ex * ex_rotation.inverse()).vec();
        Eigen::MatrixXd sqrt_info = Eigen::LLT<Eigen::Matrix<double, 24, 24>>(covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 24, 18, Eigen::RowMajor>> jacobian_error_state(jacobians[0]);
                jacobian_error_state.setZero();
                jacobian_error_state.block<18, 18>(0, 0) = Eigen::MatrixXd::Identity(18, 18);
                jacobian_error_state = sqrt_info * jacobian_error_state;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 24, 7, Eigen::RowMajor>> jacobian_ex(jacobians[1]);
                jacobian_ex.setZero();
                jacobian_ex.block<3, 3>(18, 0) = Eigen::Matrix3d::Identity();
                jacobian_ex.block<3, 3>(21, 3) = utils::Qright(q_ex * ex_rotation.inverse()).bottomRightCorner<3, 3>();
                jacobian_ex = sqrt_info * jacobian_ex;
            }
        }
        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[24];
        double **jaco = new double *[2];
        jaco[0] = new double[24 * 18];
        jaco[1] = new double[24 * 7];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 24, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 24, 18, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 24, 7, Eigen::RowMajor>>(jaco[1]) << std::endl << std::endl;

        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_V{parameters[0][3], parameters[0][4], parameters[0][5]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d delta_Ba{parameters[0][9], parameters[0][10], parameters[0][11]};
        Eigen::Vector3d delta_Bg{parameters[0][12], parameters[0][13], parameters[0][14]};
        Eigen::Vector3d delta_g{parameters[0][15], parameters[0][16], parameters[0][17]};

        Eigen::Quaterniond q_ex{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
        Eigen::Vector3d t_ex{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Matrix<double, 24, 1> residual;
        residual.block<3, 1>(0, 0) = delta_P;
        residual.block<3, 1>(3, 0) = delta_V;
        residual.block<3, 1>(6, 0) = delta_theta;
        residual.block<3, 1>(9, 0) = delta_Ba;
        residual.block<3, 1>(12, 0) = delta_Bg;
        residual.block<3, 1>(15, 0) = delta_g;
        residual.block<3, 1>(18, 0) = t_ex - ex_translation;
        residual.block<3, 1>(21, 0) = 2 * (q_ex * ex_rotation.inverse()).vec();
        Eigen::MatrixXd sqrt_info = Eigen::LLT<Eigen::Matrix<double, 24, 24>>(covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual;

        puts("num");
        std::cout << residual.transpose() << std::endl << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 24, 18> num_jacobian_error_state;
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

            Eigen::Matrix<double, 24, 1> tmp_residual;
            tmp_residual.block<3, 1>(0, 0) = delta_P_new;
            tmp_residual.block<3, 1>(3, 0) = delta_V_new;
            tmp_residual.block<3, 1>(6, 0) = delta_theta_new;
            tmp_residual.block<3, 1>(9, 0) = delta_Ba_new;
            tmp_residual.block<3, 1>(12, 0) = delta_Bg_new;
            tmp_residual.block<3, 1>(15, 0) = delta_g_new;
            tmp_residual.block<3, 1>(18, 0) = t_ex - ex_translation;
            tmp_residual.block<3, 1>(21, 0) = 2 * (q_ex * ex_rotation.inverse()).vec();
            tmp_residual = sqrt_info * tmp_residual;

            num_jacobian_error_state.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_error_state << std::endl << std::endl;

        Eigen::Matrix<double, 24, 7> num_jacobian_ex;
        for (int k = 0; k < 6; k++) {
            Eigen::Quaterniond q_ex_new{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
            Eigen::Vector3d t_ex_new{parameters[1][0], parameters[1][1], parameters[1][2]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                t_ex_new += delta;
            }
            else if (a == 1) {
                q_ex_new = utils::deltaQ(delta) * q_ex_new;
            }
            /*Eigen::Quaterniond dq = utils::axis2Quat(delta);
            Eigen::AngleAxisd dq(delta.norm(), delta / delta.norm());
            q_ex_new = dq * q_ex_new;
            q_ex_new.normalized(); */

            Eigen::Matrix<double, 24, 1> tmp_residual;
            tmp_residual.block<3, 1>(0, 0) = delta_P;
            tmp_residual.block<3, 1>(3, 0) = delta_V;
            tmp_residual.block<3, 1>(6, 0) = delta_theta;
            tmp_residual.block<3, 1>(9, 0) = delta_Ba;
            tmp_residual.block<3, 1>(12, 0) = delta_Bg;
            tmp_residual.block<3, 1>(15, 0) = delta_g;
            tmp_residual.block<3, 1>(18, 0) = t_ex_new - ex_translation;
            tmp_residual.block<3, 1>(21, 0) = 2 * (q_ex_new * ex_rotation.inverse()).vec();
            tmp_residual = sqrt_info * tmp_residual;

            num_jacobian_ex.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_ex << std::endl << std::endl;
    }

    Eigen::Matrix<double, 24, 24> covariance;
    Eigen::Quaterniond ex_rotation;
    Eigen::Vector3d ex_translation;
};

class LidarEdgeFactorExSF: public ceres::SizedCostFunction<1, 18, 7>
{
public:
    LidarEdgeFactorExSF(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_a_,
                        const Eigen::Vector3d &last_point_b_, double s_, const Eigen::Vector3d &rn_,
                        const Eigen::Quaterniond &qbn_, double cov_): curr_point(curr_point_),
        last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Quaterniond q_b_l{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
        Eigen::Vector3d t_b_l{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        } 
        //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta);

        Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
        q_last_curr = q_identity.slerp(s, q_last_curr);

        nominal_rn = dq * nominal_rn + delta_P;

        Eigen::Vector3d t_last_curr = s * nominal_rn;
        Eigen::Matrix<double, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

        Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<double, 3, 1> de = lpa - lpb;

        Eigen::Map<Eigen::Matrix<double, 1, 1>> residual(residuals);
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / (de.norm() * cov) );

        Eigen::Vector3d coeff = nu.transpose() * utils::skew(lpb - lpa) / (nu.norm() * de.norm());

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>> jacobian_error_state(jacobians[0]);
                jacobian_error_state.setZero();
                // Use Slerp(q0, q1, t) = q0 * (q0^{-1} * q1)^t
                Eigen::Vector3d theta = utils::vee(nominal_qbn.toRotationMatrix().log());
                Eigen::Matrix3d J_l_inv = utils::Jinvleft(theta);
                Eigen::Matrix3d J_l_s = utils::Jleft(s * theta);
                Eigen::Matrix3d J_l_d = utils::Jleft(delta_theta);
                Eigen::Matrix3d J_l_coeff = utils::Jleft(s * J_l_s * J_l_inv * delta_theta);
                jacobian_error_state.block<1, 3>(0, 0) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * s;
                jacobian_error_state.block<1, 3>(0, 6) = -coeff.transpose() * q_b_l.inverse().toRotationMatrix() *
                        (utils::skew(q_last_curr * (q_b_l * cp + t_b_l)) * J_l_coeff * J_l_s * s * J_l_inv +
                        s * utils::skew(dq * rn) * J_l_d);

                // Use Slerp(q0, q1, t) = sin((1 - t) * theta) / sin(theta) * q0 + sin(t * theta) / sin(theta) * q1
                /*Eigen::Quaterniond q0{1.0, 0.0, 0.0, 0.0}, q1 = (dq * nominal_qbn).normalized();
                Eigen::Matrix<double, 2, 1> scalar = utils::getSlerpCoeff(q0, q1, s);
                Eigen::Matrix3d J_l = utils::Jleft(delta_theta);
                jacobian_error_state.block<1, 3>(0, 0) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * s;
                jacobian_error_state.block<1, 3>(0, 6) = -coeff.transpose() * q_b_l.inverse().toRotationMatrix() *
                        (utils::skew(dq * nominal_qbn * (q_b_l * cp + t_b_l)) * J_l * scalar(1) +
                        s * utils::skew(dq * rn) * J_l); */
                jacobian_error_state = jacobian_error_state / cov;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_ex(jacobians[1]);
                jacobian_ex.setZero();
                Eigen::Vector3d P = q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l;
                jacobian_ex.block<1, 3>(0, 0) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * (q_last_curr.toRotationMatrix() - Eigen::Matrix3d::Identity());
                jacobian_ex.block<1, 3>(0, 3) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * (-q_last_curr.toRotationMatrix() * utils::skew(q_b_l * cp) + utils::skew(P));
                jacobian_ex = jacobian_ex / cov;
            }
        }

        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[1];
        double **jaco = new double *[2];
        jaco[0] = new double[1 * 18];
        jaco[1] = new double[1 * 7];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[1]) << std::endl << std::endl;

        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Quaterniond q_b_l{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
        Eigen::Vector3d t_b_l{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        } 
        //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta);

        Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
        q_last_curr = q_identity.slerp(s, q_last_curr);

        nominal_rn = dq * nominal_rn + delta_P;

        Eigen::Vector3d t_last_curr = s * nominal_rn;
        Eigen::Matrix<double, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

        Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<double, 3, 1> de = lpa - lpb;
        Eigen::Matrix<double, 1, 1> residual;
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / (de.norm() * cov) );

        puts("num");
        std::cout << residual.transpose() << std::endl <<  std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 18> num_jacobian_error_state;
        for (int k = 0; k < 18; ++k) {
            Eigen::Vector3d delta_P_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Vector3d delta_theta_new{parameters[0][6], parameters[0][7], parameters[0][8]};
            Eigen::Vector3d nominal_rn = rn;
            Eigen::Quaterniond nominal_qbn = qbn;

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
            //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta_new);

            Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
            Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
            q_last_curr = q_identity.slerp(s, q_last_curr);

            nominal_rn = dq * nominal_rn + delta_P_new;

            Eigen::Vector3d t_last_curr = s * nominal_rn;
            Eigen::Matrix<double, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

            Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
            Eigen::Matrix<double, 3, 1> de = lpa - lpb;
            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / (de.norm() * cov) );

            num_jacobian_error_state.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_error_state << std::endl << std::endl;

        Eigen::Matrix<double, 1, 7> num_jacobian_ex;
        for (int k = 0; k < 6; k++) {
            Eigen::Vector3d nominal_rn = rn;
            Eigen::Quaterniond nominal_qbn = qbn;
            Eigen::Quaterniond q_b_l_new{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
            Eigen::Vector3d t_b_l_new{parameters[1][0], parameters[1][1], parameters[1][2]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                t_b_l_new += delta;
            }
            else if (a == 1) {
                q_b_l_new = utils::deltaQ(delta) * q_b_l_new;
            }

            Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
            if (delta_theta.norm() > 1e-10) {
                Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
                dq = axis_dq;
            } 
            //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta);

            Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
            Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
            q_last_curr = q_identity.slerp(s, q_last_curr);

            nominal_rn = dq * nominal_rn + delta_P;

            Eigen::Vector3d t_last_curr = s * nominal_rn;
            Eigen::Matrix<double, 3, 1> lp = q_b_l_new.inverse() * (q_last_curr * (q_b_l_new * cp + t_b_l_new) + t_last_curr - t_b_l_new);

            Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
            Eigen::Matrix<double, 3, 1> de = lpa - lpb;

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / (de.norm() * cov) );

            num_jacobian_ex.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_ex << std::endl << std::endl;
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};

class LidarPlaneFactorExSF: public ceres::SizedCostFunction<1, 18, 7>
{
public:
    LidarPlaneFactorExSF(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_a_,
                         const Eigen::Vector3d &last_point_b_, const Eigen::Vector3d &last_point_c_, double s_,
                         const Eigen::Vector3d &rn_, const Eigen::Quaterniond &qbn_, double cov_):
        curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_),
        last_point_c(last_point_c_), s(s_), rn(rn_), qbn(qbn_), cov(cov_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Quaterniond q_b_l{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
        Eigen::Vector3d t_b_l{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;
        Eigen::Vector3d lpc = last_point_c;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        } 
        //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta);

        Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
        q_last_curr = q_identity.slerp(s, q_last_curr);

        nominal_rn = dq * nominal_rn + delta_P;

        Eigen::Vector3d t_last_curr = s * nominal_rn;
        Eigen::Matrix<double, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

        Eigen::Matrix<double, 3, 1> M = (lpa - lpb).cross(lpa - lpc);

        Eigen::Map<Eigen::Matrix<double, 1, 1>> residual(residuals);
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( ((lp - lpa).transpose() * M).norm() / (M.norm() * cov) );

        Eigen::MatrixXd tmp = (lp - lpa).transpose() * M;
        Eigen::Vector3d coeff = utils::sign(tmp(0, 0)) * M.transpose() / M.norm();

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>> jacobian_error_state(jacobians[0]);
                jacobian_error_state.setZero();
                // Use Slerp(q0, q1, t) = q0 * (q0^{-1} * q1)^t
                Eigen::Vector3d theta = utils::vee(nominal_qbn.toRotationMatrix().log());
                Eigen::Matrix3d J_l_inv = utils::Jinvleft(theta);
                Eigen::Matrix3d J_l_s = utils::Jleft(s * theta);
                Eigen::Matrix3d J_l_d = utils::Jleft(delta_theta);
                Eigen::Matrix3d J_l_coeff = utils::Jleft(s * J_l_s * J_l_inv * delta_theta);
                jacobian_error_state.block<1, 3>(0, 0) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * s;
                jacobian_error_state.block<1, 3>(0, 6) = -coeff.transpose() * q_b_l.inverse().toRotationMatrix() *
                        (utils::skew(q_last_curr * (q_b_l * cp + t_b_l)) * J_l_coeff * J_l_s * s * J_l_inv +
                        s * utils::skew(dq * rn) * J_l_d);

                // Use Slerp(q0, q1, t) = sin((1 - t) * theta) / sin(theta) * q0 + sin(t * theta) / sin(theta) * q1
                /*Eigen::Quaterniond q0{1.0, 0.0, 0.0, 0.0}, q1 = (dq * nominal_qbn).normalized();
                Eigen::Matrix<double, 2, 1> scalar = utils::getSlerpCoeff(q0, q1, s);
                Eigen::Matrix3d J_l = utils::Jleft(delta_theta);
                jacobian_error_state.block<1, 3>(0, 0) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * s;
                jacobian_error_state.block<1, 3>(0, 6) = -coeff.transpose() * q_b_l.inverse().toRotationMatrix() *
                        (utils::skew(dq * nominal_qbn * (q_b_l * cp + t_b_l)) * J_l * scalar(1) +
                        s * utils::skew(dq * rn) * J_l); */
                jacobian_error_state = jacobian_error_state / cov;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_ex(jacobians[1]);
                jacobian_ex.setZero();
                Eigen::Vector3d P = q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l;
                jacobian_ex.block<1, 3>(0, 0) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * (q_last_curr.toRotationMatrix() - Eigen::Matrix3d::Identity());
                jacobian_ex.block<1, 3>(0, 3) = coeff.transpose() * q_b_l.inverse().toRotationMatrix() * (-q_last_curr.toRotationMatrix() * utils::skew(q_b_l * cp) + utils::skew(P));
                jacobian_ex = jacobian_ex / cov;
            }
        }

        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[1];
        double **jaco = new double *[2];
        jaco[0] = new double[1 * 18];
        jaco[1] = new double[1 * 7];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 18, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[1]) << std::endl << std::endl;

        Eigen::Vector3d delta_P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Vector3d delta_theta{parameters[0][6], parameters[0][7], parameters[0][8]};
        Eigen::Vector3d nominal_rn = rn;
        Eigen::Quaterniond nominal_qbn = qbn;

        Eigen::Quaterniond q_b_l{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
        Eigen::Vector3d t_b_l{parameters[1][0], parameters[1][1], parameters[1][2]};

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;
        Eigen::Vector3d lpc = last_point_c;

        Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
        if (delta_theta.norm() > 1e-10) {
            Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
            dq = axis_dq;
        } 
        //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta);

        Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
        Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
        q_last_curr = q_identity.slerp(s, q_last_curr);

        nominal_rn = dq * nominal_rn + delta_P;

        Eigen::Vector3d t_last_curr = s * nominal_rn;
        Eigen::Matrix<double, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

        Eigen::Matrix<double, 3, 1> M = (lpa - lpb).cross(lpa - lpc);

        Eigen::Matrix<double, 1, 1> residual;
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( ((lp - lpa).transpose() * M).norm() / (M.norm() * cov) );

        puts("num");
        std::cout << residual.transpose() << std::endl <<  std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 18> num_jacobian_error_state;
        for (int k = 0; k < 18; ++k) {
            Eigen::Vector3d delta_P_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Vector3d delta_theta_new{parameters[0][6], parameters[0][7], parameters[0][8]};
            Eigen::Vector3d nominal_rn = rn;
            Eigen::Quaterniond nominal_qbn = qbn;

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
            //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta_new);

            Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
            Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
            q_last_curr = q_identity.slerp(s, q_last_curr);

            nominal_rn = dq * nominal_rn + delta_P_new;

            Eigen::Vector3d t_last_curr = s * nominal_rn;
            Eigen::Matrix<double, 3, 1> lp = q_b_l.inverse() * (q_last_curr * (q_b_l * cp + t_b_l) + t_last_curr - t_b_l);

            Eigen::Matrix<double, 3, 1> M = (lpa - lpb).cross(lpa - lpc);

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( ((lp - lpa).transpose() * M).norm() / (M.norm() * cov) );

            num_jacobian_error_state.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_error_state << std::endl << std::endl;

        Eigen::Matrix<double, 1, 7> num_jacobian_ex;
        for (int k = 0; k < 6; k++) {
            Eigen::Vector3d nominal_rn = rn;
            Eigen::Quaterniond nominal_qbn = qbn;
            Eigen::Quaterniond q_b_l_new{parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};
            Eigen::Vector3d t_b_l_new{parameters[1][0], parameters[1][1], parameters[1][2]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                t_b_l_new += delta;
            }
            else if (a == 1) {
                q_b_l_new = utils::deltaQ(delta) * q_b_l_new;
            }

            Eigen::Quaterniond dq{1.0, 0.0, 0.0, 0.0};
            if (delta_theta.norm() > 1e-10) {
                Eigen::AngleAxisd axis_dq(delta_theta.norm(), delta_theta / delta_theta.norm());
                dq = axis_dq;
            } 
            //Eigen::Quaterniond dq = utils::axis2Quat(delta_theta);

            Eigen::Quaterniond q_last_curr = (dq * nominal_qbn).normalized();
            Eigen::Quaterniond q_identity{1.0, 0.0, 0.0, 0.0};
            q_last_curr = q_identity.slerp(s, q_last_curr);

            nominal_rn = dq * nominal_rn + delta_P;

            Eigen::Vector3d t_last_curr = s * nominal_rn;
            Eigen::Matrix<double, 3, 1> lp = q_b_l_new.inverse() * (q_last_curr * (q_b_l_new * cp + t_b_l_new) + t_last_curr - t_b_l_new);

            Eigen::Matrix<double, 3, 1> M = (lpa - lpb).cross(lpa - lpc);

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( ((lp - lpa).transpose() * M).norm() / (M.norm() * cov) );

            num_jacobian_ex.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_ex << std::endl << std::endl;
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b, last_point_c;
    double s;
    Eigen::Vector3d rn;
    Eigen::Quaterniond qbn;
    double cov;
};

class LidarMapEdgeFactor: public ceres::SizedCostFunction<1, 7>
{
public:
    LidarMapEdgeFactor(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_a_,
                       const Eigen::Vector3d &last_point_b_): curr_point(curr_point_),
        last_point_a(last_point_a_), last_point_b(last_point_b_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d t_w_curr{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond q_w_curr{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;

        Eigen::Matrix<double, 3, 1> lp = q_w_curr * cp + t_w_curr;

        Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<double, 3, 1> de = lpa - lpb;

        residuals[0] = nu.norm() / de.norm();

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                Eigen::Vector3d coeff = nu.transpose() * utils::skew(lpb - lpa) / (nu.norm() * de.norm());
                jacobian_pose.block<1, 3>(0, 0) = coeff.transpose();
                jacobian_pose.block<1, 3>(0, 3) = -coeff.transpose() * utils::skew(q_w_curr * cp);
            }
        }

        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;

        Eigen::Vector3d t_w_curr{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond q_w_curr{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

        Eigen::Vector3d cp = curr_point;
        Eigen::Vector3d lpa = last_point_a;
        Eigen::Vector3d lpb = last_point_b;

        Eigen::Matrix<double, 3, 1> lp = q_w_curr * cp + t_w_curr;

        Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<double, 3, 1> de = lpa - lpb;
        Eigen::Matrix<double, 1, 1> residual;
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / de.norm() );

        puts("num");
        std::cout << residual.transpose() << std::endl <<  std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 7> num_jacobian_pose;
        for (int k = 0; k < 6; k++) {
            Eigen::Vector3d t_w_curr_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Quaterniond q_w_curr_new{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                t_w_curr_new += delta;
            }
            else if (a == 1) {
                q_w_curr_new = utils::deltaQ(delta) * q_w_curr_new;
            }

            Eigen::Matrix<double, 3, 1> lp = q_w_curr_new * cp + t_w_curr_new;

            Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
            Eigen::Matrix<double, 3, 1> de = lpa - lpb;

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( nu.norm() / de.norm() );

            num_jacobian_pose.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_pose << std::endl << std::endl;
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
};

class LidarMapPlaneNormFactor: public ceres::SizedCostFunction<1, 7>
{
public:
    LidarMapPlaneNormFactor(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &plane_unit_norm_,
                        double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
        negative_OA_dot_norm(negative_OA_dot_norm_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d t_w_curr{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond q_w_curr{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

        Eigen::Vector3d norm = plane_unit_norm;
        Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;

        residuals[0] = norm.dot(point_w) + negative_OA_dot_norm;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                jacobian_pose.leftCols<3>() = norm.transpose();
                jacobian_pose.block<1, 3>(0, 3) = -norm.transpose() * utils::skew(q_w_curr * curr_point);
            }
        }
        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(parameters, res, jaco);
        puts("check begin");
        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 1>>(res).transpose() << std::endl << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl << std::endl;

        Eigen::Vector3d t_w_curr{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond q_w_curr{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

        Eigen::Vector3d norm = plane_unit_norm;
        Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;

        Eigen::Matrix<double, 1, 1> residual;
        residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( norm.dot(point_w) + negative_OA_dot_norm );

        puts("num");
        std::cout << residual.transpose() << std::endl <<  std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 7> num_jacobian_pose;
        for (int k = 0; k < 6; k++) {
            Eigen::Vector3d t_w_curr_new{parameters[0][0], parameters[0][1], parameters[0][2]};
            Eigen::Quaterniond q_w_curr_new{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0) {
                t_w_curr_new += delta;
            }
            else if (a == 1) {
                q_w_curr_new = utils::deltaQ(delta) * q_w_curr_new;
            }

            Eigen::Vector3d point_w = q_w_curr_new * curr_point + t_w_curr_new;

            Eigen::Matrix<double, 1, 1> tmp_residual;
            tmp_residual.leftCols<1>() = Eigen::Matrix<double, 1, 1>( norm.dot(point_w) + negative_OA_dot_norm );

            num_jacobian_pose.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian_pose << std::endl << std::endl;
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
};