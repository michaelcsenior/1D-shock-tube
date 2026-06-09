// created 2026-06-08 by Michael
#pragma region libraries
#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "vector_conversion.hpp"
namespace vec = vector_conversion; // name aliasing for imported modules
#include "exact_riemann_solver.hpp"
namespace ers = exact_riemann_solver;
#pragma endregion

using Array3Xd = Eigen::Array3Xd;

namespace finite_volume_riemann_solver {

#pragma schemes
class Scheme {
    const std::string name_;
    const float cfl_limit_; // theoretical CFL limit
    const bool tvd_; // flase by default (scheme is not TVD)
    std::string limiter_; // "None" by default

public:
    Scheme(std::string name, float cfl_limit,
           bool tvd = false, std::string limiter = "None"
          ) : name_(name), cfl_limit_(cfl_limit), tvd_(tvd), limiter_(limiter)
          {}
    
    virtual ~Scheme() = default;

    // advanced the solution (conservative variables) by one time step
    virtual Array3Xd march(const Array3Xd& u, // (3, n_cells)
                           double dx,
                           double dt,
                           double gamma) = 0;
                           // pure virtual method
};

// Godunov's flux difference splitting method for solving the 1D Euler equations.
// This is a first order accurate upwind scheme that computes interface fluxes 
// by computing the exact solution of the Riemann problem from left and right 
// states at each cell interface. (if left and right states are equal, the 
// Riemann solver is skipped)
// zero-gradient boundary condition is used (assume boundaries are situated in 
// quiescent gas, so the fluxes are calculated using values from the control 
// volume adjacent to the boundary)
class Godunov : public Scheme {
public:
    Godunov() : Scheme("Godunov", 1.0) {}

    ~Godunov() = default;

    Array3Xd march(const Array3Xd& u, 
                   double dx, double dt, double gamma) override {
        const Eigen::Index n_cells = u.cols();
        Array3Xd w = Array3Xd::Zero(3, n_cells); // primitive variable array
        vec::primitive_from_conservative(u, w, gamma); // populate w
        // using auto so slicing returns a view, rather than a copy
        const auto wlefts  = w.leftCols(n_cells-1); // (3, n_cells-1)
        const auto wrights = w.rightCols(n_cells-1); // (3, n_cells-1)
        
        // if left and right states are equal, don't need exact Riemann solver
        const double abs_tol = 1e-6;
        const double rel_tol = 1e-4;
        const auto diff = (wlefts - wrights).abs();
            // auto preserves lazy expression
        const auto scale = wlefts.abs().max(wrights.abs()); // (3, n_cells-1)
        const Eigen::Array<bool,1,Eigen::Dynamic> // (1, n_cells-1)
            col_is_equal = (diff < (abs_tol + rel_tol*scale)).colwise().all();

        Array3Xd w_interfaces = Array3Xd::Zero(3, n_cells-1);
        // true: no Riemann solver needed. make interface value the average of 
        // left and right value
        // false: evolve w at interfaces using exact Riemann solver
        for (Eigen::Index j = 0; j < n_cells-1; j++) {
            if (col_is_equal(j)) {
                w_interfaces.col(j) = 0.5 * (wlefts.col(j)+wrights.col(j));
            } else {
                Eigen::Vector3d wleft = wlefts.col(j);
                Eigen::Vector3d wright = wrights.col(j);
                w_interfaces.col(j) = ers::get_s_state(ers::get_wave_config(
                                                                        wleft, 
                                                                        wright,
                                                                        gamma)
                                                      );
            }
        }

        // convert to f for finite volume
        Array3Xd f_center = Array3Xd::Zero(3, n_cells-1);
        vec::flux_from_primitive(w_interfaces, f_center, gamma); // populate f
        
        // treatment for boundary nodes: assume boundaries are situated in 
        // quiescent gas, so the fluxes are calculated using values from the 
        // control volume adjacent to the boundary
        Array3Xd f_leftboundary(3, 1);
        Array3Xd f_rightboundary(3, 1);
        vec::flux_from_conservative(u.col(0), f_leftboundary, gamma);
        vec::flux_from_conservative(u.col(n_cells-1), f_rightboundary, gamma);

        Array3Xd f(3, n_cells+1);
        f << f_leftboundary, f_center, f_rightboundary;

        const auto flux_diff = f.rightCols(n_cells) - f.leftCols(n_cells);
        const Array3Xd u_new = u - dt/dx * flux_diff; // (3, n_cells)

        return u_new;
    }
};

class AUSMPlus : public Scheme {
    
}
#pragma endregion










#pragma region functions
// runtime warning
void warning(const std::string& message) {
    std::cerr << "Warning: " << message << std::endl;
}
#pragma endregion










} // end of namespace finite_volume_riemann_solver