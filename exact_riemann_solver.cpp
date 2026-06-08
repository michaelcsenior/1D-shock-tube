// created 2026-06-02 by Michael
#pragma region libraries
#include <iostream>
#include <string>
#include <cmath>
#include <utility>
#include <optional>
#include <stdexcept>

#include <Eigen/Core>

#include "exact_riemann_solver.hpp"
#pragma endregion

using Array31d = Eigen::Array<double, 3, 1>;

#pragma region class implementations
// class IniCond
    double IniCond::calculate_speed_of_sound(double rho, double p, double gamma)
    {
        return std::sqrt(gamma * p / rho);
    }

    IniCond::IniCond(const Array31d& wleft, const Array31d& wright, 
                     double gamma_in)
         : rholeft(wleft(0)), 
           uleft(wleft(1)),
           pleft(wleft(2)),
           rhoright(wright(0)),
           uright(wright(1)),
           pright(wright(2)),
           cleft(calculate_speed_of_sound(rholeft, pleft, gamma_in)),
           cright(calculate_speed_of_sound(rhoright, pright, gamma_in)),
           gamma(gamma_in)
        { // validate input (this is after sqrt operation but ok)
            if (rholeft <= 0 || pleft <= 0 || rhoright <= 0 || pright <= 0) {
                throw std::invalid_argument(
                             "Density and pressure must be strictly positive.");
            }
            if (gamma <= 1) { throw std::invalid_argument(
                              "Ratio of specific heat much be greater than 1.");
            }

            // verify that IC's does not give a vaccum solution
            if (uright - uleft - 2*(cleft+cright)/(gamma-1) >= 0) {
                throw std::invalid_argument("Initial conditions result in "
                                            "a vacuum solution.");
            }
        }
    
    // Accessers
    Array31d IniCond::left_state() const {
        return Array31d(rholeft, uleft, pleft);
    }

    Array31d IniCond::right_state() const {
        return Array31d(rhoright, uright, pright);
    }

// class WaveConfig
    // using factory pattern, constructor is a private member
    WaveConfig::WaveConfig(double pstar_in, double ustar_in, 
                           bool is_leftshock_in, bool is_rightshock_in)
                          : pstar(pstar_in), ustar(ustar_in), 
                            is_leftshock(is_leftshock_in), 
                            is_rightshock(is_rightshock_in)
                          {}
    
    // helper functions for solving configuration
    #pragma region helper funcs
    // get f and fprime values for each wave type
    std::pair<double, double> WaveConfig::f_leftshock(double pstar,
                                                      double rholeft, 
                                                      double pleft, 
                                                      double gamma) {
        double common_term = std::sqrt(
            2 / ((gamma+1) * rholeft * (pstar + (gamma-1)*pleft/(gamma+1)))   );

        double f = (pstar-pleft) * common_term;
        double fprime = common_term * (1 - 0.5*(pstar-pleft)/
                                           (pstar + pleft*(gamma-1)/(gamma+1))
                                       );

        return {f, fprime};
    }

    std::pair<double, double> WaveConfig::f_leftfan(double pstar, 
                                                    double rholeft,
                                                    double pleft,
                                                    double cleft,
                                                    double gamma) {
        double f = 2 * cleft *
            (std::pow(pstar/pleft, (gamma-1)/(2*gamma)) - 1) / (gamma-1) ;
        double fprime = std::pow(pstar/pleft, -(gamma+1)/(2*gamma)) 
                        / (rholeft*cleft);

        return {f, fprime};
    }

    std::pair<double, double> WaveConfig::f_rightshock(double pstar,
                                                       double rhoright,
                                                       double pright,
                                                       double gamma) {
        double common_term = std::sqrt(
            2 / ((gamma+1) * rhoright * (pstar + (gamma-1)*pright/(gamma+1))) );

        double f = (pstar-pright) * common_term;
        double fprime = common_term * (1 - 0.5*(pstar-pright)/
                                           (pstar + pright*(gamma-1)/(gamma+1))
                                       );

        return {f, fprime};
    }

    std::pair<double, double> WaveConfig::f_rightfan(double pstar,
                                                     double rhoright,
                                                     double pright,
                                                     double cright,
                                                     double gamma) {
        double f = 2 * cright *
            (std::pow(pstar/pright, (gamma-1)/(2*gamma)) - 1) / (gamma-1) ;
        double fprime = std::pow(pstar/pright, -(gamma+1)/(2*gamma)) 
                        / (rhoright*cright);

        return {f, fprime};
    }

    // values for the iterative equation for Newton-Raphson method
    std::pair<double, double> WaveConfig::leftshock_rightshock(
                                    double pstar_current, const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
                f_leftshock(pstar_current, ic.rholeft, ic.pleft, ic.gamma);
        auto [fright, frightprime] = // double, double 
                f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }

    std::pair<double, double> WaveConfig::leftshock_rightfan(
                                    double pstar_current, const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
                    f_leftshock(pstar_current, ic.rholeft, ic.pleft, ic.gamma);
        auto [fright, frightprime] = // double, double
                    f_rightfan(pstar_current, ic.rhoright, 
                               ic.pright, ic.cright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;

        return {eqn_value, derivative_value};
    }

    std::pair<double, double> WaveConfig::leftfan_rightshock(
                                    double pstar_current, const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
            f_leftfan(pstar_current, ic.rholeft, ic.pleft, ic.cleft, ic.gamma);
        auto [fright, frightprime] = // double, double 
            f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }

    std::pair<double, double> WaveConfig::leftfan_rightfan(double pstar_current,
                                                            const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
            f_leftfan(pstar_current, ic.rholeft, ic.pleft, ic.cleft, ic.gamma);
        auto [fright, frightprime] = // double, double
            f_rightfan(pstar_current, ic.rhoright, 
                       ic.pright, ic.cright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }
    #pragma endregion

    WaveConfig WaveConfig::solve_config(const IniCond& ic) { // factory method
        double pstar_current = // initial guess using acoustic approximation
            (ic.pleft*ic.rhoright*ic.cright + ic.pright*ic.rholeft*ic.cleft + 
             (ic.uleft-ic.uright)*ic.rholeft*ic.cleft*ic.rhoright*ic.cright
            ) / (ic.rhoright*ic.cright + ic.rholeft*ic.cleft);
        // under some IC's the acoustic approximation may be negative:
        if (pstar_current <= 0) {pstar_current = 1e-6;}

        bool converged = false;
        int num_iter = 0;
        while (num_iter++ < 30) {
            double eqn_value = -1.0;
            double derivative_value = -1.0;
            double pstar_next = -1.0;
            double eqn_value_prev = -1.0;

            if (pstar_current > ic.pleft) { // left shock
                if (pstar_current > ic.pright) { // right shock
                    auto result = leftshock_rightshock(pstar_current, ic);
                    eqn_value = result.first;
                    derivative_value = result.second;
                } else { // right fan
                    auto result = leftshock_rightfan(pstar_current, ic);
                    eqn_value = result.first;
                    derivative_value = result.second;
                }
            } else { // left fan
                if (pstar_current > ic.pright) { // right shock
                    auto result = leftfan_rightshock(pstar_current, ic);
                    eqn_value = result.first;
                    derivative_value = result.second;
                } else { // right fan
                    auto result = leftfan_rightfan(pstar_current, ic);
                    eqn_value = result.first;
                    derivative_value = result.second;
                }
            }

            if (std::abs(derivative_value) < 1e-10) {
                throw std::runtime_error("Division by zero encountered in "
                                         "Newton-Raphson method.");
            }
            // Newton-Raphson step
            pstar_next = pstar_current - eqn_value / derivative_value;

            if (std::abs(pstar_next - pstar_current)/pstar_current < 1e-6
                && std::abs(eqn_value) < 1e-3) {            // convergence
                pstar_current = pstar_next;
                converged = true;
                break;
            }

            if (pstar_next <= 0) {pstar_next = pstar_current/2;}
            pstar_current = pstar_next;
        }
        
        // check iteration limit
        if (!converged) {
            throw std::runtime_error("Newton-Raphson failed to converge within "
                                      "30 iterations.");
        }

        // now check wave config and calculate ustar
        double fleft = 0.0;
        double fright = 0.0;
        bool is_leftshock = false;
        bool is_rightshock = false;

        if (pstar_current > ic.pleft) {
            is_leftshock = true;
            auto result = 
                    f_leftshock(pstar_current, ic.rholeft, ic.pleft, ic.gamma);
            fleft = result.first;
        } else {
            auto result = f_leftfan(pstar_current, ic.rholeft, 
                                    ic.pleft, ic.cleft, ic.gamma);
            fleft = result.first;
        }

        if (pstar_current > ic.pright) {
            is_rightshock = true;
            auto result = 
                f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
            fright = result.first;
        } else {
            auto result = f_rightfan(pstar_current, ic.rhoright, 
                                     ic.pright, ic.cright, ic.gamma);
            fright = result.first;
        }

        double ustar = 0.5 * (ic.uleft + ic.uright + fright - fleft);

        return WaveConfig(pstar_current, ustar, is_leftshock, is_rightshock);
    }

// class WaveSolution
    // using factory pattern, constructor is a private member
    WaveSolution::WaveSolution(const IniCond& ic_in, const WaveConfig& cfg_in,
                               double rhostarleft_in, double rhostarright_in,
                               std::optional<double> left_shock_speed_in,
                               std::optional<double> right_shock_speed_in,
                               std::optional<double> left_fan_head_speed_in,
                               std::optional<double> left_fan_tail_speed_in,
                               std::optional<double> right_fan_head_speed_in,
                               std::optional<double> right_fan_tail_speed_in
                ) // optional b/c existance depends on the wave configuration
                  : ic(ic_in),
                    cfg(cfg_in),
                    rhostarleft(rhostarleft_in),
                    rhostarright(rhostarright_in),
                    left_shock_speed(left_shock_speed_in),
                    right_shock_speed(right_shock_speed_in),
                    left_fan_head_speed(left_fan_head_speed_in),
                    left_fan_tail_speed(left_fan_tail_speed_in),
                    right_fan_head_speed(right_fan_head_speed_in),
                    right_fan_tail_speed(right_fan_tail_speed_in)
                {}

    // member functions to compute quantities
    #pragma region helper funcs
    double WaveSolution::compute_left_shock_speed(const IniCond& ic, 
                                                  const WaveConfig& cfg) {
        return ic.uleft - ic.cleft * std::sqrt(
                    (ic.gamma+1)*cfg.pstar/(2*ic.gamma*ic.pleft) 
                    + (ic.gamma-1)/(2*ic.gamma)
                    );
    } // Rankine–Hugoniot relations

    double WaveSolution::compute_right_shock_speed(const IniCond& ic,
                                                   const WaveConfig& cfg) {
        return ic.uright + ic.cright * std::sqrt(
                    (ic.gamma+1)*cfg.pstar/(2*ic.gamma*ic.pright) 
                    + (ic.gamma-1)/(2*ic.gamma)
                    );
    } // Rankine–Hugoniot relations

    std::pair<double, double> WaveSolution::compute_left_fan_speed(
                                    const IniCond& ic, const WaveConfig& cfg) {
        double left_fan_head_speed = ic.uleft - ic.cleft;
        double left_fan_tail_speed = cfg.ustar - ic.cleft * 
                        std::pow(cfg.pstar/ic.pleft, (ic.gamma-1)/(2*ic.gamma));

        return {left_fan_head_speed, left_fan_tail_speed};
    } // isentropic relations

    std::pair<double, double> WaveSolution::compute_right_fan_speed(
                                    const IniCond& ic, const WaveConfig& cfg) {
        double right_fan_head_speed = ic.uright + ic.cright;
        double right_fan_tail_speed = cfg.ustar + ic.cright * 
                       std::pow(cfg.pstar/ic.pright, (ic.gamma-1)/(2*ic.gamma));
        
        return {right_fan_head_speed, right_fan_tail_speed};
    } // isentropic relations

    double WaveSolution::compute_rhostarleft(const IniCond& ic, 
                                             const WaveConfig& cfg) {
        // uses Rankine–Hugoniot relations (shock) 
        //  or isentropic relations (rarefaction fan)
        if (cfg.is_leftshock) {
            return ic.rholeft * (
                (cfg.pstar/ic.pleft + (ic.gamma-1)/(ic.gamma+1))
                / ((ic.gamma-1)*cfg.pstar / ((ic.gamma+1)*ic.pleft) + 1)
            );
        } else {
            return ic.rholeft * std::pow(cfg.pstar/ic.pleft, 1/ic.gamma);
        }
    }

    double WaveSolution::compute_rhostarright(const IniCond& ic, 
                                              const WaveConfig& cfg) {
        // uses Rankine–Hugoniot relations (shock) 
        //  or isentropic relations (rarefaction fan)
        if (cfg.is_rightshock) {
            return ic.rhoright * (
                (cfg.pstar/ic.pright + (ic.gamma-1)/(ic.gamma+1))
                / ((ic.gamma-1)*cfg.pstar / ((ic.gamma+1)*ic.pright) + 1)
            );
        } else {
            return ic.rhoright * std::pow(cfg.pstar/ic.pright, 1/ic.gamma);
        }
    }
    #pragma endregion

    WaveSolution WaveSolution::solve_wave(const IniCond& ic, 
                                          const WaveConfig& cfg) {
        double rhostarleft = compute_rhostarleft(ic, cfg);
        double rhostarright = compute_rhostarright(ic, cfg);
        std::optional<double> left_shock_speed;  // existance depends on 
        std::optional<double> right_shock_speed; // the wave configuration
        std::optional<double> left_fan_head_speed;
        std::optional<double> left_fan_tail_speed;
        std::optional<double> right_fan_head_speed;
        std::optional<double> right_fan_tail_speed;

        if (cfg.is_leftshock) { // calculate left wave
            left_shock_speed = compute_left_shock_speed(ic, cfg);
        } else {
            auto result = compute_left_fan_speed(ic, cfg); // pair of doubles
            left_fan_head_speed = result.first;
            left_fan_tail_speed = result.second;
        }

        if (cfg.is_rightshock) { // calculate right wave
            right_shock_speed = compute_right_shock_speed(ic, cfg);
        } else {
            auto result = compute_right_fan_speed(ic, cfg); // pair of doubles
            right_fan_head_speed = result.first;
            right_fan_tail_speed = result.second;
        }

        return WaveSolution(ic, cfg, rhostarleft, rhostarright, 
                            left_shock_speed,
                            right_shock_speed,
                            left_fan_head_speed,
                            left_fan_tail_speed,
                            right_fan_head_speed,
                            right_fan_tail_speed);
    }
#pragma endregion

#pragma region functions
WaveSolution get_wave_config(const Array31d& wleft, const Array31d& wright, 
                             double gamma) {
    // input validation is done within IniCond construction
    // it checks sign of density & pressure, and checks for vacuum solution
    IniCond ic = IniCond(wleft, wright, gamma);
    WaveConfig cfg = WaveConfig::solve_config(ic);
    WaveSolution soln = WaveSolution::solve_wave(ic, cfg);
    return soln;
    // now, the only unknown variables are states(rho, u, P) inside rarefaction
    // fans. they will be calculated later if needed
}

Array31d get_s_state(const WaveSolution& soln, double s) {
    Array31d w_along_s = Array31d::Zero();
    
    if (std::abs(s - soln.cfg.ustar) < 1e-6) { // s is the constact surface
        w_along_s << 0.5 * (soln.rhostarleft+soln.rhostarright), 
                     soln.cfg.ustar, 
                     soln.cfg.pstar;   
    }
    else if (s < soln.cfg.ustar) { // s is on left side of contact surface
        if (soln.cfg.is_leftshock) { // left wave is shock
            if (std::abs(*soln.left_shock_speed - s) <= 1e-6) { // s is shock
// note: here I dereference (*) soln.left_shock_speed because it is an optional
// variable. to perform arithmetic operation on it is must be dereferenced
// here it's safe to dereference because I know 100% for sure the value exists  
                // use average of before and after shock value
                w_along_s << 0.5 * (soln.ic.rholeft + soln.rhostarleft),
                             0.5 * (soln.ic.uleft + soln.cfg.ustar),
                             0.5 * (soln.ic.pleft + soln.cfg.pstar);
            } else if (*soln.left_shock_speed > s) { // s within left ini cond
                w_along_s << soln.ic.rholeft, soln.ic.uleft, soln.ic.pleft;
            } else { // s within left star region
                w_along_s << soln.rhostarleft, soln.cfg.ustar, soln.cfg.pstar;
            }
        } else { // left wave is rarefaction fan
            if (*soln.left_fan_head_speed >= s) { // s within left initial cond
                w_along_s << soln.ic.rholeft, soln.ic.uleft, soln.ic.pleft;
            } else if (*soln.left_fan_tail_speed <= s) { // s within left star
                w_along_s << soln.rhostarleft, soln.cfg.ustar, soln.cfg.pstar;
            } else { // s inside left fan
                w_along_s(0) = soln.ic.rholeft * std::pow(
                    (2/(soln.ic.gamma+1) + (soln.ic.gamma-1)*(soln.ic.uleft-s)
                                           / ((soln.ic.gamma+1)*soln.ic.cleft)),
                    (2/(soln.ic.gamma-1))
                );
                w_along_s(1) = 2 / (soln.ic.gamma+1) 
                    * (soln.ic.cleft + soln.ic.uleft*(soln.ic.gamma-1)/2 + s);
                w_along_s(2) = soln.ic.pleft
                    * std::pow(w_along_s(0)/soln.ic.rholeft, soln.ic.gamma);
            }
        }
    }
    else { // s is on the right side of contact surface
        if (soln.cfg.is_rightshock) { // right wave is shock
            if (std::abs(*soln.right_shock_speed - s) <= 1e-6) { // s is shock
// note: here I dereference (*) soln.left_shock_speed because it is an optional
// variable. to perform arithmetic operation on it is must be dereferenced
// here it's safe to dereference because I know 100% for sure the value exists
                // use average of before and after shock value
                w_along_s << 0.5 * (soln.ic.rhoright + soln.rhostarright),
                             0.5 * (soln.ic.uright + soln.cfg.ustar),
                             0.5 * (soln.ic.pright + soln.cfg.pstar);
            } else if (*soln.right_shock_speed < s) { // s in right initial cond
                w_along_s << soln.ic.rhoright, soln.ic.uright, soln.ic.pright;
            } else { // s within right star region
                w_along_s << soln.rhostarright, soln.cfg.ustar, soln.cfg.pstar;
            }
        } else { // right wave is rarefaction fan
            if (*soln.right_fan_head_speed <= s) { // s within right ini cond
                w_along_s << soln.ic.rhoright, soln.ic.uright, soln.ic.pright;
            } else if (*soln.right_fan_tail_speed >= s) { // s within right star
                w_along_s << soln.rhostarright, soln.cfg.ustar, soln.cfg.pstar;
            } else { // s inside right fan
                w_along_s(0) = soln.ic.rhoright * std::pow(
                    (2/(soln.ic.gamma+1) - (soln.ic.gamma-1)*(soln.ic.uright-s)
                                           /((soln.ic.gamma+1)*soln.ic.cright)),
                    (2/(soln.ic.gamma-1))
                );
                w_along_s(1) = 2 / (soln.ic.gamma+1) 
                   * (-soln.ic.cright + soln.ic.uright*(soln.ic.gamma-1)/2 + s);
                w_along_s(2) = soln.ic.pright
                    * std::pow(w_along_s(0)/soln.ic.rhoright, soln.ic.gamma);
            }
        }
    }

    return w_along_s;
}
#pragma endregion

// int main() {
//     std::cout << "Hello, world!" << std::endl;
//     Array31d wleft(1.0,0.0,0.01);
//     Array31d wright(1,0,100);
//     double gamma = 1.4;
//     WaveSolution soln = get_wave_config(wleft, wright, gamma);
//     // std::cout << soln.rhostarleft << "\n" <<soln.rhostarright << std::endl;
//     // std::cout << "ustar " << soln.cfg.ustar << std::endl;
//     // std::cout << std::to_string(*soln.left_fan_head_speed) << "\n" 
//     // << std::to_string(*soln.left_fan_tail_speed) << "\n" << 
//     // std::to_string(*soln.right_fan_head_speed) << "\n"
//     // << std::to_string(*soln.right_fan_tail_speed) << std::endl;

//     Array31d interface_state = get_s_state(soln);
//     std::cout << interface_state << std::endl;

//     return 0;
// }