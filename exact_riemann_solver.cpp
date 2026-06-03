// created 2026-06-02 by Michael
#pragma region libraries
#include <iostream>
#include <string>
#include <cmath>
#include <utility>

#include <Eigen/Core>
#pragma endregion

using Vector3d = Eigen::Vector3d;
using MatrixXd = Eigen::MatrixXd;

#pragma region classes
/**
 * @class IniCond
 * @brief Stores left and right initial conditions for a 1D Riemann problem, 
 *        and ratio of specific heat.
 *
 * This class stores primitive variables (density, velocity, pressure) for both
 * the left and right states, and the ratio of specific heat gamma. Speed of 
 * sound is computed and stored as well.
 */
class IniCond {
    static double calculate_speed_of_sound(double rho, double p, double gamma) {
        return std::sqrt(gamma * p / rho);
    }

public: // since they're all constant, no risk of accidental modification
    const double rholeft;
    const double uleft;
    const double pleft;
    const double rhoright;
    const double uright;
    const double pright;
    const double cleft;
    const double cright;
    const double gamma; // gas constant

    /**
     * @brief Constructor.
     *
     * @param wleft  Primitive variables for left state [rho, u, p], of type 
     *               Vector3d.
     * @param wright Primitive variables for right state [rho, u, p], of type 
     *               Vector3d.
     * @param gamma_in  Ratio of specific heats.
     *
     * The constructor computes the speed of sound for both states.
     */
    IniCond(Vector3d wleft, Vector3d wright, double gamma_in)
         : rholeft(wleft(0)), 
           uleft(wleft(1)),
           pleft(wleft(2)),
           rhoright(wright(0)),
           uright(wright(1)),
           pright(wright(2)),
           gamma(gamma_in),
           cleft(calculate_speed_of_sound(rholeft, pleft, gamma)),
           cright(calculate_speed_of_sound(rhoright, pright, gamma))
        {}
    
    // Accessers
    Vector3d left_state() const {
        return Vector3d(rholeft, uleft, pleft);
    }

    Vector3d right_state() const {
        return Vector3d(rhoright, uright, pright);
    }
};

/**
 * @class WaveConfig
 * @brief Solves and stores the wave confiuration of a Riemann problem.
 * 
 * The wave configuration is found using a iterative Newton-Raphson method.
 * The construction is done via a Factory Pattern.
 */
class WaveConfig {
    // using factor pattern, constructor is a private member
    WaveConfig(double pstar_in, double ustar_in, 
               bool is_leftshock_in, bool is_rightshock_in)
               : pstar(pstar_in), ustar(ustar_in), 
                 is_leftshock(is_leftshock_in), is_rightshock(is_rightshock_in)
               {}
    
    // helper functions for solving configuration
    #pragma region helper funcs
    // get f and fprime values for each wave type
    static std::pair<double, double> f_leftshock(double pstar, double rholeft, 
                                                 double pleft, double gamma) {
        double common_term = std::sqrt(
            2 / ((gamma+1) * rholeft * (pstar + (gamma-1)*pleft/(gamma+1)))   );

        double f = (pstar-pleft) * common_term;
        double fprime = common_term * (1 - 0.5*(pstar-pleft)/
                                           (pstar + pleft*(gamma-1)/(gamma+1))
                                       );

        return {f, fprime};
    }

    static std::pair<double, double> f_leftfan(double pstar, double pleft,
                                               double cleft, double gamma) {
        double f = 2 * cleft *
            (std::pow(pstar/pleft, (gamma-1)/(2*gamma)) - 1) / (gamma-1) ;
        double fprime = std::pow(pstar/pleft, -(gamma+1)/(2*gamma)) 
                        / (pleft*cleft);

        return {f, fprime};
    }

    static std::pair<double, double> f_rightshock(double pstar, double rhoright,
                                                  double pright, double gamma) {
        double common_term = std::sqrt(
            2 / ((gamma+1) * rhoright * (pstar + (gamma-1)*pright/(gamma+1))) );

        double f = (pstar-pright) * common_term;
        double fprime = common_term * (1 - 0.5*(pstar-pright)/
                                           (pstar + pright*(gamma-1)/(gamma+1))
                                       );

        return {f, fprime};
    }

    static std::pair<double, double> f_rightfan(double pstar, double pright,
                                                double cright, double gamma) {
        double f = 2 * cright *
            (std::pow(pstar/pright, (gamma-1)/(2*gamma)) - 1) / (gamma-1) ;
        double fprime = std::pow(pstar/pright, -(gamma+1)/(2*gamma)) 
                        / (pright*cright);
        
        return {f, fprime};
    }

    // values for the iterative equation for Newton-Raphson method
    static std::pair<double, double> leftshock_rightshock(double pstar_current,
                                                          const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
                f_leftshock(pstar_current, ic.rholeft, ic.pleft, ic.gamma);
        auto [fright, frightprime] = // double, double 
                f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }

    static std::pair<double, double> leftshock_rightfan(double pstar_current, 
                                                        const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
                    f_leftshock(pstar_current, ic.rholeft, ic.pleft, ic.gamma);
        auto [fright, frightprime] = // double, double
                    f_rightfan(pstar_current, ic.pright, ic.cright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }

    static std::pair<double, double> leftfan_rightshock(double pstar_current,
                                                        const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
                f_leftfan(pstar_current, ic.pleft, ic.cleft, ic.gamma);
        auto [fright, frightprime] = // double, double 
                f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }

    static std::pair<double, double> leftfan_rightfan(double pstar_current,
                                                      const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
                    f_leftfan(pstar_current, ic.pleft, ic.cleft, ic.gamma);
        auto [fright, frightprime] = // double, double
                    f_rightfan(pstar_current, ic.pright, ic.cright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }
    #pragma endregion

public:
    const double pstar;
    const double ustar;
    const bool is_leftshock;
    const bool is_rightshock;

    static WaveConfig create(const IniCond& ic) { // constructor
        double pstar_current = // initial guess using acoustic approximation
            (ic.pleft*ic.rhoright*ic.cright + ic.pright*ic.rholeft*ic.cleft + 
             (ic.uleft-ic.uright)*ic.rholeft*ic.cleft*ic.rhoright*ic.cright
            ) / (ic.rhoright*ic.cright + ic.rholeft*ic.cleft);
        // under some IC's the acoustic approximation may be negative:
        if (pstar_current <= 0) {pstar_current = 1e-6;}
        
        bool converged = false;
        int num_iter = 0;
        while (num_iter++ < 30) {
            double eqn_value = 0.0;
            double derivative_value = 0.0;

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

            if (std::abs(derivative_value) < 1e-12) {
                throw std::runtime_error("Division by zero encountered in "
                                         "Newton-Raphson method.");
            }
            // Newton-Raphson step
            double pstar_next = pstar_current - eqn_value / derivative_value;

            if (std::abs(pstar_next - pstar_current) < 1e-8
                && std::abs(eqn_value < 1e-8)) {            // convergence
                pstar_current = pstar_next;
                converged = true;
                break;
            }

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
            auto result = 
                    f_leftfan(pstar_current, ic.pleft, ic.cleft, ic.gamma);
            fleft = result.first;
        }

        if (pstar_current > ic.pright) {
            is_rightshock = true;
            auto result = 
                f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
            fright = result.first;
        } else {
            auto result = 
                    f_rightfan(pstar_current, ic.pright, ic.cright, ic.gamma);
            fright = result.first;
        }

        double ustar = 0.5 * (ic.uleft + ic.uright + fright - fleft);

        return WaveConfig(pstar_current, ustar, is_leftshock, is_rightshock);
    }
};

#pragma endregion

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}