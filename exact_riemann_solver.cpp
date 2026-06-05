// created 2026-06-02 by Michael
#pragma region libraries
#include <iostream>
#include <string>
#include <cmath>
#include <utility>
#include <optional>

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
 * 
 * The class is immutable after construction.
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
 * This class computes the "star" region pressure and velocity in a 
 * compressible flow Riemann problem. The solution is obtained using 
 * a Newton–Raphson iteration on the pressure equation.
 * 
 * The class is immutable after construction. Construction is restricted 
 * through a factory method ("create") so that each object is initialized 
 * with a fully solved wave configuration.
 * 
 * Public data members:
 * - pstar: double
 *      star-region pressure
 * - ustar: double
 *      star-region velocity
 * - is_leftshock: bool
 *      whether the left wave is a shock or rarefaction fan
 * - is_rightshock: bool 
 *      whether the right wave is a shock or rarefaction fan
 * 
 * Newton–Raphson details:
 * - Initial guess uses the acoustic approximation
 * - Iteration stops when both:
 *      |p_{n+1} - p_n| < 1e-8
 *      |f(p_n)| < 1e-8
 * - Throws std::runtime_error if:
 *      - derivative becomes numerically zero
 *      - convergence is not achieved within 30 iterations
 */
class WaveConfig {
    // using factory pattern, constructor is a private member
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

    static std::pair<double, double> f_leftfan(double pstar, 
                                               double rholeft, double pleft,
                                               double cleft, double gamma) {
        double f = 2 * cleft *
            (std::pow(pstar/pleft, (gamma-1)/(2*gamma)) - 1) / (gamma-1) ;
        double fprime = std::pow(pstar/pleft, -(gamma+1)/(2*gamma)) 
                        / (rholeft*cleft);

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

    static std::pair<double, double> f_rightfan(double pstar, 
                                                double rhoright, double pright,
                                                double cright, double gamma) {
        double f = 2 * cright *
            (std::pow(pstar/pright, (gamma-1)/(2*gamma)) - 1) / (gamma-1) ;
        double fprime = std::pow(pstar/pright, -(gamma+1)/(2*gamma)) 
                        / (rhoright*cright);

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
                    f_rightfan(pstar_current, ic.rhoright, 
                               ic.pright, ic.cright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;

        return {eqn_value, derivative_value};
    }

    static std::pair<double, double> leftfan_rightshock(double pstar_current,
                                                        const IniCond& ic) {
        auto [fleft, fleftprime] = // double, double
            f_leftfan(pstar_current, ic.rholeft, ic.pleft, ic.cleft, ic.gamma);
        auto [fright, frightprime] = // double, double 
            f_rightshock(pstar_current, ic.rhoright, ic.pright, ic.gamma);
        
        double eqn_value = fleft + fright + ic.uright - ic.uleft;
        double derivative_value =  fleftprime + frightprime;
        
        return {eqn_value, derivative_value};
    }

    static std::pair<double, double> leftfan_rightfan(double pstar_current,
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

public:
    const double pstar;
    const double ustar;
    const bool is_leftshock;
    const bool is_rightshock;

    static WaveConfig solve_config(const IniCond& ic) { // factory method
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
};

/**
 * @class WaveSolution
 * @brief Solution of a 1D Euler Riemann problem, including variables in 
 *        the star region, and wave speeds. Variables within rarefaction 
 *        fans are not computed.
 *
 * This class represents the solution state of a Riemann problem
 * between left and right initial conditions (IniCond), given a wave
 * configuration (WaveConfig).
 *
 * Stores:
 * - "Star" region quantities (rho*, p*, u*)
 * - Speeds of shock waves, or of head and tail of rarefaction fans, 
 *   depending on wave configuration. Since not all wave quantities 
 *   exist for every configuration, these data members are marked 
 *   optional.
 *
 * The class is immutable after construction. Object is constructed 
 * with a factory design pattern, via the static factory method "solve".
 * 
 * Example
 * ------------------------------------------------------------------------
 * IniCond ic = ...;
 * WaveConfig cfg = WaveConfig::solve_config(ic);
 * WaveSolution sol = WaveSolution::solve(ic, cfg);
 *
 * std::cout << sol.rhostarright << " ," << sol.cfg.ustar << "\n";
 */
class WaveSolution {
    // using factory pattern, constructor is a private member
    WaveSolution(const IniCond& ic_in, const WaveConfig& cfg_in,
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
    static double compute_left_shock_speed(const IniCond& ic, 
                                           const WaveConfig& cfg) {
        return ic.uleft - ic.cleft * std::sqrt(
                    (ic.gamma+1)*cfg.pstar/(2*ic.gamma*ic.pleft) 
                    + (ic.gamma-1)/(2*ic.gamma)
                    );
    } // Rankine–Hugoniot relations

    static double compute_right_shock_speed(const IniCond& ic,
                                            const WaveConfig& cfg) {
        return ic.uright + ic.cright * std::sqrt(
                    (ic.gamma+1)*cfg.pstar/(2*ic.gamma*ic.pright) 
                    + (ic.gamma-1)/(2*ic.gamma)
                    );
    } // Rankine–Hugoniot relations

    static std::pair<double, double> compute_left_fan_speed(const IniCond& ic,
                                                        const WaveConfig& cfg) {
        double left_fan_head_speed = ic.uleft - ic.cleft;
        double left_fan_tail_speed = cfg.ustar - ic.cleft * 
                        std::pow(cfg.pstar/ic.pleft, (ic.gamma-1)/(2*ic.gamma));

        return {left_fan_head_speed, left_fan_tail_speed};
    } // isentropic relations

    static std::pair<double, double> compute_right_fan_speed(const IniCond& ic,
                                                        const WaveConfig& cfg) {
        double right_fan_head_speed = ic.uright + ic.cright;
        double right_fan_tail_speed = cfg.ustar + ic.cright * 
                       std::pow(cfg.pstar/ic.pright, (ic.gamma-1)/(2*ic.gamma));
        
        return {right_fan_head_speed, right_fan_tail_speed};
    } // isentropic relations

    static double compute_rhostarleft(const IniCond& ic, const WaveConfig& cfg) 
    { // uses Rankine–Hugoniot relations (shock) 
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

    static double compute_rhostarright(const IniCond& ic, const WaveConfig& cfg)
    { // uses Rankine–Hugoniot relations (shock) 
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

public:
    const IniCond ic; // must copy objects, cannot store by reference, b/c
    const WaveConfig cfg; // they will be out of scope when function returns
    const double rhostarleft;
    const double rhostarright;
    const std::optional<double> left_shock_speed;  // existance depends on the
    const std::optional<double> right_shock_speed; // wave configuration
    const std::optional<double> left_fan_head_speed;
    const std::optional<double> left_fan_tail_speed;
    const std::optional<double> right_fan_head_speed;
    const std::optional<double> right_fan_tail_speed;

    /**
     * @brief Factory method used for object construction. Computes the 
     * full Riemann solution.
     *
     * This function:
     * 1. Computes star-region densities (rho*)
     * 2. Computes shock or rarefaction wave speeds according to wave type
     *    from WaveConfig
     * 3. Assembles a immutable WaveSolution object
     * 4. Note: does not compute variables inside rarefaction fans
     *
     * @param ic Initial conditions (left/right states)
     * @param cfg Precomputed wave configuration
     *
     * @return Fully constructed WaveSolution
     */
    static WaveSolution solve_wave(const IniCond& ic, const WaveConfig& cfg) {
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
};
#pragma endregion

#pragma region functions
WaveSolution get_wave_config(Vector3d wleft, Vector3d wright, double gamma) {
    // input validation is done within IniCond construction
    // it checks sign of density & pressure, and checks for vacuum solution
    IniCond ic = IniCond(wleft, wright, gamma);
    WaveConfig cfg = WaveConfig::solve_config(ic);
    WaveSolution soln = WaveSolution::solve_wave(ic, cfg);
    return soln;
    // now, the only unknown variables are states(rho, u, P) inside rarefaction
    // fans. they will be calculated later if needed
}

int main() {
    std::cout << "Hello, world!" << std::endl;
    Vector3d wleft(1,-2,0.4);
    Vector3d wright(1,2,0.4);
    double gamma = 1.4;
    WaveSolution soln = get_wave_config(wleft, wright, gamma);
    std::cout << soln.rhostarleft << "\n" <<soln.rhostarright << std::endl;
    std::cout << "ustar " << soln.cfg.ustar << std::endl;
    std::cout << std::to_string(*soln.left_fan_head_speed) << "\n" 
    << std::to_string(*soln.left_fan_tail_speed) << "\n" << 
    std::to_string(*soln.right_fan_head_speed) << "\n"
    << std::to_string(*soln.right_fan_tail_speed) << std::endl;
    return 0;
}