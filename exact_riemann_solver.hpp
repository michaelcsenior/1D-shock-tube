// include guard
#ifndef EXACT_RIEMANN_SOLVER_HPP
#define EXACT_RIEMANN_SOLVER_HPP

#include <utility>
#include <optional>

#include <Eigen/Core>

using Array31d = Eigen::Array<double, 3, 1>;
// using Array rather than Vector/Matrix because in this project 
// all operations are element-wise array operation. there is no 
// matrix operations

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
    static double calculate_speed_of_sound(double rho, double p, double gamma);

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
     *               Array<double, 3, 1>.
     * @param wright Primitive variables for right state [rho, u, p], of type 
     *               Array<double, 3, 1>.
     * @param gamma_in  Ratio of specific heats.
     *
     * The constructor computes the speed of sound for both states.
     */
    IniCond(const Array31d& wleft, const Array31d& wright, double gamma_in);
    
    // Accessers
    Array31d left_state() const;

    Array31d right_state() const;
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
               bool is_leftshock_in, bool is_rightshock_in);
    
    // helper functions for solving configuration
    #pragma region helper funcs
    // get f and fprime values for each wave type
    static std::pair<double, double> f_leftshock(double pstar, double rholeft, 
                                                 double pleft, double gamma);

    static std::pair<double, double> f_leftfan(double pstar, 
                                               double rholeft, double pleft,
                                               double cleft, double gamma);

    static std::pair<double, double> f_rightshock(double pstar, double rhoright,
                                                  double pright, double gamma);

    static std::pair<double, double> f_rightfan(double pstar, 
                                                double rhoright, double pright,
                                                double cright, double gamma);
    
    // values for the iterative equation for Newton-Raphson method
    static std::pair<double, double> leftshock_rightshock(double pstar_current,
                                                          const IniCond& ic);

    static std::pair<double, double> leftshock_rightfan(double pstar_current, 
                                                        const IniCond& ic);

    static std::pair<double, double> leftfan_rightshock(double pstar_current,
                                                        const IniCond& ic);

    static std::pair<double, double> leftfan_rightfan(double pstar_current,
                                                      const IniCond& ic);
    #pragma endregion

public:
    const double pstar;
    const double ustar;
    const bool is_leftshock;
    const bool is_rightshock;

    static WaveConfig solve_config(const IniCond& ic);
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
                ); // optional b/c existance depends on the wave configuration

    // member functions to compute quantities
    #pragma region helper funcs
    static double compute_left_shock_speed(const IniCond& ic, 
                                           const WaveConfig& cfg);
                    
    static double compute_right_shock_speed(const IniCond& ic,
                                            const WaveConfig& cfg);

    static std::pair<double, double> compute_left_fan_speed(const IniCond& ic,
                                                        const WaveConfig& cfg);

    static std::pair<double, double> compute_right_fan_speed(const IniCond& ic,
                                                        const WaveConfig& cfg);

    static double compute_rhostarleft(const IniCond& ic, const WaveConfig& cfg);

    static double compute_rhostarright(const IniCond& ic, 
                                       const WaveConfig& cfg);
    #pragma endregion

public:
    const IniCond ic;     // must copy objects, cannot store by reference, b/c
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
    static WaveSolution solve_wave(const IniCond& ic, const WaveConfig& cfg);
};
#pragma endregion

#pragma region functions
/**
 * @brief Solves the 1D Riemann problem for the Euler equations.
 *
 * Constructs the initial-condition object from the left and right
 * primitive states, computes the star-region wave configuration,
 * and returns the complete wave solution structure.
 *
 * The returned solution contains wave types and speeds and states  
 * in the star region. States inside rarefaction fans are not 
 * stored here and can be computed later as needed.
 *
 * @param wleft  Left primitive state vector (rho, u, p).
 * @param wright Right primitive state vector (rho, u, p).
 * @param gamma  Ratio of specific heats.
 *
 * @return WaveSolution object: stores complete solution of the 
 * Riemann problem.
 *
 * @throws std::invalid_argument If either input state contains
 *         nonphysical values (density/pressure <= 0) or if
 *         they produces a vacuum solution. (The exception is 
 *         checked not within the function it self but within 
 *         IniCond object construction)
 */
WaveSolution get_wave_config(const Array31d& wleft, const Array31d& wright, 
                             double gamma);

/**
 * @brief Samples the state of a 1D Euler Riemann problem at a given
 *        similarity coordinate.
 *
 * Evaluates the Riemann solution at a given similarity value of
 * s = x / t and returns the corresponding state of primitive variables
 * (density, velocity, pressure).
 * The default value s = 0 is provided for use in Godunov-type schemes,
 * where the solution is sampled at cell interfaces during the evolution 
 * step.
 *
 * @param soln Precomputed Riemann solution containing wave configuration,
 *             wave speeds, and states in star regions.
 * @param s    Similarity coordinate (x/t) at which to sample the solution.
 *             Default is 0.
 *
 * @return Array<double,3,1> Primitive variables (rho, u, p) at the specified s.
 */
Array31d get_s_state(const WaveSolution& soln, double s = 0);
#pragma endregion

#endif