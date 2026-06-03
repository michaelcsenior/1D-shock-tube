// created 2026-06-02 by Michael
#pragma region libraries
#include <iostream>
#include <string>
#include <cmath>

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
    WaveConfig(double pstar_in, double ustar_in, 
               bool is_leftshock_in, bool is_rightshock_in)
               : pstar(pstar_in), ustar(ustar_in), 
                 is_leftshock(is_leftshock_in), is_rightshock(is_rightshock_in)
               {}

public:
    const double pstar;
    const double ustar;
    const bool is_leftshock;
    const bool is_rightshock;

    static WaveConfig create(IniCond ic) { // constructor
        


        return WaveConfig(pstar_in, ustar_in, 
                          is_leftshock_in, is_rightshock_in);
    }

}

#pragma endregion

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}