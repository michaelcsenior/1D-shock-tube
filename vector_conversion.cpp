// created 2026-06-07 by Michael
// utility module used to calculate state variables in vector notations, 
// i.e. W, U, F, from one another

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

using Array3Xd = Eigen::Array3Xd;

void check_dimension(const Array3Xd& a, const Array3Xd& b) {
    if (a.cols() != b.cols()) {
        throw std::invalid_argument("Expected input matrices of the same shape."
            " Got " + std::to_string(a.cols()) + " columns and "
            + std::to_string(b.cols()) + " columns.");
    }
}

void primitive_from_conservative(const Array3Xd& u, // conservative
                                 Array3Xd& w, // primitive
                                 double gamma) {
    // calculate W from given U
    check_dimension(u, w);
    w.row(0) = u.row(0);
    w.row(1) = u.row(1).array() / u.row(0).array();
    w.row(2) = (gamma-1) * (u.row(2).array() 
                            - 0.5 * u.row(1).array().square() / u.row(0).array()
                            );
}

void conservative_from_primitive(const Array3Xd& w,
                                 Array3Xd& u,
                                 double gamma) {
    // calculate U from given W
    check_dimension(w, u);
    u.row(0) = w.row(0);
    u.row(1) = w.row(0).array() * w.row(1).array();
    u.row(2) = (w.row(2).array()/(gamma-1) 
                + 0.5 * w.row(0).array() * w.row(1).array().square()
               );
}

void flux_from_conservative(const Array3Xd& u, Array3Xd& f, double gamma) {
    // calculate F from given U
    check_dimension(u, f);
    auto rho = u.row(0); // type "auto" makes this simply a view of the u array,
    auto mom = u.row(1); // rather than a copy
    auto energy = u.row(2);

    auto p = (gamma-1) * (energy - 0.5*mom.square()/rho); // lazy expression:
    // here p is not yet evaluated. A lazy expression represents a computation, 
    // but does not compute or store the result yet. It is only evaluated
    // when called afterwards

    f.row(0) = mom;
    f.row(1) = mom.square() / rho + p;   // p evaluated here
    f.row(2) = (energy + p) * mom / rho; // p evaluated here
}

void flux_from_primitive(const Array3Xd& w, Array3Xd& f, double gamma) {
    // calculate F from given W
    check_dimension(w, f);
    auto rho = w.row(0);
    auto velo = w.row(1);
    auto p = w.row(2);
    
    auto rhoE = p/(gamma-1) + 0.5*rho*velo.square(); // dens * specific energy

    f.row(0) = rho * velo;
    f.row(1) = rho * velo.square() + p;
    f.row(2) = (rhoE + p) * velo;
}