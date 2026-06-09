// utility module used to calculate state variables in vector notations, 
// i.e. W, U, F, from one another

// include guard
#ifndef VECTOR_CONVERSION_HPP
#define VECTOR_CONVERSION_HPP

#include <Eigen/Core>

using Array3Xd = Eigen::Array3Xd;

namespace vector_conversion {

void check_dimension(const Array3Xd& a, const Array3Xd& b);

void primitive_from_conservative(const Array3Xd& u, // conservative
                                 Array3Xd& w, // primitive
                                 double gamma);
    // calculate W from given U

void conservative_from_primitive(const Array3Xd& w,
                                 Array3Xd& u,
                                 double gamma);
    // calculate U from given W

void flux_from_conservative(const Array3Xd& u, Array3Xd& f, double gamma);
    // calculate F from given U

void flux_from_primitive(const Array3Xd& w, Array3Xd& f, double gamma);
    // calculate F from given W

} // end of namespace vector_conversion

#endif