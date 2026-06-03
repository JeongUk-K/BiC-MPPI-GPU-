#pragma once

#include <Eigen/Dense>

#include <fstream>

inline void writeQuadrotorPathHeader(std::ofstream &csv) {
  csv << "variant,map,iter,x,y,z,vx,vy,vz\n";
}

inline void writeQuadrotorPathRow(std::ofstream &csv, const char *variant,
                                  int map, int iter,
                                  const Eigen::VectorXd &x) {
  csv << variant << ',' << map << ',' << iter;
  for (int d = 0; d < x.rows(); ++d) {
    csv << ',' << x(d);
  }
  csv << '\n';
}
