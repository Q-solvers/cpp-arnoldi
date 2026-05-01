#ifndef ARNOLDI_DETAIL_STATS_HPP
#define ARNOLDI_DETAIL_STATS_HPP

#include <chrono>

namespace arnoldi::detail {

  struct Stats {
    int    nopx = 0, nbx = 0, nrorth = 0, nitref = 0, nrstrt = 0;

    double aupd = 0, aup2 = 0, aitr = 0, eigh = 0;
    double gets = 0, apps = 0, conv = 0;
    double mvopx = 0, mvbx = 0, getv0 = 0;
    double itref = 0, rvec = 0;

    void   reset() { *this = Stats{}; }
  };

  inline Stats stats{};

  inline void  arscnd(double& t) {
     using clk = std::chrono::steady_clock;
     t         = std::chrono::duration<double>(clk::now().time_since_epoch()).count();
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_STATS_HPP
