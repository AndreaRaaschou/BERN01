#include <iostream>
#include <mdspan>
#include <memory>
#include <random>
#include <tuple>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

class PottsModel {
  using StateType = uint8_t;
public:
  auto totalEnergy() -> int { return TotalEnergy; }

  // auto averageEnergy() -> double {
  //   return static_cast<double>(TotalEnergy) / static_cast<double>(State.size());
  // }


private:
  int L;
  StateType Q;
  double Temperature;
  double Beta;

  int TotalEnergy = 0;

};

NB_MODULE(qpotts_ext, m) {
    nb::class_<PottsModel>(m, "PottsModel")
        .def(nb::init<>())
        .def("total_energy", &PottsModel::totalEnergy);
}