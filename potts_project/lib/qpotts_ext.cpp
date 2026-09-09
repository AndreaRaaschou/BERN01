#include <iostream>
#include <mdspan>
#include <memory>
#include <random>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

enum class Start { Cold, Hot };

class PottsModel {
  using StateType = uint8_t;

public:
  PottsModel(int L, double T, StateType Q = 2, Start Start = Start::Cold)
      : L(L), Q(Q), Temperature(T), Beta(1.0 / T),
        StateData(std::make_unique<StateType[]>(L * L)),
        State(StateData.get(), L, L), Gen(Device()), LDist(0, L - 1),
        QDist(0, Q - 1) {
    if (Start == Start::Hot) {
      for (int Row = 0; Row < State.extent(0); Row++)
        for (int Col = 0; Col < State.extent(1); Col++)
          State[Row, Col] = QDist(Gen);
    }

    TotalEnergy = computeTotalEnergy();
  }

  auto totalEnergy() -> int { return TotalEnergy; }

  auto averageEnergy() -> double {
    return static_cast<double>(TotalEnergy) / static_cast<double>(State.size());
  }

  auto density(int E) const -> double { return std::exp(-Beta * E); }

  auto tryMetropolisUpdate() -> bool {
    const auto [Row, Col, NewState] = propose();

    int EnergyChange = computeEnergyChange(Row, Col, NewState);

    if (ADist(Gen) > std::exp(-Beta * EnergyChange))
      return false;

    State[Row, Col] = NewState;
    TotalEnergy += EnergyChange;
    return true;
  }

  auto sampleMetropolis(int NumSamples, int NumBurnIns)
      -> nb::ndarray<nb::numpy, double> {
    double *Energies = new double[NumSamples];

    for (int I = 0; I < NumBurnIns; I++)
      tryMetropolisUpdate();

    for (int I = 0; I < NumSamples; I++) {
      tryMetropolisUpdate();
      Energies[I] = averageEnergy();
    }
    
    nb::capsule Owner(Energies, [](void *P) noexcept {
      delete[] static_cast<double *>(P);
    });

    return nb::ndarray<nb::numpy, double>(
      Energies,
      {static_cast<size_t>(NumSamples)},
      Owner
    );
  }

private:
  int L;
  StateType Q;
  double Temperature;
  double Beta;

  int TotalEnergy = 0;

  std::unique_ptr<StateType[]> StateData;
  std::mdspan<StateType, std::dextents<int, 2>> State;

  std::random_device Device{};
  std::mt19937 Gen;
  std::uniform_int_distribution<int> LDist;
  std::uniform_int_distribution<StateType> QDist;
  std::uniform_real_distribution<double> ADist{0.0, 1.0};

  auto propose() -> std::tuple<int, int, StateType> {
    int Row = LDist(Gen);
    int Col = LDist(Gen);
    StateType NewState = QDist(Gen);

    return {Row, Col, NewState};
  }

  enum class Direction { North, South, East, West };

  auto adjacentState(int Row, int Col, Direction Direction) const -> StateType {
    switch (Direction) {
    case Direction::North:
      Row--;
      break;
    case Direction::South:
      Row++;
      break;
    case Direction::East:
      Col++;
      break;
    case Direction::West:
      Col--;
      break;
    }
    Row = (Row + State.extent(0)) % State.extent(0);
    Col = (Col + State.extent(1)) % State.extent(1);
    return State[Row, Col];
  }

  auto computeEnergyChange(int Row, int Col, StateType NewState) const -> int {
    if (NewState == State[Row, Col])
      return 0;

    StateType StateHere = State[Row, Col];
    StateType StateNorth = adjacentState(Row, Col, Direction::North);
    StateType StateSouth = adjacentState(Row, Col, Direction::South);
    StateType StateEast = adjacentState(Row, Col, Direction::East);
    StateType StateWest = adjacentState(Row, Col, Direction::West);

    int EnergyChange = 0;
    EnergyChange -= (NewState == StateNorth) - (StateHere == StateNorth);
    EnergyChange -= (NewState == StateSouth) - (StateHere == StateSouth);
    EnergyChange -= (NewState == StateEast) - (StateHere == StateEast);
    EnergyChange -= (NewState == StateWest) - (StateHere == StateWest);

    return EnergyChange;
  }

  auto applyChange(int Row, int Col, StateType NewState, int EnergyChange)
      -> void {
    TotalEnergy += EnergyChange;
    State[Row, Col] = NewState;
  }

  auto localEnergy(int Row, int Col) const -> int {
    const StateType StateHere = State[Row, Col];
    const StateType StateSouth = adjacentState(Row, Col, Direction::South);
    const StateType StateEast = adjacentState(Row, Col, Direction::East);

    return -(StateHere == StateSouth) - (StateHere == StateEast);
  }

  auto computeTotalEnergy() const -> int {
    int Energy = 0;
    for (int Row = 0; Row < State.extent(0); Row++)
      for (int Col = 0; Col < State.extent(1); Col++)
        Energy += localEnergy(Row, Col);

    return Energy;
  }
};

NB_MODULE(qpotts_ext, M) {
  nb::enum_<Start>(M, "Start")
      .value("Cold", Start::Cold)
      .value("Hot", Start::Hot);
  nb::class_<PottsModel>(M, "PottsModel")
      .def(nb::init<int, double, uint8_t, Start>())
      .def("try_metropolis_update", &PottsModel::tryMetropolisUpdate)
      .def("sample_metropolis", &PottsModel::sampleMetropolis)
      .def("average_energy", &PottsModel::averageEnergy);
}