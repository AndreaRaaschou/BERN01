#include <mdspan>
#include <memory>
#include <random>
#include <tuple>

#ifdef HAVE_BOOST_RANDOM
#include <boost/random/xoshiro.hpp>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

enum class Start { Cold, Hot };

class PottsModel {
  using StateType = uint8_t;

public:
  PottsModel(int L, double T, StateType Q, Start Start)
      : L(L), Q(Q), Temperature(T), Beta(1.0 / T),
        StateData(std::make_unique<StateType[]>(L * L)),
        State(StateData.get(), L, L), Gen(Device()), LDist(0, L - 1),
        QDist(0, Q - 1), NewStateDist(0, Q - 2) {
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

  auto setTemperature(double T) -> void { Beta = 1.0 / T; }

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

    nb::capsule Owner(
        Energies, [](void *P) noexcept { delete[] static_cast<double *>(P); });

    return nb::ndarray<nb::numpy, double>(
        Energies, {static_cast<size_t>(NumSamples)}, Owner);
  }

  auto tryGibbsUpdate() -> bool {
    int Row = LDist(Gen);
    int Col = LDist(Gen);

    // TODO: change to class member
    std::vector<int> NeighbourCounts(Q);
    std::vector<double> PDF(Q);

    for (const auto Direction : AllDirections) {
      StateType State = adjacentState(Row, Col, Direction);
      NeighbourCounts[State]++;
    }

    for (int I = 0; I < Q; I++)
      PDF[I] = std::exp(Beta * NeighbourCounts[I]);

    std::discrete_distribution<int> NewStateDist{PDF.begin(), PDF.end()};

    StateType OldState = State[Row, Col];
    StateType NewState = NewStateDist(Gen);

    int EnergyChange = -NeighbourCounts[NewState] + NeighbourCounts[OldState];
    State[Row, Col] = NewState;
    TotalEnergy += EnergyChange;

    return true;
  }

  auto sampleGibbs(int NumSamples, int NumBurnIns)
      -> nb::ndarray<nb::numpy, double> {
    double *Energies = new double[NumSamples];

    for (int I = 0; I < NumBurnIns; I++)
      tryGibbsUpdate();

    for (int I = 0; I < NumSamples; I++) {
      tryGibbsUpdate();
      Energies[I] = averageEnergy();
    }

    nb::capsule Owner(
        Energies, [](void *P) noexcept { delete[] static_cast<double *>(P); });

    return nb::ndarray<nb::numpy, double>(
        Energies, {static_cast<size_t>(NumSamples)}, Owner);
  }

  auto stateView() -> nb::ndarray<StateType, nb::numpy> {
    size_t NRows = static_cast<size_t>(State.extent(0));
    size_t NCols = static_cast<size_t>(State.extent(1));
    return nb::ndarray<StateType, nb::numpy>(StateData.get(), {NRows, NCols});
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
#ifdef HAVE_BOOST_RANDOM
  boost::random::xoshiro256pp Gen;
#else
  std::mt19937 Gen;
#endif
  std::uniform_int_distribution<int> LDist;
  std::uniform_int_distribution<int> QDist;
  std::uniform_int_distribution<int> NewStateDist;
  std::uniform_real_distribution<double> ADist{0.0, 1.0};

  auto propose() -> std::tuple<int, int, StateType> {
    int Row = LDist(Gen);
    int Col = LDist(Gen);
    StateType NewState = NewStateDist(Gen);

    [[unlikely]] if (NewState == State[Row, Col])
      NewState = Q - 1;

    return {Row, Col, NewState};
  }

  enum class Direction { North, South, East, West };
  static constexpr std::array<Direction, 4> AllDirections = {
      Direction::North,
      Direction::South,
      Direction::East,
      Direction::West,
  };

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
    StateType StateHere = State[Row, Col];

    if (NewState == StateHere)
      return 0;

    int EnergyChange = 0;
    for (const auto Direction : AllDirections) {
      StateType StateThere = adjacentState(Row, Col, Direction);
      EnergyChange -= (NewState == StateThere) - (StateHere == StateThere);
    }

    return EnergyChange;
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

} // namespace

NB_MODULE(qpotts_ext, M) {
  nb::enum_<Start>(M, "Start")
      .value("Cold", Start::Cold)
      .value("Hot", Start::Hot);
  nb::class_<PottsModel>(M, "PottsModel")
      .def(nb::init<int, double, uint8_t, Start>(), "L"_a, "T"_a, "q"_a = 2,
           "start"_a = Start::Cold,
           R"doc(
            Constructs a new PottsModel instance.

            Parameters
            ----------
            L : int
                Specifies lattice size N = `L` × `L`.
            T : float
                Temperature.
            q : int
                Number of states. Must be between 2 and 255 (the default is 2).
            start : Start
                Specifies cold- or hot-start. Valid values: `Start.Cold` and `Start.Hot (the default is `Start.Cold`).
            )doc")
      .def("try_metropolis_update", &PottsModel::tryMetropolisUpdate)
      .def("sample_metropolis", &PottsModel::sampleMetropolis, "num_samples"_a,
           "num_burn_ins"_a = 0,
           R"doc(
            Samples energy states using Metropolis MCMC.

            Parameters
            ----------
            num_samples : int
                Number of samples to record.
            num_burn_ins : int
                Number of samples before starting recording (the default is 0).

            Returns
            -------
            np.ndarray
                Average energy level (E / N) for each sample.
            )doc")
      .def("sample_gibbs", &PottsModel::sampleGibbs, "num_samples"_a,
           "num_burn_ins"_a = 0,
           R"doc(
            Samples energy states using Gibbs MCMC.

            Parameters
            ----------
            num_samples : int
                Number of samples to record.
            num_burn_ins : int
                Number of samples before starting recording (the default is 0).

            Returns
            -------
            np.ndarray
                Average energy level (E / N) for each sample.
            )doc")
      .def("average_energy", &PottsModel::averageEnergy,
           "Get the average energy E / N.")
      .def("set_temperature", &PottsModel::setTemperature, "T"_a,
           "Updates the temperature while keeping the state.")
      .def("state_view", &PottsModel::stateView,
           nb::rv_policy::reference_internal,
           "Returns a `ndarray.view` to the state storage");
}
