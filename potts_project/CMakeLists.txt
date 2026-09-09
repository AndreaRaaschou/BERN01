cmake_minimum_required(VERSION 3.15...3.26)

project(qpotts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

find_package(Boost QUIET COMPONENTS random)

if(Boost_random_FOUND)
  message(STATUS "Boost.Random: FOUND; using boost::random::xoshiro256pp")
else()
  message(STATUS "Boost.Random: NOT FOUND; defaulting to std::mt19937")
endif()

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  qpotts_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  lib/qpotts_ext.cpp
)

if (Boost_random_FOUND)
  target_link_libraries(qpotts_ext PRIVATE Boost::random)
  target_compile_definitions(qpotts_ext PRIVATE HAVE_BOOST_RANDOM)
endif()

# Install directive for scikit-build-core
install(TARGETS qpotts_ext LIBRARY DESTINATION qpotts)
install(FILES lib/qpotts/__init__.py DESTINATION qpotts)
