#ifndef BOOST_FIX_H
#define BOOST_FIX_H

#include <cstdint>
#include <cstddef>

// Boost math fix
namespace boost { namespace math { typedef ::uintmax_t uintmax_t; } }

// Filesystem namespace alias for GCC 7/8
#if __has_include(<filesystem>)
  #include <filesystem>
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace std { namespace filesystem = experimental::filesystem; }
#endif

#endif
