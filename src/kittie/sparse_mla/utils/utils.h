#pragma once

#ifdef KITTIE_DEBUG
#define DEBUG_PRINT_IF(condition, ...)                                         \
  do {                                                                         \
    if (condition)                                                             \
      printf(__VA_ARGS__);                                                     \
  } while (0)
#define DEBUG_PRINT(...)                                                       \
  DEBUG_PRINT_IF(blockIdx.x == 0 && threadIdx.x == 0, __VA_ARGS__)
#else
#define DEBUG_PRINT_IF(...)                                                    \
  do {                                                                         \
  } while (0)
#define DEBUG_PRINT(...)                                                       \
  do {                                                                         \
  } while (0)
#endif
