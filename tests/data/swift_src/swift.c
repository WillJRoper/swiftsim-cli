/* Synthetic SWIFT-like source for the swiftsim-cli timer tests.
 *
 * Generated from the timers in tests/data/logs: every function carries the
 * timer format strings of its SWIFT counterpart, and its calls reproduce the
 * nesting of those timers in the logs (at each log's SWIFT revision). The
 * function bodies are otherwise meaningless.
 */

#include "src/timers_stub.h"

int main(void) {
  {
    const ticks tic_65 = getticks();
    if (verbose)
      message("Reading initial conditions took %.3f %s.",
              clocks_from_ticks(getticks() - tic_65), clocks_getunit());
  }
  {
    const ticks tic_66 = getticks();
    if (verbose)
      message("space_init took %.3f %s.",
              clocks_from_ticks(getticks() - tic_66), clocks_getunit());
  }
}
