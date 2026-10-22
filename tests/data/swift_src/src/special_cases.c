/* Hand-written cases the timer parser must handle (see test_timer_db.py). */

#include "timers_stub.h"

struct cell;

/* A timed function returning a pointer. */
struct cell *special_pointer_function(int n) {
  const ticks tic = getticks();
  struct cell *c = 0;
  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  return c;
}

/* A brace inside a comment: { must not confuse anything. */
void special_comment_brace(void) {
  const ticks tic = getticks();
  /* opening brace in a comment: { */
  // and another one {
  const char *s = "a brace in a string {";
  if (verbose)
    message("Doing the thing took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/* Timers in both branches of an #ifdef with unbalanced braces. */
void special_ifdef(void) {
  const ticks tic = getticks();
#ifdef WITH_MPI
  if (verbose) {
    message("MPI branch took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
#else
  if (verbose) {
    message("Serial branch took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
#endif
  }
}

/* Function pointers handed to the threadpool count as calls. */
void special_mapped_timed(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

void special_mapper(void *data) { special_mapped_timed(); }

void special_threadpool_caller(void) {
  const ticks tic = getticks();
  threadpool_map(0, special_mapper, 0);
  if (verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/* Text after the time: an operation unless parenthesised. */
void special_suffixes(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s to find active cells.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("took %.3f %s (including reweight).",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

/* The same message in two places is one logical timer. */
void special_duplicate(int a) {
  ticks tic = getticks();
  if (a) {
    if (verbose)
      message("Duplicated took %.3f %s.", clocks_from_ticks(getticks() - tic),
              clocks_getunit());
  } else {
    if (verbose)
      message("Duplicated took %.3f %s.", clocks_from_ticks(getticks() - tic),
              clocks_getunit());
  }
}

/* A commented-out timer is not a timer. */
void special_commented(void) {
  // message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
  //         clocks_getunit());
}
