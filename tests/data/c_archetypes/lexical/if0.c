/* Code in #if 0 is never compiled. */

#if 0
void dead_function(struct engine *e) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
#endif

void live_function(struct engine *e) {
  const ticks tic = getticks();
#if 0
  dead_helper(e);
  message("Old method took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
#elif defined(WITH_MPI)
  live_helper(e);
  message("MPI method took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
#else
  live_helper(e);
  message("New method took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
#endif
#if 1
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
#endif
}

void dead_helper(struct engine *e) { e->step++; }
void live_helper(struct engine *e) { e->step++; }
