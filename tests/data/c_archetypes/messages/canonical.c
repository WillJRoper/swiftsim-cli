/* The canonical SWIFT function timer, guarded and unguarded. */

void engine_rebuild(struct engine *e) {
  const ticks tic = getticks();
  space_rebuild(e->s);
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

void space_rebuild(struct space *s) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void engine_braced_guard(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose) {
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  }
}

void engine_no_period(struct engine *e) {
  ticks tic = getticks();
  if (e->verbose) message("took %.3f %s", clocks_from_ticks(getticks() - tic), clocks_getunit());
}
