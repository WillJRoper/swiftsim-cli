/* Reachability through untimed functions and operation line ranges. */

void engine_step(struct engine *e) {
  const ticks tic = getticks();
  before_ops(e);
  ticks tic2 = getticks();
  untimed_middle(e);
  if (e->verbose)
    message("Gravity took %.3f %s.", clocks_from_ticks(getticks() - tic2),
            clocks_getunit());
  between_ops(e);
  tic2 = getticks();
  hydro_timed(e);
  if (e->verbose)
    message("Hydro took %.3f %s.", clocks_from_ticks(getticks() - tic2),
            clocks_getunit());
  after_ops(e);
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

void untimed_middle(struct engine *e) { gravity_timed(e); }

void gravity_timed(struct engine *e) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void hydro_timed(struct engine *e) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void before_ops(struct engine *e) {}
void between_ops(struct engine *e) {}
void after_ops(struct engine *e) {}
