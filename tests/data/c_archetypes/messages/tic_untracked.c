/* A start time kept in a struct member (engine.c's e->tic_step). */

void engine_step(struct engine *e) {
  e->tic_step = getticks();
  make_hydro_tasks(e);
  if (e->verbose)
    message("Stored tic took %.3f %s.", e->wallclock_time, clocks_getunit());
}

void make_hydro_tasks(struct engine *e) { e->step++; }
