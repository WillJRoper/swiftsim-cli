/* Which tic a timer measures from (engine.c, proxy.c, swift.c). */

void engine_maketasks(struct engine *e) {
  const ticks tic_total = getticks();
  ticks tic = getticks();
  make_hydro_tasks(e);
  if (e->verbose)
    message("Making hydro tasks took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  tic = getticks();
  make_gravity_tasks(e);
  if (e->verbose)
    message("Making gravity tasks took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  ticks tic2 = getticks();
  link_tasks(e);
  if (e->verbose)
    message("Linking tasks took %.3f %s.",
            clocks_from_ticks(getticks() - tic2), clocks_getunit());
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic_total),
            clocks_getunit());
}

void loops_and_blocks(struct engine *e) {
  for (int k = 0; k < 3; k++) {
    const ticks tic = getticks();
    make_hydro_tasks(e);
    if (e->verbose) {
      if (k == 0) {
        message("Loop body took %.3f %s.",
                clocks_from_ticks(getticks() - tic), clocks_getunit());
      }
    }
  }
  while (e->step < 10) {
    ticks tic3;
    tic3 = getticks();
    link_tasks(e);
    message("While body took %.3f %s.", clocks_from_ticks(getticks() - tic3),
            clocks_getunit());
  }
}

void clocks_style(struct engine *e) {
  struct clocks_time tic, toc;
  clocks_gettime(&tic);
  make_hydro_tasks(e);
  clocks_gettime(&toc);
  message("space_init took %.3f %s.", clocks_diff(&tic, &toc),
          clocks_getunit());
  clocks_gettime(&tic);
  link_tasks(e);
  clocks_gettime(&toc);
  message("took %.3f %s.", clocks_diff(&tic, &toc), clocks_getunit());
}

void make_hydro_tasks(struct engine *e) { e->step++; }
void make_gravity_tasks(struct engine *e) { e->step++; }
void link_tasks(struct engine *e) { e->step++; }
