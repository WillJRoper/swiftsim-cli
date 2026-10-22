/* Spellings of "took" found in SWIFT (fof.c, gravity.c, engine.c). */

void fof_calc_group_props(struct fof_props *props, int verbose) {
  ticks tic = getticks();
  if (verbose)
    message("took: %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  if (verbose)
    message("Communication took : %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("fof_link_attachable_particles() took (FOF SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("Finding the total no. of groups took: (FOF SCALING): %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void gravity_exact_force_compute(struct space *s, const struct engine *e) {
  const ticks tic = getticks();
  int counter = 0;
  message("Computed exact gravity for %d gparts (took %.3f %s). ", counter,
          clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void ewald_init(int verbose) {
  const ticks tic = getticks();
  if (verbose)
    message("Ewald correction table computed (took %.3f %s). ",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}
