/* Message calls split over lines and adjacent string literals (proxy.c,
 * fof.c). */

void proxy_tags_exchange(struct space *s) {
  ticks tic2 = getticks();
  if (s->e->verbose) {
    message(
        "Setting up, packing, and posting proxy aggregated tag exchanges took "
        "%.3f %s.",
        clocks_from_ticks(getticks() - tic2), clocks_getunit());
  }
}

void fof_search_tree(struct fof_props *props, const int verbose) {
  ticks tic = getticks();
  if (verbose)
    message(
        "Finding max no. of cells + offset IDs"
        "took: %.3f %s.",
        clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("Calculating the total no. of local groups took: (FOF SCALING): %.3f "
            "%s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("Split "
            /* a "quoted" comment between literals */
            "label took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("Escaped \"quotes\" and a\ttab took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}
