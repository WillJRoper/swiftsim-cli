/* Braces unbalanced across #ifdef branches: tree-sitter cannot place the
 * code, so the brace-counting fallback attributes it. */

#if defined(WITH_MPI) && defined(HAVE_METIS)
void unbalanced(struct engine *e, int a) {
  const ticks tic = getticks();
#ifdef WITH_MPI
  if (a) {
    exchange(e);
    message("MPI branch took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
#else
  if (a) {
    exchange(e);
    message("Serial branch took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
#endif
  }
  while (a--) {
    exchange(e);
  }
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
#endif

void exchange(struct engine *e) {
  const char *s = "{";
  e->step++;
}

void balanced_after(struct engine *e) {
  const ticks tic = getticks();
  exchange(e);
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
