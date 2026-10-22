/* The same function defined in both branches of #ifdef WITH_MPI. */

#ifdef WITH_MPI
void engine_exchange(struct engine *e) {
  const ticks tic = getticks();
  mpi_exchange(e);
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
#else
void engine_exchange(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
#endif

#if defined(WITH_MPI) && defined(HAVE_PARMETIS)
void partition_repart(struct engine *e) {
  const ticks tic = getticks();
  mpi_exchange(e);
  if (e->verbose)
    message("Repartitioning took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}
#elif defined(WITH_MPI)
void partition_repart(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("Serial repartitioning took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}
#endif

void mpi_exchange(struct engine *e) { e->step++; }
