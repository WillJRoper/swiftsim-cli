/* Labels before "took", with runtime %s/%d parts (engine.c, io, lightcone). */

void engine_launch(struct engine *e, const char *call) {
  const ticks tic = getticks();
  if (e->verbose)
    message("(%s) took %.3f %s.", call, clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

void write_array_single(const char *name, int verbose) {
  const ticks tic = getticks();
  if (verbose)
    message("Copying for '%s' took %.3f %s.", name,
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("'%s' took %.3f %s.", name, clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

void write_array_parallel(const char *name, int size, int rank) {
  const ticks tic = getticks();
  message("H5Dwrite for '%s' (%d MB) on rank %d took %.3f %s (speed = %f MB/s).",
          name, size, rank, clocks_from_ticks(getticks() - tic),
          clocks_getunit(), 1.0);
}

void lightcone_flush(int index, int verbose) {
  const ticks tic = getticks();
  if (verbose)
    message("lightcone %d: Flushing particle buffers took %.3f %s.", index,
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void mesh_patches(int verbose) {
  const ticks tic = getticks();
  if (verbose)
    message(" - Sorting of mesh cells took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (verbose)
    message("100%% of %s {done} took %.3f %s.", "cells",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}
