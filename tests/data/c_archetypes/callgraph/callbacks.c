/* Functions handed over as callbacks (threadpool_map, qsort, ...). */

void space_rebuild_mapper(void *map_data, int num, void *extra) {
  cell_compute(map_data);
}

void space_split_mapper(void *map_data, int num, void *extra) {}

int cmp_cells(const void *a, const void *b) { return 0; }

void cell_compute(void *c) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void space_rebuild(struct space *s) {
  const ticks tic = getticks();
  threadpool_map(&s->e->threadpool, space_rebuild_mapper, s->cells_top,
                 s->nr_cells, sizeof(struct cell), threadpool_auto_chunk_size,
                 s);
  if (s->e->verbose)
    message("Rebuilding took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  const ticks tic2 = getticks();
  threadpool_map(&s->e->threadpool, &space_split_mapper, s->cells_top,
                 s->nr_cells, sizeof(struct cell), 0, s);
  qsort(s->cells_top, s->nr_cells, sizeof(struct cell), cmp_cells);
  if (s->e->verbose)
    message("Splitting took %.3f %s.", clocks_from_ticks(getticks() - tic2),
            clocks_getunit());
}
