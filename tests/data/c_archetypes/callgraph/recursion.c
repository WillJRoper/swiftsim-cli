/* Recursion and mutual recursion (cell tree walks). */

void cell_split_recursive(struct cell *c) {
  const ticks tic = getticks();
  if (c->split) cell_split_recursive(c->progeny[0]);
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void tree_walk_a(struct cell *c) {
  if (c->split) tree_walk_b(c->progeny[0]);
}

void tree_walk_b(struct cell *c) {
  const ticks tic = getticks();
  if (c->split) tree_walk_a(c->progeny[0]);
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
