/* Time conversions and units: %f/%e/%g/%lf, literal units, integers. */

void zoom_region_init(int verbose) {
  const ticks tic = getticks();
  if (verbose)
    message("Zoom region initialisation took %f %s",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void spec_variants(void) {
  const ticks tic = getticks();
  message("Exp took %e %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  message("General took %g %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  message("Long took %lf %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  message("Width took %10.4f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void literal_units(void) {
  const ticks tic = getticks();
  message("Literal ms took %.3f ms.", clocks_from_ticks(getticks() - tic));
  message("Literal s took %.3f s.", clocks_from_ticks(getticks() - tic) / 1e3);
  message("Literal seconds took %.2f seconds", clocks_from_ticks(getticks() - tic) / 1e3);
  message("Unitless took %.3f.", clocks_from_ticks(getticks() - tic));
}

void integer_time(void) {
  const ticks tic = getticks();
  message("Integer took %d ms.", (int)clocks_from_ticks(getticks() - tic));
  message("Ticks took %lld ticks.", (long long)(getticks() - tic));
}
