/* Synthetic SWIFT-like source for the swiftsim-cli timer tests.
 *
 * Generated from the timers in tests/data/logs: every function carries the
 * timer format strings of its SWIFT counterpart, and its calls reproduce the
 * nesting of those timers in the logs (at each log's SWIFT revision). The
 * function bodies are otherwise meaningless.
 */

#include "timers_stub.h"

void compute_potential_global(void) {
  {
    const ticks tic_1 = getticks();
    if (verbose)
      message("Mesh MPI-reduction took %.3f %s.",
              clocks_from_ticks(getticks() - tic_1), clocks_getunit());
  }
  {
    const ticks tic_2 = getticks();
    if (verbose)
      message("Gpart assignment took %.3f %s.",
              clocks_from_ticks(getticks() - tic_2), clocks_getunit());
  }
  {
    const ticks tic_3 = getticks();
    if (verbose)
      message("Forward Fourier transform took %.3f %s.",
              clocks_from_ticks(getticks() - tic_3), clocks_getunit());
  }
  {
    const ticks tic_4 = getticks();
    if (verbose)
      message("Applying Green function took %.3f %s.",
              clocks_from_ticks(getticks() - tic_4), clocks_getunit());
  }
  {
    const ticks tic_5 = getticks();
    if (verbose)
      message("Reverse Fourier transform took %.3f %s.",
              clocks_from_ticks(getticks() - tic_5), clocks_getunit());
  }
  {
    const ticks tic_6 = getticks();
    if (verbose)
      message("Computing mesh accelerations took %.3f %s.",
              clocks_from_ticks(getticks() - tic_6), clocks_getunit());
  }
}

void engine_activate_fof_attach_tasks(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_activate_fof_tasks(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_allocate_foreign_particles(void) {
  {
    const ticks tic_7 = getticks();
    if (verbose)
      message("Counting number of foreign particles took %.3f %s.",
              clocks_from_ticks(getticks() - tic_7), clocks_getunit());
  }
  {
    const ticks tic_8 = getticks();
    if (verbose)
      message("Recursively linking foreign arrays took %.3f %s.",
              clocks_from_ticks(getticks() - tic_8), clocks_getunit());
  }
}

void engine_collect_end_of_step(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_config(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_drift_all(void) {
  const ticks tic = getticks();
  space_synchronize_particle_positions();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_drift_top_multipoles(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_exchange_cells(void) {
  const ticks tic = getticks();
  proxy_cells_exchange();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_exchange_strays(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_exchange_top_multipoles(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_first_init_particles(void) {
  const ticks tic = getticks();
  space_first_init_bparts();
  space_first_init_gparts();
  space_first_init_parts();
  space_first_init_sinks();
  space_first_init_sparts();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_fof(void) {
  {
    const ticks tic_9 = getticks();
    engine_activate_fof_tasks();
    engine_launch();
    engine_make_fof_tasks();
    engine_print_task_counts();
    fof_allocate();
    fof_compute_group_props();
    fof_compute_local_sizes();
    fof_link_attachable_particles();
    if (verbose)
      message("Complete FOF search took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_9), clocks_getunit());
  }
}

void engine_init(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_init_particles(void) {
  const ticks tic = getticks();
  compute_potential_global();
  engine_collect_end_of_step();
  engine_first_init_particles();
  engine_launch();
  engine_print_task_counts();
  engine_rebuild();
  engine_recompute_displacement_constraint();
  scheduler_write_dependencies();
  space_convert_quantities();
  space_init_bparts();
  space_init_gparts();
  space_init_parts();
  space_init_sinks();
  space_init_sparts();
  space_post_init_parts();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_launch(void) {
  const ticks tic = getticks();
  if (verbose)
    message("(%s) took %.3f %s.", "label0",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_make_fof_tasks(void) {
  const ticks tic = getticks();
  {
    const ticks tic_10 = getticks();
    if (verbose)
      message("Making FOF tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_10), clocks_getunit());
  }
  {
    const ticks tic_11 = getticks();
    if (verbose)
      message("Splitting FOF tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_11), clocks_getunit());
  }
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_makeproxies_with_zoom_region(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_maketasks(void) {
  const ticks tic = getticks();
  {
    const ticks tic_12 = getticks();
    if (verbose)
      message("Making hydro tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_12), clocks_getunit());
  }
  {
    const ticks tic_13 = getticks();
    zoom_engine_make_self_gravity_tasks();
    if (verbose)
      message("Making gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_13), clocks_getunit());
  }
  {
    const ticks tic_14 = getticks();
    if (verbose)
      message("Splitting tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_14), clocks_getunit());
  }
  {
    const ticks tic_15 = getticks();
    if (verbose)
      message("Counting and linking tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_15), clocks_getunit());
  }
  {
    const ticks tic_16 = getticks();
    if (verbose)
      message("Setting super-pointers took %.3f %s.",
              clocks_from_ticks(getticks() - tic_16), clocks_getunit());
  }
  {
    const ticks tic_17 = getticks();
    if (verbose)
      message("Making extra hydroloop tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_17), clocks_getunit());
  }
  {
    const ticks tic_18 = getticks();
    if (verbose)
      message("Linking gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_18), clocks_getunit());
  }
  {
    const ticks tic_19 = getticks();
    if (verbose)
      message("Setting unlocks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_19), clocks_getunit());
  }
  {
    const ticks tic_20 = getticks();
    if (verbose)
      message("Ranking the tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_20), clocks_getunit());
  }
  {
    const ticks tic_21 = getticks();
    if (verbose)
      message("Making zoom->zoom gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_21), clocks_getunit());
  }
  {
    const ticks tic_22 = getticks();
    if (verbose)
      message("Making bkg->bkg gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_22), clocks_getunit());
  }
  {
    const ticks tic_23 = getticks();
    if (verbose)
      message("Making buffer->buffer gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_23), clocks_getunit());
  }
  {
    const ticks tic_24 = getticks();
    if (verbose)
      message("Making zoom->buffer gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_24), clocks_getunit());
  }
  {
    const ticks tic_25 = getticks();
    if (verbose)
      message("Making buffer->bkg gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_25), clocks_getunit());
  }
  {
    const ticks tic_26 = getticks();
    if (verbose)
      message("Creating send tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_26), clocks_getunit());
  }
  {
    const ticks tic_27 = getticks();
    proxy_tags_exchange();
    if (verbose)
      message("Exchanging cell tags took %.3f %s.",
              clocks_from_ticks(getticks() - tic_27), clocks_getunit());
  }
  {
    const ticks tic_28 = getticks();
    if (verbose)
      message("Creating recv tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_28), clocks_getunit());
  }
  engine_allocate_foreign_particles();
  scheduler_reweight();
  zoom_engine_make_hierarchical_void_tasks();
  if (verbose)
    message("took %.3f %s (including reweight).",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_marktasks(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_prepare(void) {
  const ticks tic = getticks();
  {
    const ticks tic_29 = getticks();
    if (verbose)
      message("Communicating rebuild flag took %.3f %s.",
              clocks_from_ticks(getticks() - tic_29), clocks_getunit());
  }
  engine_drift_all();
  engine_fof();
  engine_rebuild();
  engine_repartition();
  engine_split_gas_particles();
  engine_unskip();
  if (verbose)
    message("took %.3f %s (including unskip, rebuild and reweight).",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_print_stats(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_print_task_counts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_rebuild(void) {
  const ticks tic = getticks();
  {
    const ticks tic_30 = getticks();
    if (verbose)
      message("updating particle counts took %.3f %s.",
              clocks_from_ticks(getticks() - tic_30), clocks_getunit());
  }
  engine_exchange_cells();
  engine_exchange_top_multipoles();
  engine_maketasks();
  engine_marktasks();
  engine_print_task_counts();
  engine_unskip();
  scheduler_report_task_times();
  space_list_useful_top_level_cells();
  space_rebuild();
  void_space_split();
  zoom_void_space_split();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_recompute_displacement_constraint(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_redistribute(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_repartition(void) {
  const ticks tic = getticks();
  engine_makeproxies_with_zoom_region();
  engine_redistribute();
  partition_repartition();
  scheduler_report_task_times();
  space_free_cells();
  task_dump_stats();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_repartition_trigger(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_split(void) {
  const ticks tic = getticks();
  engine_makeproxies_with_zoom_region();
  engine_redistribute();
  partition_initial_partition();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_split_gas_particles(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_step(void) {
  {
    const ticks tic_31 = getticks();
    if (verbose)
      message("Writing step info to files took %.3f %s",
              clocks_from_ticks(getticks() - tic_31), clocks_getunit());
  }
  {
    const ticks tic_32 = getticks();
    if (verbose)
      message("Updating general quantities took %.3f %s",
              clocks_from_ticks(getticks() - tic_32), clocks_getunit());
  }
  pm_mesh_compute_potential();
  /* What a step can run (the log nesting is reproduced elsewhere). */
  engine_activate_fof_attach_tasks();
  engine_activate_fof_tasks();
  engine_allocate_foreign_particles();
  engine_collect_end_of_step();
  engine_drift_all();
  engine_drift_top_multipoles();
  engine_exchange_cells();
  engine_exchange_strays();
  engine_exchange_top_multipoles();
  engine_fof();
  engine_launch();
  engine_make_fof_tasks();
  engine_makeproxies_with_zoom_region();
  engine_maketasks();
  engine_marktasks();
  engine_prepare();
  engine_print_task_counts();
  engine_rebuild();
  engine_recompute_displacement_constraint();
  engine_redistribute();
  engine_repartition();
  engine_repartition_trigger();
  engine_split_gas_particles();
  engine_unskip();
  fof_allocate();
  fof_compute_group_props();
  fof_compute_local_sizes();
  fof_finalise_attachables();
  fof_link_attachable_particles();
  partition_repartition();
  proxy_cells_exchange();
  proxy_tags_exchange();
  repart_edge_metis_zoom();
  scheduler_report_task_times();
  scheduler_reweight();
  space_bparts_get_cell_index();
  space_free_cells();
  space_gparts_get_cell_index();
  space_list_useful_top_level_cells();
  space_parts_get_cell_index();
  space_rebuild();
  space_regrid();
  space_regrid_zoom();
  space_sparts_get_cell_index();
  space_split();
  space_synchronize_particle_positions();
  void_space_split();
  zoom_engine_make_hierarchical_void_tasks();
  zoom_engine_make_self_gravity_tasks();
  zoom_get_region_dim_and_shift();
  zoom_void_space_split();
  zoom_void_timestep_collect();
}

void engine_unskip(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void fof_allocate(void) {
  const ticks tic = getticks();
  {
    const ticks tic_33 = getticks();
    if (verbose)
      message("Setting initial group index took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_33), clocks_getunit());
  }
  {
    const ticks tic_34 = getticks();
    if (verbose)
      message("Setting initial attach index took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_34), clocks_getunit());
  }
  {
    const ticks tic_35 = getticks();
    if (verbose)
      message("Setting initial distances took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_35), clocks_getunit());
  }
  {
    const ticks tic_36 = getticks();
    if (verbose)
      message("Setting initial group sizes took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_36), clocks_getunit());
  }
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void fof_compute_group_props(void) {
  const ticks tic = getticks();
  {
    const ticks tic_37 = getticks();
    if (verbose)
      message("Computing group properties took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_37), clocks_getunit());
  }
  {
    const ticks tic_38 = getticks();
    if (verbose)
      message("Calculating the total no. of local groups took: (FOF SCALING): %.3f %s.",
              clocks_from_ticks(getticks() - tic_38), clocks_getunit());
  }
  {
    const ticks tic_39 = getticks();
    if (verbose)
      message("Finding the total no. of groups took: (FOF SCALING): %.3f %s.",
              clocks_from_ticks(getticks() - tic_39), clocks_getunit());
  }
  {
    const ticks tic_40 = getticks();
    if (verbose)
      message("Setting default group ID took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_40), clocks_getunit());
  }
  {
    const ticks tic_41 = getticks();
    if (verbose)
      message("Group sorting took: %.3f %s.",
              clocks_from_ticks(getticks() - tic_41), clocks_getunit());
  }
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void fof_compute_local_sizes(void) {
  const ticks tic = getticks();
  {
    const ticks tic_42 = getticks();
    if (verbose)
      message("FOF calc group size took (FOF SCALING): %.3f %s.",
              clocks_from_ticks(getticks() - tic_42), clocks_getunit());
  }
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void fof_finalise_attachables(void) {
  {
    const ticks tic_43 = getticks();
    if (verbose)
      message("fof_finalise_attachables() took (FOF SCALING): %.3f %s.",
              clocks_from_ticks(getticks() - tic_43), clocks_getunit());
  }
}

void fof_link_attachable_particles(void) {
  {
    const ticks tic_44 = getticks();
    engine_activate_fof_attach_tasks();
    engine_launch();
    if (verbose)
      message("fof_link_attachable_particles() took (FOF SCALING): %.3f %s.",
              clocks_from_ticks(getticks() - tic_44), clocks_getunit());
  }
}

void partition_initial_partition(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void partition_repartition(void) {
  const ticks tic = getticks();
  repart_edge_metis_zoom();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void pm_mesh_compute_potential(void) {
  compute_potential_global();
}

void proxy_cells_exchange(void) {
  {
    const ticks tic_45 = getticks();
    if (verbose)
      message("Counting cells to send took %.3f %s.",
              clocks_from_ticks(getticks() - tic_45), clocks_getunit());
  }
  {
    const ticks tic_46 = getticks();
    if (verbose)
      message("Packing cells took %.3f %s.",
              clocks_from_ticks(getticks() - tic_46), clocks_getunit());
  }
  {
    const ticks tic_47 = getticks();
    if (verbose)
      message("Un-packing cells took %.3f %s.",
              clocks_from_ticks(getticks() - tic_47), clocks_getunit());
  }
}

void proxy_tags_exchange(void) {
  {
    const ticks tic_48 = getticks();
    if (verbose)
      message("Cell pack tags took %.3f %s.",
              clocks_from_ticks(getticks() - tic_48), clocks_getunit());
  }
  {
    const ticks tic_49 = getticks();
    if (verbose)
      message("Cell unpack tags took %.3f %s.",
              clocks_from_ticks(getticks() - tic_49), clocks_getunit());
  }
}

void repart_edge_metis_zoom(void) {
  {
    const ticks tic_50 = getticks();
    if (verbose)
      message("weight mapper took %.3f %s.",
              clocks_from_ticks(getticks() - tic_50), clocks_getunit());
  }
}

void restart_read(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void scheduler_report_task_times(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void scheduler_reweight(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void scheduler_write_dependencies(void) {
  {
    const ticks tic_51 = getticks();
    if (verbose)
      message("Printing task graph took %.3f %s.",
              clocks_from_ticks(getticks() - tic_51), clocks_getunit());
  }
}

void space_bparts_get_cell_index(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_convert_quantities(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_first_init_bparts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_first_init_gparts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_first_init_parts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_first_init_sinks(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_first_init_sparts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_free_cells(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_gparts_get_cell_index(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_init_bparts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_init_gparts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_init_parts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_init_sinks(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_init_sparts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_list_useful_top_level_cells(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_parts_get_cell_index(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_post_init_parts(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_rebuild(void) {
  const ticks tic = getticks();
  {
    const ticks tic_52 = getticks();
    if (verbose)
      message("Moving non-local particles took %.3f %s.",
              clocks_from_ticks(getticks() - tic_52), clocks_getunit());
  }
  {
    const ticks tic_53 = getticks();
    if (verbose)
      message("hooking up cells took %.3f %s.",
              clocks_from_ticks(getticks() - tic_53), clocks_getunit());
  }
  engine_exchange_strays();
  space_bparts_get_cell_index();
  space_gparts_get_cell_index();
  space_parts_get_cell_index();
  space_regrid();
  space_regrid_zoom();
  space_sparts_get_cell_index();
  space_split();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_regrid(void) {
  const ticks tic = getticks();
  space_free_cells();
  zoom_construct_tl_cells();
  zoom_get_region_dim_and_shift();
  zoom_region_init();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_regrid_zoom(void) {
  const ticks tic = getticks();
  space_free_cells();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_sparts_get_cell_index(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_split(void) {
  const ticks tic = getticks();
  {
    const ticks tic_54 = getticks();
    if (verbose)
      message("Buffer cell tree and multipole construction took %.3f %s.",
              clocks_from_ticks(getticks() - tic_54), clocks_getunit());
  }
  {
    const ticks tic_55 = getticks();
    if (verbose)
      message("Zoom cell tree and multipole construction took %.3f %s.",
              clocks_from_ticks(getticks() - tic_55), clocks_getunit());
  }
  {
    const ticks tic_56 = getticks();
    if (verbose)
      message("Background cell tree and multipole construction took %.3f %s.",
              clocks_from_ticks(getticks() - tic_56), clocks_getunit());
  }
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void space_synchronize_particle_positions(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void void_space_split(void) {
  {
    const ticks tic_57 = getticks();
    if (verbose)
      message("Void cell tree and multipole construction took %.3f %s.",
              clocks_from_ticks(getticks() - tic_57), clocks_getunit());
  }
}

void zoom_apply_zoom_shift_to_particles(void) {
  {
    const ticks tic_58 = getticks();
    if (verbose)
      message("Applying zoom region shift took %f %s",
              clocks_from_ticks(getticks() - tic_58), clocks_getunit());
  }
}

void zoom_construct_tl_cells(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void zoom_engine_make_hierarchical_void_tasks(void) {
  {
    const ticks tic_59 = getticks();
    if (verbose)
      message("Making void cell tree tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_59), clocks_getunit());
  }
}

void zoom_engine_make_self_gravity_tasks(void) {
  {
    const ticks tic_60 = getticks();
    if (verbose)
      message("Making bkg->bkg gravity tasks took %.3f %s.",
              clocks_from_ticks(getticks() - tic_60), clocks_getunit());
  }
}

void zoom_get_region_dim_and_shift(void) {
  {
    const ticks tic_61 = getticks();
    if (verbose)
      message("Computing high resolution particle dim and shift took %f %s",
              clocks_from_ticks(getticks() - tic_61), clocks_getunit());
  }
}

void zoom_props_init(void) {
  {
    const ticks tic_62 = getticks();
    if (verbose)
      message("Initialising zoom region properties took %f %s",
              clocks_from_ticks(getticks() - tic_62), clocks_getunit());
  }
}

void zoom_region_init(void) {
  {
    const ticks tic_63 = getticks();
    zoom_apply_zoom_shift_to_particles();
    zoom_get_region_dim_and_shift();
    if (verbose)
      message("Zoom region initialisation took %f %s",
              clocks_from_ticks(getticks() - tic_63), clocks_getunit());
  }
}

void zoom_void_space_split(void) {
  {
    const ticks tic_64 = getticks();
    if (verbose)
      message("Void cell tree and multipole construction took %.3f %s.",
              clocks_from_ticks(getticks() - tic_64), clocks_getunit());
  }
}

void zoom_void_timestep_collect(void) {
  const ticks tic = getticks();
  if (verbose)
    message("took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}
