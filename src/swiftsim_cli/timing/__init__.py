"""Timer analysis of SWIFT logs.

Submodules:
    timer_db: Timer sites and static call graph parsed from SWIFT source.
    log_scan: Single streaming pass over a log (timers, steps, tasks).
    timer_nesting: Runtime call tree of the timers in each step.
    timing_hierarchy: Aggregation of the trees into tables.
    timing_plots: Charts drawn from the aggregates.
"""
