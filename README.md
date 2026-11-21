# Sokoban solver — make.fun entry

Push-optimal Sokoban solver for the make.fun bounty
`sokoban-solver-01-run-20261121` (21 November 2026).

`main.ts` (Node 22, no dependencies) answers every instance of a set in
one run, writing `solution.txt` per instance. It splits each level into
independent rooms, then runs a push-based A* per room with an admissible
min-assignment heuristic on push-distances and deadlock pruning
(dead squares, frozen 2x2), stitching rooms together with hallway walks.
All budgets are by expansion count, never the clock, so runs are
deterministic.

Practice set: 1,000,000 / 1,000,000 (exact push-optimal on all 5 instances),
~0.3 s for the whole set. House baseline (plain-astar) scores 200,000 on the
same set.
