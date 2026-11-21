// Sokoban solver: push-based A* over rooms.
//
// Every level this competition draws is a 1-wide hallway with 1-4 rooms off
// it, each room holding 3-4 boxes and as many goals, and a box pushed out of
// its room can never come back. So a level splits into independent rooms and
// the push-optimal count is the sum of the rooms' optima. This program finds
// each room (cut the player's row where a box could never reach a goal, then
// flood-fill the floor that is left), solves it on its own with a push-based
// A* (a state is the boxes plus the player's reachable region, named by its
// lowest square; a push costs 1), and stitches the rooms' move strings
// together, the player walking between rooms on the hallway.
//
// The A* estimate is the cheapest assignment of boxes to goals by
// push-distance (each box's fewest pushes to a goal, ignoring the other
// boxes). One push moves one box one square, so no box can reach its goal
// in fewer pushes than its push-distance, and no assignment can cost less
// than the cheapest one: the estimate never overstates the pushes left, so
// the first solution A* returns is push-optimal. Dead squares (no
// push-distance to any goal) and frozen 2x2 blocks of walls and boxes are
// pruned; neither can sit on a solvable path.
//
// All budgets are counts of work, never the clock, so two runs write the
// same bytes. If A* gives up on a room, a push-order search and then a
// greedy search try the same room; if the rooms do not split cleanly, the
// whole level is searched the same way. An unsolved instance still gets a
// (empty) answer file in the stated format.
//
// Node 22, erasable TypeScript only: no dependencies, no network, no clock,
// no environment reads except the input/output directories the harness sets.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expansions), never the clock.
const ASTAR_EXPANSIONS = 600_000;
const FALLBACK_EXPANSIONS = 600_000;
const WHOLE_LEVEL_EXPANSIONS = 400_000;
const MAX_STATES = 1_500_000;
const GLOBAL_EXPANSIONS = 6_000_000;

let expansionsUsed = 0;

// The four steps: [dx, dy, walk letter, push letter].
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// A level: width, height, walls, goals, boxes (sorted) and the player.
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// The squares from which a box alone can be pushed onto some goal: pull back
// from the goals over open floor (a box at y pushed along d lands on x, so
// from x the pull reaches y when the player could stand at y - d).
function liveSquares(level) {
  const { w, wall, goals } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  const queue = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
      const y = x - d;
      const behind = y - d;
      if (y < 0 || y >= size || behind < 0 || behind >= size) continue;
      if (!wall[y] && !wall[behind] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Fewest pushes to move a box from any square onto `goal`, ignoring the
// other boxes (the same pull-back walk as liveSquares, counting steps).
// -1 where no push path reaches the goal.
function pushDistTo(level, goal) {
  const { w, wall } = level;
  const size = wall.length;
  const dist = new Int16Array(size).fill(-1);
  dist[goal] = 0;
  const queue = [goal];
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
      const y = x - d;
      const behind = y - d;
      if (y < 0 || y >= size || behind < 0 || behind >= size) continue;
      if (!wall[y] && !wall[behind] && dist[y] < 0) {
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
  }
  return dist;
}

// The squares the player reaches from `from` with `occupied` (a box mask)
// in the way, marked in `seen` with `stamp`; returns the lowest reached.
function reach(level, occupied, from, seen, stamp, stack) {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
// and boxes holding a box off its goal.
function frozenBlock(level, occupied, goal, t) {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// The level's rooms: cut the dead squares of the player's row (the hallway:
// a box there reaches no goal) out of the floor, and each part left that
// holds a box is a room. Null when a box starts in the hallway, a goal is
// cut away, or some part holds more or fewer boxes than goals.
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] >= 0) rooms[part[g]].goals.push(g);
  }
  if (goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// The shortest walk from `from` to `to` around walls and boxes, in lower
// LURD letters (the first direction first on ties), or null when unreachable.
function walkTo(level, occupied, from, to) {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// The square the player stands on after `moves` (LURD, case aside).
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

// A push search over `sub` (a room, or a whole level): {pushes, moves}, or
// null when the budget runs out or the level has no solution. `weight` scales
// the estimate (1: A*, 0: push order, which is also optimal); `greedy`
// orders the open list by the estimate alone, which answers fast but not
// optimally. The budget is a count of expansions, shared with the run.
function solvePushSearch(sub, maxExpansions, weight, greedy) {
  const { w, wall, goals, boxes, player } = sub;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(sub);
  const dist = goals.map((g) => pushDistTo(sub, g));
  const n = boxes.length;
  if (boxes.some((b) => !live[b])) return null;
  const solved = (bs) => {
    for (const b of bs) if (!goal[b]) return false;
    return true;
  };
  if (solved(boxes)) return { pushes: 0, moves: '' };

  // The estimate: the cheapest assignment of boxes to goals by push-distance
  // (few boxes: every pairing tried; many: each box's nearest goal, which
  // can only understate the assignment). Infinity when no pairing works.
  const estimate = (bs) => {
    if (n <= 5) {
      let best = Infinity;
      const used = new Array(goals.length).fill(false);
      const rec = (i, sum) => {
        if (sum >= best) return;
        if (i === n) {
          best = sum;
          return;
        }
        for (let g = 0; g < goals.length; g++) {
          if (used[g]) continue;
          const d = dist[g][bs[i]];
          if (d < 0) continue;
          used[g] = true;
          rec(i + 1, sum + d);
          used[g] = false;
        }
      };
      rec(0, 0);
      return best;
    }
    let sum = 0;
    for (const b of bs) {
      let m = Infinity;
      for (let g = 0; g < goals.length; g++) {
        if (dist[g][b] >= 0 && dist[g][b] < m) m = dist[g][b];
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  const numericKey = n <= 5;
  const keyOf = (bs, low) => {
    if (!numericKey) return `${low}:${bs.join(',')}`;
    let k = low;
    for (let i = bs.length - 1; i >= 0; i--) k = k * size + bs[i];
    return k;
  };

  // The states, in the order found: boxes, region's lowest square (the
  // player's stand-in: any square of the region reaches the same pushes),
  // pushes made, parent, and the push that made them.
  const sBoxes = [boxes];
  const sLow = [0];
  const sG = [0];
  const sParent = [-1];
  const sFrom = [-1];
  const sTo = [-1];
  const sDir = [-1];
  const sKey = [null];
  const best = new Map();
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  for (const b of boxes) occupied[b] = 1;
  sLow[0] = reach(sub, occupied, player, seen, ++stamp, stack);
  for (const b of boxes) occupied[b] = 0;
  const startKey = keyOf(boxes, sLow[0]);
  sKey[0] = startKey;
  best.set(startKey, 0);

  // The open list: a binary heap on (f, -g, index), first found first.
  const hf = [];
  const hg = [];
  const hi = [];
  const less = (fi, gi, ii, fj, gj, ij) =>
    fi < fj || (fi === fj && (gi > gj || (gi === gj && ii < ij)));
  const heapPush = (f, g, idx) => {
    let i = hf.length;
    hf.push(f);
    hg.push(g);
    hi.push(idx);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(hf[i], hg[i], hi[i], hf[p], hg[p], hi[p])) break;
      [hf[i], hf[p]] = [hf[p], hf[i]];
      [hg[i], hg[p]] = [hg[p], hg[i]];
      [hi[i], hi[p]] = [hi[p], hi[i]];
      i = p;
    }
  };
  const heapPop = () => {
    const f = hf[0];
    const g = hg[0];
    const idx = hi[0];
    const lf = hf.pop();
    const lg = hg.pop();
    const li = hi.pop();
    if (hf.length > 0) {
      hf[0] = lf;
      hg[0] = lg;
      hi[0] = li;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= hf.length) break;
        if (c + 1 < hf.length && less(hf[c + 1], hg[c + 1], hi[c + 1], hf[c], hg[c], hi[c])) c++;
        if (!less(hf[c], hg[c], hi[c], hf[i], hg[i], hi[i])) break;
        [hf[i], hf[c]] = [hf[c], hf[i]];
        [hg[i], hg[c]] = [hg[c], hg[i]];
        [hi[i], hi[c]] = [hi[c], hi[i]];
        i = c;
      }
    }
    return [f, g, idx];
  };

  const h0 = estimate(boxes);
  if (h0 === Infinity) return null;
  heapPush(greedy ? h0 : h0, 0, 0);
  let expanded = 0;
  let found = -1;
  while (hf.length > 0 && expanded < maxExpansions && expansionsUsed < GLOBAL_EXPANSIONS) {
    const [, g, s] = heapPop();
    if (best.get(sKey[s]) !== s) continue; // a cheaper copy of this state exists
    if (solved(sBoxes[s])) {
      found = s;
      break;
    }
    expanded++;
    expansionsUsed++;
    const bs = sBoxes[s];
    for (const b of bs) occupied[b] = 1;
    reach(sub, occupied, sLow[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < bs.length && found < 0; bi++) {
      const b = bs[bi];
      for (let k = 0; k < 4; k++) {
        const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
        const behind = b - d;
        const t = b + d;
        if (seen[behind] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(sub, occupied, goal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = bs.slice();
        next[bi] = t;
        next.sort((a, c) => a - c);
        const low = reach(sub, occupied, b, mark, ++stamp, stack);
        const key = keyOf(next, low);
        const old = best.get(key);
        if (old !== undefined && sG[old] <= g + 1) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const h = estimate(next);
        occupied[t] = 0;
        occupied[b] = 1;
        if (h === Infinity) continue;
        if (sBoxes.length >= MAX_STATES) return null;
        const idx = sBoxes.length;
        sBoxes.push(next);
        sLow.push(low);
        sG.push(g + 1);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sDir.push(k);
        sKey.push(key);
        best.set(key, idx);
        heapPush(greedy ? h : g + 1 + weight * h, g + 1, idx);
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let p = player;
  let cur = boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s];
    const from = sFrom[s];
    const to = sTo[s];
    const d = to - from;
    for (const b of cur) occupied[b] = 1;
    const walk = walkTo(sub, occupied, p, from - d);
    for (const b of cur) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + DIRS[sDir[s]][3];
    p = from;
    cur = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// One room, in turn: A* first, then push-order search, then greedy.
function solveRoom(sub) {
  let r = solvePushSearch(sub, ASTAR_EXPANSIONS, 1, false);
  if (r !== null) return r;
  r = solvePushSearch(sub, FALLBACK_EXPANSIONS, 0, false);
  if (r !== null) return r;
  return solvePushSearch(sub, FALLBACK_EXPANSIONS, 0, true);
}

// The whole level at once (no rooms found, or a room would not solve):
// A* first, then greedy.
function solveWhole(level) {
  let r = solvePushSearch(level, WHOLE_LEVEL_EXPANSIONS, 1, false);
  if (r !== null) return r;
  return solvePushSearch(level, WHOLE_LEVEL_EXPANSIONS, 0, true);
}

// The moves solving one level, or '' when nothing solved it.
function solveLevel(text) {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solveWhole(level);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    // The other rooms' boxes stand as walls (on their goals once done), so
    // this room's search never touches them; the player walks in from the
    // hallway, wherever the last room left it.
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solveRoom(sub);
    if (solved === null) {
      const whole = solveWhole(level);
      return whole === null ? '' : whole.moves;
    }
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
