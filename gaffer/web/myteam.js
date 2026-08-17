/* gaffer — your own squad: import it, edit it, and find out what it costs you.
 *
 * Everything here runs in the browser against the already-loaded projections,
 * and that is a deliberate constraint rather than a shortcut. The published
 * site has no backend at all, and the one thing a phone genuinely needs before
 * a deadline is not another optimal squad — it is an answer to "I was going to
 * do this; is it worse?". That question is cheap to answer from data already in
 * memory, so it works offline, instantly, on a static host.
 *
 * What is genuinely NOT possible here, and is not faked:
 *
 *   - Re-running the optimiser. Picking the best 15 out of 587 under a budget
 *     is a MILP; the honest move is to say so and point at the local server.
 *     Evaluating a squad you already chose is just arithmetic, which is why
 *     that half works and the other half does not.
 *   - Fetching your team by entry id from the published site. The FPL API sends
 *     no CORS headers (verified: no access-control-allow-origin on
 *     /api/entry/{id}/), so the browser is not allowed to read it from another
 *     origin. The local server can, because it is not a browser. On the static
 *     site the export bakes the picks in instead.
 *
 * The best XI is exact, not a heuristic: within a fixed formation the choice is
 * independent per position, so taking the top N of each position by expected
 * points and maximising over the eight legal shapes is the true optimum.
 */
(function () {
  "use strict";

  var G = window.G || (window.G = {});
  var U = G.ui;
  var el = U.el, num = U.num, money = U.money, signed = U.signed;

  var STORAGE_KEY = "gaffer.myteam.v1";

  // 2 GKP, 5 DEF, 5 MID, 3 FWD — the squad, not the XI.
  var SQUAD_QUOTA = { 1: 2, 2: 5, 3: 5, 4: 3 };
  var POS_SHORT = { 1: "GKP", 2: "DEF", 3: "MID", 4: "FWD" };
  var BUDGET = 100.0;
  var MAX_PER_CLUB = 3;

  /* The eight legal shapes: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, ten outfield. */
  var FORMATIONS = [
    { d: 3, m: 4, f: 3 }, { d: 3, m: 5, f: 2 },
    { d: 4, m: 3, f: 3 }, { d: 4, m: 4, f: 2 }, { d: 4, m: 5, f: 1 },
    { d: 5, m: 2, f: 3 }, { d: 5, m: 3, f: 2 }, { d: 5, m: 4, f: 1 }
  ];

  // ------------------------------------------------------------- storage ----

  var state = { entryId: null, picks: [], source: "manual", savedAt: null };

  function load() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return state;
      var parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.picks)) {
        state = {
          entryId: parsed.entryId || null,
          picks: parsed.picks.map(Number).filter(function (n) { return !isNaN(n); }),
          source: parsed.source || "manual",
          savedAt: parsed.savedAt || null
        };
      }
    } catch (e) { /* private browsing, quota, corrupt value: start empty */ }
    return state;
  }

  function save() {
    state.savedAt = new Date().toISOString();
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      U.toast("Could not save your team — browser storage is unavailable.");
    }
    return state;
  }

  function clear() {
    state = { entryId: null, picks: [], source: "manual", savedAt: null };
    try { window.localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    return state;
  }

  // ----------------------------------------------------------- evaluation ---

  function byId() {
    var index = {};
    (G.store.players || []).forEach(function (p) { index[p.id] = p; });
    return index;
  }

  function resolve(ids) {
    var index = byId();
    return ids.map(function (id) { return index[id]; })
              .filter(function (p) { return !!p; });
  }

  /* Delegate to the app's own accessors rather than reading fields directly.
     The store normalises the API payload (web_name, now_cost in tenths, a gwMap
     keyed by gameweek) and duplicating that mapping here is how the two drift
     apart — which it already did once, silently, showing every price and every
     projection as zero. */
  function horizonXp(player) {
    if (!player) return 0;
    var v = G.data.horizonXp(player, G.store.gw, G.store.horizon);
    return typeof v === "number" && isFinite(v) ? v : 0;
  }

  function gwXp(player) {
    if (!player) return 0;
    var v = G.data.gwXp(player, G.store.gw);
    return typeof v === "number" && isFinite(v) ? v : 0;
  }

  /* Prices come off the wire in tenths of a million. */
  function price(player) {
    if (!player) return 0;
    return (typeof player.now_cost === "number") ? player.now_cost / 10 : 0;
  }

  function name(player) {
    return (player && (player.web_name || player.name)) || "unknown";
  }

  /* Exact best XI: maximise over the eight shapes, taking the top N per
     position within each. `key` picks the metric — the current gameweek for a
     lineup decision, the horizon for judging the squad as a whole. */
  function bestXi(players, key) {
    var score = key === "gw" ? gwXp : horizonXp;
    var pools = { 1: [], 2: [], 3: [], 4: [] };
    players.forEach(function (p) { if (pools[p.position]) pools[p.position].push(p); });
    Object.keys(pools).forEach(function (k) {
      pools[k].sort(function (a, b) { return score(b) - score(a); });
    });
    if (!pools[1].length) return null;

    var best = null;
    FORMATIONS.forEach(function (shape) {
      if (pools[2].length < shape.d || pools[3].length < shape.m || pools[4].length < shape.f) return;
      var xi = [pools[1][0]]
        .concat(pools[2].slice(0, shape.d))
        .concat(pools[3].slice(0, shape.m))
        .concat(pools[4].slice(0, shape.f));
      var total = xi.reduce(function (sum, p) { return sum + score(p); }, 0);
      if (!best || total > best.total) {
        best = { xi: xi, total: total, shape: shape,
                 label: shape.d + "-" + shape.m + "-" + shape.f };
      }
    });
    if (!best) return null;

    var bench = players.filter(function (p) { return best.xi.indexOf(p) < 0; });
    // The captain doubles, so the armband goes to the highest scorer in the XI.
    var captain = best.xi.slice().sort(function (a, b) { return score(b) - score(a); })[0];
    best.bench = bench;
    best.captain = captain;
    best.withCaptain = best.total + score(captain);
    return best;
  }

  /* Every way a squad can be illegal, each with the specific detail needed to
     fix it rather than a bare "invalid". */
  function legality(players) {
    var problems = [];
    var counts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    var clubs = {};
    var cost = 0;

    players.forEach(function (p) {
      counts[p.position] = (counts[p.position] || 0) + 1;
      clubs[p.team] = (clubs[p.team] || 0) + 1;
      cost += price(p);
    });

    if (players.length !== 15) {
      problems.push({
        kind: "size",
        text: players.length < 15
          ? "Only " + players.length + " of 15 picked — " + (15 - players.length) + " to go."
          : players.length + " players: a squad is exactly 15."
      });
    }
    Object.keys(SQUAD_QUOTA).forEach(function (pos) {
      var want = SQUAD_QUOTA[pos], got = counts[pos] || 0;
      if (got !== want) {
        problems.push({
          kind: "quota",
          text: POS_SHORT[pos] + ": " + got + " of " + want +
                (got > want ? " — " + (got - want) + " too many." : " — " + (want - got) + " short.")
        });
      }
    });
    Object.keys(clubs).forEach(function (club) {
      if (clubs[club] > MAX_PER_CLUB) {
        problems.push({
          kind: "club",
          text: club + ": " + clubs[club] + " players, and the limit is " + MAX_PER_CLUB + "."
        });
      }
    });
    if (cost > BUDGET + 1e-9) {
      problems.push({
        kind: "budget",
        text: "£" + cost.toFixed(1) + "m spent, which is £" +
              (cost - BUDGET).toFixed(1) + "m over the £100.0m budget."
      });
    }
    return { problems: problems, cost: cost, counts: counts, clubs: clubs,
             legal: problems.length === 0 };
  }

  /* Flags worth seeing next to a name: these are the things that quietly wreck
     a gameweek, and they are all already in the projection payload. */
  function risks(player) {
    var out = [];
    if (!player) return out;
    if (player.status && player.status !== "a") {
      out.push({ level: "bad", text: player.news || "flagged as unavailable" });
    }
    var p60 = player.p_60;
    if (typeof p60 === "number" && p60 < 0.55) {
      out.push({ level: "warn", text: "only " + Math.round(p60 * 100) + "% likely to play 60 minutes" });
    }
    if (player.is_blank) out.push({ level: "warn", text: "blank gameweek" });
    return out;
  }

  function evaluate(ids) {
    var players = resolve(ids);
    var check = legality(players);
    var gwBest = bestXi(players, "gw");
    var horizonBest = bestXi(players, "horizon");
    return {
      players: players,
      legality: check,
      gw: gwBest,
      horizon: horizonBest,
      totalHorizonXp: players.reduce(function (s, p) { return s + horizonXp(p); }, 0)
    };
  }

  /* Compare against the squad the optimiser actually chose, and explain each
     difference in the terms that decided it. Pairing is by position so the
     comparison is like-for-like: swapping a midfielder for a defender is not a
     sentence anyone can act on. */
  function compare(mineIds, theirsIds) {
    var index = byId();
    var mine = mineIds.slice(), theirs = theirsIds.slice();
    var mineSet = {}, theirsSet = {};
    mine.forEach(function (id) { mineSet[id] = true; });
    theirs.forEach(function (id) { theirsSet[id] = true; });

    var onlyMine = mine.filter(function (id) { return !theirsSet[id]; })
                       .map(function (id) { return index[id]; }).filter(Boolean);
    var onlyTheirs = theirs.filter(function (id) { return !mineSet[id]; })
                           .map(function (id) { return index[id]; }).filter(Boolean);

    var swaps = [];
    var remaining = onlyTheirs.slice();
    onlyMine.slice()
      .sort(function (a, b) { return horizonXp(a) - horizonXp(b); })
      .forEach(function (out) {
        // Prefer a same-position replacement; fall back to the best available.
        var candidates = remaining.filter(function (p) { return p.position === out.position; });
        var pool = candidates.length ? candidates : remaining;
        if (!pool.length) { swaps.push({ out: out, into: null }); return; }
        var into = pool.slice().sort(function (a, b) { return horizonXp(b) - horizonXp(a); })[0];
        remaining.splice(remaining.indexOf(into), 1);
        swaps.push({
          out: out,
          into: into,
          xpDelta: horizonXp(into) - horizonXp(out),
          costDelta: price(into) - price(out),
          samePosition: into.position === out.position
        });
      });
    remaining.forEach(function (into) { swaps.push({ out: null, into: into }); });

    swaps.sort(function (a, b) {
      return (b.xpDelta === undefined ? 0 : b.xpDelta) - (a.xpDelta === undefined ? 0 : a.xpDelta);
    });
    return { swaps: swaps, onlyMine: onlyMine, onlyTheirs: onlyTheirs };
  }

  /* The sentence under a swap. Names the size of the gap first, because that is
     the decision, then the reasons that produced it. */
  function explainSwap(swap) {
    if (!swap.out) return "The optimiser also holds " + name(swap.into) + ", which you do not.";
    if (!swap.into) return "You hold " + name(swap.out) + ", who is not in the optimiser's squad.";

    var bits = [];
    var d = swap.xpDelta;
    var verdict = Math.abs(d) < 0.75
      ? "Line-ball: " + name(swap.into) + " projects " + signed(d) + " points over the horizon, inside the model's own error."
      : (d > 0
          ? name(swap.into) + " projects " + signed(d) + " points more over the horizon."
          : name(swap.out) + " actually projects " + signed(-d) + " points more — keeping him is defensible.");
    bits.push(verdict);

    if (Math.abs(swap.costDelta) >= 0.05) {
      bits.push(swap.costDelta > 0
        ? "He costs £" + Math.abs(swap.costDelta).toFixed(1) + "m more, so the money has to come from somewhere."
        : "He is £" + Math.abs(swap.costDelta).toFixed(1) + "m cheaper, freeing money for elsewhere.");
    }
    if (!swap.samePosition) {
      bits.push("Different position (" + POS_SHORT[swap.out.position] + " for " +
                POS_SHORT[swap.into.position] + "), so this is a reshape, not a straight swap.");
    }
    risks(swap.out).forEach(function (r) { bits.push(name(swap.out) + ": " + r.text + "."); });
    return bits.join(" ");
  }

  G.myteam = {
    STORAGE_KEY: STORAGE_KEY,
    load: load, save: save, clear: clear,
    get state() { return state; },
    setPicks: function (ids, source) {
      state.picks = ids.slice();
      if (source) state.source = source;
      return save();
    },
    setEntryId: function (id) { state.entryId = id ? Number(id) : null; return save(); },
    evaluate: evaluate,
    price: price,
    name: name,
    gwXp: gwXp,
    compare: compare,
    explainSwap: explainSwap,
    bestXi: bestXi,
    legality: legality,
    risks: risks,
    resolve: resolve,
    horizonXp: horizonXp,
    FORMATIONS: FORMATIONS,
    SQUAD_QUOTA: SQUAD_QUOTA,
    POS_SHORT: POS_SHORT,
    BUDGET: BUDGET,
    MAX_PER_CLUB: MAX_PER_CLUB
  };
})();
