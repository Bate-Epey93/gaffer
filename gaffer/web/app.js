/* gaffer dashboard — API client, store, router, boot.
 *
 * ---------------------------------------------------------------------------
 * BACKEND CONTRACT (SPEC 3.14). Everything is same-origin under /api. Field
 * names below are what this client reads; where a name was not nailed down in
 * the spec the client accepts the listed aliases, so a reasonable FastAPI
 * serialisation of gaffer/core/types.py works without changes here.
 *
 *  GET  /api/state
 *       { season, current_gw, deadline|deadline_time, n_players, n_teams,
 *         n_fixtures, finished_gws[], first_gw, last_gw, fitted?: bool,
 *         entry_id?, teams: [{id, name, short_name, code}] }
 *
 *  GET  /api/players?gw=&horizon=
 *       { players: [ Player ] }  (a bare array is accepted too)
 *       Player = { id|player_id, web_name|name, team_id|team, team|team_short,
 *                  position|element_type, now_cost (tenths) | price (millions),
 *                  selected_by_percent|ownership, form, status, news,
 *                  chance_of_playing_next_round,
 *                  p_start, p_60, p_appear, xmins, p_defcon,
 *                  xg90|expected_goals_per_90, xa90|expected_assists_per_90,
 *                  xp   (horizon total; used only if per-GW data is absent),
 *                  gws: [ { gw, xp, sd, p_haul, fixtures: [ PlayerFixtureProjection ] } ] }
 *       If `gws` is missing the client falls back to GET /api/projections and
 *       merges the ProjectionSet in, so the per-component audit still works.
 *
 *  GET  /api/projections?first=&last=
 *       ProjectionSet as written by XPEngine.save_projections:
 *       { season, generated_at, first_gw, last_gw, model_version,
 *         projections: { "<player_id>": { "<gw>": PlayerGWProjection } },
 *         haul?: { "<player_id>|<gw>": p } }
 *
 *  GET  /api/player/{id}?gw=   -> { player?, explanation|explain, minutes_reason?, gws? }
 *  GET  /api/fixtures?gw=      -> { fixtures: [ { id, gw|event, team_h, team_a,
 *                                   kickoff_time, team_h_difficulty, team_a_difficulty,
 *                                   lambda_h, lambda_a, p_cs_h, p_cs_a } ] }
 *  POST /api/optimize          -> body { horizon, entry_id?, chips?, locks? }
 *                                 returns Plan { first_gw, horizon, objective,
 *                                 solver_status, decisions: [ GWDecision ] }
 *  GET  /api/captain?gw=       -> { gw, options: [ CaptainOption ] }
 *  GET  /api/chips             -> { recommendations: [ {chip, gw, reason, confidence?} ],
 *                                   first_half_deadline?, doubles?, blanks?, chips_used? }
 *  POST /api/refresh           -> anything; the client just reloads afterwards.
 *
 * A 4xx/5xx body's `detail` (FastAPI's HTTPException shape), `error` or
 * `message` is shown verbatim. Nothing is invented when a call fails.
 * ---------------------------------------------------------------------------
 */
(function () {
  'use strict';

  var G = (window.G = window.G || {});
  var U = G.ui;
  var el = U.el, qs = U.qs, isNum = U.isNum;

  var API = '/api';

  // The first set of chips dies at the GW19 deadline: 2027-01-02 (SPEC 1).
  // Only the date is documented, so the time is an estimate until the backend
  // hands us the real deadline from the events feed — the UI says which it is.
  var CHIP_EXPIRY_FALLBACK = '2027-01-02T11:30:00Z';

  var store = {
    mode: 'loading',          // loading | live | sample
    sampleReason: '',
    // Set when the service worker answered an /api call out of its cache
    // because the backend was unreachable. { at: ISO string, calls: n }.
    stale: null,
    origin: location.origin,
    state: null,
    teams: {},
    players: [],
    byId: {},
    fixtures: [],
    fixtureLambdas: {},
    gw: 1,
    gws: [],
    horizon: 6,
    projectionMeta: null,
    fullComponentGws: null,   // sample mode only: gws carrying a full breakdown
    plan: null, planError: null, planLoading: false,
    captainByGw: {}, captainError: null, captainLoading: false,
    chips: null, chipsError: null, chipsLoading: false,
    squad: null,
    view: 'squad'
  };
  G.store = store;

  // ------------------------------------------------------------- http ------

  function apiUrl(path) { return API + path; }

  /* How long to wait, by endpoint. One flat 20s budget used to abort the two
     endpoints that solve a MILP: /chips is measurably ~20.5s against a freshly
     built snapshot (Free Hit and Wildcard each solve their own problem on top
     of the recommended-fifteen solve), so the first call on every page load
     was cancelled and the header fell back to a *guessed* chip deadline.
     /state is the other one: on a cold backend it blocks on the first fit
     (~16s here with an empty cache) and aborting it drops the whole dashboard
     into SAMPLE mode while the backend is in fact healthy. These are waits,
     not faults, so they get room; everything else keeps the short leash. */
  var TIMEOUT_DEFAULT_MS = 20000;
  var TIMEOUT_MS = [
    [/^\/chips/, 120000],
    [/^\/optimize/, 120000],
    [/^\/refresh/, 300000],
    [/^\/state/, 90000],
    [/^\/players/, 90000],
    [/^\/projections/, 90000]
  ];

  function timeoutFor(path) {
    for (var i = 0; i < TIMEOUT_MS.length; i++) {
      if (TIMEOUT_MS[i][0].test(path)) return TIMEOUT_MS[i][1];
    }
    return TIMEOUT_DEFAULT_MS;
  }

  /* The service worker stamps X-Gaffer-Cached-At onto anything it serves from
     its cache, which it only does when the backend could not be reached. A
     response carrying that header is therefore a *replay*, and the numbers in
     it may already be worthless — a projection is only true until the next
     team news. Record the oldest one seen; boot() turns it into a banner. */
  function noteStaleness(res) {
    var at;
    try { at = res.headers.get('X-Gaffer-Cached-At'); } catch (e) { at = null; }
    if (!at) return;
    if (!store.stale) store.stale = { at: at, calls: 0 };
    else if (at < store.stale.at) store.stale.at = at;   // ISO strings sort
    store.stale.calls += 1;
    // A lazily loaded view (captain, chips) can go stale long after boot, so
    // the warning cannot live in boot() alone.
    if (store.mode === 'live') announceStale();
  }

  /* ---------------------------------------------------------- static mode --
     A published snapshot has no backend: `gaffer export` freezes every route
     to data/*.json and the site is served from a CDN. Rather than teach every
     caller about two worlds, the switch lives here — `request()` keeps its
     signature and rewrites the path to a file when a snapshot is detected.

     Detection is a probe for data/manifest.json at boot, not a build-time flag,
     so the very same index.html works served by the live API and served from a
     static host. The one thing that genuinely cannot work is re-solving against
     constraints changed on the phone, so those POSTs fail with a plain message
     rather than a network error nobody can act on. */
  var STATIC = null;          // null = not yet probed, else the manifest or false

  function staticFile(path) {
    var clean = path.split('#')[0];
    var bits = clean.split('?');
    var route = bits[0];
    var query = bits[1] || '';

    if (route === '/state') return 'data/state.json';
    if (route === '/fixtures') return 'data/fixtures.json';
    if (route === '/players') return 'data/players.json';
    if (route === '/projections') return 'data/projections.json';
    if (route === '/chips') return 'data/chips.json';
    if (route === '/optimize') return 'data/optimize.json';
    // The manager's own squad, fetched by CI (which is not a browser and so is
    // not bound by CORS) and baked into the build.
    if (route === '/squad') return 'data/entry.json';
    if (route === '/captain') {
      var gw = /gw=(\d+)/.exec(query);
      return 'data/captain-gw' + (gw ? gw[1] : (STATIC && STATIC.gw) || 1) + '.json';
    }
    var player = /^\/player\/(\d+)$/.exec(route);
    if (player) return 'data/player/' + player[1] + '.json';
    return null;
  }

  /* The per-player file carries the shared blocks once and the two fields that
     actually vary — `player` and `explanation` — keyed by gameweek, so the
     drawer gets the same shape the live API would have returned. */
  function applyPlayerOverrides(doc, path) {
    if (!doc || !doc.by_gw_overrides) return doc;
    var gw = /gw=(\d+)/.exec(path);
    var key = gw ? gw[1] : String((STATIC && STATIC.gw) || 1);
    var override = doc.by_gw_overrides[key];
    if (override) {
      if (override.player) doc.player = override.player;
      if (override.explanation) doc.explanation = override.explanation;
    }
    return doc;
  }

  /* Does this optimise ask the same question the snapshot already answered?
     Compared field by field against the manifest's record of the frozen
     request, so adding a constraint the export never saw is caught rather
     than silently answered with the default. */
  function sameOptimizeRequest(body) {
    var frozen = (STATIC && STATIC.optimize_request) || {};
    var asked = body || {};
    var keys = {};
    Object.keys(frozen).forEach(function (k) { keys[k] = true; });
    Object.keys(asked).forEach(function (k) { keys[k] = true; });
    return Object.keys(keys).every(function (k) {
      var a = frozen[k], b = asked[k];
      if (a === undefined || a === null) return b === undefined || b === null;
      return String(a) === String(b);
    });
  }

  function staticRequest(path, opts) {
    var method = (opts && opts.method) || 'GET';
    var file = staticFile(path);
    if (method !== 'GET' && path.indexOf('/optimize') === 0) {
      // One solve is frozen in the snapshot: the one the export ran. Serve it
      // when this is the same question, and say so plainly when it is not,
      // rather than quietly handing back an answer to a different query.
      if (!sameOptimizeRequest(opts && opts.body)) {
        var e = new Error(
          'This is a published snapshot, so it cannot re-run the optimiser. ' +
          'Run gaffer locally to solve with your own constraints.');
        e.status = 501; e.path = path;
        return Promise.reject(e);
      }
      file = 'data/optimize.json';
    } else if (method !== 'GET') {
      var e2 = new Error(
        'This is a published snapshot and cannot ' +
        (path.indexOf('/refresh') === 0 ? 'refresh the data' : 'do that') +
        '. It rebuilds itself on a schedule.');
      e2.status = 501; e2.path = path;
      return Promise.reject(e2);
    }
    if (!file) {
      var e3 = new Error('not available in a published snapshot');
      e3.status = 501; e3.path = path;
      return Promise.reject(e3);
    }
    return fetch(file, { headers: { 'Accept': 'application/json' } })
      .then(function (res) {
        if (!res.ok) {
          var err = new Error(res.status === 404
            ? 'this snapshot does not include ' + file
            : res.status + ' ' + res.statusText);
          err.status = res.status; err.path = path;
          throw err;
        }
        return res.json();
      })
      .then(function (body) { return applyPlayerOverrides(body, path); });
  }

  /* Probe once, before anything else runs. Resolves to the manifest or false. */
  function detectStatic() {
    if (STATIC !== null) return Promise.resolve(STATIC);
    return fetch('data/manifest.json', { headers: { 'Accept': 'application/json' } })
      .then(function (res) { return res.ok ? res.json() : false; })
      .catch(function () { return false; })
      .then(function (manifest) {
        STATIC = (manifest && manifest.static) ? manifest : false;
        return STATIC;
      });
  }
  G.detectStatic = detectStatic;
  G.isStatic = function () { return !!STATIC; };
  // The manifest itself, for views that need to explain what this build carries
  // (which team was imported, when it was generated).
  G.staticManifest = function () { return STATIC || {}; };

  function request(path, opts) {
    opts = opts || {};
    if (STATIC) return staticRequest(path, opts);
    var budget = opts.timeout || timeoutFor(path);
    var ctrl = ('AbortController' in window) ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, budget) : null;
    var init = {
      method: opts.method || 'GET',
      headers: { 'Accept': 'application/json' },
      credentials: 'same-origin'
    };
    if (ctrl) init.signal = ctrl.signal;
    if (opts.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }
    return fetch(apiUrl(path), init).then(function (res) {
      if (timer) clearTimeout(timer);
      noteStaleness(res);
      var ct = res.headers.get('content-type') || '';
      var parse = ct.indexOf('json') >= 0 ? res.json() : res.text();
      return parse.then(function (body) {
        if (!res.ok) {
          var msg = null;
          if (body && typeof body === 'object') {
            msg = body.detail || body.error || body.message;
            if (msg && typeof msg === 'object') msg = JSON.stringify(msg);
          } else if (typeof body === 'string' && body.length && body.length < 400) {
            msg = body;
          }
          var err = new Error(msg || (res.status + ' ' + res.statusText));
          err.status = res.status;
          err.path = path;
          throw err;
        }
        return body;
      }, function () {
        if (!res.ok) {
          var e2 = new Error(res.status + ' ' + res.statusText);
          e2.status = res.status; e2.path = path; throw e2;
        }
        var e3 = new Error('response was not valid JSON');
        e3.path = path; throw e3;
      });
    }, function (netErr) {
      if (timer) clearTimeout(timer);
      var e = new Error(
        netErr && netErr.name === 'AbortError'
          ? 'no response within ' + Math.round(budget / 1000) + 's'
          : (netErr && netErr.message) || 'network error'
      );
      e.network = true;
      e.path = path;
      throw e;
    });
  }
  G.request = request;

  function describeError(err) {
    var where = 'GET ' + API + (err && err.path ? err.path : '');
    var what = (err && err.message) || 'unknown error';
    if (err && err.status) what = err.status + ' — ' + what;
    return { where: where, what: what };
  }
  G.describeError = describeError;

  // -------------------------------------------------------- normalisation --

  function pick(o, names, dflt) {
    for (var i = 0; i < names.length; i++) {
      var v = o[names[i]];
      if (v !== undefined && v !== null) return v;
    }
    return dflt;
  }

  function toNum(v, dflt) {
    if (typeof v === 'number') return isFinite(v) ? v : (dflt === undefined ? null : dflt);
    if (typeof v === 'string' && v.trim() !== '') {
      var n = parseFloat(v);
      if (isFinite(n)) return n;
    }
    return dflt === undefined ? null : dflt;
  }

  var POS_NAME = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' };
  var POS_ID = { GKP: 1, DEF: 2, MID: 3, FWD: 4, GK: 1, G: 1, D: 2, M: 3, F: 4 };

  function normFixture(raw) {
    var f = {};
    f.fixture_id = toNum(pick(raw, ['fixture_id', 'id'], null));
    f.opponent_id = toNum(pick(raw, ['opponent_id', 'opponent', 'opponent_team'], null));
    f.is_home = !!pick(raw, ['is_home', 'was_home', 'home'], false);
    f.difficulty = toNum(pick(raw, ['difficulty', 'fdr'], null));
    f.team_lambda = toNum(pick(raw, ['team_lambda', 'lambda_team', 'lambda_for'], null));
    f.opponent_lambda = toNum(pick(raw, ['opponent_lambda', 'lambda_opponent', 'lambda_against'], null));
    f.xp_total = toNum(pick(raw, ['xp_total', 'xp'], null));
    f.sd_total = toNum(raw.sd_total, null);
    ['p_appear', 'p_start', 'p_60', 'xmins', 'lambda_goals', 'lambda_assists',
     'lambda_conceded', 'lambda_saves', 'p_clean_sheet', 'p_defcon', 'exp_bps'
    ].forEach(function (k) { f[k] = toNum(raw[k], null); });
    U.COMPONENTS.forEach(function (c) { f[c[0]] = toNum(raw[c[0]], null); });
    f.has_components = f.xp_appearance !== null || f.xp_goals !== null;
    return f;
  }

  function normGw(raw) {
    var fixtures = (pick(raw, ['fixtures', 'fx'], []) || []).map(normFixture);
    return {
      gw: toNum(raw.gw, 0),
      xp: toNum(pick(raw, ['xp', 'xp_total'], 0), 0),
      sd: toNum(pick(raw, ['sd', 'sd_total'], 0), 0),
      p_haul: toNum(pick(raw, ['p_haul', 'haul'], null)),
      fixtures: fixtures
    };
  }

  function normPlayer(raw) {
    var p = {};
    p.id = toNum(pick(raw, ['id', 'player_id', 'element'], null));
    p.web_name = String(pick(raw, ['web_name', 'name', 'player_name'], 'player ' + p.id));

    var team = pick(raw, ['team_id', 'team'], null);
    if (typeof team === 'string' && isNaN(parseInt(team, 10))) {
      p.team_id = null; p.team = team;
    } else {
      p.team_id = toNum(team, null);
      p.team = pick(raw, ['team', 'team_short', 'team_name', 'short_name'], null);
      if (typeof p.team !== 'string') p.team = null;
    }
    if (!p.team && p.team_id && store.teams[p.team_id]) p.team = store.teams[p.team_id].short_name;

    var pos = pick(raw, ['position', 'element_type', 'pos'], null);
    if (typeof pos === 'string') p.position = POS_ID[pos.toUpperCase()] || null;
    else p.position = toNum(pos, null);
    p.pos = POS_NAME[p.position] || '—';

    var cost = pick(raw, ['now_cost', 'cost'], null);
    if (cost === null) {
      var pr = toNum(pick(raw, ['price'], null));
      p.now_cost = pr === null ? null : Math.round(pr * 10);
    } else {
      p.now_cost = toNum(cost, null);
    }

    p.selected_by_percent = toNum(pick(raw, ['selected_by_percent', 'ownership', 'selected'], 0), 0);
    p.form = toNum(raw.form, 0);
    p.status = pick(raw, ['status'], 'a');
    p.news = pick(raw, ['news'], '') || '';
    p.chance = toNum(pick(raw, ['chance_of_playing_next_round', 'chance'], null));
    p.minutes_reason = pick(raw, ['minutes_reason', 'reason'], '') || '';
    p.total_points = toNum(raw.total_points, null);
    p.penalties_order = toNum(raw.penalties_order, null);

    p.p_start = toNum(raw.p_start, null);
    p.p_60 = toNum(pick(raw, ['p_60', 'p60'], null));
    p.p_appear = toNum(raw.p_appear, null);
    p.xmins = toNum(raw.xmins, null);
    p.p_defcon = toNum(pick(raw, ['p_defcon', 'defcon_probability'], null));
    p.xg90 = toNum(pick(raw, ['xg90', 'xG90', 'expected_goals_per_90'], null));
    p.xa90 = toNum(pick(raw, ['xa90', 'xA90', 'expected_assists_per_90'], null));

    p.xp_flat = toNum(pick(raw, ['xp', 'xp_horizon', 'xp_total'], null));
    var gws = pick(raw, ['gws', 'gameweeks', 'projections'], null);
    p.gws = Array.isArray(gws) ? gws.map(normGw) : [];
    p.gwMap = {};
    p.gws.forEach(function (g) { p.gwMap[g.gw] = g; });
    return p;
  }

  /* Attach a ProjectionSet (the on-disk / API shape keyed by player then gw). */
  function mergeProjectionSet(ps) {
    var projections = ps.projections || {};
    var haul = ps.haul || {};
    var first = toNum(ps.first_gw, null), last = toNum(ps.last_gw, null);
    var n = 0;
    Object.keys(projections).forEach(function (pidKey) {
      var pid = parseInt(pidKey, 10);
      var p = store.byId[pid];
      if (!p) return;
      var byGw = projections[pidKey];
      var list = Object.keys(byGw).map(function (gwKey) {
        var g = normGw(byGw[gwKey]);
        if (!g.gw) g.gw = parseInt(gwKey, 10);
        if (g.p_haul === null) {
          var h = haul[pid + '|' + g.gw];
          if (h !== undefined) g.p_haul = toNum(h, null);
        }
        return g;
      }).sort(function (a, b) { return a.gw - b.gw; });
      p.gws = list;
      p.gwMap = {};
      list.forEach(function (g) { p.gwMap[g.gw] = g; });
      // fill the per-fixture summary fields the table wants, if absent
      var g0 = list[0];
      if (g0 && g0.fixtures.length) {
        var f0 = g0.fixtures[0];
        ['p_start', 'p_60', 'p_appear', 'xmins', 'p_defcon'].forEach(function (k) {
          if (p[k] === null && f0[k] !== null) p[k] = f0[k];
        });
      }
      n++;
    });
    store.projectionMeta = {
      generated_at: ps.generated_at || null,
      model_version: ps.model_version || null,
      first_gw: first, last_gw: last, merged: n
    };
    if (first !== null && last !== null) {
      store.gws = [];
      for (var g = first; g <= last; g++) store.gws.push(g);
    }
    return n;
  }

  function indexPlayers(list) {
    store.players = list;
    store.byId = {};
    list.forEach(function (p) { if (p.id !== null) store.byId[p.id] = p; });
    if (!store.gws.length) {
      var seen = {};
      list.forEach(function (p) { p.gws.forEach(function (g) { seen[g.gw] = 1; }); });
      store.gws = Object.keys(seen).map(Number).sort(function (a, b) { return a - b; });
    }
  }

  function indexFixtures(list) {
    store.fixtures = list;
    store.fixtureLambdas = {};
    list.forEach(function (f) {
      var id = toNum(pick(f, ['id', 'fixture_id'], null));
      if (id === null) return;
      store.fixtureLambdas[id] = {
        gw: toNum(pick(f, ['gw', 'event'], null)),
        team_h: toNum(f.team_h, null),
        team_a: toNum(f.team_a, null),
        lambda_h: toNum(pick(f, ['lambda_h', 'lambda_home'], null)),
        lambda_a: toNum(pick(f, ['lambda_a', 'lambda_away'], null)),
        kickoff_time: f.kickoff_time || null,
        team_h_difficulty: toNum(f.team_h_difficulty, null),
        team_a_difficulty: toNum(f.team_a_difficulty, null)
      };
    });
  }

  // -------------------------------------------------------- derived reads --

  function teamShort(id) {
    var t = store.teams[id];
    return t ? t.short_name : (id === null || id === undefined ? '—' : 'T' + id);
  }
  function teamName(id) {
    var t = store.teams[id];
    return t ? t.name : teamShort(id);
  }

  /* Our own lambdas for a projected fixture, with an explicit source so the UI
     can admit when it is falling back to the official FDR. */
  function lambdasFor(fx) {
    if (isNum(fx.team_lambda) && isNum(fx.opponent_lambda)) {
      return { tl: fx.team_lambda, ol: fx.opponent_lambda, source: 'model' };
    }
    var fl = store.fixtureLambdas[fx.fixture_id];
    if (fl && isNum(fl.lambda_h) && isNum(fl.lambda_a)) {
      return fx.is_home
        ? { tl: fl.lambda_h, ol: fl.lambda_a, source: 'model' }
        : { tl: fl.lambda_a, ol: fl.lambda_h, source: 'model' };
    }
    if (isNum(fx.difficulty)) return { tl: null, ol: null, source: 'fdr', fdr: fx.difficulty };
    return { tl: null, ol: null, source: 'none' };
  }

  function easeOf(fx) {
    var l = lambdasFor(fx);
    if (l.source === 'model') {
      return { t: U.easeIndexFromLambdas(l.tl, l.ol), source: 'model', tl: l.tl, ol: l.ol };
    }
    if (l.source === 'fdr') return { t: U.easeIndexFromFdr(l.fdr), source: 'fdr' };
    return { t: null, source: 'none' };
  }

  function horizonXp(p, gw, h) {
    if (p.gws.length) {
      var s = 0, any = false;
      for (var i = 0; i < p.gws.length; i++) {
        var g = p.gws[i];
        if (g.gw >= gw && g.gw < gw + h) { s += g.xp; any = true; }
      }
      if (any) return s;
    }
    return isNum(p.xp_flat) ? p.xp_flat : 0;
  }

  function gwXp(p, gw) {
    var g = p.gwMap[gw];
    return g ? g.xp : 0;
  }

  function nextFixtures(p, gw, n) {
    var out = [];
    for (var g = gw; g < gw + n; g++) {
      var e = p.gwMap[g];
      out.push({ gw: g, fixtures: e ? e.fixtures : [], known: !!e });
    }
    return out;
  }

  G.data = {
    teamShort: teamShort, teamName: teamName, lambdasFor: lambdasFor, easeOf: easeOf,
    horizonXp: horizonXp, gwXp: gwXp, nextFixtures: nextFixtures, POS_NAME: POS_NAME,
    toNum: toNum, pick: pick, normPlayer: normPlayer, normGw: normGw
  };

  // ------------------------------------------------------------- loading --

  function setSource(kind, text) {
    var pill = qs('#source-pill');
    pill.className = 'pill pill-source ' + kind;
    qs('#source-text').textContent = text;
  }

  function banner(html, kind, actions) {
    var b = qs('#banner');
    U.clear(b);
    b.className = 'banner' + (kind ? ' ' + kind : '');
    b.hidden = false;
    b.appendChild(el('div', { html: html }));
    if (actions && actions.length) {
      var box = el('div', { class: 'banner-actions' });
      actions.forEach(function (a) {
        box.appendChild(el('button', { class: 'btn btn-sm', type: 'button', onclick: a.fn }, a.label));
      });
      b.appendChild(box);
    }
  }
  function clearBanner() { var b = qs('#banner'); b.hidden = true; U.clear(b); }

  function applyState(raw) {
    var s = {};
    s.season = raw.season || '2026/27';
    s.current_gw = toNum(pick(raw, ['current_gw', 'gw', 'next_gw'], 1), 1);
    s.deadline = pick(raw, ['deadline', 'deadline_time', 'next_deadline'], null);
    s.n_players = toNum(pick(raw, ['n_players', 'players'], null));
    s.n_teams = toNum(pick(raw, ['n_teams', 'teams_count'], null));
    s.n_fixtures = toNum(pick(raw, ['n_fixtures', 'fixtures'], null));
    s.finished_gws = raw.finished_gws || [];
    s.entry_id = toNum(raw.entry_id, null);
    s.fitted = raw.fitted === undefined ? null : !!raw.fitted;
    s.first_gw = toNum(raw.first_gw, null);
    s.last_gw = toNum(raw.last_gw, null);
    store.state = s;
    store.gw = s.current_gw;
    var teams = raw.teams || [];
    if (Array.isArray(teams) && teams.length) {
      store.teams = {};
      teams.forEach(function (t) {
        store.teams[t.id] = {
          id: t.id, name: t.name || t.short_name, short_name: t.short_name || t.name, code: t.code
        };
      });
    }
    if (s.first_gw !== null && s.last_gw !== null) {
      store.gws = [];
      for (var g = s.first_gw; g <= s.last_gw; g++) store.gws.push(g);
    }
    qs('#brand-season').textContent = s.season;
  }

  function loadLive() {
    return request('/state').then(function (stateRaw) {
      applyState(stateRaw);
      var h = store.horizon;
      return request('/players?gw=' + store.gw + '&horizon=' + h).then(function (body) {
        var list = Array.isArray(body) ? body : (body.players || body.items || []);
        if (!list.length) throw new Error('the backend returned no players');
        if (body && !Array.isArray(body) && body.teams && !Object.keys(store.teams).length) {
          body.teams.forEach(function (t) { store.teams[t.id] = t; });
        }
        indexPlayers(list.map(normPlayer));
        var needProjections = !store.players.some(function (p) { return p.gws.length; });
        var first = store.gw;
        var last = store.gw + Math.max(h, 5) - 1;
        var jobs = [];
        if (needProjections) {
          jobs.push(
            request('/projections?first=' + first + '&last=' + last)
              .then(function (ps) { mergeProjectionSet(ps.projections ? ps : (ps.projection_set || ps)); })
              .catch(function (e) {
                var d = describeError(e);
                banner('<b>Per-gameweek projections unavailable.</b> ' + U.esc(d.where) +
                       ' failed: ' + U.esc(d.what) +
                       '. Horizon totals fall back to the single <code>xp</code> field and ' +
                       'the component audit is unavailable.', 'err');
              })
          );
        }
        jobs.push(
          request('/fixtures').then(function (fb) {
            indexFixtures(Array.isArray(fb) ? fb : (fb.fixtures || []));
          }).catch(function () { /* ticker falls back to per-projection lambdas or FDR */ })
        );
        return Promise.all(jobs);
      });
    });
  }

  function enterSampleMode(err) {
    var sample = window.GAFFER_SAMPLE;
    var d = describeError(err);
    if (!sample) {
      store.mode = 'error';
      setSource('', 'offline');
      banner('<b>Backend unreachable and no sample data is embedded.</b> ' +
             U.esc(d.where) + ' failed: ' + U.esc(d.what) + '.', 'err',
             [{ label: 'Retry', fn: boot }]);
      return;
    }
    store.mode = 'sample';
    store.sampleReason = d.where + ' failed: ' + d.what;
    document.body.classList.add('is-sample');
    applyState(sample.state);
    indexPlayers((sample.players || []).map(normPlayer));
    indexFixtures(sample.fixtures || []);
    store.plan = sample.plan || null;
    store.captainByGw[sample.captain ? sample.captain.gw : store.gw] = sample.captain || null;
    store.chips = sample.chips || null;
    store.fullComponentGws = (sample.meta && sample.meta.full_component_gws) || null;
    store.projectionMeta = {
      generated_at: sample.meta ? sample.meta.generated_at : null,
      model_version: null, first_gw: store.gws[0], last_gw: store.gws[store.gws.length - 1],
      merged: store.players.length
    };
    setSource('sample', 'SAMPLE DATA');
    var meta = sample.meta || {};
    banner(
      '<b>SAMPLE DATA — not a live projection.</b> ' + U.esc(store.sampleReason) +
      '. Showing an embedded snapshot of ' + (meta.n_players_sample || store.players.length) +
      ' of ' + (meta.n_players_total || '?') + ' players generated ' +
      U.esc(U.localDate(meta.generated_at, true) || 'earlier') +
      '. ' + U.esc(meta.note || '') +
      ' Start the backend with <code>python -m gaffer.cli serve</code> and reload for live numbers.',
      '', [{ label: 'Retry backend', fn: boot }]
    );
  }

  /* Offline replay. The dashboard opened, it has numbers on it, and every one
     of them came off the phone rather than off the model. Say so loudly and in
     the two places a user looks — the source pill and a banner — because the
     failure mode this guards against is a transfer made at 17:29 against a
     projection that predates an injury. */
  function announceStale() {
    var when = U.localDate(store.stale.at, true) || 'earlier';
    document.body.classList.add('is-stale');
    setSource('stale', 'CACHED · ' + when);
    banner(
      '<b>OFFLINE — cached numbers, not a live projection.</b> The backend could ' +
      'not be reached, so ' + store.stale.calls + ' request' +
      (store.stale.calls === 1 ? '' : 's') + ' ' +
      (store.stale.calls === 1 ? 'was' : 'were') + ' answered from this phone\'s ' +
      'cache, saved ' + U.esc(when) + '. Prices, injuries and team news since ' +
      'then are missing. <b>Do not make a transfer or set a captain off this ' +
      'screen.</b> Reconnect to the Mac and reload before the deadline.',
      'err', [{ label: 'Retry', fn: boot }]
    );
  }

  /* "3 hours ago" from an ISO stamp. Deliberately coarse: the useful question
     before a deadline is "was this built before or after the team news", and a
     rounded age answers it without implying false precision. */
  function snapshotAge(iso) {
    var then = Date.parse(iso || '');
    if (!then) return 'age unknown';
    var mins = Math.max(0, Math.round((Date.now() - then) / 60000));
    if (mins < 2) return 'just now';
    if (mins < 60) return mins + ' min ago';
    var hours = Math.round(mins / 60);
    if (hours < 24) return hours + (hours === 1 ? ' hour ago' : ' hours ago');
    var days = Math.round(hours / 24);
    return days + (days === 1 ? ' day ago' : ' days ago');
  }

  /* Say it once, plainly: these numbers were true when the snapshot was built.
     Anything that moved since — an injury, a price change, a press conference —
     is not in here. */
  function announceSnapshot() {
    if (!STATIC) return;
    var age = snapshotAge(STATIC.generated_at);
    var stale = Date.parse(STATIC.generated_at || '')
      && (Date.now() - Date.parse(STATIC.generated_at)) > 12 * 3600 * 1000;
    banner(
      '<b>Published snapshot, built ' + age + '.</b> It rebuilds on a schedule, ' +
      'so team news since then is not reflected. Re-optimising with your own ' +
      'constraints needs gaffer running locally.',
      stale ? 'err' : 'info');
  }

  function boot() {
    store.mode = 'loading';
    store.stale = null;
    document.body.classList.remove('is-sample');
    document.body.classList.remove('is-stale');
    setSource('busy', 'connecting…');
    clearBanner();
    renderView(store.view, true);
    // Decide which world we are in before the first request goes out.
    return detectStatic().then(bootAfterProbe);
  }

  function bootAfterProbe() {
    if (STATIC) {
      document.body.classList.add('is-static');
      // Adopt the snapshot's horizon. The frozen files were computed for one
      // horizon, and the store's compiled-in default need not match it — when
      // they diverge every request is for a window the snapshot does not hold,
      // and the squad view fails with a 501 that looks like a bug rather than
      // a mismatch. The manifest is the authority here.
      if (STATIC.horizon) store.horizon = Number(STATIC.horizon);
      if (STATIC.gw) store.gw = Number(STATIC.gw);
    }
    return loadLive().then(function () {
      store.mode = 'live';
      if (STATIC) {
        // Calling a snapshot "LIVE" would be a lie, and the age is the one
        // thing that decides whether you can trust it before a deadline.
        setSource('live', 'SNAPSHOT · ' + snapshotAge(STATIC.generated_at));
      } else {
        setSource('live', 'LIVE · ' + store.players.length + ' players');
      }
      clearBanner();
      if (STATIC) announceSnapshot();
      if (!STATIC && store.state && store.state.fitted === false) {
        banner('<b>The backend is up but reports the model is not fitted.</b> ' +
               'Run <code>python -m gaffer.cli project --gws ' + store.gw + '-' +
               (store.gw + store.horizon - 1) + '</code>, then reload.', 'err',
               [{ label: 'Reload', fn: boot }]);
      }
      // last, so it owns the banner: nothing outranks "these numbers are old"
      if (store.stale) announceStale();
      renderView(store.view, true);
      startClocks();
      // the authoritative GW19 chip deadline lives behind /api/chips; pull it in
      // the background so the header countdown stops saying "estimated"
      loadChips().then(tickClocks, function () {});
    }, function (err) {
      enterSampleMode(err);
      renderView(store.view, true);
      startClocks();
    });
  }

  // ----------------------------------------------- lazy per-view loaders --

  function loadPlan(force) {
    if (store.mode === 'sample') return Promise.resolve(store.plan);
    if (store.plan && !force) return Promise.resolve(store.plan);
    if (store.planLoading) return store.planLoading;
    store.planError = null;
    var body = { horizon: store.horizon };
    if (store.state && store.state.entry_id) body.entry_id = store.state.entry_id;
    store.planLoading = request('/optimize', { method: 'POST', body: body })
      .then(function (res) {
        store.plan = res && res.decisions ? res : (res.plan || res);
        store.planLoading = null;
        return store.plan;
      }, function (err) {
        store.planError = err;
        store.planLoading = null;
        throw err;
      });
    return store.planLoading;
  }

  G.loadPlan = loadPlan;

  function loadCaptain(gw, force) {
    if (store.mode === 'sample') return Promise.resolve(store.captainByGw[gw] || null);
    if (store.captainByGw[gw] && !force) return Promise.resolve(store.captainByGw[gw]);
    store.captainError = null;
    return request('/captain?gw=' + gw).then(function (res) {
      store.captainByGw[gw] = res;
      return res;
    }, function (err) { store.captainError = err; throw err; });
  }

  function loadChips(force) {
    if (store.mode === 'sample') return Promise.resolve(store.chips);
    if (store.chips && !force) return Promise.resolve(store.chips);
    store.chipsError = null;
    return request('/chips').then(function (res) {
      store.chips = res;
      return res;
    }, function (err) { store.chipsError = err; throw err; });
  }

  G.load = { plan: loadPlan, captain: loadCaptain, chips: loadChips, boot: boot };

  // ------------------------------------------------------------- drawer ---

  var drawerState = { playerId: null, gw: null, detail: null, detailFor: null };

  function openPlayer(playerId, gw) {
    var p = store.byId[playerId];
    if (!p) return;
    drawerState.playerId = playerId;
    drawerState.gw = gw || store.gw;
    if (drawerState.detailFor !== playerId) { drawerState.detail = null; drawerState.detailFor = null; }
    qs('#drawer').hidden = false;
    qs('#drawer').setAttribute('aria-hidden', 'false');
    qs('#scrim').hidden = false;
    document.body.style.overflow = 'hidden';
    renderDrawer();
    if (store.mode === 'live' && drawerState.detailFor !== playerId) {
      request('/player/' + playerId + '?gw=' + drawerState.gw).then(function (res) {
        drawerState.detail = res;
        drawerState.detailFor = playerId;
        if (drawerState.playerId === playerId) renderDrawer();
      }, function () { /* the local breakdown is already shown; explanation is a bonus */ });
    }
  }

  function closeDrawer() {
    qs('#drawer').hidden = true;
    qs('#drawer').setAttribute('aria-hidden', 'true');
    qs('#scrim').hidden = true;
    document.body.style.overflow = '';
    drawerState.playerId = null;
  }

  function renderDrawer() {
    if (drawerState.playerId === null) return;
    G.views.drawer(qs('#drawer-body'), store.byId[drawerState.playerId], drawerState);
  }

  G.drawer = {
    open: openPlayer, close: closeDrawer, render: renderDrawer, state: drawerState,
    setGw: function (gw) { drawerState.gw = gw; renderDrawer(); }
  };

  // -------------------------------------------------------------- router --

  var VIEWS = ['squad', 'players', 'planner', 'captain', 'chips', 'myteam'];

  function renderView(name, force) {
    if (VIEWS.indexOf(name) < 0) name = 'squad';
    store.view = name;
    VIEWS.forEach(function (v) {
      var sec = qs('#view-' + v);
      var tab = qs('#tab-' + v);
      var on = v === name;
      sec.hidden = !on;
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      tab.tabIndex = on ? 0 : -1;
    });
    var host = qs('#view-' + name);
    if (store.mode === 'loading') {
      U.clear(host);
      host.appendChild(U.loadingBlock(6));
      return;
    }
    if (force || host.dataset.rendered !== '1' || host.dataset.stamp !== stamp()) {
      G.views[name](host);
      host.dataset.rendered = '1';
      host.dataset.stamp = stamp();
    }
  }

  function stamp() {
    return [store.mode, store.gw, store.horizon, store.players.length].join('|');
  }

  G.render = renderView;
  G.invalidate = function (name) {
    var host = qs('#view-' + name);
    if (host) host.dataset.rendered = '0';
    if (store.view === name) renderView(name, true);
  };

  function go(name) {
    // render synchronously as well as setting the hash: hashchange fires as a
    // later task, and callers (the "/" shortcut) need the view in the DOM now
    if (location.hash !== '#' + name) location.hash = name;
    renderView(name);
  }
  G.go = go;

  // -------------------------------------------------------------- clocks --

  var clockTimer = null;

  function chipExpiryIso() {
    if (store.chips && store.chips.first_half_deadline) {
      var fromSample = store.mode === 'sample';
      return {
        iso: store.chips.first_half_deadline,
        estimated: fromSample,
        source: fromSample ? 'sample' : 'backend'
      };
    }
    return { iso: CHIP_EXPIRY_FALLBACK, estimated: true, source: 'spec' };
  }
  G.chipExpiryIso = chipExpiryIso;

  function chipSeverity(days) {
    var gw = store.state ? store.state.current_gw : 1;
    if (days <= 10 || gw >= 17) return 'crit';
    if (days <= 35 || gw >= 14) return 'warn';
    return '';
  }
  G.chipSeverity = chipSeverity;

  function tickClocks() {
    var s = store.state;
    if (!s) return;
    var dl = U.timeUntil(s.deadline);
    qs('#deadline-value').textContent = dl
      ? (dl.past ? 'GW' + s.current_gw + ' locked' : U.shortCountdown(dl))
      : 'unknown';
    qs('#deadline-sub').textContent = s.deadline
      ? 'GW' + s.current_gw + ' · ' + U.localDate(s.deadline, true)
      : 'GW' + s.current_gw;
    var dlBox = qs('#clock-deadline');
    dlBox.className = 'clock' + (dl && !dl.past && dl.days < 1 ? ' warn' : '');

    var exp = chipExpiryIso();
    var cu = U.timeUntil(exp.iso);
    qs('#chipclock-value').textContent = cu ? (cu.past ? 'EXPIRED' : U.shortCountdown(cu)) : '—';
    qs('#chipclock-sub').textContent = 'GW19 · ' + U.localDate(exp.iso, false) +
      (exp.source === 'spec' ? ' (time est.)' : (exp.source === 'sample' ? ' (sample)' : ''));
    var sev = cu ? chipSeverity(cu.days) : '';
    qs('#clock-chips').className = 'clock' + (sev ? ' ' + sev : '');
  }

  function startClocks() {
    tickClocks();
    if (clockTimer) clearInterval(clockTimer);
    clockTimer = setInterval(function () {
      tickClocks();
      if (store.view === 'chips') {
        var cd = qs('#chip-countdown');
        if (cd && G.views.updateCountdown) G.views.updateCountdown(cd);
      }
    }, 1000);
  }

  // ---------------------------------------------------------------- wire --

  function wire() {
    qs('#tabs').addEventListener('click', function (e) {
      var t = e.target.closest('.tab');
      if (t) go(t.dataset.view);
    });

    window.addEventListener('hashchange', function () {
      renderView((location.hash || '#squad').slice(1));
    });

    qs('#scrim').addEventListener('click', closeDrawer);

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { closeDrawer(); return; }
      var tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var idx = ['1', '2', '3', '4', '5'].indexOf(e.key);
      if (idx >= 0) { go(VIEWS[idx]); return; }
      if (e.key === '/') {
        e.preventDefault();
        go('players');
        var box = qs('#f-search');
        if (box) box.focus();
      }
    });

    qs('#btn-reload').addEventListener('click', function () { boot(); });

    qs('#source-pill').addEventListener('click', function () {
      if (store.mode === 'sample') boot();
      else if (store.projectionMeta) {
        U.toast('projections ' +
          (store.projectionMeta.generated_at
            ? 'generated ' + U.localDate(store.projectionMeta.generated_at, true)
            : 'loaded') +
          (store.projectionMeta.model_version ? ' · model ' + store.projectionMeta.model_version : ''));
      }
    });

    qs('#btn-refresh').addEventListener('click', function () {
      var btn = this;
      if (store.mode === 'sample') { boot(); return; }
      btn.disabled = true;
      btn.textContent = 'Refreshing…';
      request('/refresh', { method: 'POST', body: {} }).then(function () {
        U.toast('backend refreshed — reloading');
        store.plan = null; store.chips = null; store.captainByGw = {};
        return boot();
      }, function (err) {
        var d = describeError(err);
        banner('<b>Refresh failed.</b> POST ' + API + '/refresh: ' + U.esc(d.what), 'err');
      }).then(function () {
        btn.disabled = false;
        btn.textContent = 'Refresh data';
      });
    });
  }

  // ------------------------------------------------- service worker / PWA --

  /* Registered from "/sw.js" so the worker's scope is the whole origin, which
     is what lets the app open from the home screen with the Mac asleep.
     The server sends it no-store, so every launch revalidates it. */
  function registerWorker() {
    if (!('serviceWorker' in navigator)) return;
    if (location.protocol !== 'http:' && location.protocol !== 'https:') return;

    // True only if this page was already under a worker's control. A first
    // install also fires controllerchange (because the worker calls
    // clients.claim), and reloading on that would loop the very first visit.
    var wasControlled = !!navigator.serviceWorker.controller;
    var reloading = false;

    navigator.serviceWorker.addEventListener('controllerchange', function () {
      if (!wasControlled || reloading) return;
      // A new build took over mid-session. The document in front of the user
      // was assembled from the old one, so reload once rather than run a mix.
      reloading = true;
      location.reload();
    });

    // Relative, not '/sw.js': the exported site is served from a project
    // subdirectory on Pages (/gaffer/), where an absolute path is a 404 and the
    // app silently loses offline support and installability.
    navigator.serviceWorker.register('sw.js', { scope: './', updateViaCache: 'none' })
      .catch(function (err) {
        // Never fatal: without a worker the dashboard is exactly what it was
        // before, an online-only page.
        if (window.console) console.warn('service worker registration failed:', err);
      });
  }

  document.addEventListener('DOMContentLoaded', function () {
    wire();
    store.view = (location.hash || '#squad').slice(1);
    if (VIEWS.indexOf(store.view) < 0) store.view = 'squad';
    boot();
    // after boot: the first paint should not queue behind a precache
    registerWorker();
  });
})();
