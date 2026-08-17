/* gaffer dashboard — the five views and the player audit drawer.
 *
 * Rendering rules that matter here:
 *  - the players table is built once as one HTML string and then mutated in
 *    place: filtering toggles a class on rows, sorting re-appends the existing
 *    nodes, changing the horizon rewrites two cells per row. Nothing rebuilds.
 *  - every projected number is clickable and opens the component audit.
 *  - SVG is written as an HTML string (the parser namespaces it correctly), so
 *    no namespace URI ever appears in this codebase.
 */
(function () {
  'use strict';

  var G = (window.G = window.G || {});
  var U = G.ui;
  var el = U.el, qs = U.qs, isNum = U.isNum, esc = U.esc;

  function S() { return G.store; }
  function D() { return G.data; }

  var TICKER_N = 5;

  // ------------------------------------------------------------- shared ---

  function head(title, notes) {
    var h = el('div', { class: 'view-head' }, [el('h1', { class: 'view-title', text: title })]);
    if (S().mode === 'sample') h.appendChild(el('span', { class: 'tag sample', text: 'sample data' }));
    (notes || []).forEach(function (n) {
      if (!n) return;
      h.appendChild(typeof n === 'string' ? el('span', { class: 'view-note', text: n }) : n);
    });
    return h;
  }

  function playerLink(p, gw, text, cls) {
    return el('button', {
      class: cls || 'linkish',
      type: 'button',
      style: 'background:none;border:0;padding:0;cursor:pointer;color:inherit;font:inherit',
      onclick: function (e) { e.stopPropagation(); G.drawer.open(p.id, gw); }
    }, text);
  }

  function fixtureTitle(fx, gw) {
    var e = D().easeOf(fx);
    var opp = D().teamName(fx.opponent_id);
    var bits = ['GW' + gw + ' ' + (fx.is_home ? 'vs ' : 'at ') + opp];
    if (e.source === 'model') {
      bits.push('our model: ' + U.num(e.tl, 2) + ' xG for, ' + U.num(e.ol, 2) + ' against');
    } else if (e.source === 'fdr') {
      bits.push('no model lambda for this fixture — official FDR ' + U.num(fx.difficulty, 0) + ' only');
    } else {
      bits.push('no difficulty data');
    }
    if (isNum(fx.xp_total)) bits.push('xP ' + U.num(fx.xp_total, 2));
    return bits.join(' · ');
  }

  function tickerHtml(p, gw, n) {
    var out = ['<div class="ticker">'];
    D().nextFixtures(p, gw, n).forEach(function (entry) {
      if (!entry.fixtures.length) {
        out.push('<span class="tk blank" title="GW' + entry.gw +
                 (entry.known ? ' — blank gameweek, no fixture' : ' — outside the projected horizon') +
                 '">' + (entry.known ? '·' : '?') + '</span>');
        return;
      }
      entry.fixtures.forEach(function (fx) {
        var e = D().easeOf(fx);
        var t = e.t === null ? 0.5 : e.t;
        var rank = U.easeRank(e.t);
        var opp = D().teamShort(fx.opponent_id);
        var label = fx.is_home ? String(opp).toUpperCase() : String(opp).toLowerCase();
        var cls = 'tk' + (e.source !== 'model' ? ' fdr' : '') +
                  (entry.fixtures.length > 1 ? ' dbl' : '');
        out.push('<span class="' + cls + '" style="--c:' + U.ramp(t) + ';--fg:' + U.inkOn(t) +
                 '" title="' + esc(fixtureTitle(fx, entry.gw)) + '">' + esc(label) +
                 '<i>' + (rank === null ? '?' : rank) + '</i></span>');
      });
    });
    out.push('</div>');
    return out.join('');
  }

  function tickerNode(p, gw, n) {
    var box = el('div');
    box.innerHTML = tickerHtml(p, gw, n);
    return box.firstChild;
  }

  function statusFlag(p) {
    if (p.status === 'a' || !p.status) return null;
    var out = ['i', 's', 'u', 'n'].indexOf(p.status) >= 0;
    return { out: out, text: p.status.toUpperCase(), news: p.news };
  }

  function missingPlayerCard(id) {
    return el('div', { class: 'pcard' }, [
      el('div', { class: 'pcard-team', text: 'id ' + id }),
      el('div', { class: 'pcard-name', text: 'not in payload' }),
      el('div', { class: 'pcard-line' }, [el('span', { text: '—' }), el('span', { class: 'pcard-xp', text: '—' })])
    ]);
  }

  function gwPicker(current, onPick) {
    var box = el('div', { class: 'gwchips' });
    S().gws.forEach(function (g) {
      box.appendChild(el('button', {
        class: 'gwchip', type: 'button', 'aria-pressed': g === current ? 'true' : 'false',
        onclick: function () { onPick(g); }
      }, 'GW' + g));
    });
    return box;
  }

  // ================================================================ SQUAD ==

  function squadView(host) {
    U.clear(host);
    host.appendChild(head('Squad', [
      'recommended XI for GW' + S().gw + ' — click any player for the full component audit'
    ]));
    var body = el('div');
    host.appendChild(body);

    function fail(err) {
      var d = G.describeError(err);
      U.clear(body);
      body.appendChild(U.errorBlock(
        'Could not build a squad.',
        esc(d.where) + ' returned: <b>' + esc(d.what) + '</b>' +
        (err && err.status === 404
          ? '<br>The optimizer endpoint is not implemented on this backend yet.'
          : '') +
        (err && err.network
          ? '<br>Start the API with <code>python -m gaffer.cli serve</code>.'
          : ''),
        [{ label: 'Retry', fn: function () { squadView(host); } }]
      ));
    }

    body.appendChild(U.loadingBlock(4));
    G.load.plan().then(function (plan) {
      if (!plan || !plan.decisions || !plan.decisions.length) {
        fail(new Error('the plan came back with no gameweek decisions'));
        return;
      }
      U.clear(body);
      renderSquad(body, plan);
    }, fail);
  }

  function renderSquad(body, plan) {
    var st = S();
    var gw = st.gw;
    var dec = null;
    plan.decisions.forEach(function (d) { if (d.gw === gw && !dec) dec = d; });
    if (!dec) dec = plan.decisions[0];

    var lineup = (dec.lineup || []).map(function (id) { return st.byId[id] || { id: id, missing: true }; });
    var bench = (dec.bench || []).map(function (id) { return st.byId[id] || { id: id, missing: true }; });

    function xpOf(p) { return p.missing ? 0 : D().gwXp(p, dec.gw); }

    var xiXp = lineup.reduce(function (a, p) { return a + xpOf(p); }, 0);
    var benchXp = bench.reduce(function (a, p) { return a + xpOf(p); }, 0);
    var capP = st.byId[dec.captain];
    var capXp = capP ? D().gwXp(capP, dec.gw) : 0;
    var chipMult = dec.chip === '3xc' ? 2 : 1;
    var total = isNum(dec.expected_points) ? dec.expected_points : (xiXp + capXp * chipMult);
    var net = isNum(dec.expected_points_net) ? dec.expected_points_net : total - 4 * (dec.hits || 0);

    var counts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    lineup.forEach(function (p) { if (!p.missing) counts[p.position]++; });
    var formation = counts[2] + '-' + counts[3] + '-' + counts[4];

    var value = 0;
    (dec.squad || []).forEach(function (id) {
      var p = st.byId[id];
      if (p && isNum(p.now_cost)) value += p.now_cost;
    });

    var stats = el('div', { class: 'stats' }, [
      U.statTile('Projected GW' + dec.gw, U.num(net, 1),
        (dec.hits ? '<span style="color:var(--neg)">' + (-4 * dec.hits) + ' hits</span> · ' : '') +
        'XI ' + U.num(xiXp, 1) + ' + capt ' + U.num(capXp * chipMult, 1), 'hero'),
      U.statTile('Captain', capP ? esc(capP.web_name) : '—',
        capP ? U.num(capXp, 2) + ' xP · ×' + (dec.chip === '3xc' ? 3 : 2) : ''),
      U.statTile('Vice', st.byId[dec.vice_captain] ? esc(st.byId[dec.vice_captain].web_name) : '—',
        st.byId[dec.vice_captain] ? U.num(D().gwXp(st.byId[dec.vice_captain], dec.gw), 2) + ' xP' : ''),
      U.statTile('Formation', formation, 'bench ' + U.num(benchXp, 1) + ' xP'),
      U.statTile('Squad value', value ? U.money(value) : '—',
        isNum(dec.bank_after) ? 'bank ' + U.money(dec.bank_after) : ''),
      U.statTile('Chip', dec.chip ? esc(String(dec.chip).toUpperCase()) : 'none',
        'FT after: ' + (isNum(dec.free_transfers_after) ? dec.free_transfers_after : '—'))
    ]);
    body.appendChild(stats);

    var grid = el('div', { class: 'squad-grid' });
    var left = el('div');
    var pitch = el('div', { class: 'pitch' });
    [1, 2, 3, 4].forEach(function (posId) {
      var row = el('div', { class: 'prow' });
      lineup.filter(function (p) { return p.missing ? false : p.position === posId; })
        .sort(function (a, b) { return xpOf(b) - xpOf(a); })
        .forEach(function (p) { row.appendChild(playerCard(p, dec, false)); });
      if (row.childNodes.length) pitch.appendChild(row);
    });
    var missing = lineup.filter(function (p) { return p.missing; });
    if (missing.length) {
      var mrow = el('div', { class: 'prow' });
      missing.forEach(function (p) { mrow.appendChild(missingPlayerCard(p.id)); });
      pitch.appendChild(mrow);
    }
    pitch.appendChild(el('div', { class: 'mixlegend' }, [
      el('span', {}, [el('i', { class: 'mix-att' }), document.createTextNode('attacking')]),
      el('span', {}, [el('i', { class: 'mix-def' }), document.createTextNode('defensive')]),
      el('span', {}, [el('i', { class: 'mix-oth' }), document.createTextNode('appearance, bonus, cards')]),
      el('span', { text: 'strip under each card = where the points come from' })
    ]));
    left.appendChild(pitch);

    var benchBox = el('div', { class: 'bench' });
    benchBox.appendChild(U.panel(
      'Bench — ' + U.num(benchXp, 1) + ' xP, in autosub order',
      null,
      [(function () {
        var row = el('div', { class: 'prow' });
        bench.forEach(function (p, i) {
          var card = p.missing ? missingPlayerCard(p.id) : playerCard(p, dec, true);
          card.appendChild(el('span', { class: 'bench-idx', text: p.position === 1 ? 'GK' : String(i) }));
          row.appendChild(card);
        });
        return row;
      })()],
      'panel-body'
    ));
    left.appendChild(benchBox);
    grid.appendChild(left);

    grid.appendChild(gwFixturePanel(dec.gw));
    body.appendChild(grid);

    if (dec.notes && dec.notes.length) {
      body.appendChild(el('div', { style: 'margin-top:12px' }, [
        U.panel('Why this team', null, [
          el('ul', { class: 'notes' }, dec.notes.map(function (n) { return el('li', { text: n }); }))
        ])
      ]));
    }
  }

  function playerCard(p, dec, isBench) {
    var gw = dec.gw;
    var xp = D().gwXp(p, gw);
    var isCap = p.id === dec.captain;
    var isVice = p.id === dec.vice_captain;
    var entry = p.gwMap[gw];
    var fx = entry && entry.fixtures.length ? entry.fixtures[0] : null;

    var card = el('div', {
      class: 'pcard' + (isCap ? ' is-cap' : ''),
      tabindex: '0',
      role: 'button',
      title: p.web_name + ' — click for the component breakdown',
      onclick: function () { G.drawer.open(p.id, gw); },
      onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); G.drawer.open(p.id, gw); } }
    }, [
      el('div', { class: 'pcard-team', text: (D().teamShort(p.team_id) || '') + ' · ' + p.pos }),
      el('div', { class: 'pcard-name', text: p.web_name }),
      el('div', { class: 'pcard-line' }, [
        el('span', { text: U.money(p.now_cost) }),
        el('span', { class: 'pcard-xp', text: U.num(xp, 1) })
      ])
    ]);

    if (isCap) card.appendChild(el('span', { class: 'pcard-badge', text: dec.chip === '3xc' ? '3C' : 'C' }));
    else if (isVice) card.appendChild(el('span', { class: 'pcard-badge vc', text: 'V' }));

    var flag = statusFlag(p);
    if (flag) {
      card.appendChild(el('span', {
        class: 'pcard-flag', text: flag.text,
        title: (flag.news || 'status ' + p.status) +
               (isNum(p.chance) ? ' — ' + p.chance + '% chance of playing' : '')
      }));
    }

    if (fx) {
      var e = D().easeOf(fx);
      var t = e.t === null ? 0.5 : e.t;
      card.appendChild(el('div', {
        class: 'pcard-fx',
        html: '<span class="tk" style="--c:' + U.ramp(t) + ';--fg:' + U.inkOn(t) + '">' +
              esc(fx.is_home ? String(D().teamShort(fx.opponent_id)).toUpperCase()
                             : String(D().teamShort(fx.opponent_id)).toLowerCase()) +
              '<i>' + (U.easeRank(e.t) === null ? '?' : U.easeRank(e.t)) + '</i></span>',
        title: fixtureTitle(fx, gw)
      }));
      if (fx.has_components) card.appendChild(U.mixBar(fx));
    } else {
      card.appendChild(el('div', { class: 'pcard-fx', text: 'blank GW' }));
    }
    if (isBench) card.style.opacity = '.92';
    return card;
  }

  function gwFixturePanel(gw) {
    var st = S();
    var rows = [];
    Object.keys(st.fixtureLambdas).forEach(function (id) {
      var f = st.fixtureLambdas[id];
      if (f.gw === gw) rows.push(f);
    });
    rows.sort(function (a, b) {
      return String(a.kickoff_time || '').localeCompare(String(b.kickoff_time || ''));
    });

    var body;
    if (!rows.length) {
      body = [el('div', { class: 'empty', text: 'No fixture lambdas loaded for GW' + gw + '.' })];
    } else {
      var tbl = el('table', { class: 'grid' });
      var html = ['<tbody>'];
      rows.forEach(function (f) {
        var lh = f.lambda_h, la = f.lambda_a;
        var th = isNum(lh) && isNum(la) ? U.easeIndexFromLambdas(lh, la) : null;
        var ta = th === null ? null : 1 - th;
        function chip(teamId, t, home) {
          var name = D().teamShort(teamId);
          var c = t === null ? '#20242c' : U.ramp(t);
          var fg = t === null ? '#98a2b1' : U.inkOn(t);
          return '<span class="tk" style="--c:' + c + ';--fg:' + fg + '">' +
                 esc(home ? String(name).toUpperCase() : String(name).toLowerCase()) + '</span>';
        }
        html.push('<tr><td>' + chip(f.team_h, th, true) + '</td>' +
          '<td class="r t-key">' + U.num(lh, 2) + '</td>' +
          '<td class="c t-dim">–</td>' +
          '<td class="t-key">' + U.num(la, 2) + '</td>' +
          '<td class="r">' + chip(f.team_a, ta, false) + '</td></tr>');
      });
      html.push('</tbody>');
      tbl.innerHTML = html.join('');
      body = [tbl];
    }
    return U.panel('GW' + gw + ' — our expected goals', null, body, 'panel-body');
  }

  // ============================================================== PLAYERS ==

  var COLS = [
    { key: 'name', label: 'Player', cls: '', sort: function (p) { return p.web_name.toLowerCase(); }, dir: 1 },
    { key: 'team', label: 'Team', cls: 't-team', sort: function (p) { return D().teamShort(p.team_id); }, dir: 1 },
    { key: 'pos', label: 'Pos', cls: 'c t-dim', sort: function (p) { return p.position || 9; }, dir: 1 },
    { key: 'price', label: '£', cls: 'r', sort: function (p) { return isNum(p.now_cost) ? p.now_cost : -1; }, dir: -1 },
    { key: 'xp', label: 'xP', cls: 'r t-key', sort: function (p) { return xpH(p); }, dir: -1 },
    { key: 'value', label: 'xP/£m', cls: 'r', sort: function (p) { return valueOf(p); }, dir: -1 },
    { key: 'own', label: 'Own%', cls: 'r t-dim', sort: function (p) { return p.selected_by_percent; }, dir: -1 },
    { key: 'p60', label: 'p60', cls: 'r', sort: function (p) { return isNum(p.p_60) ? p.p_60 : -1; }, dir: -1 },
    { key: 'xg90', label: 'xG90', cls: 'r', sort: function (p) { return isNum(p.xg90) ? p.xg90 : -1; }, dir: -1 },
    { key: 'xa90', label: 'xA90', cls: 'r', sort: function (p) { return isNum(p.xa90) ? p.xa90 : -1; }, dir: -1 },
    { key: 'defcon', label: 'DEFCON', cls: 'r', sort: function (p) { return isNum(p.p_defcon) ? p.p_defcon : -1; }, dir: -1 },
    { key: 'form', label: 'Form', cls: 'r t-dim', sort: function (p) { return p.form; }, dir: -1 },
    { key: 'fx', label: 'Next ' + TICKER_N, cls: 'no-sort', sort: null, dir: 1 }
  ];

  var tbl = {
    rows: [], sortKey: 'xp', sortDir: -1, tbody: null, countNode: null,
    filters: { q: '', pos: 0, team: 0, maxCost: null, minP60: 0, hideFlagged: false }
  };

  function xpH(p) { return D().horizonXp(p, S().gw, S().horizon); }
  function valueOf(p) {
    if (!isNum(p.now_cost) || p.now_cost <= 0) return 0;
    return xpH(p) / (p.now_cost / 10);
  }

  function rowHtml(p) {
    var flag = statusFlag(p);
    var cells = [];
    cells.push('<td class="t-name">' +
      (flag ? '<span class="flagdot' + (flag.out ? ' out' : '') + '" title="' +
              esc(flag.news || ('status ' + p.status)) + '"></span>' : '') +
      esc(p.web_name) + '</td>');
    cells.push('<td class="t-team">' + esc(D().teamShort(p.team_id)) + '</td>');
    cells.push('<td class="c t-dim">' + esc(p.pos) + '</td>');
    cells.push('<td class="r">' + U.money(p.now_cost) + '</td>');
    cells.push('<td class="r t-key">' + U.num(xpH(p), 1) + '</td>');
    cells.push('<td class="r">' + U.num(valueOf(p), 2) + '</td>');
    cells.push('<td class="r t-dim">' + U.pctv(p.selected_by_percent, 1) + '</td>');
    cells.push('<td class="r">' + U.pct(p.p_60, 0) + '</td>');
    cells.push('<td class="r">' + U.num(p.xg90, 2) + '</td>');
    cells.push('<td class="r">' + U.num(p.xa90, 2) + '</td>');
    cells.push('<td class="r">' + U.pct(p.p_defcon, 0) + '</td>');
    cells.push('<td class="r t-dim">' + U.num(p.form, 1) + '</td>');
    cells.push('<td>' + tickerHtml(p, S().gw, TICKER_N) + '</td>');
    return '<tr data-id="' + p.id + '">' + cells.join('') + '</tr>';
  }

  function playersView(host) {
    U.clear(host);
    var st = S();

    host.appendChild(head('Players', [
      st.players.length + ' players · xP summed over the next ' + st.horizon +
      ' gameweek' + (st.horizon === 1 ? '' : 's') + ' from GW' + st.gw
    ]));

    if (!st.players.length) {
      host.appendChild(U.errorBlock('No player data loaded.',
        'The backend returned an empty player list and no sample data is available.'));
      return;
    }

    host.appendChild(filterBar());

    var wrap = el('div', { class: 'tablewrap' });
    var table = el('table', { class: 'grid' });
    var thead = el('thead');
    var htr = el('tr');
    COLS.forEach(function (c) {
      var th = el('th', {
        class: c.cls + (c.sort ? '' : ' no-sort'),
        scope: 'col',
        dataset: { key: c.key },
        title: c.sort ? 'sort by ' + c.label : ''
      }, c.label);
      if (c.sort) {
        th.addEventListener('click', function () { sortBy(c.key); });
      }
      htr.appendChild(th);
    });
    thead.appendChild(htr);
    table.appendChild(thead);

    var tbody = el('tbody');
    var t0 = performance.now();
    tbody.innerHTML = st.players.map(rowHtml).join('');
    tbl.tbody = tbody;
    tbl.rows = [];
    var nodes = tbody.children;
    for (var i = 0; i < nodes.length; i++) {
      tbl.rows.push({ p: st.players[i], node: nodes[i], hidden: false });
    }
    var buildMs = performance.now() - t0;

    tbody.addEventListener('click', function (e) {
      var tr = e.target.closest('tr[data-id]');
      if (tr) G.drawer.open(parseInt(tr.dataset.id, 10), S().gw);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    host.appendChild(wrap);

    var foot = el('div', {
      style: 'display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:8px'
    }, [U.scaleLegend()]);
    foot.appendChild(el('span', {
      class: 'view-note',
      style: 'margin-left:auto',
      text: 'table built in ' + buildMs.toFixed(0) + ' ms'
    }));
    host.appendChild(foot);

    applySort();
    applyFilters();
  }

  function filterBar() {
    var st = S();
    var bar = el('div', { class: 'filters' });

    bar.appendChild(el('div', { class: 'field' }, [
      el('label', { for: 'f-search', text: 'search  ( / )' }),
      el('input', {
        class: 'input', id: 'f-search', type: 'search', placeholder: 'name or team',
        style: 'width:150px',
        oninput: function () { tbl.filters.q = this.value.trim().toLowerCase(); applyFilters(); }
      })
    ]));

    var seg = el('div', { class: 'segmented' });
    [[0, 'All'], [1, 'GKP'], [2, 'DEF'], [3, 'MID'], [4, 'FWD']].forEach(function (o) {
      seg.appendChild(el('button', {
        type: 'button', 'aria-pressed': tbl.filters.pos === o[0] ? 'true' : 'false',
        onclick: function () {
          tbl.filters.pos = o[0];
          U.qsa('button', seg).forEach(function (b) { b.setAttribute('aria-pressed', 'false'); });
          this.setAttribute('aria-pressed', 'true');
          applyFilters();
        }
      }, o[1]));
    });
    bar.appendChild(el('div', { class: 'field' }, [el('label', { text: 'position' }), seg]));

    var teamSel = el('select', {
      class: 'input',
      onchange: function () { tbl.filters.team = parseInt(this.value, 10) || 0; applyFilters(); }
    }, [el('option', { value: '0' }, 'All teams')]);
    Object.keys(st.teams).map(Number).sort(function (a, b) {
      return String(st.teams[a].short_name).localeCompare(String(st.teams[b].short_name));
    }).forEach(function (id) {
      teamSel.appendChild(el('option', { value: String(id) }, st.teams[id].short_name + ' — ' + st.teams[id].name));
    });
    bar.appendChild(el('div', { class: 'field' }, [el('label', { text: 'club' }), teamSel]));

    var maxCost = 0;
    st.players.forEach(function (p) { if (isNum(p.now_cost) && p.now_cost > maxCost) maxCost = p.now_cost; });
    var costOut = el('span', { class: 'view-note', text: 'any' });
    var costRange = el('input', {
      class: 'input', type: 'range', min: '38', max: String(maxCost || 150), step: '1',
      value: String(maxCost || 150),
      oninput: function () {
        var v = parseInt(this.value, 10);
        tbl.filters.maxCost = v >= maxCost ? null : v;
        costOut.textContent = v >= maxCost ? 'any' : '≤ ' + U.money(v);
        applyFilters();
      }
    });
    bar.appendChild(el('div', { class: 'field' }, [
      el('label', {}, ['max price ', costOut]), costRange
    ]));

    var p60Out = el('span', { class: 'view-note', text: 'any' });
    bar.appendChild(el('div', { class: 'field' }, [
      el('label', {}, ['min p60 ', p60Out]),
      el('input', {
        class: 'input', type: 'range', min: '0', max: '95', step: '5', value: '0',
        oninput: function () {
          tbl.filters.minP60 = parseInt(this.value, 10) / 100;
          p60Out.textContent = tbl.filters.minP60 ? '≥ ' + this.value + '%' : 'any';
          applyFilters();
        }
      })
    ]));

    var hSel = el('select', {
      class: 'input',
      onchange: function () {
        S().horizon = parseInt(this.value, 10);
        refreshHorizonCells();
        // the plan is solved for a horizon, so it has to be re-solved; the
        // offline snapshot has only the one plan and keeps it
        if (S().mode === 'live') S().plan = null;
        G.invalidate('planner');
      }
    });
    for (var h = 1; h <= Math.max(1, st.gws.length); h++) {
      hSel.appendChild(el('option', { value: String(h), selected: h === st.horizon }, 'next ' + h));
    }
    bar.appendChild(el('div', { class: 'field' }, [el('label', { text: 'xP horizon' }), hSel]));

    bar.appendChild(el('div', { class: 'field' }, [
      el('label', { text: 'availability' }),
      el('label', { style: 'display:flex;gap:6px;align-items:center;height:28px;font-size:12px;color:var(--ink-2);text-transform:none;letter-spacing:0' }, [
        el('input', {
          type: 'checkbox',
          onchange: function () { tbl.filters.hideFlagged = this.checked; applyFilters(); }
        }),
        document.createTextNode('hide flagged')
      ])
    ]));

    tbl.countNode = el('span', { class: 'filter-count' });
    bar.appendChild(tbl.countNode);
    return bar;
  }

  function sortBy(key) {
    var col = null;
    COLS.forEach(function (c) { if (c.key === key) col = c; });
    if (!col || !col.sort) return;
    if (tbl.sortKey === key) tbl.sortDir = -tbl.sortDir;
    else { tbl.sortKey = key; tbl.sortDir = col.dir; }
    applySort();
  }

  function applySort() {
    var col = null;
    COLS.forEach(function (c) { if (c.key === tbl.sortKey) col = c; });
    if (!col || !col.sort) return;
    var dir = tbl.sortDir;
    var keyed = tbl.rows.map(function (r) { return { r: r, k: col.sort(r.p) }; });
    keyed.sort(function (a, b) {
      if (a.k < b.k) return -dir;
      if (a.k > b.k) return dir;
      return a.r.p.id - b.r.p.id;   // stable, deterministic tiebreak
    });
    tbl.rows = keyed.map(function (x) { return x.r; });

    var frag = document.createDocumentFragment();
    tbl.rows.forEach(function (r) { frag.appendChild(r.node); });
    tbl.tbody.appendChild(frag);   // one reflow, nodes are moved not rebuilt

    U.qsa('thead th', tbl.tbody.parentNode).forEach(function (th) {
      var arrow = th.querySelector('.arrow');
      if (arrow) arrow.remove();
      if (th.dataset.key === tbl.sortKey) {
        th.appendChild(el('span', { class: 'arrow', text: dir < 0 ? '▼' : '▲' }));
      }
    });
  }

  function applyFilters() {
    var f = tbl.filters;
    var shown = 0;
    for (var i = 0; i < tbl.rows.length; i++) {
      var r = tbl.rows[i], p = r.p, ok = true;
      if (f.pos && p.position !== f.pos) ok = false;
      if (ok && f.team && p.team_id !== f.team) ok = false;
      if (ok && f.maxCost !== null && isNum(p.now_cost) && p.now_cost > f.maxCost) ok = false;
      if (ok && f.minP60 > 0 && (!isNum(p.p_60) || p.p_60 < f.minP60)) ok = false;
      if (ok && f.hideFlagged && p.status && p.status !== 'a') ok = false;
      if (ok && f.q) {
        var hay = p.web_name.toLowerCase() + ' ' + String(D().teamShort(p.team_id)).toLowerCase();
        if (hay.indexOf(f.q) < 0) ok = false;
      }
      if (ok) shown++;
      if (r.hidden === !ok) continue;          // no DOM write unless it changed
      r.hidden = !ok;
      r.node.classList.toggle('hide', !ok);
    }
    if (tbl.countNode) {
      tbl.countNode.textContent = shown + ' of ' + tbl.rows.length + ' shown';
    }
  }

  // only the two horizon-dependent cells are rewritten; no rebuild, no reflow storm
  function refreshHorizonCells() {
    var xpIdx = 4, valIdx = 5;
    tbl.rows.forEach(function (r) {
      r.node.cells[xpIdx].textContent = U.num(xpH(r.p), 1);
      r.node.cells[valIdx].textContent = U.num(valueOf(r.p), 2);
    });
    if (tbl.sortKey === 'xp' || tbl.sortKey === 'value') applySort();
    var host = qs('#view-players');
    var note = host.querySelector('.view-note');
    if (note) {
      note.textContent = S().players.length + ' players · xP summed over the next ' + S().horizon +
        ' gameweek' + (S().horizon === 1 ? '' : 's') + ' from GW' + S().gw;
    }
  }

  // ============================================================== PLANNER ==

  function plannerView(host) {
    U.clear(host);
    host.appendChild(head('Planner', ['multi-gameweek transfer plan from the optimizer']));
    var body = el('div');
    host.appendChild(body);
    body.appendChild(U.loadingBlock(4));

    G.load.plan().then(function (plan) {
      U.clear(body);
      renderPlan(body, plan);
    }, function (err) {
      var d = G.describeError(err);
      U.clear(body);
      body.appendChild(U.errorBlock('No plan available.',
        esc(d.where) + ' returned: <b>' + esc(d.what) + '</b>' +
        (err.status === 404 ? '<br>POST /api/optimize is not implemented on this backend yet.' : ''),
        [{ label: 'Retry', fn: function () { plannerView(host); } }]));
    });
  }

  function renderPlan(body, plan) {
    var st = S();
    var decisions = plan.decisions || [];
    var nTransfers = 0, nHits = 0, totalNet = 0;
    decisions.forEach(function (d) {
      nTransfers += (d.transfers || []).length;
      nHits += d.hits || 0;
      totalNet += isNum(d.expected_points_net) ? d.expected_points_net
        : (isNum(d.expected_points) ? d.expected_points - 4 * (d.hits || 0) : 0);
    });

    // With no team linked there is nothing to transfer FROM, so the backend
    // returns the initial-squad pick instead of a plan. It is still solved over
    // the full horizon, so report that horizon rather than the single decision
    // it came back with — otherwise the tile reads "GW1–1" and looks broken.
    var initial = plan.mode === 'initial_squad';
    var planGws = plan.gws || [];
    var lastGw = initial && planGws.length ? planGws[planGws.length - 1]
      : (decisions.length ? decisions[decisions.length - 1].gw : '?');
    var spanGws = initial && planGws.length ? planGws.length : decisions.length;

    if (initial) {
      body.appendChild(el('div', { class: 'view-note', style: 'margin-bottom:10px' }, [
        el('span', { html:
          '<b>No team linked</b>, so there are no transfers to plan. This is the best opening 15, ' +
          'chosen over GW' + (plan.first_gw || 1) + '–' + lastGw + '. ' +
          'To plan transfers, set your FPL entry id in <code>config.json</code> ' +
          '(<code>{"entry_id": 1234567}</code>) and reload — you can find it in the URL of your ' +
          'points page on the FPL site.' })
      ]));
    }

    body.appendChild(el('div', { class: 'stats' }, [
      U.statTile('Horizon', 'GW' + (plan.first_gw || decisions[0].gw) + '–' + lastGw,
        spanGws + ' gameweek' + (spanGws === 1 ? '' : 's') +
        (initial ? ' · opening squad' : '')),
      U.statTile('Projected net', U.num(totalNet, 1), 'after hit costs', 'hero'),
      U.statTile('Transfers', String(nTransfers), nHits ? nHits + ' hit(s) = ' + (-4 * nHits) + ' pts' : 'no hits'),
      U.statTile('Chips', decisions.filter(function (d) { return d.chip; })
        .map(function (d) { return String(d.chip).toUpperCase() + ' GW' + d.gw; }).join(', ') || 'none', ''),
      U.statTile('Solver', esc(plan.solver_status || '—'),
        isNum(plan.objective) ? 'objective ' + U.num(plan.objective, 1) : '')
    ]));

    if (st.mode === 'live') {
      body.appendChild(el('div', { style: 'margin-bottom:10px' }, [
        el('button', {
          class: 'btn btn-accent', type: 'button',
          onclick: function () {
            var b = this; b.disabled = true; b.textContent = 'Solving…';
            G.load.plan(true).then(function () { G.invalidate('planner'); },
              function () { b.disabled = false; b.textContent = 'Recompute plan'; G.invalidate('planner'); });
          }
        }, 'Recompute plan')
      ]));
    }

    var tl = el('div', { class: 'timeline' });
    decisions.forEach(function (d) {
      var transfers = d.transfers || [];
      var mid = el('div');

      if (!transfers.length) {
        mid.appendChild(el('div', { class: 'move' }, [
          el('span', { class: 'dir', style: 'background:#12161c;border:1px solid var(--line-2);color:var(--ink-2)', text: 'ROLL' }),
          el('span', { class: 'view-note', text: 'no transfer' })
        ]));
      }
      transfers.forEach(function (t) {
        var out = st.byId[t.out_id], inp = st.byId[t.in_id];
        var row = el('div', { class: 'move' }, [
          el('span', { class: 'dir out', text: 'OUT' }),
          out ? playerLink(out, d.gw, out.web_name + ' (' + U.money(t.out_price !== undefined ? t.out_price : out.now_cost) + ')', 't-name')
              : el('span', { text: 'id ' + t.out_id }),
          el('span', { class: 'arrow', text: '→' }),
          el('span', { class: 'dir in', text: 'IN' }),
          inp ? playerLink(inp, d.gw, inp.web_name + ' (' + U.money(t.in_price !== undefined ? t.in_price : inp.now_cost) + ')', 't-name')
              : el('span', { text: 'id ' + t.in_id })
        ]);
        if (isNum(t.gain)) {
          row.appendChild(el('span', { class: 'view-note', text: U.signed(t.gain, 1) + ' xP over the horizon' }));
        }
        mid.appendChild(row);
      });

      if (d.notes && d.notes.length) {
        mid.appendChild(el('ul', { class: 'notes' }, d.notes.map(function (n) { return el('li', { text: n }); })));
      }

      var cap = st.byId[d.captain];
      var right = el('div', { class: 'right' }, [
        el('div', { class: 'stat-value', text: U.num(isNum(d.expected_points_net) ? d.expected_points_net : d.expected_points, 1) }),
        el('div', { class: 'stat-sub', text: 'projected points' }),
        el('div', { class: 'stat-sub', html: d.hits ? '<span style="color:var(--neg)">' + d.hits + ' hit(s) −' + (4 * d.hits) + '</span>' : 'no hit' }),
        el('div', { class: 'stat-sub', text: 'C: ' + (cap ? cap.web_name : '—') }),
        el('div', { class: 'stat-sub', text: isNum(d.bank_after) ? 'bank ' + U.money(d.bank_after) + ' · FT ' + d.free_transfers_after : '' })
      ]);

      var card = el('div', { class: 'panel gwcard' }, [
        el('div', {}, [
          el('div', { class: 'gwno', text: 'GW' + d.gw }),
          d.chip ? el('span', { class: 'tag accent', text: String(d.chip).toUpperCase() }) : null
        ]),
        mid,
        right
      ]);
      tl.appendChild(card);
    });
    body.appendChild(tl);
  }

  // ============================================================== CAPTAIN ==

  var captainGw = null;

  function captainView(host) {
    U.clear(host);
    var st = S();
    if (captainGw === null || st.gws.indexOf(captainGw) < 0) captainGw = st.gw;

    host.appendChild(head('Captain', ['the armband is the single biggest swing in the game']));
    host.appendChild(el('div', { style: 'margin-bottom:10px' }, [
      gwPicker(captainGw, function (g) { captainGw = g; captainView(host); })
    ]));

    var body = el('div');
    host.appendChild(body);
    body.appendChild(U.loadingBlock(3));

    G.load.captain(captainGw).then(function (res) {
      U.clear(body);
      var options = (res && (res.options || res.captains)) || [];
      if (!options.length) {
        body.appendChild(el('div', { class: 'empty', text: 'The backend returned no captain options for GW' + captainGw + '.' }));
        return;
      }
      renderCaptain(body, options, captainGw);
    }, function (err) {
      var d = G.describeError(err);
      U.clear(body);
      body.appendChild(U.errorBlock('No captain ranking available.',
        esc(d.where) + ' returned: <b>' + esc(d.what) + '</b>',
        [{ label: 'Retry', fn: function () { captainView(host); } }]));
    });
  }

  function renderCaptain(body, options, gw) {
    var st = S();
    var opts = options.map(function (o) {
      var pid = D().toNum(D().pick(o, ['player_id', 'id'], null));
      var p = st.byId[pid];
      var xp = D().toNum(o.xp, 0);
      var eo = D().toNum(D().pick(o, ['effective_ownership', 'eo'], null));
      return {
        p: p, id: pid, name: p ? p.web_name : 'id ' + pid,
        xp: xp,
        sd: D().toNum(o.sd, null),
        haul: D().toNum(D().pick(o, ['p_haul', 'haul'], null)),
        eo: eo,
        ev: D().toNum(o.ev_vs_field, null),
        // expected gain on the average rival from captaining him: you bank 2x,
        // the field banks EO/100 x. Derived here, not from the backend.
        vsField: eo === null ? null : xp * 2 - xp * (eo / 100),
        rationale: o.rationale || ''
      };
    }).sort(function (a, b) { return b.xp - a.xp; });

    var maxXp = Math.max.apply(null, opts.map(function (o) { return o.xp; }));
    var maxHaul = Math.max.apply(null, opts.map(function (o) { return o.haul || 0; })) || 1;
    var maxEo = Math.max.apply(null, opts.map(function (o) { return o.eo || 0; })) || 1;

    var grid = el('div', { class: 'capgrid' });

    // ---- ranked table
    var tblEl = el('table', { class: 'grid' });
    var html = ['<thead><tr>',
      '<th class="c">#</th><th>Player</th><th>Fixture</th><th class="r">xP</th><th class="r">SD</th>',
      '<th class="r">P(haul 10+)</th><th class="r">EO</th><th class="r">vs field</th><th>Read</th>',
      '</tr></thead><tbody>'];
    opts.forEach(function (o, i) {
      var fxTxt = '—';
      if (o.p) {
        var entry = o.p.gwMap[gw];
        if (entry && entry.fixtures.length) {
          fxTxt = entry.fixtures.map(function (fx) {
            return (fx.is_home ? 'vs ' : 'at ') + D().teamShort(fx.opponent_id);
          }).join(' + ');
        } else if (entry) fxTxt = 'blank';
      }
      var kind = o.eo === null ? '' :
        (o.eo >= 50 ? '<span class="tag warn">template</span>' :
         (o.eo <= 15 ? '<span class="tag accent">differential</span>' : '<span class="tag">balanced</span>'));
      html.push('<tr data-id="' + o.id + '">' +
        '<td class="c t-dim">' + (i + 1) + '</td>' +
        '<td class="t-name">' + esc(o.name) + ' <span class="t-team">' +
          esc(o.p ? D().teamShort(o.p.team_id) : '') + '</span></td>' +
        '<td class="t-dim">' + esc(fxTxt) + '</td>' +
        '<td class="r t-key">' + U.num(o.xp, 2) + '</td>' +
        '<td class="r t-dim">' + U.num(o.sd, 2) + '</td>' +
        '<td class="r">' + barCellHtml(o.haul, maxHaul, U.pct(o.haul, 1)) + '</td>' +
        '<td class="r">' + barCellHtml(o.eo === null ? null : o.eo / 100, maxEo / 100, U.pctv(o.eo, 1), true) + '</td>' +
        '<td class="r t-key">' + (o.vsField === null ? '—' : U.signed(o.vsField, 2)) + '</td>' +
        '<td>' + kind + '</td>' +
        '</tr>');
    });
    html.push('</tbody>');
    tblEl.innerHTML = html.join('');
    tblEl.addEventListener('click', function (e) {
      var tr = e.target.closest('tr[data-id]');
      if (tr) G.drawer.open(parseInt(tr.dataset.id, 10), gw);
    });
    var left = el('div');
    left.appendChild(U.panel('GW' + gw + ' captain options', null, [
      el('div', { class: 'tablewrap', style: 'border:0;max-height:none' }, [tblEl])
    ], ''));
    left.appendChild(el('div', { class: 'view-note', style: 'margin-top:8px', text:
      '"vs field" is derived here, not by the backend: captaining a player banks 2× his xP for you ' +
      'and EO% × his xP for the average rival, so it is 2·xP − EO/100·xP. Positive means the pick ' +
      'gains rank in expectation.' }));
    grid.appendChild(left);

    // ---- trade-off panel with the quadrant plot
    var right = el('div');
    right.appendChild(U.panel('Template vs differential', null, [
      quadrantPlot(opts, maxXp, maxEo),
      el('div', { class: 'tradeoff', style: 'margin-top:10px' }, [
        el('div', { html: tradeoffText(opts) })
      ])
    ]));
    grid.appendChild(right);
    body.appendChild(grid);
  }

  function barCellHtml(v, max, label, alt) {
    if (v === null || !isNum(v)) return '<span class="t-dim">—</span>';
    var w = max > 0 ? Math.max(2, Math.min(100, (v / max) * 100)) : 0;
    return '<span class="bar-cell" style="justify-content:flex-end">' +
      '<span>' + esc(label) + '</span>' +
      '<span class="track' + (alt ? ' alt' : '') + '"><i style="width:' + w.toFixed(1) + '%"></i></span>' +
      '</span>';
  }

  function quadrantPlot(opts, maxXp, maxEo) {
    var W = 300, H = 220, ml = 34, mb = 26, mt = 10, mr = 10;
    var xMax = Math.max(10, maxEo * 1.1), yMax = Math.max(1, maxXp * 1.15);
    var plotW = W - ml - mr, plotH = H - mt - mb;
    function X(eo) { return ml + (eo / xMax) * plotW; }
    function Y(xp) { return mt + plotH - (xp / yMax) * plotH; }

    var withEo = opts.filter(function (o) { return o.eo !== null; });
    if (!withEo.length) {
      return el('div', { class: 'empty', text: 'No effective-ownership data returned, so the trade-off cannot be plotted.' });
    }
    var eoMid = withEo.map(function (o) { return o.eo; }).sort(function (a, b) { return a - b; })[Math.floor(withEo.length / 2)];
    var xpMid = withEo.map(function (o) { return o.xp; }).sort(function (a, b) { return a - b; })[Math.floor(withEo.length / 2)];

    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" height="' + H + '" role="img" ' +
             'aria-label="expected points against effective ownership">'];
    s.push('<rect x="' + ml + '" y="' + mt + '" width="' + plotW + '" height="' + plotH +
           '" fill="#0a0d11" stroke="#1b2029"/>');
    s.push('<line x1="' + X(eoMid) + '" y1="' + mt + '" x2="' + X(eoMid) + '" y2="' + (mt + plotH) +
           '" stroke="#262d38" stroke-dasharray="3 3"/>');
    s.push('<line x1="' + ml + '" y1="' + Y(xpMid) + '" x2="' + (ml + plotW) + '" y2="' + Y(xpMid) +
           '" stroke="#262d38" stroke-dasharray="3 3"/>');
    s.push('<text x="' + (ml + 5) + '" y="' + (mt + 13) + '" fill="#31d0aa" font-size="9">differential upside</text>');
    s.push('<text x="' + (ml + plotW - 5) + '" y="' + (mt + 13) + '" fill="#e0ac4f" font-size="9" text-anchor="end">template core</text>');
    s.push('<text x="' + (ml + plotW - 5) + '" y="' + (mt + plotH - 5) + '" fill="#69727f" font-size="9" text-anchor="end">must-own trap</text>');

    withEo.forEach(function (o, i) {
      var cx = X(o.eo), cy = Y(o.xp);
      var r = 4 + 4 * (o.haul || 0) / (Math.max.apply(null, withEo.map(function (q) { return q.haul || 0; })) || 1);
      s.push('<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="' + r.toFixed(1) +
             '" fill="' + (i === 0 ? '#31d0aa' : '#7aa7d8') + '" fill-opacity="0.85"/>');
      s.push('<text x="' + (cx + 6).toFixed(1) + '" y="' + (cy + 3).toFixed(1) +
             '" fill="#98a2b1" font-size="9">' + esc(o.name.slice(0, 9)) + '</text>');
    });

    s.push('<text x="' + (ml + plotW / 2) + '" y="' + (H - 6) + '" fill="#69727f" font-size="9" text-anchor="middle">effective ownership %</text>');
    s.push('<text x="10" y="' + (mt + plotH / 2) + '" fill="#69727f" font-size="9" transform="rotate(-90 10 ' + (mt + plotH / 2) + ')" text-anchor="middle">xP</text>');
    s.push('<text x="' + ml + '" y="' + (H - 14) + '" fill="#69727f" font-size="8">0</text>');
    s.push('<text x="' + (ml + plotW) + '" y="' + (H - 14) + '" fill="#69727f" font-size="8" text-anchor="end">' + xMax.toFixed(0) + '</text>');
    s.push('</svg>');

    var box = el('div');
    box.innerHTML = s.join('');
    return box;
  }

  function tradeoffText(opts) {
    var top = opts[0];
    var diff = null;
    opts.forEach(function (o) {
      if (o.eo !== null && o.eo <= 20 && (!diff || o.xp > diff.xp)) diff = o;
    });
    var lines = [];
    if (top.eo !== null) {
      lines.push('<b>' + esc(top.name) + '</b> is the highest projection at ' + U.num(top.xp, 2) +
        ' xP with ' + U.pctv(top.eo, 0) + ' effective ownership. Every point he scores hands the ' +
        'average rival ' + U.num(top.eo / 100, 2) + ' points, so captaining him is worth about ' +
        '<b>' + U.signed(top.vsField, 2) + '</b> against the field — and <i>not</i> owning him costs ' +
        'you roughly ' + U.num(top.xp * (top.eo / 100), 2) + ' points of rank per gameweek.');
    } else {
      lines.push('<b>' + esc(top.name) + '</b> is the highest projection at ' + U.num(top.xp, 2) +
        ' xP. The backend returned no effective ownership, so the rank trade-off cannot be quantified.');
    }
    if (diff && diff.id !== top.id) {
      var gap = top.xp - diff.xp;
      lines.push('<b>' + esc(diff.name) + '</b> is the live differential: ' + U.pctv(diff.eo, 0) +
        ' EO, ' + U.num(gap, 2) + ' xP behind, ' + U.pct(diff.haul, 1) + ' chance of a 10+ haul. ' +
        'You give up ' + U.num(gap * 2, 2) + ' expected points to gain on ' +
        U.num(100 - diff.eo, 0) + '% of the field when he delivers.');
    }
    lines.push('Above 100% effective ownership the armband stops being an upside play and becomes ' +
      'insurance: the haul you miss costs more rank than the haul you catch gains.');
    return lines.map(function (l) { return '<p style="margin:0 0 8px">' + l + '</p>'; }).join('');
  }

  // ================================================================ CHIPS ==

  function chipsView(host) {
    U.clear(host);
    host.appendChild(head('Chips', ['four chips die at the GW19 deadline — unused is wasted']));

    var cd = el('div', { class: 'countdown', id: 'chip-countdown' });
    updateCountdown(cd);
    host.appendChild(cd);

    var body = el('div', { style: 'margin-top:12px' });
    host.appendChild(body);
    body.appendChild(U.loadingBlock(3));

    G.load.chips().then(function (res) {
      U.clear(body);
      renderChips(body, res || {});
      updateCountdown(cd);   // the backend deadline supersedes the estimate
    }, function (err) {
      var d = G.describeError(err);
      U.clear(body);
      body.appendChild(U.errorBlock('No chip recommendations available.',
        esc(d.where) + ' returned: <b>' + esc(d.what) + '</b>',
        [{ label: 'Retry', fn: function () { chipsView(host); } }]));
    });
  }

  var CHIP_LABEL = {
    wildcard: 'Wildcard', freehit: 'Free Hit', bboost: 'Bench Boost', '3xc': 'Triple Captain'
  };
  var CHIP_WINDOW = {
    wildcard: 'GW2–19', freehit: 'GW2–19', bboost: 'GW1–19', '3xc': 'GW1–19'
  };

  function updateCountdown(node) {
    var exp = G.chipExpiryIso();
    var u = U.timeUntil(exp.iso);
    var sev = u ? G.chipSeverity(u.days) : '';
    node.className = 'countdown' + (sev ? ' ' + sev : '');
    U.clear(node);
    node.appendChild(el('div', {}, [
      el('div', { class: 'cd-label', text: 'first-half chips expire in' }),
      el('div', { class: 'cd-units' }, u && !u.past ? [
        el('div', { class: 'cd-unit' }, [el('b', { text: String(u.days) }), el('span', { text: 'days' })]),
        el('div', { class: 'cd-unit' }, [el('b', { text: String(u.hours) }), el('span', { text: 'hrs' })]),
        el('div', { class: 'cd-unit' }, [el('b', { text: String(u.mins) }), el('span', { text: 'min' })]),
        el('div', { class: 'cd-unit' }, [el('b', { text: String(u.secs) }), el('span', { text: 'sec' })])
      ] : [el('div', { class: 'cd-value', text: u ? 'EXPIRED' : 'unknown' })])
    ]));
    node.appendChild(el('div', { style: 'margin-left:auto;text-align:right' }, [
      el('div', { class: 'stat-sub', text: 'GW19 deadline · ' + U.localDate(exp.iso, true) }),
      el('div', { class: 'stat-sub', text:
        exp.source === 'backend' ? 'from the backend events feed'
        : (exp.source === 'sample'
            ? 'from the offline sample snapshot — confirm against the live events feed'
            : 'date from the spec; exact time comes from the backend when connected') }),
      el('div', { class: 'stat-sub', text: 'now GW' + (S().state ? S().state.current_gw : '?') +
        ' — wildcard, free hit, bench boost and triple captain all reset after this' })
    ]));
  }

  function renderChips(body, data) {
    var recs = data.recommendations || data.chips || [];
    var used = data.chips_used || [];
    var byChip = {};
    recs.forEach(function (r) {
      var k = String(r.chip || r.name || '').toLowerCase();
      if (!byChip[k]) byChip[k] = r;
    });

    var grid = el('div', { class: 'chipgrid' });
    ['3xc', 'bboost', 'freehit', 'wildcard'].forEach(function (key) {
      var r = byChip[key];
      var isUsed = used.indexOf(key) >= 0;
      var card = el('section', { class: 'panel chipcard' + (isUsed ? ' used' : '') }, [
        el('h3', { text: CHIP_LABEL[key] || key }),
        el('div', { class: 'stat-sub', text: 'window ' + (CHIP_WINDOW[key] || '') + (isUsed ? ' · already used' : '') }),
        el('div', { class: 'gwpick', text: r && isNum(D().toNum(r.gw, null)) ? 'GW' + r.gw : (isUsed ? 'used' : 'hold') }),
        r && r.confidence ? el('span', { class: 'tag', text: 'confidence: ' + r.confidence }) : null,
        el('p', { text: r ? (r.reason || r.rationale || '') : 'No recommendation returned for this chip.' })
      ]);
      if (r && isNum(D().toNum(r.expected_gain, null))) {
        card.appendChild(el('div', { class: 'stat-sub', text: 'expected gain ' + U.num(r.expected_gain, 1) + ' pts' }));
      }
      grid.appendChild(card);
    });
    body.appendChild(grid);

    var dbl = data.doubles || {}, blk = data.blanks || {};
    var dblGws = Object.keys(dbl).filter(function (k) { return (dbl[k] || []).length; });
    var blkGws = Object.keys(blk).filter(function (k) { return (blk[k] || []).length; });
    var lines = [];
    lines.push(dblGws.length
      ? 'Double gameweeks detected: ' + dblGws.map(function (g) {
          return 'GW' + g + ' (' + dbl[g].length + ' clubs)'; }).join(', ')
      : 'No double gameweek is in the current schedule. Doubles only appear once postponed ' +
        'fixtures are rescheduled, so this is recomputed on every run — never assume it is final.');
    lines.push(blkGws.length
      ? 'Blank gameweeks detected: ' + blkGws.map(function (g) {
          return 'GW' + g + ' (' + blk[g].length + ' clubs idle)'; }).join(', ')
      : 'No blank gameweek in the current schedule.');
    body.appendChild(el('div', { style: 'margin-top:12px' }, [
      U.panel('Schedule scan', null, [
        el('ul', { class: 'notes' }, lines.map(function (l) { return el('li', { text: l }); }))
      ])
    ]));
  }

  // =============================================================== DRAWER ==

  function drawerView(host, p, ds) {
    U.clear(host);
    if (!p) { host.appendChild(el('div', { class: 'empty', text: 'player not found' })); return; }
    var gw = ds.gw;
    var entry = p.gwMap[gw];

    var headBox = el('div', { class: 'drawer-head' });
    headBox.appendChild(el('button', {
      class: 'btn btn-sm drawer-close', type: 'button', onclick: G.drawer.close
    }, 'Close'));
    headBox.appendChild(el('div', { class: 'drawer-title' }, [
      el('h2', { text: p.web_name }),
      el('span', { class: 'tag', text: p.pos }),
      el('span', { class: 'tag', text: D().teamShort(p.team_id) }),
      el('span', { class: 'tag', text: U.money(p.now_cost) }),
      S().mode === 'sample' ? el('span', { class: 'tag sample', text: 'sample' }) : null
    ]));
    var flag = statusFlag(p);
    if (flag) {
      headBox.appendChild(el('div', {
        class: 'warnline', style: 'margin-top:6px',
        text: (flag.out ? 'UNAVAILABLE' : 'DOUBTFUL') + ' — ' +
              (p.news || 'status ' + p.status) +
              (isNum(p.chance) ? ' (' + p.chance + '% chance)' : '')
      }));
    }
    headBox.appendChild(el('div', { style: 'margin-top:8px' }, [
      gwPicker(gw, function (g) { G.drawer.setGw(g); })
    ]));
    host.appendChild(headBox);

    // ---- per-gameweek xP strip
    var sec = el('div', { class: 'drawer-sec' });
    sec.appendChild(el('h3', { text: 'projected points by gameweek' }));
    var maxXp = Math.max.apply(null, p.gws.map(function (g) { return g.xp; }).concat([1]));
    var strip = el('div', { style: 'display:flex;gap:4px;align-items:flex-end;height:64px' });
    p.gws.forEach(function (g) {
      var h = Math.max(2, (g.xp / maxXp) * 56);
      strip.appendChild(el('div', {
        style: 'flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px;cursor:pointer',
        title: 'GW' + g.gw + ': ' + U.num(g.xp, 2) + ' xP',
        onclick: function () { G.drawer.setGw(g.gw); }
      }, [
        el('div', { class: 'stat-sub', style: 'font-size:10px', text: U.num(g.xp, 1) }),
        el('div', {
          style: 'width:100%;height:' + h.toFixed(0) + 'px;border-radius:2px;background:' +
                 (g.gw === gw ? 'var(--accent)' : '#2b3441')
        }),
        el('div', { class: 'stat-sub', style: 'font-size:9px', text: g.gw })
      ]));
    });
    sec.appendChild(strip);
    host.appendChild(sec);

    if (!entry) {
      host.appendChild(el('div', { class: 'drawer-sec' }, [
        el('div', { class: 'empty', text: 'No projection for GW' + gw + '.' })
      ]));
    } else if (!entry.fixtures.length) {
      host.appendChild(el('div', { class: 'drawer-sec' }, [
        el('h3', { text: 'GW' + gw }),
        el('div', { class: 'empty', text: 'Blank gameweek — no fixture, 0 points.' })
      ]));
    } else {
      entry.fixtures.forEach(function (fx, i) {
        host.appendChild(fixtureSection(p, fx, gw, entry.fixtures.length > 1 ? i + 1 : 0));
      });
      if (entry.fixtures.length > 1) {
        host.appendChild(el('div', { class: 'drawer-sec' }, [
          el('h3', { text: 'double gameweek total' }),
          el('div', { class: 'stat-value', text: U.num(entry.xp, 2) + ' xP' })
        ]));
      }
      var risk = el('div', { class: 'drawer-sec' });
      risk.appendChild(el('h3', { text: 'risk' }));
      risk.appendChild(el('div', { class: 'kv' }, [
        el('div', {}, [el('dt', { text: 'GW xP' }), el('dd', { text: U.num(entry.xp, 2) })]),
        el('div', {}, [el('dt', { text: 'SD' }), el('dd', { text: U.num(entry.sd, 2) })]),
        el('div', {}, [el('dt', { text: 'P(haul 10+)' }), el('dd', { text: entry.p_haul === null ? '—' : U.pct(entry.p_haul, 1) })]),
        el('div', {}, [el('dt', { text: 'own' }), el('dd', { text: U.pctv(p.selected_by_percent, 1) })])
      ]));
      host.appendChild(risk);
    }

    // ---- explanation from the backend
    var exSec = el('div', { class: 'drawer-sec' });
    exSec.appendChild(el('h3', { text: 'model explanation' }));
    var detail = ds.detailFor === p.id ? ds.detail : null;
    var explanation = detail && (detail.explanation || detail.explain || detail.reason);
    if (explanation) {
      exSec.appendChild(el('div', { class: 'explain', text: String(explanation) }));
    } else if (p.minutes_reason) {
      exSec.appendChild(el('div', { class: 'explain', text: p.minutes_reason }));
    } else if (S().mode === 'sample') {
      exSec.appendChild(el('div', { class: 'view-note', text:
        'The per-player explanation comes from GET /api/player/{id} and is not part of the offline snapshot.' }));
    } else {
      exSec.appendChild(el('div', { class: 'view-note', text: 'loading explanation…' }));
    }
    host.appendChild(exSec);
  }

  function fixtureSection(p, fx, gw, idx) {
    var sec = el('div', { class: 'drawer-sec' });
    var e = D().easeOf(fx);
    var t = e.t === null ? 0.5 : e.t;
    var oppName = D().teamName(fx.opponent_id);

    sec.appendChild(el('h3', {
      text: 'GW' + gw + (idx ? ' — fixture ' + idx : '') + ' · ' +
            (fx.is_home ? 'home to ' : 'away at ') + oppName
    }));

    var fxLine = el('div', { style: 'display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px' });
    var chip = el('div');
    chip.innerHTML = '<span class="tk" style="--c:' + U.ramp(t) + ';--fg:' + U.inkOn(t) + '">' +
      esc(fx.is_home ? String(D().teamShort(fx.opponent_id)).toUpperCase()
                     : String(D().teamShort(fx.opponent_id)).toLowerCase()) +
      '<i>' + (U.easeRank(e.t) === null ? '?' : U.easeRank(e.t)) + '</i></span>';
    fxLine.appendChild(chip);
    if (e.source === 'model') {
      fxLine.appendChild(el('span', { class: 'view-note', text:
        'our model: ' + U.num(e.tl, 2) + ' expected goals for, ' + U.num(e.ol, 2) + ' against' }));
    } else if (e.source === 'fdr') {
      fxLine.appendChild(el('span', { class: 'warnline', text:
        'no model lambda for this fixture — colour falls back to official FDR ' + U.num(fx.difficulty, 0) }));
    }
    sec.appendChild(fxLine);

    if (!fx.has_components) {
      var full = S().fullComponentGws;
      sec.appendChild(el('div', { class: 'warnline', text:
        full
          ? 'The offline snapshot only carries the full component audit for GW' + full.join(', GW') +
            '. Start the backend for every gameweek.'
          : 'The backend did not return the point components for this fixture, so the breakdown ' +
            'cannot be shown. Only the total (' + U.num(fx.xp_total, 2) + ' xP) is available.' }));
      return sec;
    }

    sec.appendChild(U.componentBars(fx, fx.xp_total));

    sec.appendChild(el('h3', { style: 'margin-top:14px', text: 'underlying rates' }));
    var kv = el('div', { class: 'kv' });
    [
      ['p(start)', U.pct(fx.p_start, 0)],
      ['p(60+)', U.pct(fx.p_60, 0)],
      ['p(appear)', U.pct(fx.p_appear, 0)],
      ['xmins', U.num(fx.xmins, 0)],
      ['λ goals', U.num(fx.lambda_goals, 3)],
      ['λ assists', U.num(fx.lambda_assists, 3)],
      ['λ conceded', U.num(fx.lambda_conceded, 2)],
      ['λ saves', U.num(fx.lambda_saves, 2)],
      ['P(CS & 60+)', U.pct(fx.p_clean_sheet, 1)],
      ['P(DEFCON)', U.pct(fx.p_defcon, 1)],
      ['E[BPS]', U.num(fx.exp_bps, 1)],
      ['SD', U.num(fx.sd_total, 2)]
    ].forEach(function (r) {
      kv.appendChild(el('div', {}, [el('dt', { text: r[0] }), el('dd', { text: r[1] })]));
    });
    sec.appendChild(kv);
    return sec;
  }

  G.views = {
    squad: squadView,
    players: playersView,
    planner: plannerView,
    captain: captainView,
    chips: chipsView,
    myteam: function (host) { return G.myteamView(host); },
    drawer: drawerView,
    updateCountdown: updateCountdown
  };
})();
