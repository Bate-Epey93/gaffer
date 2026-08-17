/* gaffer — the "My Team" view: import a squad, edit it, and be told the cost.
 *
 * The logic lives in myteam.js; this file is only presentation and events. The
 * split matters because the interesting part — best XI, legality, the swap
 * comparison — is pure arithmetic that can be tested without a DOM.
 *
 * The editorial line throughout: never say "invalid" without saying which rule
 * and by how much, and never present a difference from the optimiser as a
 * mistake. A 0.4-point gap over six gameweeks is noise, and the model's own
 * backtest error is wider than that, so the copy says so rather than nudging
 * the user into a pointless transfer.
 */
(function () {
  "use strict";

  var G = window.G || (window.G = {});
  var U = G.ui;
  var el = U.el, esc = U.esc, num = U.num, signed = U.signed;

  function M() { return G.myteam; }
  function S() { return G.store; }

  function head(title, subtitle) {
    return el('header', { class: 'view-head' }, [
      el('h1', { class: 'view-title', text: title }),
      subtitle ? el('p', { class: 'view-sub', html: subtitle }) : null
    ]);
  }

  function recommendedIds() {
    var plan = S().plan;
    if (!plan || !plan.decisions || !plan.decisions.length) return [];
    var gw = S().gw, dec = null;
    plan.decisions.forEach(function (d) { if (d.gw === gw && !dec) dec = d; });
    if (!dec) dec = plan.decisions[0];
    return (dec.squad || []).slice();
  }

  // ------------------------------------------------------------- import ----

  function importPanel(rerender) {
    var st = M().state;
    var input = el('input', {
      type: 'number', class: 'input', id: 'entry-id-input',
      placeholder: 'e.g. 1234567', min: '1',
      value: st.entryId || ''
    });
    var status = el('p', { class: 'muted', style: 'margin:.5rem 0 0' });

    function importEntry() {
      var id = Number(input.value);
      if (!id || id < 1) { status.textContent = 'Enter your FPL team id first.'; return; }
      M().setEntryId(id);
      status.textContent = 'Loading team ' + id + '…';

      G.request('/squad?entry_id=' + id).then(function (res) {
        var picks = (res.squad && res.squad.picks) || res.picks || res.squad || [];
        var ids = picks.map(function (p) {
          return typeof p === 'number' ? p : (p.element || p.id);
        }).filter(Boolean);
        if (ids.length !== 15) throw new Error('got ' + ids.length + ' players, expected 15');
        M().setPicks(ids, 'entry');
        U.toast('Loaded your team from FPL.');
        rerender();
      }, function (err) {
        // The published site cannot do this, and the reason is worth stating
        // precisely rather than showing a bare network error.
        status.innerHTML = G.isStatic()
          ? '<b>The published site cannot fetch your team.</b> The FPL API sends no ' +
            'CORS headers, so a browser on another domain is not allowed to read it. ' +
            'Either build your squad below by hand, or run gaffer locally ' +
            '(<code>python -m gaffer.cli serve</code>) where it can import in one click.'
          : 'Could not load that team: ' + esc((err && err.message) || 'unknown error');
      });
    }

    var body = [
      el('p', { class: 'muted', html:
        'Your team id is in the URL when you view your points on the FPL site: ' +
        '<code>fantasy.premierleague.com/entry/<b>1234567</b>/event/1</code>' }),
      el('div', { class: 'row', style: 'gap:.5rem;align-items:center;flex-wrap:wrap' }, [
        input,
        el('button', { type: 'button', class: 'btn btn-accent', onclick: importEntry,
                       text: 'Import from FPL' }),
        st.picks.length ? el('button', {
          type: 'button', class: 'btn', text: 'Start over',
          onclick: function () { M().clear(); U.toast('Cleared.'); rerender(); }
        }) : null,
        recommendedIds().length ? el('button', {
          type: 'button', class: 'btn', text: "Copy the optimiser's 15",
          onclick: function () {
            M().setPicks(recommendedIds(), 'recommended');
            U.toast("Copied the optimiser's squad — now edit it.");
            rerender();
          }
        }) : null
      ]),
      status
    ];
    return U.panel('Your team', null, body);
  }

  // -------------------------------------------------------------- editor ---

  function playerRow(player, onRemove) {
    var flags = M().risks(player);
    return el('div', { class: 'myteam-row' }, [
      el('span', { class: 'pos-chip pos-' + player.position, text: M().POS_SHORT[player.position] }),
      el('div', { class: 'myteam-name' }, [
        el('b', { text: M().name(player) }),
        el('span', { class: 'muted', text: ' ' + player.team }),
        flags.length ? el('div', { class: 'myteam-flag ' + flags[0].level, text: flags[0].text }) : null
      ]),
      el('span', { class: 'myteam-cost', text: '£' + num(M().price(player), 1) }),
      el('span', { class: 'myteam-xp', text: num(M().horizonXp(player), 1), title: 'expected points over the horizon' }),
      onRemove ? el('button', {
        type: 'button', class: 'btn btn-sm', text: '×',
        title: 'Remove ' + M().name(player), onclick: function () { onRemove(player); }
      }) : null
    ]);
  }

  function addPanel(rerender) {
    var search = el('input', { type: 'search', class: 'input', placeholder: 'Search a player to add…' });
    var results = el('div', { class: 'myteam-results' });
    var picks = M().state.picks;

    function refresh() {
      U.clear(results);
      var q = (search.value || '').trim().toLowerCase();
      if (q.length < 2) {
        results.appendChild(el('p', { class: 'muted', text: 'Type at least two letters.' }));
        return;
      }
      var have = {};
      picks.forEach(function (id) { have[id] = true; });
      var matches = (S().players || []).filter(function (p) {
        if (have[p.id]) return false;
        return (M().name(p) || '').toLowerCase().indexOf(q) >= 0 ||
               (p.team || '').toLowerCase().indexOf(q) >= 0;
      }).sort(function (a, b) { return M().horizonXp(b) - M().horizonXp(a); }).slice(0, 12);

      if (!matches.length) {
        results.appendChild(el('p', { class: 'muted', text: 'Nobody matches, or they are already in your squad.' }));
        return;
      }
      matches.forEach(function (p) {
        results.appendChild(el('button', {
          type: 'button', class: 'myteam-add',
          onclick: function () {
            M().setPicks(picks.concat([p.id]));
            U.toast('Added ' + M().name(p) + '.');
            rerender();
          }
        }, [
          el('span', { class: 'pos-chip pos-' + p.position, text: M().POS_SHORT[p.position] }),
          el('b', { text: M().name(p) }),
          el('span', { class: 'muted', text: ' ' + p.team + ' · £' + num(M().price(p), 1) }),
          el('span', { class: 'myteam-xp', text: num(M().horizonXp(p), 1) })
        ]));
      });
    }

    search.addEventListener('input', refresh);
    refresh();
    return U.panel('Add a player', null, [search, results]);
  }

  // ------------------------------------------------------------- verdict ---

  function verdictPanel(evalResult) {
    var check = evalResult.legality;
    var body = [];

    if (!evalResult.players.length) {
      body.push(el('p', { class: 'muted', text:
        'No squad yet. Import one, copy the optimiser\'s, or add players below.' }));
      return U.panel('Verdict', null, body);
    }

    if (check.problems.length) {
      body.push(el('div', { class: 'myteam-problems' }, [
        el('b', { text: check.problems.length === 1 ? 'One rule broken:' : check.problems.length + ' rules broken:' }),
        el('ul', {}, check.problems.map(function (p) { return el('li', { text: p.text }); }))
      ]));
    } else {
      body.push(el('p', { class: 'ok', text:
        'Legal squad: 15 players, the right shape, no more than three from any club, ' +
        'and £' + check.cost.toFixed(1) + 'm of the £100.0m budget spent.' }));
    }

    var gw = evalResult.gw;
    if (gw) {
      body.push(el('div', { class: 'stats' }, [
        U.statTile('Best XI, GW' + S().gw, num(gw.withCaptain, 1),
                   gw.label + ' · captain ' + esc(M().name(gw.captain))),
        U.statTile('Squad cost', '£' + num(check.cost, 1),
                   check.cost > M().BUDGET ? 'over budget' : '£' + num(M().BUDGET - check.cost, 1) + 'm free'),
        U.statTile('Horizon xP', num(evalResult.totalHorizonXp, 1), 'all 15, next ' + S().horizon + ' GWs')
      ]));
      body.push(el('div', { class: 'myteam-list' },
        gw.xi.map(function (p) { return playerRow(p, null); })));
      if (gw.bench.length) {
        body.push(el('h3', { class: 'myteam-subhead', text: 'Bench' }));
        body.push(el('div', { class: 'myteam-list' },
          gw.bench.map(function (p) { return playerRow(p, null); })));
      }
    }
    return U.panel('Verdict', null, body);
  }

  // ---------------------------------------------------------- comparison ---

  function comparePanel(evalResult) {
    var theirs = recommendedIds();
    if (!theirs.length) {
      return U.panel('Against the optimiser', null, [
        el('p', { class: 'muted', text:
          'The recommended squad has not loaded, so there is nothing to compare against yet.' })
      ]);
    }
    var mine = M().state.picks;
    if (!mine.length) {
      return U.panel('Against the optimiser', null, [
        el('p', { class: 'muted', text: 'Build a squad above and this fills in.' })
      ]);
    }

    var diff = M().compare(mine, theirs);
    var body = [];

    if (!diff.swaps.length) {
      body.push(el('p', { class: 'ok', text: 'Identical to the optimiser\'s squad.' }));
      return U.panel('Against the optimiser', null, body);
    }

    var mineXp = evalResult.totalHorizonXp;
    var theirsXp = M().resolve(theirs).reduce(function (s, p) { return s + M().horizonXp(p); }, 0);
    var gap = mineXp - theirsXp;

    body.push(el('p', {
      class: gap < -3 ? 'warn' : (gap < 0 ? 'muted' : 'ok'),
      html: '<b>' + signed(gap, 1) + ' expected points</b> across all 15 over the next ' +
            S().horizon + ' gameweeks, against the squad the optimiser picked. ' +
            (Math.abs(gap) < 2
              ? 'That is inside the model\'s own error — these are effectively the same squad.'
              : (gap < 0
                  ? 'The differences below are where it goes.'
                  : 'Your squad projects higher, which usually means it spends money the optimiser saved.'))
    }));

    body.push(el('div', { class: 'myteam-swaps' }, diff.swaps.map(function (swap) {
      var cls = swap.xpDelta === undefined ? 'neutral'
        : (Math.abs(swap.xpDelta) < 0.75 ? 'neutral' : (swap.xpDelta > 0 ? 'bad' : 'good'));
      return el('div', { class: 'myteam-swap ' + cls }, [
        el('div', { class: 'myteam-swap-head' }, [
          swap.out ? el('span', { class: 'out', text: M().name(swap.out) }) : el('span', { class: 'muted', text: '—' }),
          el('span', { class: 'arrow', text: ' vs ' }),
          swap.into ? el('span', { class: 'into', text: M().name(swap.into) }) : el('span', { class: 'muted', text: '—' }),
          swap.xpDelta !== undefined
            ? el('span', { class: 'myteam-delta', text: signed(swap.xpDelta, 1) + ' xP' })
            : null
        ]),
        el('p', { class: 'myteam-why', text: M().explainSwap(swap) })
      ]);
    })));

    body.push(el('p', { class: 'muted', style: 'margin-top:.75rem', html:
      'This compares squads, not transfers — it ignores what a change would cost you in ' +
      'hits. A -4 is worth taking only if the swap gains more than four points over the ' +
      'gameweeks you will hold him.' }));

    return U.panel('Against the optimiser', null, body);
  }

  // ----------------------------------------------------------------- view ---

  function myteamView(host) {
    M().load();
    U.clear(host);
    host.appendChild(head('My Team',
      'Import your squad or build it by hand, then see exactly what it costs against the model.'));

    var body = el('div');
    host.appendChild(body);

    function rerender() {
      U.clear(body);
      var evalResult = M().evaluate(M().state.picks);

      body.appendChild(importPanel(rerender));
      body.appendChild(verdictPanel(evalResult));

      if (M().state.picks.length) {
        var removePanel = U.panel('Your 15', null, [
          el('div', { class: 'myteam-list' }, M().resolve(M().state.picks).map(function (p) {
            return playerRow(p, function (victim) {
              M().setPicks(M().state.picks.filter(function (id) { return id !== victim.id; }));
              U.toast('Removed ' + M().name(victim) + '.');
              rerender();
            });
          }))
        ]);
        body.appendChild(removePanel);
      }

      if (M().state.picks.length < 15) body.appendChild(addPanel(rerender));
      body.appendChild(comparePanel(evalResult));
    }

    // The comparison needs the optimiser's squad; make sure it is loaded.
    if (!S().plan) {
      G.loadPlan().then(rerender, rerender);
    }
    rerender();
  }

  G.myteamView = myteamView;
})();
