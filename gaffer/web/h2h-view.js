/* gaffer — head to head.
 *
 * Classic rank rewards total points; H2H rewards beating one person, and a
 * 90-point win counts exactly as much as a 60-point one. So every number here
 * is a win probability, not an expected score, and the two genuinely disagree:
 * behind on projection you want variance, ahead you want none.
 *
 * The whole report is computed in CI (reading an opponent's picks is a
 * cross-origin request the FPL API refuses) and read here as flat JSON.
 */
(function () {
  "use strict";

  var G = window.G || (window.G = {});
  var U = G.ui;
  var el = U.el, esc = U.esc, num = U.num;

  var cache = null;

  function pctText(p) { return Math.round(100 * (p || 0)) + '%'; }

  function head(title, sub) {
    return el('header', { class: 'view-head' }, [
      el('h1', { class: 'view-title', text: title }),
      sub ? el('p', { class: 'view-sub', html: sub }) : null
    ]);
  }

  /* One bar, three segments. Colour alone never carries it — each segment is
     labelled, so this reads the same to someone who cannot separate the hues. */
  function odds(row) {
    return el('div', {}, [
      el('div', { class: 'h2h-bar' }, [
        el('i', { class: 'win', style: 'width:' + (100 * row.win) + '%' }),
        el('i', { class: 'draw', style: 'width:' + (100 * row.draw) + '%' }),
        el('i', { class: 'loss', style: 'width:' + (100 * row.loss) + '%' })
      ]),
      el('div', { class: 'h2h-key' }, [
        el('span', { class: 'win', text: 'win ' + pctText(row.win) }),
        el('span', { class: 'draw', text: 'draw ' + pctText(row.draw) }),
        el('span', { class: 'loss', text: 'loss ' + pctText(row.loss) })
      ])
    ]);
  }

  function sideRow(label, s) {
    if (!s) return null;
    return el('div', { class: 'h2h-side' }, [
      el('span', { class: 'h2h-who', text: label }),
      el('span', { class: 'h2h-mean', text: num(s.mean, 1) }),
      el('span', { class: 'muted', text: '±' + num(s.sd, 1) }),
      el('span', { class: 'muted', text: s.p10 + '–' + s.p90 }),
      el('span', { class: 'h2h-capt', text: s.captain || '-' })
    ]);
  }

  function leaguePanel(league, mine) {
    if (league.error) {
      return U.panel(league.name || 'league', null,
        [el('p', { class: 'muted', text: league.error })]);
    }
    var opponent = league.opponent || {};
    var body = [
      el('p', { class: 'h2h-vs', html:
        'against <b>' + esc(opponent.name || '?') + '</b>' +
        (opponent.player ? ' <span class="muted">(' + esc(opponent.player) + ')</span>' : '') +
        (league.opponent && league.opponent.is_knockout ? ' — <b>knockout</b>' : '') }),
      odds(league),
      el('div', { class: 'h2h-sides' }, [
        el('div', { class: 'h2h-head' }, [
          el('span', { text: '' }), el('span', { text: 'mean' }),
          el('span', { text: 'sd' }), el('span', { text: '10–90th' }),
          el('span', { text: 'captain' })
        ]),
        sideRow('you', mine),
        sideRow('them', league.theirs)
      ]),
      el('p', { class: 'h2h-advice', text: league.advice || '' })
    ];

    var caps = league.captains || [];
    if (caps.length) {
      var best = caps[0];
      var current = null;
      caps.forEach(function (c) { if (mine && c.name === mine.captain) current = c; });
      body.push(el('h3', { class: 'myteam-subhead', text: 'armband, by win probability' }));
      body.push(el('div', { class: 'h2h-caps' }, caps.slice(0, 6).map(function (c) {
        var isNow = mine && c.name === mine.captain;
        return el('div', { class: 'h2h-cap' + (isNow ? ' is-now' : '') }, [
          el('b', { text: c.name }),
          el('span', { class: 'muted', text: num(c.mean, 1) + ' xP' }),
          el('span', { class: 'h2h-capwin', text: pctText(c.score) })
        ]);
      })));
      /* Only worth saying when it actually changes the decision, and only with
         the cost attached — a swap that gains 1% of win probability while
         costing a point of expected score is not obviously right. */
      if (current && best.name !== current.name && (best.score - current.score) > 0.005) {
        body.push(el('p', { class: 'h2h-swap', html:
          '<b>' + esc(best.name) + '</b> over ' + esc(current.name) + ' is worth ' +
          '+' + Math.round(100 * (best.score - current.score)) + ' points of win ' +
          'probability here, at a cost of ' + num(best.mean - current.mean, 1) +
          ' expected points.' }));
      }
    }
    return U.panel(league.name || 'league', null, body);
  }

  function render(host, data) {
    U.clear(host);
    host.appendChild(head('Head to head',
      'Every tie this gameweek, scored on the chance of <b>winning</b> — which is not ' +
      'the same as scoring the most points.'));

    if (data.unavailable) {
      host.appendChild(U.panel('Not available yet', null, [
        el('p', { class: 'muted', text: data.unavailable })
      ]));
      return;
    }
    var leagues = data.leagues || [];
    if (!leagues.length) {
      host.appendChild(U.panel('No head-to-head leagues', null, [
        el('p', { class: 'muted', text:
          'This build found no H2H leagues for the configured team.' })
      ]));
      return;
    }
    if (leagues.some(function (l) { return l.stale; })) {
      host.appendChild(el('p', { class: 'muted h2h-stale', text:
        'GW' + data.gw + ' squads are not public until the deadline, so where a ' +
        'squad is missing this uses the last one FPL published. Estimates, not ' +
        'confirmed line-ups.' }));
    }
    leagues.forEach(function (l) { host.appendChild(leaguePanel(l, data.mine)); });
  }

  function h2hView(host) {
    if (cache) { render(host, cache); return; }
    U.clear(host);
    host.appendChild(U.loadingBlock(5));
    fetch('data/h2h.json', { headers: { 'Accept': 'application/json' } })
      .then(function (r) { return r.ok ? r.json() : { unavailable: 'not in this build' }; })
      .catch(function () {
        return { unavailable: 'This snapshot carries no head-to-head data. It is ' +
                              'built only when a team id is configured.' };
      })
      .then(function (data) { cache = data; render(host, data); });
  }

  G.h2hView = h2hView;
})();
