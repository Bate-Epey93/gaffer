/* gaffer dashboard — shared UI primitives.
 *
 * No framework, no build step, no network references. SVG is always produced by
 * assigning an HTML string to innerHTML (the HTML parser puts it in the SVG
 * namespace for us) so nothing here has to name the SVG namespace URI.
 */
(function () {
  'use strict';

  var G = (window.G = window.G || {});

  // ------------------------------------------------------------------ dom --

  function qs(sel, root) { return (root || document).querySelector(sel); }
  function qsa(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') n.className = v;
        else if (k === 'html') n.innerHTML = v;
        else if (k === 'text') n.textContent = v;
        else if (k === 'dataset') { for (var d in v) n.dataset[d] = v[d]; }
        else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), v);
        else if (v === true) n.setAttribute(k, '');
        else n.setAttribute(k, v);
      }
    }
    if (kids !== undefined && kids !== null) {
      (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
        if (c === null || c === undefined || c === false) return;
        n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      });
    }
    return n;
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // ----------------------------------------------------------- formatting --

  function isNum(x) { return typeof x === 'number' && isFinite(x); }

  function num(x, d) {
    if (!isNum(x)) return '—';
    return x.toFixed(d === undefined ? 1 : d);
  }

  function signed(x, d) {
    if (!isNum(x)) return '—';
    var s = Math.abs(x).toFixed(d === undefined ? 2 : d);
    if (x > 0) return '+' + s;
    if (x < 0) return '-' + s;
    return s;
  }

  function money(tenths) {
    if (!isNum(tenths)) return '—';
    return '£' + (tenths / 10).toFixed(1);
  }

  function pct(frac, d) {
    if (!isNum(frac)) return '—';
    return (frac * 100).toFixed(d === undefined ? 0 : d) + '%';
  }

  function pctv(v, d) {
    if (!isNum(v)) return '—';
    return v.toFixed(d === undefined ? 1 : d) + '%';
  }

  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  var DAY = 86400000;

  function timeUntil(iso) {
    if (!iso) return null;
    var t = Date.parse(iso);
    if (isNaN(t)) return null;
    var ms = t - Date.now();
    var a = Math.abs(ms);
    return {
      ms: ms,
      past: ms < 0,
      days: Math.floor(a / DAY),
      hours: Math.floor(a / 3600000) % 24,
      mins: Math.floor(a / 60000) % 60,
      secs: Math.floor(a / 1000) % 60
    };
  }

  function shortCountdown(u) {
    if (!u) return '—';
    if (u.past) return 'expired';
    if (u.days >= 1) return u.days + 'd ' + u.hours + 'h';
    if (u.hours >= 1) return u.hours + 'h ' + u.mins + 'm';
    return u.mins + 'm ' + u.secs + 's';
  }

  function localDate(iso, withTime) {
    if (!iso) return '';
    var t = new Date(iso);
    if (isNaN(t.getTime())) return '';
    var opts = { weekday: 'short', day: 'numeric', month: 'short' };
    if (withTime) { opts.hour = '2-digit'; opts.minute = '2-digit'; }
    try { return t.toLocaleString(undefined, opts); } catch (e) { return t.toISOString(); }
  }

  // ------------------------------------------------------- difficulty scale --
  //
  // ONE continuous scale, driven by our own model: net expected goals for the
  // player's team in that fixture (team lambda minus opponent lambda). It runs
  // deep navy (hard) through grey to gold (easy) — a monotone lightness ramp, so
  // it reads correctly under any colour vision deficiency — and every cell also
  // prints the 1-5 rank, so colour is redundant, never load-bearing on its own.

  var STOPS = [
    [0.00, [0x17, 0x23, 0x3f]],
    [0.25, [0x2f, 0x47, 0x63]],
    [0.50, [0x5c, 0x66, 0x72]],
    [0.75, [0x97, 0x86, 0x5a]],
    [1.00, [0xdc, 0xc0, 0x5a]]
  ];

  function ramp(t) {
    t = clamp(t, 0, 1);
    for (var i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i];
        var f = (t - a[0]) / (b[0] - a[0]);
        var c = [0, 1, 2].map(function (j) {
          return Math.round(a[1][j] + f * (b[1][j] - a[1][j]));
        });
        return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
      }
    }
    return 'rgb(220,192,90)';
  }

  function inkOn(t) {
    // luminance of the ramp rises monotonically; flip the text past the midpoint
    return t > 0.62 ? '#141007' : '#e7edf5';
  }

  // net lambda -> 0 (hardest) .. 1 (easiest)
  var NET_SPAN = 1.8;
  function easeIndexFromLambdas(teamLambda, oppLambda) {
    if (!isNum(teamLambda) || !isNum(oppLambda)) return null;
    return clamp((teamLambda - oppLambda + NET_SPAN) / (2 * NET_SPAN), 0, 1);
  }

  // fallback when the model has no lambda for a fixture: official FDR 1..5
  function easeIndexFromFdr(fdr) {
    if (!isNum(fdr)) return null;
    return clamp((5 - fdr) / 4, 0, 1);
  }

  // 1 = easiest, 5 = hardest (matches the FPL convention the user already knows)
  function easeRank(t) {
    if (t === null) return null;
    return clamp(6 - Math.max(1, Math.ceil(t * 5)), 1, 5);
  }

  function scaleLegend() {
    var bar = 'linear-gradient(90deg,' + [0, 0.25, 0.5, 0.75, 1].map(ramp).join(',') + ')';
    return el('div', { class: 'scale-legend' }, [
      el('span', { text: 'fixture scale: hard' }),
      el('span', { class: 'scale-bar', style: 'background:' + bar }),
      el('span', { text: 'easy' }),
      el('span', { class: 'tag', text: 'our λ, not FDR' }),
      el('span', { text: 'digit = 1 (easiest) to 5 (hardest); dotted underline = official FDR fallback' })
    ]);
  }

  // ------------------------------------------------------- component bars --

  var COMPONENTS = [
    ['xp_appearance', 'appearance'],
    ['xp_goals', 'goals'],
    ['xp_assists', 'assists'],
    ['xp_clean_sheet', 'clean sheet'],
    ['xp_goals_conceded', 'conceded'],
    ['xp_saves', 'saves'],
    ['xp_defcon', 'DEFCON'],
    ['xp_bonus', 'bonus'],
    ['xp_cards', 'cards'],
    ['xp_penalty', 'pen save/miss']
  ];

  /* Labelled diverging bars around a zero axis. Positive right, negative left,
     every row carrying its signed value so the sign never depends on colour. */
  function componentBars(fx, total) {
    var vals = COMPONENTS.map(function (c) {
      return { key: c[0], label: c[1], v: isNum(fx[c[0]]) ? fx[c[0]] : 0 };
    });
    var maxPos = 0, maxNeg = 0;
    vals.forEach(function (r) {
      if (r.v > maxPos) maxPos = r.v;
      if (-r.v > maxNeg) maxNeg = -r.v;
    });
    var span = maxPos + maxNeg;
    if (span <= 0) span = 1;
    var zeroPct = (maxNeg / span) * 100;

    var wrap = el('div', { class: 'comp' });
    vals.forEach(function (r) {
      var track = el('div', { class: 'track' });
      track.appendChild(el('span', { class: 'zero', style: 'left:' + zeroPct.toFixed(2) + '%' }));
      var w = (Math.abs(r.v) / span) * 100;
      if (w > 0.15) {
        var bar;
        if (r.v >= 0) {
          bar = el('i', { style: 'left:' + zeroPct.toFixed(2) + '%;width:' + w.toFixed(2) + '%' });
        } else {
          bar = el('i', {
            class: 'neg',
            style: 'left:' + (zeroPct - w).toFixed(2) + '%;width:' + w.toFixed(2) + '%'
          });
        }
        track.appendChild(bar);
      }
      wrap.appendChild(el('div', { class: 'comprow' }, [
        el('div', { class: 'lab', text: r.label }),
        track,
        el('div', { class: 'val' + (r.v < 0 ? ' neg' : ''), text: signed(r.v, 2) })
      ]));
    });

    var sum = vals.reduce(function (a, r) { return a + r.v; }, 0);
    var shown = isNum(total) ? total : sum;
    wrap.appendChild(el('div', { class: 'comprow total' }, [
      el('div', { class: 'lab', text: 'total xP' }),
      el('div', {}),
      el('div', { class: 'val', text: num(shown, 2) })
    ]));
    if (isNum(total) && Math.abs(total - sum) > 0.005) {
      wrap.appendChild(el('div', {
        class: 'warnline',
        text: 'components sum to ' + num(sum, 3) + ', total says ' + num(total, 3) +
              ' — the backend is not returning a consistent breakdown'
      }));
    }
    return wrap;
  }

  /* Three-way composition strip used on the pitch cards: attacking / defensive /
     the rest. Deliberately only three buckets — a ten-colour stack would be
     decoration, not information. The full audit is one click away in the drawer. */
  function mixBar(fx) {
    var att = (fx.xp_goals || 0) + (fx.xp_assists || 0) + (fx.xp_penalty || 0);
    var def = (fx.xp_clean_sheet || 0) + (fx.xp_saves || 0) + (fx.xp_defcon || 0)
            + (fx.xp_goals_conceded || 0);
    var oth = (fx.xp_appearance || 0) + (fx.xp_bonus || 0) + (fx.xp_cards || 0);
    att = Math.max(0, att); def = Math.max(0, def); oth = Math.max(0, oth);
    var tot = att + def + oth;
    var bar = el('div', {
      class: 'mixbar',
      title: 'attacking ' + num(att, 2) + ' · defensive ' + num(def, 2) + ' · other ' + num(oth, 2)
    });
    if (tot <= 0) return bar;
    [['mix-att', att], ['mix-def', def], ['mix-oth', oth]].forEach(function (p) {
      bar.appendChild(el('i', {
        class: p[0], style: 'width:' + ((p[1] / tot) * 100).toFixed(2) + '%'
      }));
    });
    return bar;
  }

  // ------------------------------------------------------------- feedback --

  function loadingBlock(lines) {
    var box = el('div', { class: 'loading' });
    for (var i = 0; i < (lines || 4); i++) box.appendChild(el('div', { class: 'skel' }));
    return box;
  }

  function errorBlock(title, detail, actions) {
    var box = el('div', { class: 'errbox' }, [el('b', { text: title })]);
    if (detail) box.appendChild(el('div', { style: 'margin-top:6px', html: detail }));
    if (actions && actions.length) {
      var row = el('div', { style: 'margin-top:10px;display:flex;gap:6px;flex-wrap:wrap' });
      actions.forEach(function (a) {
        row.appendChild(el('button', { class: 'btn btn-sm', type: 'button', onclick: a.fn }, a.label));
      });
      box.appendChild(row);
    }
    return box;
  }

  var toastTimer = null;
  function toast(msg) {
    var t = qs('#toast');
    if (!t) { t = el('div', { class: 'toast', id: 'toast' }); document.body.appendChild(t); }
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.hidden = true; }, 3200);
  }

  // ---------------------------------------------------------------- misc --

  function statTile(label, value, sub, cls) {
    return el('div', { class: 'stat' + (cls ? ' ' + cls : '') }, [
      el('div', { class: 'stat-label', text: label }),
      el('div', { class: 'stat-value', html: value }),
      sub ? el('div', { class: 'stat-sub', html: sub }) : null
    ]);
  }

  function panel(title, right, bodyNodes, bodyClass) {
    var head = el('div', { class: 'panel-head' }, [el('h2', { class: 'panel-title', text: title })]);
    if (right) { right.style.marginLeft = 'auto'; head.appendChild(right); }
    return el('section', { class: 'panel' }, [
      head, el('div', { class: bodyClass || 'panel-body' }, bodyNodes)
    ]);
  }

  G.ui = {
    qs: qs, qsa: qsa, el: el, esc: esc, clear: clear,
    isNum: isNum, num: num, signed: signed, money: money, pct: pct, pctv: pctv, clamp: clamp,
    timeUntil: timeUntil, shortCountdown: shortCountdown, localDate: localDate,
    ramp: ramp, inkOn: inkOn, easeIndexFromLambdas: easeIndexFromLambdas,
    easeIndexFromFdr: easeIndexFromFdr, easeRank: easeRank, scaleLegend: scaleLegend,
    COMPONENTS: COMPONENTS, componentBars: componentBars, mixBar: mixBar,
    loadingBlock: loadingBlock, errorBlock: errorBlock, toast: toast,
    statTile: statTile, panel: panel
  };
})();
