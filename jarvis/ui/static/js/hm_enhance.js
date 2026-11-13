/* HM Algo dashboard enhancements — presentation only, no API calls.
 *
 * 1. Sortable tables: every `table.tt-table` with a thead gets header buttons
 *    (click / Enter / Space) and aria-sort. The sort is re-applied after each
 *    live re-render, so polling never resets the user's chosen order.
 * 2. Row filter: an <input data-hm-filter="<tbody id>"> hides non-matching rows,
 *    also re-applied after each re-render.
 * 3. Focus retention: a re-render that replaces the focused row control moves
 *    focus to the new control with the same data-* identity (keyboard users
 *    no longer lose their place every poll).
 * 4. Confirm modal: Esc / backdrop cancel, initial focus on Cancel, Tab
 *    trapped inside, focus restored to the opener on close.
 */
(function () {
  'use strict';

  var NUM_RE = /[-+]?\d[\d,]*(?:\.\d+)?/;
  var DATE_RE = /^\d{4}-\d{2}-\d{2}/;

  function cellKey(row, idx) {
    var c = row.cells[idx];
    var t = c ? c.textContent.replace(/\s+/g, ' ').trim() : '';
    if (!t || t === '—' || t === '-') return null;
    if (DATE_RE.test(t)) return t;
    var m = t.match(NUM_RE);
    // Only numeric when the number leads the cell ("+1.20 (0.4R)", "55%"), not "XAUUSD".
    if (m && t.indexOf(m[0]) <= 1) return parseFloat(m[0].replace(/,/g, ''));
    return t.toLowerCase();
  }

  function compare(a, b) {
    if (a === null) return b === null ? 0 : 1; // blanks always last
    if (b === null) return -1;
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    return String(a).localeCompare(String(b));
  }

  function dataRows(tbody) {
    if (tbody.hasAttribute('data-state')) return []; // loading / empty placeholder
    return Array.prototype.filter.call(tbody.rows, function (r) {
      return !(r.cells.length === 1 && r.cells[0].colSpan > 1);
    });
  }

  function wireTable(table) {
    var thead = table.tHead;
    var tbody = table.tBodies[0];
    if (!thead || !tbody || table.hasAttribute('data-hm-sort-wired')) return;
    table.setAttribute('data-hm-sort-wired', '');

    var sort = { sig: '', col: -1, dir: 1 };
    var filterInput = tbody.id ? document.querySelector('[data-hm-filter="' + tbody.id + '"]') : null;
    var observer;

    function headerSig() { return thead.textContent.replace(/\s+/g, ' ').trim(); }

    function decorate() {
      var sig = headerSig();
      if (sig !== sort.sig) { sort = { sig: sig, col: -1, dir: 1 }; } // tab switched columns
      var row = thead.rows[0];
      if (!row) return;
      Array.prototype.forEach.call(row.cells, function (th, idx) {
        var label = th.textContent.trim();
        if (!label || /actions/i.test(label)) { th.removeAttribute('aria-sort'); return; }
        var btn = th.querySelector('.hm-sort-btn');
        if (!btn) {
          btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'hm-sort-btn';
          while (th.firstChild) btn.appendChild(th.firstChild);
          th.appendChild(btn);
          btn.addEventListener('click', function () {
            sort.dir = sort.col === idx ? -sort.dir : (th.classList.contains('tt-num') ? -1 : 1);
            sort.col = idx;
            decorate();
            apply();
          });
        }
        var active = sort.col === idx;
        th.setAttribute('aria-sort', active ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none');
        btn.setAttribute('data-dir', active ? (sort.dir > 0 ? 'asc' : 'desc') : '');
        btn.title = 'Sort by ' + label;
      });
      if (observer) observer.takeRecords(); // wrapping headers is not a new render
    }

    function apply() {
      var rows = dataRows(tbody);
      if (sort.col >= 0 && rows.length > 1) {
        // shortcut: history is paginated client-side, so this sorts the visible page only; sort state.history in dashboard.js if a global order is needed.
        var keyed = rows.map(function (r, i) { return { r: r, k: cellKey(r, sort.col), i: i }; });
        keyed.sort(function (a, b) {
          var c = compare(a.k, b.k);
          if (a.k !== null && b.k !== null) c *= sort.dir;
          return c || a.i - b.i;
        });
        var frag = document.createDocumentFragment();
        keyed.forEach(function (x) { frag.appendChild(x.r); });
        tbody.appendChild(frag);
      }
      if (filterInput) {
        var q = filterInput.value.trim().toLowerCase();
        rows.forEach(function (r) {
          r.classList.toggle('hm-filtered-out', !!q && r.textContent.toLowerCase().indexOf(q) === -1);
        });
      }
      if (observer) observer.takeRecords(); // our own reorder is not a new render
    }

    observer = new MutationObserver(function (records) {
      var headChanged = records.some(function (r) { return thead.contains(r.target); });
      if (headChanged) decorate();
      apply();
      restoreFocus();
    });
    observer.observe(thead, { childList: true, subtree: true });
    observer.observe(tbody, { childList: true });
    if (filterInput) filterInput.addEventListener('input', apply);
    decorate();
    apply();
  }

  /* ── Focus retention across re-renders ─────────────────────────────── */
  var lastFocus = null; // a selector for the last focused row control

  document.addEventListener('focusin', function (e) {
    var el = e.target;
    if (!el.closest || !el.closest('tbody')) return;
    var attr = Array.prototype.filter.call(el.attributes, function (a) { return /^data-/.test(a.name); })[0];
    lastFocus = attr ? '[' + attr.name + '="' + CSS.escape(attr.value) + '"]' : null;
  });

  function restoreFocus() {
    if (!lastFocus) return;
    var a = document.activeElement;
    if (a && a !== document.body && a.isConnected) return;
    var el = document.querySelector(lastFocus);
    if (el) el.focus({ preventScroll: true });
  }

  /* ── Confirm modal accessibility ────────────────────────────────────── */
  function wireModal() {
    var modal = document.getElementById('cockpit-confirm-modal');
    if (!modal) return;
    var opener = null;
    var cancel = document.getElementById('confirm-modal-cancel');

    new MutationObserver(function () {
      if (!modal.hidden) {
        opener = document.activeElement;
        if (cancel) cancel.focus(); // the safe choice is the default
      } else if (opener && opener.isConnected) {
        opener.focus({ preventScroll: true });
        opener = null;
      }
    }).observe(modal, { attributes: true, attributeFilter: ['hidden'] });

    modal.addEventListener('click', function (e) {
      if (e.target === modal && cancel) cancel.click();
    });
    modal.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && cancel) { e.preventDefault(); cancel.click(); return; }
      if (e.key !== 'Tab') return;
      var f = modal.querySelectorAll('button:not([disabled])');
      if (!f.length) return;
      var first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll('table.tt-table'), wireTable);
    wireModal();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
