/* ═══════════════════════════════════════════════════════════
   FAILSAFE — script.js
   Connects to FastAPI backend at http://localhost:8000
   ═══════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';

/* ── TABS ──────────────────────────────────────────────────── */

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

/* ══════════════════════════════════════════════════════════════
   SINGLE STUDENT
══════════════════════════════════════════════════════════════ */

document.getElementById('predictBtn').addEventListener('click', async () => {
  const btn     = document.getElementById('predictBtn');
  const errEl   = document.getElementById('singleError');
  const resultEl = document.getElementById('singleResult');

  setLoading(btn, true);
  errEl.hidden = true;
  resultEl.hidden = true;

  const payload = {
    sex:        val('sex'),
    age:        num('age'),
    address:    val('address'),
    Medu:       num('Medu'),
    Fedu:       num('Fedu'),
    studytime:  num('studytime'),
    failures:   num('failures'),
    schoolsup:  val('schoolsup'),
    famsup:     val('famsup'),
    paid:       val('paid'),
    activities: val('activities'),
    nursery:    val('nursery'),
    higher:     val('higher'),
    absences:   num('absences'),
    G1:         num('G1'),
    G2:         num('G2'),
  };

  try {
    const res  = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    renderSingleResult(data);
  } catch (e) {
    showError(errEl, e.message);
  } finally {
    setLoading(btn, false);
  }
});

function renderSingleResult(data) {
  /* badge */
  const badge = document.getElementById('riskBadge');
  badge.textContent = data.risk_tier;
  badge.className   = `risk-badge ${data.risk_tier}`;

  /* probability number + bar */
  document.getElementById('probNum').textContent  = `${data.risk_prob}%`;
  const barColor = { HIGH: 'var(--high)', MEDIUM: 'var(--med)', LOW: 'var(--low)' }[data.risk_tier];
  const bar = document.getElementById('probBar');
  bar.style.width      = '0%';
  bar.style.background = barColor;
  setTimeout(() => { bar.style.width = `${data.risk_prob}%`; }, 50);

  /* SHAP */
  const shapContainer = document.getElementById('shapBars');
  shapContainer.innerHTML = '';
  const maxShap = Math.max(...data.shap_reasons.map(r => Math.abs(r.shap)));

  data.shap_reasons.forEach(r => {
    const pct  = ((Math.abs(r.shap) / maxShap) * 100).toFixed(1);
    const isPos = r.shap > 0;
    const row  = document.createElement('div');
    row.className = 'shap-row';
    row.innerHTML = `
      <div class="shap-meta">
        <span class="shap-name">${r.label}</span>
        <span class="shap-dir ${isPos ? 'pos' : 'neg'}">${r.direction}</span>
      </div>
      <div class="shap-track">
        <div class="shap-fill ${isPos ? 'pos' : 'neg'}" style="width:0%" data-w="${pct}%"></div>
      </div>`;
    shapContainer.appendChild(row);
  });

  /* animate SHAP bars after paint */
  requestAnimationFrame(() => {
    shapContainer.querySelectorAll('.shap-fill').forEach(el => {
      el.style.width = el.dataset.w;
    });
  });

  /* LLM */
  const llmCard = document.getElementById('llmCard');
  if (data.llm_summary) {
    document.getElementById('llmBody').textContent = data.llm_summary;
    llmCard.hidden = false;
  } else {
    llmCard.hidden = true;
  }

  const resultEl = document.getElementById('singleResult');
  resultEl.hidden = false;
  resultEl.classList.add('fade-in');
}

/* ══════════════════════════════════════════════════════════════
   BULK UPLOAD
══════════════════════════════════════════════════════════════ */

let allStudents = [];

/* file input label update */
document.getElementById('csvFile').addEventListener('change', e => {
  const name = e.target.files[0]?.name ?? 'Drop CSV here or click to browse';
  document.getElementById('fileText').textContent = name;
});

document.getElementById('bulkBtn').addEventListener('click', async () => {
  const file   = document.getElementById('csvFile').files[0];
  const btn    = document.getElementById('bulkBtn');
  const errEl  = document.getElementById('bulkError');
  const resEl  = document.getElementById('bulkResult');

  if (!file) { showError(errEl, 'Please select a CSV file first.'); return; }

  setLoading(btn, true);
  errEl.hidden = true;
  resEl.hidden  = true;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res  = await fetch(`${API}/predict/bulk`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    allStudents = data.students;
    renderBulkResult(data);
  } catch (e) {
    showError(errEl, e.message);
  } finally {
    setLoading(btn, false);
  }
});

document.getElementById('tierFilter').addEventListener('change', e => {
  renderTable(e.target.value);
});

function renderBulkResult(data) {
  /* stats */
  const statsRow = document.getElementById('statsRow');
  const stats = [
    { cls: 'total',  num: data.summary.total,  lbl: 'Total' },
    { cls: 'high',   num: data.summary.high,   lbl: 'High Risk' },
    { cls: 'medium', num: data.summary.medium, lbl: 'Medium Risk' },
    { cls: 'low',    num: data.summary.low,    lbl: 'Low Risk' },
  ];
  statsRow.innerHTML = stats.map(s => `
    <div class="stat-box ${s.cls} fade-in">
      <div class="stat-num">${s.num}</div>
      <div class="stat-lbl">${s.lbl}</div>
    </div>`).join('');

  renderTable('');

  const resEl = document.getElementById('bulkResult');
  resEl.hidden = false;
}

function renderTable(filter) {
  const rows = filter
    ? allStudents.filter(s => s.risk_tier === filter)
    : allStudents;

  document.getElementById('bulkBody').innerHTML = rows.map(s => `
    <tr>
      <td>${s.student_id}</td>
      <td style="font-family:var(--mono)">${s.risk_prob}%</td>
      <td><span class="tier-chip ${s.risk_tier}">${s.risk_tier}</span></td>
      <td>${s.G1}</td>
      <td>${s.G2}</td>
      <td>${s.absences}</td>
    </tr>`).join('');
}

/* ── HELPERS ───────────────────────────────────────────────── */

const val = id => document.getElementById(id).value;
const num = id => parseFloat(document.getElementById(id).value);

function setLoading(btn, on) {
  btn.disabled = on;
  btn.querySelector('.btn-text').hidden   = on;
  btn.querySelector('.btn-spinner').hidden = !on;
}

function showError(el, msg) {
  el.textContent = `⚠ ${msg}`;
  el.hidden = false;
}