/* ============================================================
   DartsCamera – app.js
   Full client: setup, game state, board canvas, manual entry,
   camera, Socket.IO real-time updates.
============================================================ */

'use strict';

// ── Dartboard constants (mirror of core/board.py) ──────────────────
const SEGMENTS       = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5];
const BULL_INNER_R   = 6.35  / 170;
const BULL_OUTER_R   = 15.9  / 170;
const TRIPLE_INNER_R = 99.0  / 170;
const TRIPLE_OUTER_R = 107.0 / 170;
const DOUBLE_INNER_R = 162.0 / 170;
const DOUBLE_OUTER_R = 1.0;
const SEG_ANGLE      = 2 * Math.PI / 20;
const SEG_HALF_ANGLE = Math.PI / 20;          // 9°
const START_ANGLE    = -Math.PI / 2;          // top (12 o'clock) in canvas coords

const CRICKET_NUMBERS = [20,19,18,17,16,15,25];

// ── Segment colours (even / odd position in SEGMENTS array) ────────
const SEG_EVEN = { single: '#1a1a1f', triple: '#1b6e3b', double: '#1b6e3b' };
const SEG_ODD  = { single: '#e8e0c8', triple: '#8b1a1a', double: '#8b1a1a' };

// ── State ──────────────────────────────────────────────────────────
let gameState       = null;
let currentMult     = 1;
let selectedMode    = '501';
let cameraAvailable = false;
let autoDetectActive = false;
let calMode         = false;   // true while 2-click manual calibration is active
let calStep         = 0;       // 0 = awaiting centre click, 1 = awaiting edge click
let calCentre       = null;    // {fx, fy, cssX, cssY} – first click in frame + css px

// ── DOM refs ───────────────────────────────────────────────────────
const modalSetup      = document.getElementById('modal-setup');
const appDiv          = document.getElementById('app');
const boardCanvas     = document.getElementById('board-canvas');
const ctx             = boardCanvas.getContext('2d');
const clickHint       = document.getElementById('click-hint');
const camFeed         = document.getElementById('camera-feed');
const camPip          = document.getElementById('cam-pip');
const noCameraMsg     = document.getElementById('no-camera-msg');
const camDot          = document.getElementById('cam-indicator');
const camWrap         = document.getElementById('cam-wrap');
const camOverlay      = document.getElementById('cam-overlay');
let   ovCtx           = null;
const winnerOverlay   = document.getElementById('winner-overlay');
const winnerNameEl    = document.getElementById('winner-name');
const toastContainer  = document.createElement('div');
toastContainer.className = 'toast-container';
document.body.appendChild(toastContainer);

// ── Socket.IO ──────────────────────────────────────────────────────
const socket = io();

socket.on('connect', () => console.log('[socket] connected'));

socket.on('camera_status', ({ available }) => {
  cameraAvailable = available;
  camDot.className = 'cam-dot ' + (available ? 'online' : 'offline');
  camDot.title = available ? 'Caméra connectée' : 'Caméra non disponible';
  if (available) {
    camFeed.src = '/video_feed';
    noCameraMsg.classList.add('hidden');
    camFeed.classList.remove('hidden');
    camPip.src = '/video_feed';
    camPip.classList.remove('hidden');
  } else {
    noCameraMsg.classList.remove('hidden');
    camFeed.classList.add('hidden');
    camPip.classList.add('hidden');
    camPip.src = '';
  }
});

socket.on('game_state', state => {
  gameState = state;
  renderAll();
});

socket.on('dart_thrown', result => {
  if (result.busted) showToast('Bust ! Score annulé', 'bust');
  else if (result.won) showWinner(result.player || gameState?.winner || '');
});

socket.on('turn_ended', result => {
  if (result.next_player) showToast(`Au tour de ${result.next_player}`, 'info');
});

socket.on('detection_result', ({ darts, error }) => {
  if (error) { showToast(error, 'bust'); return; }
  if (!darts) return;
  // Flash the camera border to signal detection
  triggerDetectFlash();
  // Only register darts BEYOND what's already counted this turn
  const already = gameState?.current_turn_darts?.length ?? 0;
  const newOnes = darts.slice(already);
  if (darts.length === 0 && !autoDetectActive) showToast('Aucune fléchette détectée', 'info');
  newOnes.forEach(d => apiThrow(d.score, d.label, d.x ?? null, d.y ?? null));
  if (newOnes.length > 0) setDetectState('detected', `${darts.length} fléchette${darts.length > 1 ? 's' : ''} détectée${darts.length > 1 ? 's' : ''}`);
});

socket.on('auto_detect_status', ({ active }) => {
  autoDetectActive = active;
  const btn = document.getElementById('btn-auto-detect');
  if (btn) btn.dataset.active = active ? 'true' : 'false';
  setDetectState(active ? 'listening' : 'idle', active ? 'Écoute en cours…' : 'Prêt');
});

socket.on('calibration_result', result => {
  const msg = document.getElementById('cal-status-text');
  if (result.error) {
    if (msg) msg.textContent = '⚠ ' + result.error;
    showToast('Calibration : ' + result.error, 'bust');
  } else {
    if (msg) msg.textContent = `Cible calibrée ✓  r=${result.radius}px`;
    showToast(`Calibration auto ✓  r=${result.radius}px`, 'info');
  }
});

// ============================================================
//   SETUP MODAL
// ============================================================

// Mode buttons
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedMode = btn.dataset.mode;
    const isX01 = selectedMode === '501' || selectedMode === '301';
    document.getElementById('x01-options').style.display = isX01 ? 'flex' : 'none';
  });
});

// Add player
function addPlayerRow(name = '', team = '') {
  const list = document.getElementById('player-list');
  const row  = document.createElement('div');
  row.className = 'player-input-row';
  row.innerHTML = `
    <input type="text" class="player-name-input" placeholder="Nom du joueur" value="${name}"/>
    <button class="btn-remove-player" title="Retirer">✕</button>`;
  row.querySelector('.btn-remove-player').addEventListener('click', () => {
    if (document.querySelectorAll('.player-input-row').length > 1) row.remove();
  });
  list.appendChild(row);
}
addPlayerRow('Joueur 1');

document.getElementById('btn-add-player').addEventListener('click', () => {
  addPlayerRow(`Joueur ${document.querySelectorAll('.player-input-row').length + 1}`);
});

document.getElementById('btn-start-game').addEventListener('click', startGame);

async function startGame() {
  const names = [...document.querySelectorAll('.player-name-input')]
    .map(i => i.value.trim()).filter(Boolean);
  if (names.length === 0) { showToast('Ajoutez au moins un joueur', 'bust'); return; }

  const payload = {
    mode:       selectedMode,
    players:    names,
    double_out: document.getElementById('opt-double-out').checked,
    double_in:  document.getElementById('opt-double-in').checked,
  };

  const res  = await apiFetch('/api/game/new', 'POST', payload);
  gameState  = res;
  modalSetup.classList.remove('active');
  appDiv.classList.remove('hidden');
  document.getElementById('hdr-mode').textContent = selectedMode.toUpperCase();
  requestAnimationFrame(() => resizeCanvas());
  renderAll();
}

// ============================================================
//   RENDER
// ============================================================

function renderAll() {
  if (!gameState) return;
  updateHeader();
  updateThrowsBar();
  updateCurrentPlayerCard();
  updateScoreboard();
  updateHistory();
  drawBoard();
}

function updateHeader() {
  document.getElementById('hdr-mode').textContent  = gameState.mode.toUpperCase();
  document.getElementById('hdr-round').textContent = `Round ${gameState.round}`;
}

function updateThrowsBar() {
  const darts  = gameState.current_turn_darts || [];
  const busted = gameState.current_turn_busted;
  let total = 0;
  for (let i = 1; i <= 3; i++) {
    const slot   = document.getElementById(`slot-${i}`);
    if (!slot) continue;
    const lblEl  = slot.querySelector('.ts-label');
    const ptsEl  = slot.querySelector('.ts-pts');
    const dart   = darts[i - 1];
    slot.className = 'throw-slot';
    if (dart) {
      lblEl.textContent = dart.label;
      ptsEl.textContent = dart.score + ' pts';
      total += dart.score;
      if (busted) {
        slot.classList.add('bust');
      } else if (dart.label?.startsWith('T')) {
        slot.classList.add('filled', 'triple');
      } else if (dart.label?.startsWith('D') && dart.label !== 'DB') {
        slot.classList.add('filled', 'double');
      } else {
        slot.classList.add('filled');
      }
    } else {
      lblEl.textContent = '–';
      ptsEl.textContent = '';
    }
  }
  const totalEl = document.getElementById('throw-total-val');
  const bustEl  = document.getElementById('throw-bust-label');
  if (totalEl) totalEl.textContent = `${total} pts`;
  if (bustEl)  bustEl.classList.toggle('hidden', !busted);
  if (totalEl) totalEl.classList.toggle('hidden', !!busted);
}

function updateCurrentPlayerCard() {
  const p = currentPlayerData();
  if (!p) return;

  document.getElementById('cp-name').textContent = p.name + (p.team ? ` · ${p.team}` : '');

  const scoreEl = document.getElementById('cp-score');
  if (gameState.mode === '501' || gameState.mode === '301') {
    scoreEl.textContent = p.remaining;
    scoreEl.style.color = gameState.current_turn_busted ? 'var(--warn)' : 'var(--accent)';
  } else if (gameState.mode === 'cricket') {
    scoreEl.textContent = p.cricket_score;
    scoreEl.style.color = 'var(--accent)';
  } else if (gameState.mode === 'around_the_clock') {
    scoreEl.textContent = `→ ${p.atc_target === 25 ? 'Bull' : p.atc_target}`;
    scoreEl.style.color = 'var(--accent)';
  } else {
    scoreEl.textContent = p.total_counted;
  }

  // Checkout suggestion
  const coEl = document.getElementById('cp-checkout');
  const co   = gameState.checkout_suggestion;
  if (co && co.length > 0) {
    coEl.innerHTML = co.map((d, i) => {
      const cls = i === co.length - 1 ? 'checkout-finish' : '';
      return `<span class="${cls}">${d}</span>`;
    }).join('<span class="checkout-arrow">→</span>');
  } else {
    coEl.innerHTML = '';
  }

  // Stats
  document.getElementById('cp-stats').textContent =
    `Moy/fléchette: ${p.avg_per_dart} | Moy/round: ${p.avg_per_round}`;
}

function currentPlayerData() {
  if (!gameState) return null;
  return gameState.players[gameState.current_player_idx] || null;
}

// ── Scoreboard ──────────────────────────────────────────────────────

function updateScoreboard() {
  const mode = gameState.mode;
  document.getElementById('scoreboard-x01').classList.toggle('hidden', mode === 'cricket' || mode === 'around_the_clock');
  document.getElementById('scoreboard-cricket').classList.toggle('hidden', mode !== 'cricket');
  document.getElementById('scoreboard-atc').classList.toggle('hidden', mode !== 'around_the_clock');

  if (mode === '501' || mode === '301' || mode === 'free') renderX01Scoreboard();
  else if (mode === 'cricket') renderCricketScoreboard();
  else if (mode === 'around_the_clock') renderAtcScoreboard();
}

function renderX01Scoreboard() {
  const el = document.getElementById('scoreboard-x01');
  el.innerHTML = '';
  gameState.players.forEach((p, idx) => {
    const active = idx === gameState.current_player_idx && !gameState.game_over;
    const winner = gameState.winner === p.name;
    const card   = document.createElement('div');
    card.className = 'player-score-card' + (active ? ' active-player' : '') + (winner ? ' winner-card' : '');

    const lastTurn = p.rounds_played > 0 ? '' : '';
    card.innerHTML = `
      <div class="psc-header">
        <span class="psc-name">${esc(p.name)}</span>
        ${p.team ? `<span class="psc-team">${esc(p.team)}</span>` : ''}
      </div>
      <div class="psc-remaining ${gameState.current_turn_busted && active ? 'bust' : ''}">
        ${gameState.mode !== 'free' ? p.remaining : p.total_counted + ' pts'}
      </div>
      <div class="psc-stats">
        Moy: ${p.avg_per_dart} / fléchette &nbsp;|&nbsp; ${p.total_darts} fléchettes
      </div>
      ${lastTurn}`;
    el.appendChild(card);
  });
}

function renderCricketScoreboard() {
  const el = document.getElementById('scoreboard-cricket');
  const ps = gameState.players;

  let html = `<table class="cricket-table">
    <thead><tr>
      <th>N°</th>
      ${ps.map(p => `<th>${esc(p.name)}</th>`).join('')}
    </tr></thead><tbody>`;

  CRICKET_NUMBERS.forEach(num => {
    html += `<tr><td class="number-col">${num === 25 ? 'Bull' : num}</td>`;
    ps.forEach(p => {
      const hits = p.cricket_hits?.[num] ?? 0;
      const marks = cricketMark(hits);
      const cls   = hits >= 3 ? ' closed' : '';
      html += `<td><span class="cricket-hit${cls}">${marks}</span></td>`;
    });
    html += '</tr>';
  });

  html += `</tbody><tfoot><tr><td>Score</td>
    ${ps.map(p => `<td style="color:var(--accent);font-weight:800">${p.cricket_score}</td>`).join('')}
  </tr></tfoot></table>`;

  // Active player indicator
  html += `<div style="margin-top:8px;font-size:11px;color:var(--muted)">
    Au tour de : <strong style="color:var(--text)">${esc(gameState.current_player)}</strong>
  </div>`;

  el.innerHTML = html;
}

function cricketMark(hits) {
  if (hits === 0) return '&nbsp;&nbsp;';
  if (hits === 1) return '/';
  if (hits === 2) return 'X';
  return '⊗';  // closed
}

function renderAtcScoreboard() {
  const el = document.getElementById('scoreboard-atc');
  el.innerHTML = '';
  gameState.players.forEach((p, idx) => {
    const active = idx === gameState.current_player_idx && !gameState.game_over;
    const card   = document.createElement('div');
    card.className = 'atc-card' + (active ? ' active-player' : '');
    card.innerHTML = `
      <div class="psc-name">${esc(p.name)}</div>
      <div class="atc-target-num">${p.atc_target === 25 ? 'Bull' : p.atc_target}</div>
      <div class="psc-stats">Cible suivante</div>`;
    el.appendChild(card);
  });
}

// ============================================================
//   DARTBOARD CANVAS
// ============================================================

function resizeCanvas() {
  const wrapper = boardCanvas.parentElement;
  const size    = Math.min(wrapper.clientWidth, wrapper.clientHeight) - 32;
  boardCanvas.width  = size;
  boardCanvas.height = size;
  drawBoard();
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawBoard() {
  const W  = boardCanvas.width;
  const H  = boardCanvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const R  = (Math.min(W, H) / 2) * 0.88;   // board drawing radius

  ctx.clearRect(0, 0, W, H);

  // outer miss ring
  ctx.fillStyle = '#111';
  ctx.beginPath();
  ctx.arc(cx, cy, R * 1.12, 0, Math.PI * 2);
  ctx.fill();

  // Draw segments
  for (let i = 0; i < 20; i++) {
    const angleStart = START_ANGLE + i * SEG_ANGLE - SEG_HALF_ANGLE;
    const angleEnd   = angleStart + SEG_ANGLE;
    const col        = i % 2 === 0 ? SEG_EVEN : SEG_ODD;

    // Double ring
    drawArc(cx, cy, R * DOUBLE_INNER_R, R * DOUBLE_OUTER_R, angleStart, angleEnd, col.double);
    // Outer single
    drawArc(cx, cy, R * TRIPLE_OUTER_R, R * DOUBLE_INNER_R, angleStart, angleEnd, col.single);
    // Triple ring
    drawArc(cx, cy, R * TRIPLE_INNER_R, R * TRIPLE_OUTER_R, angleStart, angleEnd, col.triple);
    // Inner single
    drawArc(cx, cy, R * BULL_OUTER_R,   R * TRIPLE_INNER_R, angleStart, angleEnd, col.single);
  }

  // Wire lines between segments
  ctx.strokeStyle = '#333';
  ctx.lineWidth   = 1;
  for (let i = 0; i < 20; i++) {
    const angle = START_ANGLE + i * SEG_ANGLE - SEG_HALF_ANGLE;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * R * BULL_OUTER_R,
               cy + Math.sin(angle) * R * BULL_OUTER_R);
    ctx.lineTo(cx + Math.cos(angle) * R * DOUBLE_OUTER_R,
               cy + Math.sin(angle) * R * DOUBLE_OUTER_R);
    ctx.stroke();
  }
  // Ring borders
  [BULL_OUTER_R, TRIPLE_INNER_R, TRIPLE_OUTER_R, DOUBLE_INNER_R, DOUBLE_OUTER_R].forEach(fr => {
    ctx.beginPath();
    ctx.arc(cx, cy, R * fr, 0, Math.PI * 2);
    ctx.stroke();
  });

  // Outer bull (25)
  ctx.fillStyle = '#2e7d32';
  ctx.beginPath();
  ctx.arc(cx, cy, R * BULL_OUTER_R, 0, Math.PI * 2);
  ctx.fill();

  // Inner bull (50)
  ctx.fillStyle = '#b71c1c';
  ctx.beginPath();
  ctx.arc(cx, cy, R * BULL_INNER_R, 0, Math.PI * 2);
  ctx.fill();

  // Segment numbers
  ctx.fillStyle   = '#fff';
  ctx.font        = `bold ${Math.max(10, R * 0.09)}px Segoe UI, sans-serif`;
  ctx.textAlign   = 'center';
  ctx.textBaseline= 'middle';
  for (let i = 0; i < 20; i++) {
    const angle = START_ANGLE + i * SEG_ANGLE;
    const nr    = R * 1.07;
    ctx.fillText(SEGMENTS[i], cx + Math.cos(angle) * nr, cy + Math.sin(angle) * nr);
  }

  // Checkout highlights
  highlightCheckout(cx, cy, R);

  // Dart positions
  drawDarts(cx, cy, R);
}

function drawArc(cx, cy, r1, r2, a1, a2, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, r2, a1, a2);
  ctx.arc(cx, cy, r1, a2, a1, true);
  ctx.closePath();
  ctx.fill();
}

// Highlight checkout target segments
function highlightCheckout(cx, cy, R) {
  if (!gameState) return;
  const co = gameState.checkout_suggestion;
  if (!co || co.length === 0) return;

  co.forEach((label, idx) => {
    const isFinish = idx === co.length - 1;
    const color    = isFinish ? 'rgba(255,107,53,0.55)' : 'rgba(0,229,255,0.35)';
    const [r1, r2, aStart, aEnd] = labelToArc(label, R);
    if (r1 === null) return;

    ctx.save();
    ctx.fillStyle = color;
    if (aStart !== null) {
      ctx.beginPath();
      ctx.arc(cx, cy, r2, aStart, aEnd);
      ctx.arc(cx, cy, r1, aEnd, aStart, true);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, r2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  });
}

function labelToArc(label, R) {
  // Returns [r_inner, r_outer, angle_start, angle_end] or [null,...]
  if (label === 'DB') return [0, R * BULL_INNER_R, null, null];
  if (label === 'B')  return [R * BULL_INNER_R, R * BULL_OUTER_R, null, null];
  if (label === 'MISS') return [null, null, null, null];

  const prefix = label[0];
  const num    = parseInt(label.slice(1), 10);
  if (isNaN(num)) return [null, null, null, null];

  const segIdx = SEGMENTS.indexOf(num);
  if (segIdx === -1) return [null, null, null, null];

  const aCenter = START_ANGLE + segIdx * SEG_ANGLE;
  const a1 = aCenter - SEG_HALF_ANGLE;
  const a2 = aCenter + SEG_HALF_ANGLE;

  const radii = {
    D: [R * DOUBLE_INNER_R, R * DOUBLE_OUTER_R],
    T: [R * TRIPLE_INNER_R, R * TRIPLE_OUTER_R],
    S: [R * BULL_OUTER_R,   R * TRIPLE_INNER_R],
  };
  const [r1, r2] = radii[prefix] || radii.S;
  return [r1, r2, a1, a2];
}

// Draw current turn darts on canvas
function drawDarts(cx, cy, R) {
  if (!gameState) return;
  const darts = gameState.current_turn_darts || [];
  const colors = ['#00e5ff', '#7c4dff', '#ff6b35'];

  darts.forEach((d, i) => {
    if (d.x == null || d.y == null) return;
    const px = cx + d.x * R;
    const py = cy + d.y * R;

    ctx.save();
    ctx.shadowBlur  = 12;
    ctx.shadowColor = colors[i];
    ctx.fillStyle   = colors[i];
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.fillStyle  = '#fff';
    ctx.font       = 'bold 9px sans-serif';
    ctx.textAlign  = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i + 1, px, py);
    ctx.restore();
  });
}

// ── Click-to-score ─────────────────────────────────────────────────
boardCanvas.addEventListener('click', e => {
  if (!gameState || gameState.game_over) return;
  if (gameState.current_turn_darts?.length >= 3) {
    showToast('Ce tour a déjà 3 fléchettes. Validez le tour.', 'bust');
    return;
  }

  const rect  = boardCanvas.getBoundingClientRect();
  const W = boardCanvas.width;
  const H = boardCanvas.height;
  const scaleX = W / rect.width;
  const scaleY = H / rect.height;
  const px = (e.clientX - rect.left)  * scaleX;
  const py = (e.clientY - rect.top)   * scaleY;
  const cx = W / 2, cy = H / 2;
  const R  = (Math.min(W, H) / 2) * 0.88;

  const { score, label } = pixelToScore(px, py, cx, cy, R);
  const xn = (px - cx) / R;
  const yn = (py - cy) / R;
  apiThrow(score, label, xn, yn);
});

function pixelToScore(px, py, cx, cy, R) {
  const dx = px - cx, dy = py - cy;
  const rn = Math.sqrt(dx*dx + dy*dy) / R;

  if (rn <= BULL_INNER_R)  return { score: 50, label: 'DB' };
  if (rn <= BULL_OUTER_R)  return { score: 25, label: 'B'  };

  let angle = Math.atan2(dx, -dy) * 180 / Math.PI;
  if (angle < 0) angle += 360;
  const segIdx = Math.floor((angle + 9) / 18) % 20;
  const seg    = SEGMENTS[segIdx];

  if (rn <= TRIPLE_INNER_R) return { score: seg,     label: `S${seg}` };
  if (rn <= TRIPLE_OUTER_R) return { score: seg * 3, label: `T${seg}` };
  if (rn <= DOUBLE_INNER_R) return { score: seg,     label: `S${seg}` };
  if (rn <= DOUBLE_OUTER_R) return { score: seg * 2, label: `D${seg}` };
  return { score: 0, label: 'MISS' };
}

// ============================================================
//   MANUAL ENTRY
// ============================================================

// Build number grid
const numGrid = document.getElementById('number-grid');
for (let n = 1; n <= 20; n++) {
  const btn = document.createElement('button');
  btn.className   = 'num-btn';
  btn.textContent = n;
  btn.addEventListener('click', () => {
    const score = currentMult * n;
    const prefix = ['S','D','T'][currentMult - 1];
    const label  = `${prefix}${n}`;
    apiThrow(score, label, null, null);
  });
  numGrid.appendChild(btn);
}

// Multiplier buttons
document.querySelectorAll('.mult-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mult-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMult = parseInt(btn.dataset.mult, 10);
  });
});

// Special buttons (Bull, Bullseye, Miss)
document.querySelectorAll('.spec-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const score = parseInt(btn.dataset.score, 10);
    const label = btn.dataset.label;
    apiThrow(score, label, null, null);
  });
});

// End turn / Undo
document.getElementById('btn-end-turn').addEventListener('click', apiEndTurn);
document.getElementById('btn-undo').addEventListener('click', apiUndo);

// New game
document.getElementById('btn-new-game').addEventListener('click', () => {
  // Stop auto-detect if running
  if (autoDetectActive) {
    apiFetch('/api/camera/auto_detect', 'POST', { active: false });
  }
  appDiv.classList.add('hidden');
  modalSetup.classList.add('active');
  gameState = null;
});
document.getElementById('btn-play-again').addEventListener('click', () => {
  winnerOverlay.classList.add('hidden');
  appDiv.classList.add('hidden');
  modalSetup.classList.add('active');
  gameState = null;
});

// ============================================================
//   TABS
// ============================================================
document.querySelectorAll('.tab-bar .tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab-bar .tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const id = tab.dataset.tab;
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
  });
});

// ============================================================
//   CAMERA OVERLAY – manual calibration helpers
// ============================================================

function initOverlay() {
  if (!camOverlay || !camWrap) return;
  camOverlay.width  = camWrap.offsetWidth  || camWrap.clientWidth;
  camOverlay.height = camWrap.offsetHeight || camWrap.clientHeight;
  ovCtx = camOverlay.getContext('2d');
}

function clearOverlay() {
  if (!ovCtx) initOverlay();
  if (ovCtx) ovCtx.clearRect(0, 0, camOverlay.width, camOverlay.height);
}

// Convert a mouse event on #cam-wrap to video-frame pixel coords.
function clickToFrame(e) {
  const rect  = camWrap.getBoundingClientRect();
  const cssX  = e.clientX - rect.left;
  const cssY  = e.clientY - rect.top;
  const frameW = camFeed.naturalWidth  || 1280;
  const frameH = camFeed.naturalHeight || 720;
  return {
    fx:   cssX / rect.width  * frameW,
    fy:   cssY / rect.height * frameH,
    cssX, cssY,
    scaleX: rect.width  / frameW,
    scaleY: rect.height / frameH,
  };
}

function drawCrossHair(cx, cy) {
  if (!ovCtx) return;
  ovCtx.save();
  ovCtx.strokeStyle = '#00e5ff';
  ovCtx.lineWidth   = 2;
  ovCtx.shadowColor = '#00e5ff';
  ovCtx.shadowBlur  = 6;
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 22, cy); ovCtx.lineTo(cx + 22, cy);
  ovCtx.moveTo(cx, cy - 22); ovCtx.lineTo(cx, cy + 22);
  ovCtx.stroke();
  ovCtx.beginPath();
  ovCtx.arc(cx, cy, 7, 0, Math.PI * 2);
  ovCtx.stroke();
  ovCtx.restore();
}

function drawRadiusPreview(cx, cy, rCss, labelText) {
  if (!ovCtx) return;
  ovCtx.save();
  ovCtx.strokeStyle = '#ff6b35';
  ovCtx.lineWidth   = 2;
  ovCtx.setLineDash([5, 5]);
  ovCtx.shadowColor = '#ff6b35';
  ovCtx.shadowBlur  = 6;
  ovCtx.beginPath();
  ovCtx.arc(cx, cy, rCss, 0, Math.PI * 2);
  ovCtx.stroke();
  ovCtx.setLineDash([]);
  ovCtx.fillStyle = '#ff6b35';
  ovCtx.font      = 'bold 12px monospace';
  ovCtx.fillText(labelText, cx + rCss * 0.72 + 6, cy - rCss * 0.72 - 4);
  ovCtx.restore();
}

function cancelCal() {
  calMode = false;
  calStep = 0;
  calCentre = null;
  clearOverlay();
  if (camWrap) camWrap.style.cursor = '';
  const msg = document.getElementById('cal-status-text');
  if (msg) msg.textContent = '';
}

// ESC cancels calibration
document.addEventListener('keydown', e => { if (e.key === 'Escape') cancelCal(); });

// ── Live radius preview while hovering after centre click ────────
camWrap?.addEventListener('mousemove', e => {
  if (!calMode || calStep !== 1 || !calCentre) return;
  const { fx, fy, cssX, cssY, scaleX, scaleY } = clickToFrame(e);
  const rFr  = Math.hypot(fx - calCentre.fx, fy - calCentre.fy);
  const rCss = Math.hypot(cssX - calCentre.cssX, cssY - calCentre.cssY);
  clearOverlay();
  drawCrossHair(calCentre.cssX, calCentre.cssY);
  drawRadiusPreview(calCentre.cssX, calCentre.cssY, rCss, `r=${Math.round(rFr)}px`);
});

// ── Click handler: 2-click calibration ───────────────────────────
camWrap?.addEventListener('click', async e => {
  if (!calMode) return;
  const { fx, fy, cssX, cssY } = clickToFrame(e);
  const msg = document.getElementById('cal-status-text');

  if (calStep === 0) {
    calCentre = { fx, fy, cssX, cssY };
    calStep   = 1;
    if (msg) msg.textContent = '2/2 : Cliquez sur le bord extérieur (double ring)…';
    clearOverlay();
    drawCrossHair(cssX, cssY);
  } else {
    const saved = { ...calCentre };   // copy before cancelCal clears it
    const rFr   = Math.hypot(fx - saved.fx, fy - saved.fy);
    cancelCal();  // clears overlay and resets state
    const r = await apiFetch('/api/camera/calibrate_manual', 'POST', {
      center_x: saved.fx,
      center_y: saved.fy,
      radius:   rFr,
    });
    if (msg) msg.textContent = r.error ? '⚠ ' + r.error : `Calibré ✓  r=${Math.round(rFr)}px`;
    showToast(r.error ? r.error : `Calibré ✓  r=${Math.round(rFr)}px`, r.error ? 'bust' : 'info');
  }
});

// ============================================================
//   CAMERA CONTROLS
// ============================================================

document.getElementById('btn-set-bg').addEventListener('click', async () => {
  const r = await apiFetch('/api/camera/background', 'POST');
  const msg = document.getElementById('cal-status-text');
  if (msg) msg.textContent = r.error ? '' : 'Fond défini ✓';
  showToast(r.error ? r.error : 'Fond défini ✓', r.error ? 'bust' : 'info');
});

// Calibrer = start 2-click manual calibration mode
document.getElementById('btn-calibrate').addEventListener('click', () => {
  initOverlay();
  calMode  = true;
  calStep  = 0;
  calCentre = null;
  clearOverlay();
  camWrap.style.cursor = 'crosshair';
  const msg = document.getElementById('cal-status-text');
  if (msg) msg.textContent = '1/2 : Cliquez sur le centre (bull)…';
  // Switch to camera tab if not already visible
  const camTab = document.querySelector('.tab[data-tab="camera"]');
  if (camTab && !camTab.classList.contains('active')) camTab.click();
});

document.getElementById('btn-detect').addEventListener('click', async () => {
  const r = await apiFetch('/api/camera/detect', 'GET');
  if (r.error) showToast(r.error, 'bust');
  else showToast('Détection lancée…', 'info');
});
document.getElementById('btn-auto-detect').addEventListener('click', async () => {
  const r = await apiFetch('/api/camera/auto_detect', 'POST');
  if (r.error) showToast(r.error, 'bust');
});

window.addEventListener('resize', () => { if (calMode) initOverlay(); });


// ============================================================
//   API HELPERS
// ============================================================

async function apiThrow(score, label, xn, yn) {
  const body = { score, label, x_norm: xn, y_norm: yn };
  const res  = await apiFetch('/api/game/throw', 'POST', body);
  if (res.error) showToast(res.error, 'bust');
  // state update comes via socket
}

async function apiEndTurn() {
  const res = await apiFetch('/api/game/end_turn', 'POST');
  if (res.error) showToast(res.error, 'bust');
}

async function apiUndo() {
  const res = await apiFetch('/api/game/undo', 'POST');
  if (res.error) showToast(res.error, 'bust');
  else showToast(`Annulé : ${res.result?.undone}`, 'info');
}

// ============================================================
//   HISTORY
// ============================================================

function updateHistory() {
  const el = document.getElementById('hist-rows');
  if (!el || !gameState) return;
  const players = gameState.players || [];
  // Collect all recent turns from all players
  const rows = [];
  players.forEach(p => {
    (p.recent_turns || []).forEach(t => {
      rows.push({ player: p.name, turn: t });
    });
  });
  // Show last 8, most-recent first
  el.innerHTML = '';
  rows.slice(-8).reverse().forEach(({ player, turn }) => {
    const div = document.createElement('div');
    div.className = 'hist-row' + (turn.busted ? ' busted' : '');
    const labels = (turn.darts || []).map(d => d.label).join('  ');
    div.innerHTML = `
      <span class="hist-player">${esc(player.slice(0, 10))}</span>
      <span class="hist-throws">${labels || '–'}</span>
      <span class="hist-total">${turn.busted ? 'BUST' : turn.scored}</span>`;
    el.appendChild(div);
  });
}

// ============================================================
//   DETECT STATE UI
// ============================================================

function setDetectState(state, text) {
  const el    = document.getElementById('detect-state');
  const label = document.getElementById('ds-label');
  if (el)    el.className = 'detect-state ' + state;
  if (label) label.textContent = text;
}

function triggerDetectFlash() {
  const el = document.getElementById('detect-flash');
  if (!el) return;
  el.classList.remove('active');
  void el.offsetWidth; // force reflow
  el.classList.add('active');
  setTimeout(() => el.classList.remove('active'), 700);
}


async function apiFetch(url, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  try {
    const r = await fetch(url, opts);
    return await r.json();
  } catch (e) {
    console.error(e);
    return { error: 'Erreur réseau' };
  }
}

// ============================================================
//   WINNER
// ============================================================

function showWinner(name) {
  winnerNameEl.textContent = name || gameState?.winner || '?';
  winnerOverlay.classList.remove('hidden');
}

// ============================================================
//   TOASTS
// ============================================================

function showToast(msg, type = 'info') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  toastContainer.appendChild(t);
  requestAnimationFrame(() => {
    t.classList.add('show');
    setTimeout(() => {
      t.classList.remove('show');
      setTimeout(() => t.remove(), 350);
    }, 2500);
  });
}

// ============================================================
//   UTILITY
// ============================================================

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Fetch initial state on load (if a game is already running server-side)
(async () => {
  const st = await apiFetch('/api/game/state');
  if (!st.error) {
    gameState = st;
    modalSetup.classList.remove('active');
    appDiv.classList.remove('hidden');
    requestAnimationFrame(() => resizeCanvas());
    renderAll();
  }
})();
