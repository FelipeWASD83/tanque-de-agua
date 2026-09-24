// Liga tudo: escolhe o motor (GPU ou CPU), monta o tanque do tamanho da
// janela, transforma mouse/dedo em paredes e água, e roda o relógio.

import {
  METROS_POR_PX, QUALIDADES, CORANTES, PARTICULAS_POR_CELULA, PROFUNDIDADE, MOLDURA,
  PINCEL_MIN_CELULAS, PINCEL_MAX_CELULAS, GRAVIDADE, dimensoesDaGrade, passoMaximo,
} from './constantes.js';
import { Cena } from './cena.js';
import { CAMPOS, semearRetangulo } from './fisica-cpu.js';
import { MODELOS, ORDEM_MODELOS, semearAgua } from './modelos.js';
import { abrirGPU, MotorGPU, MotorCPU } from './motor.js';

const $ = (s) => document.querySelector(s);
const parametros = new URLSearchParams(location.search);

// ------------------------------------------------------------ preferências

const guardado = {
  ler(chave, padrao) {
    try {
      const v = localStorage.getItem('tanque:' + chave);
      return v === null ? padrao : JSON.parse(v);
    } catch { return padrao; }
  },
  gravar(chave, valor) {
    try { localStorage.setItem('tanque:' + chave, JSON.stringify(valor)); return true; } catch { return false; }
  },
  remover(chave) {
    try { localStorage.removeItem('tanque:' + chave); } catch { /* sem armazenamento */ }
  },
};

const estado = {
  ferramenta: guardado.ler('ferramenta', 'parede'),
  tamanho: guardado.ler('tamanho', 35),
  cor: guardado.ler('cor', 'limpa'),
  qualidade: guardado.ler('qualidade', 'alto'),
  pausado: false,
  lento: false,
  congelado: false, // só para medir pelo console: para o relógio inteiro
};
if (!CORANTES[estado.cor]) estado.cor = 'limpa';

// ------------------------------------------------------------ motor e tanque

let canvas = $('#tanque');
const guia = $('#guia');
const gctx = guia.getContext('2d');
let gpu = null;
let cena = null;
let motor = null;
let reconstruindo = false;
const desfazer = [];

/** Tamanho da área do tanque (o canvas; no celular ele para acima das barras de baixo). */
function tamanhoDaTela() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return { cssW: canvas.clientWidth, cssH: canvas.clientHeight, dpr };
}

/**
 * Monta (ou remonta) o tanque do tamanho da janela, mantendo paredes e água.
 * Janela sem tamanho (aba escondida, painel fechado) não monta: fica esperando
 * o próximo redimensionamento. Devolve se montou.
 */
let montagemPendente = null;
function montar({ tracos = [], particulas = null, hAntigo = null } = {}) {
  const { cssW, cssH, dpr } = tamanhoDaTela();
  if (cssW < 16 || cssH < 16) {
    montagemPendente = { tracos, particulas, hAntigo };
    return false;
  }
  montagemPendente = null;
  for (const c of [canvas, guia]) {
    c.width = Math.round(cssW * dpr);
    c.height = Math.round(cssH * dpr);
  }
  const qualidade = gpu ? estado.qualidade : 'cpu';
  const { nx, ny, h } = dimensoesDaGrade(cssW * METROS_POR_PX, cssH * METROS_POR_PX, qualidade);
  cena = new Cena(nx, ny, h);
  if (tracos.length) {
    cena.refazer(tracos);
    cena.atualizar();
  }
  motor?.destruir();
  motor = gpu ? new MotorGPU(gpu, canvas, cena) : new MotorCPU(canvas, cena);
  if (particulas && particulas.length) motor.adicionar(reamostrar(particulas, hAntigo ?? h, h), false);
  atualizarRegua();
  $('#motor').textContent = gpu ? `GPU · ${gpu.nome}` : 'CPU';
  return true;
}

/** Ajusta a quantidade de partículas quando a célula muda de tamanho, e descarta as de fora. */
function reamostrar(lista, hAntigo, hNovo) {
  const razao = (hAntigo / hNovo) ** 2;
  const n = lista.length / CAMPOS;
  const saida = [];
  const W = cena.largura, H = cena.altura, m = MOLDURA * hNovo;
  for (let k = 0; k < n; k++) {
    const o = k * CAMPOS;
    let copias = Math.floor(razao) + (Math.random() < razao % 1 ? 1 : 0);
    for (let c = 0; c < copias; c++) {
      const p = lista.slice(o, o + CAMPOS);
      if (c > 0) { p[0] += (Math.random() - 0.5) * hNovo * 0.5; p[1] += (Math.random() - 0.5) * hNovo * 0.5; }
      if (p[0] <= m || p[1] <= m || p[0] >= W - m || p[1] >= H - m) continue;
      saida.push(p);
    }
  }
  const junta = new Float32Array(saida.length * CAMPOS);
  saida.forEach((p, k) => junta.set(p, k * CAMPOS));
  return junta;
}

async function remontar(opcoes = {}) {
  if (montagemPendente) { montar(montagemPendente); return; }
  if (reconstruindo || !motor) return;
  reconstruindo = true;
  try {
    const particulas = await motor.particulas();
    const tracos = cena.tracos.slice();
    const hAntigo = cena.h;
    montar({ tracos, particulas, hAntigo, ...opcoes });
  } finally {
    reconstruindo = false;
  }
}

function atualizarRegua() {
  const el = $('#regua');
  el.textContent = '';
  const { cssW, cssH } = tamanhoDaTela();
  const m = MOLDURA * cena.h;
  // a interface dos cantos começa depois da moldura do tanque
  document.documentElement.style.setProperty('--moldura', `${(m / cena.altura) * cssH}px`);
  const x = ((m + 0.017) / cena.largura) * cssW;
  for (let k = 1; m + k * 0.1 < cena.altura - m - 0.02; k++) {
    const y = (1 - (m + k * 0.1) / cena.altura) * cssH;
    const s = document.createElement('span');
    s.textContent = String(k * 10);
    s.style.left = `${x}px`;
    s.style.top = `${y}px`;
    el.append(s);
  }
}

// ------------------------------------------------------------ ferramentas

const DICAS = {
  parede: 'Arraste para desenhar parede · Shift faz linha reta · Ctrl+Z desfaz',
  reta: 'Arraste de uma ponta à outra',
  borracha: 'Passe por cima da parede para apagar',
  torneira: 'Segure para despejar · mexa o mouse para jogar o jato',
  balde: 'Arraste um retângulo: ele vira um bloco de água',
  esponja: 'Segure em cima da água para tirar',
  mao: 'Arraste dentro da água para empurrar',
  corante: 'Passe por cima da água para tingir',
};

/** Raio do pincel de parede, em metros. */
function raioParede() {
  const f = estado.tamanho / 100;
  return (PINCEL_MIN_CELULAS + (PINCEL_MAX_CELULAS - PINCEL_MIN_CELULAS) * f * f) * cena.h;
}
/** Raio das ferramentas de água, em metros. */
function raioAgua() {
  const f = estado.tamanho / 100;
  const base = estado.ferramenta === 'torneira' ? 1.5 + 7 * f : 3 + 22 * f;
  return base * cena.h;
}
function raioDaFerramenta() {
  return ['parede', 'reta', 'borracha'].includes(estado.ferramenta) ? raioParede() : raioAgua();
}

const ponteiro = {
  dentro: false, apertado: false,
  x: 0, y: 0, // metros
  vx: 0, vy: 0, // m/s, suavizada
  t: 0,
  inicio: null, // para reta e balde
  anterior: null, // último ponto do traço
  reta: false,
};

function paraMundo(ev) {
  const r = canvas.getBoundingClientRect();
  return [((ev.clientX - r.left) / r.width) * cena.largura, (1 - (ev.clientY - r.top) / r.height) * cena.altura];
}

function guardarDesfazer() {
  desfazer.push(cena.tracos.slice());
  if (desfazer.length > 60) desfazer.shift();
}

let salvarDepois = 0;
function salvarParedes() {
  clearTimeout(salvarDepois);
  salvarDepois = setTimeout(() => {
    const t = cena.tracos.map((q) => [q.x0, q.y0, q.x1, q.y1, q.raio, q.apagar ? 1 : 0].map((v) => Math.round(v * 1e5) / 1e5));
    // Grande demais (ou armazenamento cheio): apaga o que estava guardado, para
    // não voltar um desenho velho na próxima visita.
    if (t.length > 40000 || !guardado.gravar('paredes', t)) guardado.remover('paredes');
  }, 400);
}

function carregarParedesGuardadas() {
  const t = guardado.ler('paredes', null);
  if (!Array.isArray(t)) return null;
  return t.map(([x0, y0, x1, y1, raio, apagar]) => ({ x0, y0, x1, y1, raio, apagar: !!apagar }));
}

function tracar(x0, y0, x1, y1) {
  cena.tracar(x0, y0, x1, y1, raioParede(), estado.ferramenta === 'borracha');
}

canvas.addEventListener('pointerdown', aoApertar);
function aoApertar(ev) {
  if (!motor) return;
  if (ev.button !== 0 && ev.pointerType === 'mouse') return;
  try { canvas.setPointerCapture(ev.pointerId); } catch { /* ponteiro sintético ou já solto */ }
  const [x, y] = paraMundo(ev);
  Object.assign(ponteiro, { apertado: true, dentro: true, x, y, vx: 0, vy: 0, t: performance.now() });
  const f = estado.ferramenta;
  if (f === 'parede' || f === 'borracha') {
    guardarDesfazer();
    ponteiro.reta = f === 'parede' && ev.shiftKey;
    if (ponteiro.reta) ponteiro.inicio = [x, y];
    else { tracar(x, y, x, y); ponteiro.anterior = [x, y]; }
  } else if (f === 'reta' || f === 'balde') {
    ponteiro.inicio = [x, y];
  }
}

canvas.addEventListener('pointermove', (ev) => {
  const eventos = ev.getCoalescedEvents?.() ?? [ev];
  for (const e of eventos.length ? eventos : [ev]) mover(e);
});
function mover(ev) {
  if (!cena) return;
  const [x, y] = paraMundo(ev);
  const agora = performance.now();
  const dt = Math.max(1, agora - ponteiro.t) / 1000;
  if (ponteiro.dentro) {
    const a = Math.min(1, dt * 25);
    ponteiro.vx += ((x - ponteiro.x) / dt - ponteiro.vx) * a;
    ponteiro.vy += ((y - ponteiro.y) / dt - ponteiro.vy) * a;
  }
  Object.assign(ponteiro, { x, y, t: agora, dentro: true });
  if (!ponteiro.apertado) return;
  const f = estado.ferramenta;
  if ((f === 'parede' || f === 'borracha') && !ponteiro.reta && ponteiro.anterior) {
    const [ax, ay] = ponteiro.anterior;
    if (Math.hypot(x - ax, y - ay) > cena.h * 0.35) {
      tracar(ax, ay, x, y);
      ponteiro.anterior = [x, y];
    }
  }
}

canvas.addEventListener('pointerup', soltar);
canvas.addEventListener('pointercancel', soltar);
function soltar(ev) {
  if (!ponteiro.apertado) return;
  const [x, y] = paraMundo(ev);
  const f = estado.ferramenta;
  if ((f === 'reta' || (f === 'parede' && ponteiro.reta)) && ponteiro.inicio) {
    if (f === 'reta') guardarDesfazer();
    const [ax, ay] = ponteiro.inicio;
    tracar(ax, ay, x, y);
  } else if (f === 'balde' && ponteiro.inicio) {
    const [ax, ay] = ponteiro.inicio;
    if (Math.abs(x - ax) > cena.h && Math.abs(y - ay) > cena.h) {
      motor.adicionar(semearRetangulo(ax, ay, x, y, cena.h, CORANTES[estado.cor].absorcao), true);
    }
  }
  if (['parede', 'reta', 'borracha'].includes(f)) salvarParedes();
  Object.assign(ponteiro, { apertado: false, inicio: null, anterior: null, reta: false });
}

canvas.addEventListener('pointerleave', () => { if (!ponteiro.apertado) ponteiro.dentro = false; });

/** O que as ferramentas contínuas fazem neste quadro. */
function ferramentaDoQuadro(dtSim) {
  if (!ponteiro.apertado) return null;
  const f = estado.ferramenta;
  const raio = raioAgua();
  if (f === 'torneira') {
    if (dtSim > 0) emitirTorneira(raio, dtSim);
    return null;
  }
  if (f === 'esponja') return { tipo: 'esponja', x: ponteiro.x, y: ponteiro.y, raio };
  if (f === 'mao') {
    const lim = 6;
    const v = Math.hypot(ponteiro.vx, ponteiro.vy);
    const k = v > lim ? lim / v : 1;
    return { tipo: 'mao', x: ponteiro.x, y: ponteiro.y, raio, vx: ponteiro.vx * k, vy: ponteiro.vy * k };
  }
  if (f === 'corante') return { tipo: 'corante', x: ponteiro.x, y: ponteiro.y, raio, absorcao: CORANTES[estado.cor].absorcao };
  return null;
}

function emitirTorneira(raio, dt) {
  const h = cena.h;
  let vx = ponteiro.vx * 0.5, vy = ponteiro.vy * 0.5 - 0.7;
  const v = Math.hypot(vx, vy);
  if (v > 4) { vx *= 4 / v; vy *= 4 / v; }
  const vazao = 2 * raio * Math.max(0.7, Math.hypot(vx, vy)); // m²/s
  const n = Math.min(40000, Math.round((vazao * dt) / (h * h) * PARTICULAS_POR_CELULA));
  if (n <= 0) return;
  const lista = new Float32Array(n * CAMPOS);
  const abs = CORANTES[estado.cor].absorcao;
  for (let k = 0; k < n; k++) {
    const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * raio;
    const o = k * CAMPOS;
    lista[o] = ponteiro.x + Math.cos(a) * r;
    lista[o + 1] = ponteiro.y + Math.sin(a) * r;
    lista[o + 2] = vx;
    lista[o + 3] = vy;
    lista[o + 9] = abs[0]; lista[o + 10] = abs[1]; lista[o + 11] = abs[2];
  }
  motor.adicionar(lista, true);
}

// ------------------------------------------------------------ guia (cursor e prévia)

function desenharGuia() {
  const { dpr } = tamanhoDaTela();
  gctx.setTransform(1, 0, 0, 1, 0, 0);
  gctx.clearRect(0, 0, guia.width, guia.height);
  if (!ponteiro.dentro && !ponteiro.apertado) return;
  const sx = guia.width / cena.largura, sy = guia.height / cena.altura;
  const tela = (x, y) => [x * sx, (cena.altura - y) * sy];
  const [cx, cy] = tela(ponteiro.x, ponteiro.y);
  const f = estado.ferramenta;
  const laranja = '#d4531c';
  gctx.lineWidth = 1.5 * dpr;

  if ((f === 'reta' || (f === 'parede' && ponteiro.reta)) && ponteiro.inicio) {
    const [ax, ay] = tela(...ponteiro.inicio);
    gctx.strokeStyle = 'rgba(212, 83, 28, 0.35)';
    gctx.lineCap = 'round';
    gctx.lineWidth = raioParede() * 2 * sx;
    gctx.beginPath(); gctx.moveTo(ax, ay); gctx.lineTo(cx, cy); gctx.stroke();
    gctx.lineWidth = 1.5 * dpr;
  }
  if (f === 'balde' && ponteiro.inicio) {
    const [ax, ay] = tela(...ponteiro.inicio);
    gctx.setLineDash([6 * dpr, 4 * dpr]);
    gctx.strokeStyle = '#2b7f95';
    gctx.fillStyle = 'rgba(46, 143, 166, 0.12)';
    gctx.fillRect(ax, ay, cx - ax, cy - ay);
    gctx.strokeRect(ax, ay, cx - ax, cy - ay);
    gctx.setLineDash([]);
  }
  // círculo do pincel
  const r = raioDaFerramenta() * sx;
  gctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  gctx.lineWidth = 3 * dpr;
  gctx.beginPath(); gctx.arc(cx, cy, Math.max(r, 3 * dpr), 0, Math.PI * 2); gctx.stroke();
  gctx.strokeStyle = laranja;
  gctx.lineWidth = 1.5 * dpr;
  gctx.beginPath(); gctx.arc(cx, cy, Math.max(r, 3 * dpr), 0, Math.PI * 2); gctx.stroke();
  gctx.fillStyle = laranja;
  gctx.beginPath(); gctx.arc(cx, cy, 1.5 * dpr, 0, Math.PI * 2); gctx.fill();
}

// ------------------------------------------------------------ modelos

function carregarModelo(nome) {
  const m = MODELOS[nome];
  if (!m || !motor) return;
  guardarDesfazer();
  const montado = m.montar(cena.largura, cena.altura, cena.h);
  cena.limpar();
  for (const t of montado.tracos) cena.tracar(t.x0, t.y0, t.x1, t.y1, t.raio);
  cena.atualizar();
  motor.paredesMudaram();
  motor.limparAgua();
  const abs = CORANTES[estado.cor].absorcao;
  for (const a of montado.agua) motor.adicionar(semearAgua(a, cena.h, abs), false);
  salvarParedes();
}

// ------------------------------------------------------------ interface

let tanqueVazio = false;
function mostrarDica() {
  // a água não fica guardada entre uma visita e outra: tanque vazio ganha instrução
  const vazio = tanqueVazio && !['torneira', 'balde'].includes(estado.ferramenta);
  const dica = $('#dica');
  dica.textContent = vazio ? 'Tanque sem água · Torneira (4) ou Balde (5) para pôr, ou um cenário em Montar' : DICAS[estado.ferramenta] ?? '';
  dica.classList.toggle('dica--vazio', vazio);
}

function escolherFerramenta(nome) {
  estado.ferramenta = nome;
  guardado.gravar('ferramenta', nome);
  for (const b of document.querySelectorAll('[data-ferramenta]')) b.setAttribute('aria-checked', String(b.dataset.ferramenta === nome));
  mostrarDica();
}

function escolherCor(nome) {
  estado.cor = nome;
  guardado.gravar('cor', nome);
  for (const b of document.querySelectorAll('.cor')) b.setAttribute('aria-checked', String(b.dataset.cor === nome));
}

function alternarPausa(v = !estado.pausado) {
  estado.pausado = v;
  $('#pausar').setAttribute('aria-pressed', String(v));
  $('#pausar span').textContent = v ? 'Continuar' : 'Pausar';
}
function alternarLento(v = !estado.lento) {
  estado.lento = v;
  $('#lento').setAttribute('aria-pressed', String(v));
}

function desfazerParede() {
  if (!motor) return;
  const anterior = desfazer.pop();
  if (!anterior) return;
  cena.refazer(anterior);
  cena.atualizar();
  motor.paredesMudaram();
  salvarParedes();
}

function montarInterface() {
  for (const b of document.querySelectorAll('[data-ferramenta]')) b.addEventListener('click', () => escolherFerramenta(b.dataset.ferramenta));
  const cores = $('#cores');
  for (const [nome, c] of Object.entries(CORANTES)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'cor';
    b.dataset.cor = nome;
    b.setAttribute('role', 'radio');
    b.title = c.nome;
    b.setAttribute('aria-label', c.nome);
    b.style.setProperty('--amostra', c.amostra);
    b.addEventListener('click', () => escolherCor(nome));
    cores.append(b);
  }
  const modelos = $('#modelos');
  for (const nome of ORDEM_MODELOS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'modelo';
    b.textContent = MODELOS[nome].nome;
    b.addEventListener('click', () => carregarModelo(nome));
    modelos.append(b);
  }
  const tam = $('#tamanho');
  tam.value = estado.tamanho;
  tam.addEventListener('input', () => { estado.tamanho = Number(tam.value); guardado.gravar('tamanho', estado.tamanho); });

  const sel = $('#qualidade');
  const opcoes = gpu ? ['maximo', 'alto', 'medio', 'leve'] : ['cpu'];
  for (const q of opcoes) {
    const o = document.createElement('option');
    o.value = q;
    o.textContent = QUALIDADES[q].nome;
    sel.append(o);
  }
  if (!opcoes.includes(estado.qualidade)) estado.qualidade = opcoes[Math.min(1, opcoes.length - 1)];
  sel.value = gpu ? estado.qualidade : 'cpu';
  sel.disabled = !gpu;
  sel.addEventListener('change', () => {
    estado.qualidade = sel.value;
    guardado.gravar('qualidade', sel.value);
    remontar();
  });

  $('#pausar').addEventListener('click', () => alternarPausa());
  $('#lento').addEventListener('click', () => alternarLento());
  $('#desfazer').addEventListener('click', desfazerParede);
  $('#limpar-agua').addEventListener('click', () => motor?.limparAgua());
  $('#limpar-paredes').addEventListener('click', () => {
    if (!motor) return;
    guardarDesfazer();
    cena.limpar();
    cena.atualizar();
    motor.paredesMudaram();
    salvarParedes();
  });
  $('#tela-cheia').addEventListener('click', alternarTelaCheia);

  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
    const lista = ['parede', 'reta', 'borracha', 'torneira', 'balde', 'esponja', 'mao', 'corante'];
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') { ev.preventDefault(); desfazerParede(); return; }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (/^[1-8]$/.test(ev.key)) escolherFerramenta(lista[Number(ev.key) - 1]);
    else if (ev.key === ' ') { ev.preventDefault(); alternarPausa(); }
    else if (ev.key.toLowerCase() === 'l') alternarLento();
    else if (ev.key.toLowerCase() === 'f') alternarTelaCheia();
    else if (ev.key === '[' || ev.key === ']') {
      estado.tamanho = Math.max(0, Math.min(100, estado.tamanho + (ev.key === ']' ? 8 : -8)));
      tam.value = estado.tamanho;
      guardado.gravar('tamanho', estado.tamanho);
    }
  });

  escolherFerramenta(DICAS[estado.ferramenta] ? estado.ferramenta : 'parede');
  escolherCor(estado.cor);
}

function alternarTelaCheia() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

let esperaRedimensionar = 0;
window.addEventListener('resize', () => {
  clearTimeout(esperaRedimensionar);
  esperaRedimensionar = setTimeout(() => remontar(), 250);
});

function avisar(texto) {
  const el = $('#aviso');
  el.textContent = texto;
  el.hidden = !texto;
}

// ------------------------------------------------------------ relógio

const SUB_MAX_GPU = 24;
const SUB_MAX_CPU = 6;
/** Tempo (ms) que o quadro pode gastar com a física antes de a água começar a andar mais devagar. */
const ORCAMENTO_MS = 18;
let ultimo = performance.now();
let quadros = 0, marcaQps = performance.now();
let custoSub = 0.4; // ms por subpasso, média móvel medida

function registrarCusto(ms, sub) {
  const custo = ms / sub;
  if (sub <= 0 || !Number.isFinite(custo)) return;
  custoSub += (custo - custoSub) * 0.15;
}

function quadro(agora) {
  requestAnimationFrame(quadro);
  if (reconstruindo || !motor || estado.congelado) return;
  const real = Math.min((agora - ultimo) / 1000, 1 / 20);
  ultimo = agora;
  let dt = estado.pausado ? 0 : real * (estado.lento ? 0.25 : 1);
  // a maior velocidade medida define o passo; ferramenta que joga água rápido também conta
  let vMax = motor.vMax;
  if (vMax != null && ponteiro.apertado && (estado.ferramenta === 'mao' || estado.ferramenta === 'torneira')) {
    vMax = Math.max(vMax, Math.min(6, Math.hypot(ponteiro.vx, ponteiro.vy)) + 0.7);
  }
  const dtMax = passoMaximo(cena.h, cena.altura, vMax, dt);
  let sub = dt > 0 ? Math.ceil(dt / dtMax - 1e-6) : 0;
  // Teto de subpassos pelo que a máquina aguenta: se a placa está ocupada
  // (um jogo aberto, por exemplo), a água anda mais devagar em vez de travar a tela.
  const tetoFixo = motor.tipo === 'gpu' ? SUB_MAX_GPU : SUB_MAX_CPU;
  const teto = Math.max(2, Math.min(tetoFixo, Math.floor(ORCAMENTO_MS / custoSub)));
  if (sub > teto) { sub = teto; dt = sub * dtMax; }
  const ferramenta = ferramentaDoQuadro(dt);
  if (cena.atualizar()) motor.paredesMudaram();
  const t0 = performance.now();
  motor.quadro(dt, sub, ferramenta);
  if (motor.tipo === 'gpu') motor.esperar().then(() => registrarCusto(performance.now() - t0, sub));
  else registrarCusto(performance.now() - t0, sub);
  if (motor.tipo === 'gpu') motor.contagem(); // traz a contagem e a maior velocidade, sem esperar
  desenharGuia();
  quadros++;
  if (agora - marcaQps > 500) {
    const qps = (quadros * 1000) / (agora - marcaQps);
    quadros = 0;
    marcaQps = agora;
    $('#qps').textContent = `${Math.round(qps)} qps`;
    atualizarLeitura();
  }
}

const fmt = new Intl.NumberFormat('pt-BR');
const fmt1 = new Intl.NumberFormat('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
async function atualizarLeitura() {
  const n = await motor.contagem();
  const litros = (n / PARTICULAS_POR_CELULA) * cena.h * cena.h * PROFUNDIDADE * 1000;
  $('#litros').textContent = `${fmt1.format(litros)} L`;
  $('#particulas').textContent = `${fmt.format(n)} partículas`;
  if ((n === 0) !== tanqueVazio) {
    tanqueVazio = n === 0;
    mostrarDica();
  }
}

// ------------------------------------------------------------ partida

async function iniciar() {
  if (!parametros.has('cpu')) {
    const r = await abrirGPU();
    if (r.device) {
      gpu = r;
      r.device.lost.then((info) => {
        if (info.reason === 'destroyed') return;
        cairParaCpu('A placa de vídeo parou de responder. Continuando na CPU, com menos água.');
      });
    } else {
      avisar(`${r.erro} Rodando na CPU, com menos água e menos detalhe.`);
    }
  }
  montarInterface();
  const paredes = carregarParedesGuardadas();
  montar({ tracos: paredes ?? [] });
  const pedido = parametros.get('modelo');
  if (pedido && MODELOS[pedido]) carregarModelo(pedido);
  else if (!paredes) carregarModelo('copo');
  requestAnimationFrame((t) => { ultimo = t; quadro(t); });
}

function cairParaCpu(texto) {
  const tracos = cena?.tracos.slice() ?? [];
  gpu = null;
  const novo = document.createElement('canvas');
  novo.id = 'tanque';
  novo.className = canvas.className;
  novo.setAttribute('role', 'img');
  novo.setAttribute('aria-label', canvas.getAttribute('aria-label'));
  canvas.replaceWith(novo);
  canvas = novo;
  for (const t of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel', 'pointerleave']) {
    // os ouvintes foram presos ao canvas antigo: repassa os eventos do novo
    novo.addEventListener(t, (ev) => {
      if (t === 'pointerdown') aoApertar(ev);
      else if (t === 'pointermove') (ev.getCoalescedEvents?.() ?? [ev]).forEach(mover);
      else if (t === 'pointerleave') { if (!ponteiro.apertado) ponteiro.dentro = false; }
      else soltar(ev);
    });
  }
  motor = null;
  const sel = $('#qualidade');
  sel.textContent = '';
  const o = document.createElement('option');
  o.value = 'cpu'; o.textContent = QUALIDADES.cpu.nome;
  sel.append(o);
  sel.disabled = true;
  montar({ tracos });
  avisar(texto);
}

// Para depurar e medir pelo console: window.tanque
window.tanque = {
  get motor() { return motor; },
  get cena() { return cena; },
  estado,
  carregarModelo,
  remontar,
  GRAVIDADE,
};

iniciar();
