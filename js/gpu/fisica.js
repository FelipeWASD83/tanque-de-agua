// A física da água na placa de vídeo (WebGPU). Mesmos passos, mesma ordem e
// mesmas constantes do motor da CPU (../fisica-cpu.js), que é a referência
// testada. A diferença é que aqui cada passo é um compute shader, e a soma das
// partículas na grade usa inteiros atômicos (ponto fixo) porque a GPU não tem
// soma atômica de float.
//
// Por quadro: ferramenta → partículas novas → compactação (tira as mortas)
// → N subpassos (P2G, faces, células, pressão, gradiente, extrapolação, G2P).

import {
  GRAVIDADE, PARTICULAS_POR_CELULA, TAU_DENSIDADE, TENSAO_SUPERFICIAL, DENSIDADE_AGUA, MAX_BOLHAS,
  VOLTA_VOLUME_BOLHA, PRESSAO_BOLHA_MAX, MICROPASSO, MAX_MICROPASSOS,
  CAMADAS_EXTRAPOLACAO, RAIO_PARTICULA, SDF_MAX, ESPUMA_V0, ESPUMA_V1, ESPUMA_TAXA, ESPUMA_VIDA, DIFUSAO_CORANTE,
  RES_MASCARA, PARTICULAS_MAX_POR_CELULA, ITERACOES_PCG, EXCESSO_MIN, EXCESSO_MAX,
} from '../constantes.js';
import { CAMPOS } from '../fisica-cpu.js';
import { PressaoPCG } from './pcg.js';

const GRUPO = 256;
const NOVAS_MAX = 1 << 20; // partículas novas por quadro (um cenário inteiro cabe de uma vez)
const BYTES_PARTICULA = CAMPOS * 4;

// --------------------------------------------------------------------- WGSL

const CABECALHO = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, mw: u32, mh: u32,
  h: f32, dt: f32, gx: f32, gy: f32,
  tauDens: f32, tensao: f32, rPart: f32, sdfMax: f32,
  escalaAr: f32, espDecai: f32, maxPart: u32, nNovas: u32,
  fTipo: u32, fX: f32, fY: f32, fRaio: f32,
  fVx: f32, fVy: f32, fA0: f32, fA1: f32,
  fA2: f32, ppc: f32, checar: u32, medir: u32,
};
struct Particula { pos: vec2f, vel: vec2f, c: vec4f, extra: vec4f };
const ESCALA: f32 = 65536.0;
const SOLIDO: u32 = 0u;
const AGUA: u32 = 1u;
const AR: u32 = 2u;
`;

// Funções do campo de distância (precisam de "sdf" e "P" declarados).
const FUNCOES_SDF = /* wgsl */ `
fn sdfPx(x: i32, y: i32) -> f32 {
  let xx = clamp(x, 0, i32(P.mw) - 1);
  let yy = clamp(y, 0, i32(P.mh) - 1);
  return sdf[u32(yy) * P.mw + u32(xx)];
}
fn sdfEm(p: vec2f) -> f32 {
  let f = clamp(p / P.h * ${RES_MASCARA}.0 - 0.5, vec2f(0.0), vec2f(f32(P.mw) - 1.0, f32(P.mh) - 1.0));
  let x0 = min(i32(floor(f.x)), i32(P.mw) - 2);
  let y0 = min(i32(floor(f.y)), i32(P.mh) - 2);
  let t = f - vec2f(f32(x0), f32(y0));
  return mix(mix(sdfPx(x0, y0), sdfPx(x0 + 1, y0), t.x), mix(sdfPx(x0, y0 + 1), sdfPx(x0 + 1, y0 + 1), t.x), t.y);
}
fn normalEm(p: vec2f) -> vec2f {
  let e = P.h / ${RES_MASCARA}.0;
  let g = vec2f(sdfEm(p + vec2f(e, 0.0)) - sdfEm(p - vec2f(e, 0.0)), sdfEm(p + vec2f(0.0, e)) - sdfEm(p - vec2f(0.0, e)));
  let n = length(g);
  if (n < 1e-9) { return vec2f(0.0, 1.0); }
  return g / n;
}
fn celulaSolida(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(P.nx) || j >= i32(P.ny)) { return true; }
  let a = i * 2; let b = j * 2;
  return sdfPx(a, b) + sdfPx(a + 1, b) + sdfPx(a, b + 1) + sdfPx(a + 1, b + 1) < 0.0;
}
`;

const vivas = /* wgsl */ `
  let k = g.x;
  if (k >= min(atomicLoad(&contagem[0]), P.maxPart)) { return; }
`;

const KERNELS = {
  // ---------------------------------------------------------------- P2G
  p2g: {
    b: [['pIn', 'array<Particula>'], ['contagem', 'array<atomic<u32>>'], ['uAcc', 'array<atomic<i32>>'],
      ['vAcc', 'array<atomic<i32>>'], ['celAcc', 'array<atomic<i32>>'], ['corAcc', 'array<atomic<i32>>']],
    corpo: /* wgsl */ `
fn soma(w: f32, val: f32) -> vec2i { return vec2i(i32(round(w * val * ESCALA)), i32(round(w * ESCALA))); }
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  ${vivas}
  let pa = pIn[k];
  if (pa.pos.x < 0.0) { return; }
  let x = pa.pos;
  let inv = 1.0 / P.h;
  let nx = i32(P.nx); let ny = i32(P.ny);
  {
    let f = vec2f(x.x * inv, x.y * inv - 0.5);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 1); let j0 = clamp(i32(floor(f.y)), 0, ny - 2);
    let t = f - vec2f(f32(i0), f32(j0));
    for (var b = 0; b < 2; b++) { for (var a = 0; a < 2; a++) {
      let w = select(1.0 - t.x, t.x, a == 1) * select(1.0 - t.y, t.y, b == 1);
      let d = vec2f(f32(i0 + a) * P.h - x.x, (f32(j0 + b) + 0.5) * P.h - x.y);
      let s = soma(w, pa.vel.x + dot(pa.c.xy, d));
      let idx = u32(i0 + a + (j0 + b) * (nx + 1));
      atomicAdd(&uAcc[2u * idx], s.x); atomicAdd(&uAcc[2u * idx + 1u], s.y);
    } }
  }
  {
    let f = vec2f(x.x * inv - 0.5, x.y * inv);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 2); let j0 = clamp(i32(floor(f.y)), 0, ny - 1);
    let t = f - vec2f(f32(i0), f32(j0));
    for (var b = 0; b < 2; b++) { for (var a = 0; a < 2; a++) {
      let w = select(1.0 - t.x, t.x, a == 1) * select(1.0 - t.y, t.y, b == 1);
      let d = vec2f((f32(i0 + a) + 0.5) * P.h - x.x, f32(j0 + b) * P.h - x.y);
      let s = soma(w, pa.vel.y + dot(pa.c.zw, d));
      let idx = u32(i0 + a + (j0 + b) * nx);
      atomicAdd(&vAcc[2u * idx], s.x); atomicAdd(&vAcc[2u * idx + 1u], s.y);
    } }
  }
  {
    let f = vec2f(x.x * inv - 0.5, x.y * inv - 0.5);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 2); let j0 = clamp(i32(floor(f.y)), 0, ny - 2);
    let t = f - vec2f(f32(i0), f32(j0));
    let c = u32(i0 + j0 * nx);
    atomicAdd(&celAcc[2u * c], i32(round((1.0 - t.x) * (1.0 - t.y) * ESCALA)));
    atomicAdd(&celAcc[2u * (c + 1u)], i32(round(t.x * (1.0 - t.y) * ESCALA)));
    atomicAdd(&celAcc[2u * (c + P.nx)], i32(round((1.0 - t.x) * t.y * ESCALA)));
    atomicAdd(&celAcc[2u * (c + P.nx + 1u)], i32(round(t.x * t.y * ESCALA)));
  }
  let ci = clamp(i32(floor(x.x * inv)), 0, nx - 1); let cj = clamp(i32(floor(x.y * inv)), 0, ny - 1);
  let cc = u32(ci + cj * nx);
  atomicAdd(&celAcc[2u * cc + 1u], 1);
  if (any(pa.extra.yzw != vec3f(0.0))) { // sem tinta não precisa somar (é a maioria)
    atomicAdd(&corAcc[3u * cc], i32(round(pa.extra.y * 256.0)));
    atomicAdd(&corAcc[3u * cc + 1u], i32(round(pa.extra.z * 256.0)));
    atomicAdd(&corAcc[3u * cc + 2u], i32(round(pa.extra.w * 256.0)));
  }
}`,
  },

  // --------------------------------------- faces: média, gravidade e parede
  faces: {
    b: [['uAcc', 'array<i32>'], ['vAcc', 'array<i32>'], ['u', 'array<f32>'], ['v', 'array<f32>'], ['sdf', 'array<f32>']],
    sdf: true,
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let nx = i32(P.nx); let ny = i32(P.ny);
  let nu = u32((nx + 1) * ny); let nv = u32(nx * (ny + 1));
  if (k < nu) {
    let i = i32(k % (P.nx + 1u)); let j = i32(k / (P.nx + 1u));
    let w = uAcc[2u * k + 1u];
    var val = 0.0;
    if (w > 0) { val = f32(uAcc[2u * k]) / f32(w) + P.gx * P.dt; }
    if (i == 0 || i == nx || celulaSolida(i - 1, j) || celulaSolida(i, j)) { val = 0.0; }
    u[k] = val;
  }
  if (k < nv) {
    let i = i32(k % P.nx); let j = i32(k / P.nx);
    let w = vAcc[2u * k + 1u];
    var val = 0.0;
    if (w > 0) { val = f32(vAcc[2u * k]) / f32(w) + P.gy * P.dt; }
    if (j == 0 || j == ny || celulaSolida(i, j - 1) || celulaSolida(i, j)) { val = 0.0; }
    v[k] = val;
  }
}`,
  },

  // --------------------------------------- células: tipo e divergência
  // --------------------------------------- tipo de cada célula e fração de água
  tipos: {
    b: [['celAcc', 'array<i32>'], ['sdf', 'array<f32>'], ['tipo', 'array<u32>'], ['faseA', 'array<f32>']],
    sdf: true,
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let i = i32(c % P.nx); let j = i32(c / P.nx);
  var t = AR;
  if (celulaSolida(i, j)) { t = SOLIDO; } else if (celAcc[2u * c + 1u] > 0) { t = AGUA; }
  tipo[c] = t;
  var f = -1.0;
  if (t != SOLIDO) { f = min(1.0, f32(celAcc[2u * c]) / ESCALA / P.ppc); }
  faseA[c] = f;
}`,
  },

  // --------------------------------------- fração de água estendida para dentro da parede
  // (camada 0: a partir das células livres; 1 e 2: das já preenchidas). −1 = sem valor.
  estender: {
    b: [['tipo', 'array<u32>'], ['fI', 'array<f32>'], ['fO', 'array<f32>']],
    fase: true,
    corpo: /* wgsl */ `
fn tem(n: u32, camada: u32) -> bool {
  if (tipo[n] != SOLIDO) { return true; }
  return camada > 0u && fI[n] >= 0.0;
}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  if (tipo[c] != SOLIDO) { fO[c] = fI[c]; return; }
  let camada = F.cor;
  if (camada > 0u && fI[c] >= 0.0) { fO[c] = fI[c]; return; }
  let i = c % P.nx; let j = c / P.nx;
  var s = 0.0; var k = 0.0;
  if (i > 0u && tem(c - 1u, camada)) { s += fI[c - 1u]; k += 1.0; }
  if (i < P.nx - 1u && tem(c + 1u, camada)) { s += fI[c + 1u]; k += 1.0; }
  if (j > 0u && tem(c - P.nx, camada)) { s += fI[c - P.nx]; k += 1.0; }
  if (j < P.ny - 1u && tem(c + P.nx, camada)) { s += fI[c + P.nx]; k += 1.0; }
  var v = -1.0;
  if (k > 0.0) { v = s / k; }
  fO[c] = v;
}`,
  },

  // --------------------------------------- borrão (1 2 1) numa direção (F.cor = 0: x, 1: y)
  borrar: {
    b: [['fI', 'array<f32>'], ['fO', 'array<f32>']],
    fase: true,
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let i = c % P.nx; let j = c / P.nx;
  var a = c; var b = c;
  if (F.cor == 0u) {
    if (i > 0u) { a = c - 1u; }
    if (i < P.nx - 1u) { b = c + 1u; }
  } else {
    if (j > 0u) { a = c - P.nx; }
    if (j < P.ny - 1u) { b = c + P.nx; }
  }
  fO[c] = (max(fI[a], 0.0) + 2.0 * max(fI[c], 0.0) + max(fI[b], 0.0)) * 0.25;
}`,
  },

  copiarFase: {
    b: [['faseB', 'array<f32>'], ['faseA', 'array<f32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  faseA[c] = faseB[c];
}`,
  },

  // --------------------------------------- normal da superfície (gradiente da fração)
  normais: {
    b: [['faseA', 'array<f32>'], ['normal', 'array<vec2f>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let i = c % P.nx; let j = c / P.nx;
  if (i < 1u || j < 1u || i >= P.nx - 1u || j >= P.ny - 1u) { normal[c] = vec2f(0.0); return; }
  let gr = vec2f(max(faseA[c + 1u], 0.0) - max(faseA[c - 1u], 0.0), max(faseA[c + P.nx], 0.0) - max(faseA[c - P.nx], 0.0)) * 0.5;
  let n = length(gr);
  var r = vec2f(0.0);
  if (n > 1e-3) { r = gr / n; }
  normal[c] = r;
}`,
  },

  // --------------------------------------- curvatura κ = −div(normal)
  curvatura: {
    b: [['normal', 'array<vec2f>'], ['kappa', 'array<f32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let i = c % P.nx; let j = c / P.nx;
  if (i < 1u || j < 1u || i >= P.nx - 1u || j >= P.ny - 1u) { kappa[c] = 0.0; return; }
  kappa[c] = -((normal[c + 1u].x - normal[c - 1u].x) + (normal[c + P.nx].y - normal[c - P.nx].y)) / (2.0 * P.h);
}`,
  },

  // --------------------------------------- pressão das células de ar (em unidades de q)
  // o salto σ·κ onde o ar encosta na água (a pressão própria das bolhas sai do solver)
  pressaoAr: {
    b: [['tipo', 'array<u32>'], ['kappa', 'array<f32>'], ['qAr', 'array<f32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let i = c % P.nx; let j = c / P.nx;
  if (tipo[c] != AR || i < 1u || j < 1u || i >= P.nx - 1u || j >= P.ny - 1u) { qAr[c] = 0.0; return; }
  var p = 0.0;
  if (P.tensao > 0.0) {
    var s = kappa[c]; var k = 1.0;
    if (tipo[c - 1u] == AGUA) { s += kappa[c - 1u]; k += 1.0; }
    if (tipo[c + 1u] == AGUA) { s += kappa[c + 1u]; k += 1.0; }
    if (tipo[c - P.nx] == AGUA) { s += kappa[c - P.nx]; k += 1.0; }
    if (tipo[c + P.nx] == AGUA) { s += kappa[c + P.nx]; k += 1.0; }
    if (k > 1.0) { p += P.tensao * clamp(s / k, -1.0 / P.h, 1.0 / P.h); }
  }
  qAr[c] = p * P.escalaAr;
}`,
  },

  // --------------------------------------- rótulos do ar: quem virou água perde, quem virou ar herda da vizinha
  propagarRotulo: {
    b: [['tipo', 'array<u32>'], ['rotulo', 'array<u32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  if (tipo[c] != AR) { rotulo[c] = 0u; return; }
  if (rotulo[c] != 0u) { return; }
  let i = c % P.nx; let j = c / P.nx;
  if (i < 1u || j < 1u || i >= P.nx - 1u || j >= P.ny - 1u) { return; }
  var r = 0u;
  if (tipo[c - 1u] == AR) { r = max(r, rotulo[c - 1u]); }
  if (tipo[c + 1u] == AR) { r = max(r, rotulo[c + 1u]); }
  if (tipo[c - P.nx] == AR) { r = max(r, rotulo[c - P.nx]); }
  if (tipo[c + P.nx] == AR) { r = max(r, rotulo[c + P.nx]); }
  rotulo[c] = r;
}`,
  },

  // --------------------------------------- lado direito de cada bolha: a água que está entrando
  // (antes da projeção), menos a tensão superficial das faces; volume e número de faces
  ladoBolhas: {
    b: [['tipo', 'array<u32>'], ['rotulo', 'array<u32>'], ['u', 'array<f32>'], ['v', 'array<f32>'], ['qAr', 'array<f32>'],
      ['volB', 'array<atomic<u32>>'], ['facesB', 'array<atomic<u32>>'], ['acc', 'array<atomic<u32>>']],
    corpo: /* wgsl */ `
fn somarF(i: u32, v: f32) {
  var velho = atomicLoad(&acc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&acc[i], velho, bitcast<u32>(bitcast<f32>(velho) + v));
    if (r.exchanged) { break; }
    velho = r.old_value;
  }
}
// uma soma atômica por trecho contíguo da mesma bolha no grupo, não uma por
// célula (milhares de células no mesmo endereço viram fila)
const NENHUMA: u32 = 0xffffffffu;
var<workgroup> gLab: array<u32, ${GRUPO}>;
var<workgroup> gVal: array<f32, ${GRUPO}>;
var<workgroup> gFac: array<u32, ${GRUPO}>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let c = g.x;
  var alvo = NENHUMA; var s = 0.0; var k = 0u;
  if (c < P.nx * P.ny) {
    let b = rotulo[c];
    let i = c % P.nx; let j = c / P.nx;
    if (tipo[c] == AR && b >= 2u && b < ${MAX_BOLHAS}u && i >= 1u && j >= 1u && i < P.nx - 1u && j < P.ny - 1u) {
      alvo = b;
      let t = qAr[c];
      let nx = P.nx;
      if (tipo[c - 1u] == AGUA) { s += u[i + j * (nx + 1u)] - t; k++; }
      if (tipo[c + 1u] == AGUA) { s += -u[i + 1u + j * (nx + 1u)] - t; k++; }
      if (tipo[c - nx] == AGUA) { s += v[i + j * nx] - t; k++; }
      if (tipo[c + nx] == AGUA) { s += -v[i + (j + 1u) * nx] - t; k++; }
    }
  }
  gLab[li] = alvo; gVal[li] = s; gFac[li] = k;
  workgroupBarrier();
  if (alvo != NENHUMA && (li == 0u || gLab[li - 1u] != alvo)) {
    var soma = s; var faces = k; var vol = 1u;
    var m = li + 1u;
    loop {
      if (m >= ${GRUPO}u) { break; }
      if (gLab[m] != alvo) { break; }
      soma += gVal[m]; faces += gFac[m]; vol++;
      m++;
    }
    atomicAdd(&volB[alvo], vol);
    if (faces > 0u) { atomicAdd(&facesB[alvo], faces); somarF(alvo, soma); }
  }
}`,
  },

  // --------------------------------------- fecha o lado direito (volume que volta) e o chute inicial de y
  fecharBolhas: {
    b: [['volB', 'array<u32>'], ['facesB', 'array<u32>'], ['rhsB', 'array<f32>'], ['alvoB', 'array<f32>'], ['pBolha', 'array<f32>'], ['yB', 'array<f32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b >= ${MAX_BOLHAS}u) { return; }
  if (facesB[b] == 0u) { yB[b] = 0.0; return; }
  rhsB[b] = rhsB[b] + ${VOLTA_VOLUME_BOLHA.toFixed(4)} * (alvoB[b] - f32(volB[b])) * P.h / P.dt;
  yB[b] = pBolha[b] * P.escalaAr;
}`,
  },

  // --------------------------------------- pressão da bolha somada ao ar (depois do solver)
  somarBolhaNoAr: {
    b: [['tipo', 'array<u32>'], ['rotulo', 'array<u32>'], ['facesB', 'array<u32>'], ['yB', 'array<f32>'], ['qAr', 'array<f32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let b = rotulo[c];
  if (tipo[c] != AR || b < 2u || b >= ${MAX_BOLHAS}u) { return; }
  if (facesB[b] == 0u) { return; }
  qAr[c] = qAr[c] + yB[b];
}`,
  },

  // --------------------------------------- guarda a pressão de cada bolha em Pa (vale para o próximo subpasso)
  guardarBolhas: {
    b: [['facesB', 'array<u32>'], ['yB', 'array<f32>'], ['pBolha', 'array<f32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b >= ${MAX_BOLHAS}u || facesB[b] == 0u) { return; }
  pBolha[b] = clamp(yB[b] / P.escalaAr, -${PRESSAO_BOLHA_MAX.toFixed(1)}, ${PRESSAO_BOLHA_MAX.toFixed(1)});
}`,
  },

  // --------------------------------------- tipos em bytes (4 células por palavra), para a CPU ler
  empacotarTipos: {
    b: [['tipo', 'array<u32>'], ['tipo8', 'array<u32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let n = P.nx * P.ny;
  if (k * 4u >= n) { return; }
  var w = 0u;
  for (var a = 0u; a < 4u; a++) {
    let c = k * 4u + a;
    if (c < n) { w = w | (tipo[c] << (8u * a)); }
  }
  tipo8[k] = w;
}`,
  },

  celulas: {
    b: [['celAcc', 'array<i32>'], ['u', 'array<f32>'], ['v', 'array<f32>'],
      ['tipo', 'array<u32>'], ['div', 'array<f32>'], ['q', 'array<f32>'], ['qAr', 'array<f32>']],
    corpo: /* wgsl */ `
fn temAgua(i: i32, j: i32) -> bool {
  return tipo[u32(i + j * i32(P.nx))] == AGUA;
}
fn miolo(i: i32, j: i32) -> bool {
  if (i < 1 || j < 1 || i >= i32(P.nx) - 1 || j >= i32(P.ny) - 1) { return false; }
  for (var b = -1; b <= 1; b++) { for (var a = -1; a <= 1; a++) {
    if (!temAgua(i + a, j + b)) { return false; }
  } }
  return true;
}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= P.nx * P.ny) { return; }
  let i = i32(c % P.nx); let j = i32(c / P.nx);
  if (tipo[c] != AGUA) { div[c] = 0.0; q[c] = 0.0; return; }
  let nx = P.nx;
  let d = u[u32(i) + 1u + u32(j) * (nx + 1u)] - u[u32(i) + u32(j) * (nx + 1u)] +
    v[u32(i) + u32(j + 1) * nx] - v[u32(i) + u32(j) * nx];
  var excesso = f32(celAcc[2u * c]) / ESCALA / P.ppc - 1.0;
  if (excesso < 0.0 && !miolo(i, j)) { excesso = 0.0; }
  excesso = clamp(excesso, ${EXCESSO_MIN.toFixed(3)}, ${EXCESSO_MAX.toFixed(3)});
  // ar vizinho com pressão própria entra como valor conhecido (lado direito)
  var ar = 0.0;
  if (tipo[c - 1u] == AR) { ar += qAr[c - 1u]; }
  if (tipo[c + 1u] == AR) { ar += qAr[c + 1u]; }
  if (tipo[c - nx] == AR) { ar += qAr[c - nx]; }
  if (tipo[c + nx] == AR) { ar += qAr[c + nx]; }
  div[c] = d - P.h / max(P.tauDens, P.dt) * excesso - ar;
}`,
  },

  // --------------------------------------- gradiente de pressão
  gradiente: {
    b: [['tipo', 'array<u32>'], ['q', 'array<f32>'], ['u', 'array<f32>'], ['v', 'array<f32>'],
      ['uVal', 'array<u32>'], ['vVal', 'array<u32>'], ['qAr', 'array<f32>']],
    corpo: /* wgsl */ `
// pressão da célula: a resolvida na água, a própria no ar
fn pr(c: u32) -> f32 {
  if (tipo[c] == AR) { return qAr[c]; }
  return q[c];
}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let nx = P.nx; let ny = P.ny;
  if (k < (nx + 1u) * ny) {
    let i = k % (nx + 1u); let j = k / (nx + 1u);
    var tl = SOLIDO; var tr = SOLIDO;
    if (i > 0u) { tl = tipo[i - 1u + j * nx]; }
    if (i < nx) { tr = tipo[i + j * nx]; }
    if (tl == SOLIDO || tr == SOLIDO) { u[k] = 0.0; uVal[k] = 1u; }
    else if (tl == AGUA || tr == AGUA) { u[k] = u[k] - (pr(i + j * nx) - pr(i - 1u + j * nx)); uVal[k] = 1u; }
    else { uVal[k] = 0u; }
  }
  if (k < nx * (ny + 1u)) {
    let i = k % nx; let j = k / nx;
    var tb = SOLIDO; var tt = SOLIDO;
    if (j > 0u) { tb = tipo[i + (j - 1u) * nx]; }
    if (j < ny) { tt = tipo[i + j * nx]; }
    if (tb == SOLIDO || tt == SOLIDO) { v[k] = 0.0; vVal[k] = 1u; }
    else if (tb == AGUA || tt == AGUA) { v[k] = v[k] - (pr(i + j * nx) - pr(i + (j - 1u) * nx)); vVal[k] = 1u; }
    else { vVal[k] = 0u; }
  }
}`,
  },

  // --------------------------------------- extrapolação para o ar
  extrapolar: {
    b: [['uI', 'array<f32>'], ['vI', 'array<f32>'], ['uValI', 'array<u32>'], ['vValI', 'array<u32>'],
      ['uO', 'array<f32>'], ['vO', 'array<f32>'], ['uValO', 'array<u32>'], ['vValO', 'array<u32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let nx = P.nx; let ny = P.ny;
  let wu = nx + 1u;
  if (k < wu * ny) {
    let i = k % wu; let j = k / wu;
    if (uValI[k] == 1u) { uO[k] = uI[k]; uValO[k] = 1u; }
    else {
      var s = 0.0; var n = 0.0;
      if (i > 0u && uValI[k - 1u] == 1u) { s += uI[k - 1u]; n += 1.0; }
      if (i < wu - 1u && uValI[k + 1u] == 1u) { s += uI[k + 1u]; n += 1.0; }
      if (j > 0u && uValI[k - wu] == 1u) { s += uI[k - wu]; n += 1.0; }
      if (j < ny - 1u && uValI[k + wu] == 1u) { s += uI[k + wu]; n += 1.0; }
      if (n > 0.0) { uO[k] = s / n; uValO[k] = 1u; } else { uO[k] = uI[k]; uValO[k] = 0u; }
    }
  }
  let hv = ny + 1u;
  if (k < nx * hv) {
    let i = k % nx; let j = k / nx;
    if (vValI[k] == 1u) { vO[k] = vI[k]; vValO[k] = 1u; }
    else {
      var s = 0.0; var n = 0.0;
      if (i > 0u && vValI[k - 1u] == 1u) { s += vI[k - 1u]; n += 1.0; }
      if (i < nx - 1u && vValI[k + 1u] == 1u) { s += vI[k + 1u]; n += 1.0; }
      if (j > 0u && vValI[k - nx] == 1u) { s += vI[k - nx]; n += 1.0; }
      if (j < hv - 1u && vValI[k + nx] == 1u) { s += vI[k + nx]; n += 1.0; }
      if (n > 0.0) { vO[k] = s / n; vValO[k] = 1u; } else { vO[k] = vI[k]; vValO[k] = 0u; }
    }
  }
}`,
  },

  // ---------------------------------------------------------------- G2P
  g2p: {
    b: [['pIn', 'array<Particula>'], ['contagem', 'array<atomic<u32>>'], ['u', 'array<f32>'], ['v', 'array<f32>'], ['sdf', 'array<f32>'],
      ['celAcc', 'array<i32>'], ['corAcc', 'array<i32>']],
    sdf: true,
    corpo: /* wgsl */ `
// velocidade da grade (já projetada) na posição, interpolada; limitada a vmax
fn velocidadeEm(x: vec2f, vmax: f32) -> vec2f {
  let inv = 1.0 / P.h;
  let nx = i32(P.nx); let ny = i32(P.ny);
  var r = vec2f(0.0);
  {
    let f = vec2f(x.x * inv, x.y * inv - 0.5);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 1); let j0 = clamp(i32(floor(f.y)), 0, ny - 2);
    let t = f - vec2f(f32(i0), f32(j0));
    let k = u32(i0 + j0 * (nx + 1));
    r.x = mix(mix(u[k], u[k + 1u], t.x), mix(u[k + P.nx + 1u], u[k + P.nx + 2u], t.x), t.y);
  }
  {
    let f = vec2f(x.x * inv - 0.5, x.y * inv);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 2); let j0 = clamp(i32(floor(f.y)), 0, ny - 1);
    let t = f - vec2f(f32(i0), f32(j0));
    let k = u32(i0 + j0 * nx);
    r.y = mix(mix(v[k], v[k + 1u], t.x), mix(v[k + P.nx], v[k + P.nx + 1u], t.x), t.y);
  }
  let l = length(r);
  if (l > vmax) { r *= vmax / l; }
  return r;
}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  ${vivas}
  var pa = pIn[k];
  if (pa.pos.x < 0.0) { return; }
  let inv = 1.0 / P.h;
  let nx = i32(P.nx); let ny = i32(P.ny);
  var x = pa.pos;
  var vel = vec2f(0.0);
  var cu = vec2f(0.0); var cv = vec2f(0.0);
  {
    let f = vec2f(x.x * inv, x.y * inv - 0.5);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 1); let j0 = clamp(i32(floor(f.y)), 0, ny - 2);
    let t = f - vec2f(f32(i0), f32(j0));
    for (var b = 0; b < 2; b++) { for (var a = 0; a < 2; a++) {
      let val = u[u32(i0 + a + (j0 + b) * (nx + 1))];
      let wx = select(1.0 - t.x, t.x, a == 1); let wy = select(1.0 - t.y, t.y, b == 1);
      vel.x += wx * wy * val;
      cu += vec2f(select(-1.0, 1.0, a == 1) * wy, wx * select(-1.0, 1.0, b == 1)) * inv * val;
    } }
  }
  {
    let f = vec2f(x.x * inv - 0.5, x.y * inv);
    let i0 = clamp(i32(floor(f.x)), 0, nx - 2); let j0 = clamp(i32(floor(f.y)), 0, ny - 1);
    let t = f - vec2f(f32(i0), f32(j0));
    for (var b = 0; b < 2; b++) { for (var a = 0; a < 2; a++) {
      let val = v[u32(i0 + a + (j0 + b) * nx)];
      let wx = select(1.0 - t.x, t.x, a == 1); let wy = select(1.0 - t.y, t.y, b == 1);
      vel.y += wx * wy * val;
      cv += vec2f(select(-1.0, 1.0, a == 1) * wy, wx * select(-1.0, 1.0, b == 1)) * inv * val;
    } }
  }
  let vmax = ${(MICROPASSO * MAX_MICROPASSOS).toFixed(3)} * P.h / P.dt;
  let rap = length(vel);
  // espuma: rápida e rarefeita
  let ci = clamp(i32(floor(x.x * inv)), 0, nx - 1); let cj = clamp(i32(floor(x.y * inv)), 0, ny - 1);
  let dens = f32(celAcc[2u * u32(ci + cj * nx)]) / ESCALA / P.ppc;
  let fonte = smoothstep(${ESPUMA_V0.toFixed(4)}, ${ESPUMA_V1.toFixed(4)}, rap) * (1.0 - smoothstep(0.45, 0.95, dens));
  let espuma = min(1.0, pa.extra.x * P.espDecai + fonte * ${ESPUMA_TAXA.toFixed(4)} * P.dt);
  // corante: difunde puxando a cor para a média da célula
  let cc = u32(ci + cj * nx);
  let nCel = celAcc[2u * cc + 1u];
  if (nCel > 1) {
    let media = vec3f(f32(corAcc[3u * cc]), f32(corAcc[3u * cc + 1u]), f32(corAcc[3u * cc + 2u])) / (256.0 * f32(nCel));
    pa.extra = vec4f(pa.extra.x, pa.extra.yzw + (media - pa.extra.yzw) * min(1.0, P.dt / ${DIFUSAO_CORANTE.toFixed(4)}));
  }

  atomicMax(&contagem[2], bitcast<u32>(rap));
  if (rap > vmax) { vel *= vmax / rap; }
  // anda em micro-passos pela velocidade da grade, batendo na parede a cada um
  let micro = u32(clamp(ceil(min(rap, vmax) * P.dt / (${MICROPASSO.toFixed(3)} * P.h)), 1.0, ${MAX_MICROPASSOS.toFixed(1)}));
  let dm = P.dt / f32(micro);
  for (var m = 0u; m < micro; m++) {
    var w = vel;
    if (micro > 1u) { w = velocidadeEm(x, vmax); }
    x += w * dm;
    let d = sdfEm(x);
    if (d < P.rPart) {
      if (d < -P.sdfMax + 0.5) { pIn[k].pos = vec2f(-1.0, -1.0); return; }
      let n = normalEm(x);
      x += n * (P.rPart - d) * P.h;
      let vn = dot(vel, n);
      if (vn < 0.0) { vel -= vn * n; }
    }
  }
  pa.pos = x; pa.vel = vel; pa.c = vec4f(cu, cv); pa.extra.x = espuma;
  pIn[k] = pa;
}`,
  },

  // --------------------------------------- ferramentas (uma vez por quadro)
  ferramenta: {
    b: [['pIn', 'array<Particula>'], ['contagem', 'array<atomic<u32>>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  ${vivas}
  if (P.fTipo == 0u) { return; }
  var pa = pIn[k];
  if (pa.pos.x < 0.0) { return; }
  let d2 = dot(pa.pos - vec2f(P.fX, P.fY), pa.pos - vec2f(P.fX, P.fY));
  let r2 = P.fRaio * P.fRaio;
  if (d2 >= r2) { return; }
  if (P.fTipo == 1u) { pIn[k].pos = vec2f(-1.0, -1.0); return; }
  let peso = 1.0 - sqrt(d2 / r2);
  if (P.fTipo == 2u) {
    let a = min(1.0, peso * 0.9);
    pa.vel += (vec2f(P.fVx, P.fVy) - pa.vel) * a;
  } else if (P.fTipo == 3u) {
    pa.extra = vec4f(pa.extra.x, P.fA0, P.fA1, P.fA2);
  }
  pIn[k] = pa;
}`,
  },

  // --------------------------------------- partículas novas
  emitir: {
    b: [['pIn', 'array<Particula>'], ['contagem', 'array<atomic<u32>>'], ['novas', 'array<Particula>'],
      ['celAcc', 'array<i32>'], ['sdf', 'array<f32>']],
    sdf: true,
    grupo: 64,
    corpo: /* wgsl */ `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  if (k >= P.nNovas) { return; }
  let np = novas[k];
  if (!(sdfEm(np.pos) > P.rPart + 0.05)) { return; }
  if (P.checar == 1u) {
    let i = i32(floor(np.pos.x / P.h)); let j = i32(floor(np.pos.y / P.h));
    if (i < 0 || j < 0 || i >= i32(P.nx) || j >= i32(P.ny)) { return; }
    if (f32(celAcc[2u * u32(i + j * i32(P.nx)) + 1u]) >= P.ppc - 1.0) { return; }
  }
  let idx = atomicAdd(&contagem[0], 1u);
  if (idx >= P.maxPart) { atomicSub(&contagem[0], 1u); return; }
  pIn[idx] = np;
}`,
  },

  // --------------------------------------- compactação: tira as mortas
  compactar: {
    b: [['pIn', 'array<Particula>'], ['pOut', 'array<Particula>'], ['contagem', 'array<atomic<u32>>']],
    corpo: /* wgsl */ `
fn ruim(f: f32) -> bool { return (bitcast<u32>(f) & 0x7f800000u) == 0x7f800000u; }
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  ${vivas}
  let pa = pIn[k];
  if (pa.pos.x < 0.0 || ruim(pa.pos.x) || ruim(pa.pos.y) || ruim(pa.vel.x) || ruim(pa.vel.y)) { return; }
  let idx = atomicAdd(&contagem[1], 1u);
  pOut[idx] = pa;
}`,
  },

  // --------------------------------------- argumentos de despacho indireto
  argumentos: {
    b: [['contagem', 'array<u32>'], ['despacho', 'array<u32>'], ['desenho', 'array<u32>']],
    fase: true,
    grupo: 1,
    corpo: /* wgsl */ `
@compute @workgroup_size(1) fn main() {
  // fase 1: fim da compactação (a contagem nova vira a atual)
  if (F.cor == 1u) { contagem[0] = contagem[1]; contagem[1] = 0u; }
  else if (P.medir == 1u) { contagem[2] = 0u; } // maior velocidade: medida de novo em cada quadro que anda (pausado, fica a última)
  let n = min(contagem[0], P.maxPart);
  contagem[0] = n;
  despacho[0] = (n + ${GRUPO - 1}u) / ${GRUPO}u; despacho[1] = 1u; despacho[2] = 1u;
  desenho[0] = 6u; desenho[1] = n; desenho[2] = 0u; desenho[3] = 0u;
}`,
  },
};

function montarWgsl(nome, k) {
  const linhas = [CABECALHO, '@group(0) @binding(0) var<uniform> P: Params;'];
  let b = 1;
  if (k.fase) linhas.push(`struct Fase { cor: u32, _a: u32, _b: u32, _c: u32 };\n@group(0) @binding(${b++}) var<uniform> F: Fase;`);
  for (const [n, t] of k.b) linhas.push(`@group(0) @binding(${b++}) var<storage, read_write> ${n}: ${t};`);
  if (k.sdf) linhas.push(FUNCOES_SDF);
  linhas.push(k.corpo);
  return linhas.join('\n');
}

// ----------------------------------------------------------------- motor

export class FisicaGPU {
  static instancias = 0;

  /**
   * @param {GPUDevice} device
   * @param {import('../cena.js').Cena} cena
   */
  constructor(device, cena) {
    if (RES_MASCARA !== 2) throw new Error('O motor da GPU assume 2×2 pixels de máscara por célula.');
    this.device = device;
    this.cena = cena;
    this.nx = cena.nx;
    this.ny = cena.ny;
    this.h = cena.h;
    this.gravidade = [0, -GRAVIDADE];
    this.nu = (this.nx + 1) * this.ny;
    this.nv = this.nx * (this.ny + 1);
    this.nc = this.nx * this.ny;
    this.maxParticulas = Math.ceil(this.nc * PARTICULAS_MAX_POR_CELULA);
    this.paridade = 0;
    this.vMax = null;
    this.pendentes = [];
    this.pendentesChecar = true;
    this.contagemAproximada = 0;
    this.idRender = ++FisicaGPU.instancias;
    this.#criarBuffers();
    this.#criarPipelines();
    this.montarPressao({ iteracoes: ITERACOES_PCG });
    this.subirParedes(true);
  }

  #buffer(nome, bytes, uso = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const b = this.device.createBuffer({ label: nome, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: uso });
    this.buffers[nome] = b;
    return b;
  }

  #criarBuffers() {
    this.buffers = {};
    const { nu, nv, nc } = this;
    const cena = this.cena;
    this.#buffer('uAcc', nu * 8);
    this.#buffer('vAcc', nv * 8);
    this.#buffer('celAcc', nc * 8);
    this.#buffer('corAcc', nc * 12);
    for (const n of ['u', 'uTmp', 'uVal', 'uValTmp']) this.#buffer(n, nu * 4);
    for (const n of ['v', 'vTmp', 'vVal', 'vValTmp']) this.#buffer(n, nv * 4);
    for (const n of ['tipo', 'div', 'q', 'faseA', 'faseB', 'kappa', 'qAr', 'rotulo']) this.#buffer(n, nc * 4);
    this.#buffer('normal', nc * 8);
    for (const n of ['pBolha', 'volB', 'facesB', 'rhsB', 'yB', 'alvoB']) this.#buffer(n, MAX_BOLHAS * 4);
    this.#buffer('tipo8', Math.ceil(nc / 4) * 4);
    this.#buffer('sdf', cena.mw * cena.mh * 4);
    this.#buffer('pA', this.maxParticulas * BYTES_PARTICULA);
    this.#buffer('pB', this.maxParticulas * BYTES_PARTICULA);
    this.#buffer('novas', NOVAS_MAX * BYTES_PARTICULA);
    this.#buffer('contagem', 16);
    this.#buffer('despacho', 16, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.#buffer('desenho', 16, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.#buffer('params', 112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.#buffer('fase0', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.#buffer('fase1', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(this.buffers.fase0, 0, new Uint32Array([0, 0, 0, 0]));
    this.device.queue.writeBuffer(this.buffers.fase1, 0, new Uint32Array([1, 0, 0, 0]));
    this.#buffer('fase2', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(this.buffers.fase2, 0, new Uint32Array([2, 0, 0, 0]));
    this.leitura = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.lendo = false;
    // Custo de verdade da física, pelo relógio da própria placa (sem contar o
    // tempo que o quadro espera na fila, que era o que deixava tudo lento).
    if (this.device.features.has('timestamp-query')) {
      this.carimbos = this.device.createQuerySet({ type: 'timestamp', count: 2 });
      this.carimbosResolvidos = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.carimbosLeitura = [0, 1, 2].map(() => ({
        buffer: this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        livre: true,
      }));
    }
    /** Milissegundos de GPU por subpasso (média móvel), ou null se a placa não mede. */
    this.custoPorSubpasso = null;
    // ar: os tipos de célula vão para um Web Worker (ar-trabalhador.js), que
    // devolve quem é bolha e quanto ar cada uma tem
    const bytesTipos = Math.ceil(this.nc / 4) * 4;
    this.ar = {
      trabalhador: new Worker(new URL('../ar-trabalhador.js', import.meta.url), { type: 'module' }),
      leitura: this.device.createBuffer({ size: bytesTipos + MAX_BOLHAS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      bytesTipos,
      ocupado: false,
      pendente: false,
      geracao: 0,
    };
    this.ar.trabalhador.onmessage = (ev) => this.#receberRotulos(ev.data);
    this.ar.trabalhador.onerror = (ev) => { console.error('ar-trabalhador:', ev.message); this.ar.ocupado = false; };
    // motor recriado (janela redimensionada, qualidade trocada) com água dentro:
    // espera o primeiro retrato do ar, senão o ar fechado vira vácuo por uns quadros
    this.reiniciarAr();
  }

  #criarPipelines() {
    this.kernels = {};
    for (const [nome, k] of Object.entries(KERNELS)) {
      const modulo = this.device.createShaderModule({ label: nome, code: montarWgsl(nome, k) });
      const pipeline = this.device.createComputePipeline({ label: nome, layout: 'auto', compute: { module: modulo, entryPoint: 'main' } });
      this.kernels[nome] = { ...k, pipeline, grupos: new Map() };
    }
  }

  /** Bind group de um kernel, com os nomes trocados conforme a variante. */
  #grupo(nome, troca = {}, fase = 0) {
    const k = this.kernels[nome];
    const chave = JSON.stringify(troca) + fase;
    let g = k.grupos.get(chave);
    if (g) return g;
    const entradas = [{ binding: 0, resource: { buffer: this.buffers.params } }];
    let b = 1;
    if (k.fase) entradas.push({ binding: b++, resource: { buffer: this.buffers[`fase${fase}`] } });
    for (const [n] of k.b) entradas.push({ binding: b++, resource: { buffer: this.buffers[troca[n] ?? n] } });
    g = this.device.createBindGroup({ label: nome + chave, layout: k.pipeline.getBindGroupLayout(0), entries: entradas });
    k.grupos.set(chave, g);
    return g;
  }

  get trocaParticulas() {
    return this.paridade === 0 ? { pIn: 'pA', pOut: 'pB' } : { pIn: 'pB', pOut: 'pA' };
  }

  /** (Re)monta o solver de pressão (as opções servem para medir variações pelo console). */
  montarPressao(opcoes) {
    this.pressao?.destruir();
    const B = this.buffers;
    this.pressao = new PressaoPCG(this.device, {
      nx: this.nx, ny: this.ny, tipo: B.tipo, div: B.div, q: B.q,
      rotulo: B.rotulo, facesBolha: B.facesB, rhsBolha: B.rhsB, yBolha: B.yB, volBolha: B.volB,
    }, opcoes);
  }

  /** Buffer com as partículas atuais (o render lê daqui). */
  get particulas() {
    return this.buffers[this.paridade === 0 ? 'pA' : 'pB'];
  }

  /** Sobe para a GPU as linhas do campo de distância que mudaram. */
  subirParedes(tudo = false) {
    const cena = this.cena;
    let faixa = cena.pegarLinhasAlteradas();
    if (tudo) faixa = [0, cena.mh - 1];
    if (!faixa) return;
    const [y0, y1] = faixa;
    this.device.queue.writeBuffer(this.buffers.sdf, y0 * cena.mw * 4, cena.sdf, y0 * cena.mw, (y1 - y0 + 1) * cena.mw);
  }

  /** Enfileira partículas novas (formato de 12 floats) para o próximo quadro. */
  adicionar(lista, checar = true) {
    if (!lista.length) return;
    if (this.pendentes.length && this.pendentesChecar !== checar) {
      // lotes com regra diferente não se misturam: o segundo espera o próximo quadro
      (this.adiados ??= []).push([lista, checar]);
      return;
    }
    this.pendentesChecar = checar;
    this.pendentes.push(lista);
  }

  /** Rótulos novos que o trabalhador mandou: sobem para a placa. */
  #receberRotulos(m) {
    this.ar.ocupado = false;
    if (m.descartado || m.geracao !== this.ar.geracao || this.destruido) return; // cena mudou no meio: descarta
    this.ar.esperando = false;
    const q = this.device.queue;
    q.writeBuffer(this.buffers.rotulo, 0, m.rotulo);
    q.writeBuffer(this.buffers.alvoB, 0, m.alvo);
    for (const [r, p] of m.nascidas) q.writeBuffer(this.buffers.pBolha, r * 4, new Float32Array([p]));
    const tem = m.alvo.some((v) => v > 0);
    if (this.ar.temBolhas && !tem) this.ar.zerarVetores = true; // as bolhas acabaram: zera antes do próximo solver
    this.ar.temBolhas = tem;
  }

  /** Cenário novo: esquece as bolhas; o ar fechado que existir vira bolha com o volume que tiver. */
  reiniciarAr() {
    this.ar.geracao++;
    if (this.ar.temBolhas) this.ar.zerarVetores = true;
    this.ar.temBolhas = false;
    this.ar.esperando = true;
    this.ar.quadrosEsperando = 0;
    this.ar.trabalhador.postMessage({ tipo: 'reiniciar', geracao: this.ar.geracao });
    const q = this.device.queue;
    q.writeBuffer(this.buffers.rotulo, 0, new Uint32Array(this.nc));
    q.writeBuffer(this.buffers.pBolha, 0, new Float32Array(MAX_BOLHAS));
    q.writeBuffer(this.buffers.alvoB, 0, new Float32Array(MAX_BOLHAS));
  }

  /** Tira toda a água. */
  limparAgua() {
    const q = this.device.queue;
    q.writeBuffer(this.buffers.contagem, 0, new Uint32Array(4));
    q.writeBuffer(this.buffers.despacho, 0, new Uint32Array([0, 1, 1, 0]));
    q.writeBuffer(this.buffers.desenho, 0, new Uint32Array([6, 0, 0, 0]));
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.buffers.celAcc);
    q.submit([enc.finish()]);
    this.pendentes = [];
    this.adiados = [];
    this.contagemAproximada = 0;
    this.reiniciarAr();
  }

  #escreverParams(dt, ferramenta, nNovas, checar, medir) {
    const f = ferramenta ?? { tipo: 'nenhuma' };
    const tipos = { nenhuma: 0, esponja: 1, mao: 2, corante: 3 };
    const buf = new ArrayBuffer(112);
    const u = new Uint32Array(buf), fl = new Float32Array(buf);
    const c = this.cena;
    u[0] = this.nx; u[1] = this.ny; u[2] = c.mw; u[3] = c.mh;
    fl[4] = this.h; fl[5] = dt; fl[6] = this.gravidade[0]; fl[7] = this.gravidade[1];
    fl[8] = this.tauDensidade ?? TAU_DENSIDADE; fl[9] = this.tensao ?? TENSAO_SUPERFICIAL; fl[10] = RAIO_PARTICULA; fl[11] = SDF_MAX;
    fl[12] = dt / (DENSIDADE_AGUA * this.h); fl[13] = Math.exp(-dt / ESPUMA_VIDA); u[14] = this.maxParticulas; u[15] = nNovas;
    u[16] = tipos[f.tipo] ?? 0; fl[17] = f.x ?? 0; fl[18] = f.y ?? 0; fl[19] = f.raio ?? 0;
    fl[20] = f.vx ?? 0; fl[21] = f.vy ?? 0;
    const a = f.absorcao ?? [0, 0, 0];
    fl[22] = a[0]; fl[23] = a[1]; fl[24] = a[2];
    fl[25] = PARTICULAS_POR_CELULA; u[26] = checar ? 1 : 0; u[27] = medir ? 1 : 0;
    this.device.queue.writeBuffer(this.buffers.params, 0, buf);
  }

  #despachar(pass, nome, total, troca, fase) {
    const k = this.kernels[nome];
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, this.#grupo(nome, troca, fase));
    pass.dispatchWorkgroups(Math.ceil(total / (k.grupo ?? GRUPO)));
  }

  #despacharParticulas(pass, nome, troca) {
    const k = this.kernels[nome];
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, this.#grupo(nome, troca));
    pass.dispatchWorkgroupsIndirect(this.buffers.despacho, 0);
  }

  /**
   * Grava um quadro inteiro no encoder: ferramenta, partículas novas,
   * compactação e os subpassos. `dt` é o tempo simulado do quadro.
   */
  quadro(enc, dt, subpassos, ferramenta) {
    // partículas novas deste quadro
    let novas = 0;
    const checar = this.pendentesChecar;
    if (this.pendentes.length) {
      const total = this.pendentes.reduce((s, l) => s + l.length / CAMPOS, 0);
      const junta = new Float32Array(Math.min(total, NOVAS_MAX) * CAMPOS);
      let o = 0;
      const sobra = [];
      for (const l of this.pendentes) {
        const cabe = Math.min(l.length, junta.length - o);
        junta.set(l.subarray(0, cabe), o);
        o += cabe;
        if (cabe < l.length) sobra.push(l.subarray(cabe));
      }
      novas = o / CAMPOS;
      this.device.queue.writeBuffer(this.buffers.novas, 0, junta, 0, o);
      this.pendentes = sobra;
      if (!sobra.length && this.adiados?.length) {
        const [l, c] = this.adiados.shift();
        this.pendentes = [l];
        this.pendentesChecar = c;
      }
    }
    // Cenário novo: segura a física até o rastreador ver o ar fechado pela
    // primeira vez (senão uma bolha sem ar fecha antes de ser vista, em ~1 quadro).
    const esperandoAr = this.ar.esperando && this.ar.quadrosEsperando++ < 30;
    if (esperandoAr) subpassos = 0;
    const dtSub = subpassos > 0 ? dt / subpassos : 1 / 240;
    this.#escreverParams(dtSub, ferramenta, novas, checar, subpassos > 0);

    const t = this.trocaParticulas;
    let pass = enc.beginComputePass({ label: 'quadro' });
    this.#despacharParticulas(pass, 'ferramenta', t);
    if (novas) this.#despachar(pass, 'emitir', novas, t);
    this.#despachar(pass, 'argumentos', 1, {}, 0);
    this.#despacharParticulas(pass, 'compactar', t);
    this.#despachar(pass, 'argumentos', 1, {}, 1);
    pass.end();
    this.paridade ^= 1;
    this.contagemAproximada += novas;

    const tp = this.trocaParticulas;
    const { nu, nv, nc } = this;
    const nf = Math.max(nu, nv);
    // carimbo no começo do primeiro subpasso e no fim do último
    const leitura = subpassos > 0 && this.carimbos ? this.carimbosLeitura.find((l) => l.livre) : null;
    for (let s = 0; s < subpassos; s++) {
      enc.clearBuffer(this.buffers.uAcc);
      enc.clearBuffer(this.buffers.vAcc);
      enc.clearBuffer(this.buffers.celAcc);
      enc.clearBuffer(this.buffers.corAcc);
      for (const b of ['volB', 'facesB', 'rhsB']) enc.clearBuffer(this.buffers[b]);
      if (this.ar.zerarVetores) { this.pressao.zerarBolhas(enc); this.ar.zerarVetores = false; }
      const bolhas = !!this.ar.temBolhas;
      const carimbo = leitura && (s === 0 || s === subpassos - 1) ? { querySet: this.carimbos } : null;
      if (carimbo && s === 0) carimbo.beginningOfPassWriteIndex = 0;
      if (carimbo && s === subpassos - 1) carimbo.endOfPassWriteIndex = 1;
      pass = enc.beginComputePass(carimbo ? { label: 'subpasso', timestampWrites: carimbo } : { label: 'subpasso' });
      // modo perfil (só para medir pelo console): cada etapa num passe com carimbo de tempo
      const perfil = this.perfil;
      const marca = (nome) => {
        if (!perfil) return;
        pass.end();
        pass = enc.beginComputePass({ label: nome, timestampWrites: { querySet: perfil.carimbos, beginningOfPassWriteIndex: perfil.nomes.length } });
        perfil.nomes.push(nome);
      };
      marca('p2g');
      this.#despacharParticulas(pass, 'p2g', tp);
      marca('tipos e rótulos');
      this.#despachar(pass, 'faces', nf);
      this.#despachar(pass, 'tipos', nc);
      this.#despachar(pass, 'propagarRotulo', nc);
      this.#despachar(pass, 'propagarRotulo', nc);
      marca('superfície');
      this.#superficie(pass);
      marca('divergência e bolhas');
      this.#despachar(pass, 'pressaoAr', nc);
      this.#despachar(pass, 'celulas', nc);
      if (bolhas) {
        this.#despachar(pass, 'ladoBolhas', nc, { acc: 'rhsB' });
        this.#despachar(pass, 'fecharBolhas', MAX_BOLHAS);
      }
      marca('pressão (MGPCG)');
      this.pressao.gravar(pass, bolhas);
      marca('gradiente e extrapolação');
      if (bolhas) {
        this.#despachar(pass, 'somarBolhaNoAr', nc);
        this.#despachar(pass, 'guardarBolhas', MAX_BOLHAS);
      }
      this.#despachar(pass, 'gradiente', nf);
      for (let c = 0; c < CAMADAS_EXTRAPOLACAO; c++) {
        const ida = c % 2 === 0;
        this.#despachar(pass, 'extrapolar', nf, ida
          ? { uI: 'u', vI: 'v', uValI: 'uVal', vValI: 'vVal', uO: 'uTmp', vO: 'vTmp', uValO: 'uValTmp', vValO: 'vValTmp' }
          : { uI: 'uTmp', vI: 'vTmp', uValI: 'uValTmp', vValI: 'vValTmp', uO: 'u', vO: 'v', uValO: 'uVal', vValO: 'vVal' });
      }
      if (CAMADAS_EXTRAPOLACAO % 2 === 1) throw new Error('CAMADAS_EXTRAPOLACAO precisa ser par (o resultado volta para u e v).');
      marca('g2p');
      this.#despacharParticulas(pass, 'g2p', tp);
      marca('fim');
      pass.end();
    }
    if (esperandoAr && !this.ar.ocupado) {
      // só os tipos, sem andar: partículas → grade → tipo de célula
      enc.clearBuffer(this.buffers.uAcc);
      enc.clearBuffer(this.buffers.vAcc);
      enc.clearBuffer(this.buffers.celAcc);
      enc.clearBuffer(this.buffers.corAcc);
      const p0 = enc.beginComputePass({ label: 'tipos (esperando o ar)' });
      this.#despacharParticulas(p0, 'p2g', this.trocaParticulas);
      this.#despachar(p0, 'tipos', this.nc);
      p0.end();
    }
    if ((subpassos > 0 || esperandoAr) && !this.ar.ocupado) {
      const passTipos = enc.beginComputePass({ label: 'tipos para o ar' });
      this.#despachar(passTipos, 'empacotarTipos', Math.ceil(this.nc / 4));
      passTipos.end();
      enc.copyBufferToBuffer(this.buffers.tipo8, 0, this.ar.leitura, 0, this.ar.bytesTipos);
      enc.copyBufferToBuffer(this.buffers.pBolha, 0, this.ar.leitura, this.ar.bytesTipos, MAX_BOLHAS * 4);
      this.ar.ocupado = true;
      this.ar.pendente = true;
    }
    if (leitura) {
      enc.resolveQuerySet(this.carimbos, 0, 2, this.carimbosResolvidos, 0);
      enc.copyBufferToBuffer(this.carimbosResolvidos, 0, leitura.buffer, 0, 16);
      leitura.livre = false;
      this.carimboPendente = { leitura, subpassos };
    }
  }

  /** Chamar depois de enviar o quadro: lê o carimbo de tempo e os tipos quando ficarem prontos. */
  aposEnviar() {
    if (this.ar.pendente) {
      this.ar.pendente = false;
      const geracao = this.ar.geracao;
      this.ar.leitura.mapAsync(GPUMapMode.READ).then(() => {
        const dados = this.ar.leitura.getMappedRange();
        const tipos = dados.slice(0, this.ar.bytesTipos);
        const pressoes = dados.slice(this.ar.bytesTipos);
        this.ar.leitura.unmap();
        this.ar.trabalhador.postMessage({ nx: this.nx, ny: this.ny, tipos, pressoes, geracao }, [tipos, pressoes]);
      }).catch(() => { this.ar.ocupado = false; });
    }
    const p = this.carimboPendente;
    if (!p) return;
    this.carimboPendente = null;
    p.leitura.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(p.leitura.buffer.getMappedRange());
      const ms = Number(t[1] - t[0]) / 1e6 / p.subpassos;
      p.leitura.buffer.unmap();
      p.leitura.livre = true;
      if (ms > 0 && Number.isFinite(ms)) {
        this.custoPorSubpasso = this.custoPorSubpasso == null ? ms : this.custoPorSubpasso + (ms - this.custoPorSubpasso) * 0.2;
      }
    }).catch(() => { p.leitura.livre = true; });
  }

  /**
   * Curvatura da superfície, igual à da CPU: fração de água estendida para
   * dentro da parede (3 camadas), borrão (1 2 1) em x e y, duas vezes, e de
   * novo a extensão; depois normal e κ.
   */
  #superficie(pass) {
    const AB = { fI: 'faseA', fO: 'faseB' }, BA = { fI: 'faseB', fO: 'faseA' };
    const nc = this.nc;
    const estender = (ida) => {
      // três camadas, alternando os buffers; começa em "ida" e termina no outro
      this.#despachar(pass, 'estender', nc, ida ? AB : BA, 0);
      this.#despachar(pass, 'estender', nc, ida ? BA : AB, 1);
      this.#despachar(pass, 'estender', nc, ida ? AB : BA, 2);
    };
    for (let k = 0; k < 2; k++) {
      estender(true); // A → B
      this.#despachar(pass, 'borrar', nc, BA, 0); // x: B → A
      this.#despachar(pass, 'borrar', nc, AB, 1); // y: A → B
      this.#despachar(pass, 'copiarFase', nc); // B → A
    }
    estender(true); // A → B
    this.#despachar(pass, 'copiarFase', nc); // B → A
    this.#despachar(pass, 'normais', nc);
    this.#despachar(pass, 'curvatura', nc);
  }

  /**
   * Só para medir pelo console: roda um quadro de um subpasso com cada etapa
   * num passe separado e devolve os milissegundos de GPU de cada uma.
   */
  async perfilar(dt = 1 / 1000) {
    if (!this.device.features.has('timestamp-query')) return null;
    const carimbos = this.device.createQuerySet({ type: 'timestamp', count: 16 });
    this.perfil = { carimbos, nomes: [] };
    const enc = this.device.createCommandEncoder();
    this.quadro(enc, dt, 1, null);
    const nomes = this.perfil.nomes;
    const n = nomes.length;
    this.perfil = null;
    const res = this.device.createBuffer({ size: 16 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const leitura = this.device.createBuffer({ size: 16 * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.resolveQuerySet(carimbos, 0, n, res, 0);
    enc.copyBufferToBuffer(res, 0, leitura, 0, n * 8);
    this.device.queue.submit([enc.finish()]);
    await leitura.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(leitura.getMappedRange().slice(0));
    leitura.unmap();
    const saida = {};
    for (let k = 0; k < n - 1; k++) saida[nomes[k]] = +(Number(t[k + 1] - t[k]) / 1e6).toFixed(3);
    carimbos.destroy(); res.destroy(); leitura.destroy();
    return saida;
  }

  /** Pede a contagem de partículas (chega com um ou dois quadros de atraso). */
  async lerContagem() {
    if (this.lendo) return this.contagemAproximada;
    this.lendo = true;
    try {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.buffers.contagem, 0, this.leitura, 0, 16);
      this.device.queue.submit([enc.finish()]);
      await this.leitura.mapAsync(GPUMapMode.READ);
      const dados = this.leitura.getMappedRange();
      const n = new Uint32Array(dados)[0];
      this.vMax = new Float32Array(dados)[2];
      this.leitura.unmap();
      this.contagemAproximada = n;
      return n;
    } finally {
      this.lendo = false;
    }
  }

  /**
   * Copia todas as partículas de volta para a CPU (12 floats cada). Faz a
   * própria leitura da contagem: a do relógio pode estar atrasada um quadro.
   */
  async lerParticulas() {
    const conta = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e0 = this.device.createCommandEncoder();
    e0.copyBufferToBuffer(this.buffers.contagem, 0, conta, 0, 16);
    this.device.queue.submit([e0.finish()]);
    await conta.mapAsync(GPUMapMode.READ);
    const n = new Uint32Array(conta.getMappedRange())[0];
    conta.unmap();
    conta.destroy();
    const bytes = Math.max(16, n * BYTES_PARTICULA);
    const destino = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    if (n) enc.copyBufferToBuffer(this.particulas, 0, destino, 0, n * BYTES_PARTICULA);
    this.device.queue.submit([enc.finish()]);
    await destino.mapAsync(GPUMapMode.READ);
    const dados = new Float32Array(destino.getMappedRange().slice(0, n * BYTES_PARTICULA));
    destino.unmap();
    destino.destroy();
    return dados;
  }

  destruir() {
    this.destruido = true;
    this.ar.trabalhador.terminate();
    this.ar.leitura.destroy();
    this.pressao.destruir();
    this.carimbos?.destroy();
    this.carimbosResolvidos?.destroy();
    for (const l of this.carimbosLeitura ?? []) l.buffer.destroy();
    for (const b of Object.values(this.buffers)) b.destroy();
    this.leitura.destroy();
  }
}
