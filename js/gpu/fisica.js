// A física da água na placa de vídeo (WebGPU). Mesmos passos, mesma ordem e
// mesmas constantes do motor da CPU (../fisica-cpu.js), que é a referência
// testada. A diferença é que aqui cada passo é um compute shader, e a soma das
// partículas na grade usa inteiros atômicos (ponto fixo) porque a GPU não tem
// soma atômica de float.
//
// Por quadro: ferramenta → partículas novas → compactação (tira as mortas)
// → N subpassos (P2G, faces, células, pressão, gradiente, extrapolação, G2P).

import {
  GRAVIDADE, PARTICULAS_POR_CELULA, TAU_DENSIDADE,
  CAMADAS_EXTRAPOLACAO, RAIO_PARTICULA, SDF_MAX, ESPUMA_V0, ESPUMA_V1, ESPUMA_TAXA, ESPUMA_VIDA, DIFUSAO_CORANTE,
  RES_MASCARA, PARTICULAS_MAX_POR_CELULA, ITERACOES_PCG,
} from '../constantes.js';
import { CAMPOS } from '../fisica-cpu.js';
import { PressaoPCG } from './pcg.js';

const GRUPO = 256;
const NOVAS_MAX = 1 << 18; // partículas novas por quadro
const BYTES_PARTICULA = CAMPOS * 4;

// --------------------------------------------------------------------- WGSL

const CABECALHO = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, mw: u32, mh: u32,
  h: f32, dt: f32, gx: f32, gy: f32,
  tauDens: f32, _livre1: f32, rPart: f32, sdfMax: f32,
  _livre2: f32, espDecai: f32, maxPart: u32, nNovas: u32,
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
  atomicAdd(&corAcc[3u * cc], i32(round(pa.extra.y * 256.0)));
  atomicAdd(&corAcc[3u * cc + 1u], i32(round(pa.extra.z * 256.0)));
  atomicAdd(&corAcc[3u * cc + 2u], i32(round(pa.extra.w * 256.0)));
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
  celulas: {
    b: [['celAcc', 'array<i32>'], ['u', 'array<f32>'], ['v', 'array<f32>'], ['sdf', 'array<f32>'],
      ['tipo', 'array<u32>'], ['div', 'array<f32>'], ['q', 'array<f32>']],
    sdf: true,
    corpo: /* wgsl */ `
fn temAgua(i: i32, j: i32) -> bool {
  return celAcc[2u * u32(i + j * i32(P.nx)) + 1u] > 0 && !celulaSolida(i, j);
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
  var t = AR;
  if (celulaSolida(i, j)) { t = SOLIDO; } else if (celAcc[2u * c + 1u] > 0) { t = AGUA; }
  tipo[c] = t;
  if (t != AGUA) { div[c] = 0.0; q[c] = 0.0; return; }
  let nx = P.nx;
  let d = u[u32(i) + 1u + u32(j) * (nx + 1u)] - u[u32(i) + u32(j) * (nx + 1u)] +
    v[u32(i) + u32(j + 1) * nx] - v[u32(i) + u32(j) * nx];
  var excesso = f32(celAcc[2u * c]) / ESCALA / P.ppc - 1.0;
  if (excesso < 0.0 && !miolo(i, j)) { excesso = 0.0; }
  div[c] = d - P.h / max(P.tauDens, P.dt) * excesso;
}`,
  },

  // --------------------------------------- gradiente de pressão
  gradiente: {
    b: [['tipo', 'array<u32>'], ['q', 'array<f32>'], ['u', 'array<f32>'], ['v', 'array<f32>'],
      ['uVal', 'array<u32>'], ['vVal', 'array<u32>']],
    corpo: /* wgsl */ `
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let nx = P.nx; let ny = P.ny;
  if (k < (nx + 1u) * ny) {
    let i = k % (nx + 1u); let j = k / (nx + 1u);
    var tl = SOLIDO; var tr = SOLIDO;
    if (i > 0u) { tl = tipo[i - 1u + j * nx]; }
    if (i < nx) { tr = tipo[i + j * nx]; }
    if (tl == SOLIDO || tr == SOLIDO) { u[k] = 0.0; uVal[k] = 1u; }
    else if (tl == AGUA || tr == AGUA) { u[k] = u[k] - (q[i + j * nx] - q[i - 1u + j * nx]); uVal[k] = 1u; }
    else { uVal[k] = 0u; }
  }
  if (k < nx * (ny + 1u)) {
    let i = k % nx; let j = k / nx;
    var tb = SOLIDO; var tt = SOLIDO;
    if (j > 0u) { tb = tipo[i + (j - 1u) * nx]; }
    if (j < ny) { tt = tipo[i + j * nx]; }
    if (tb == SOLIDO || tt == SOLIDO) { v[k] = 0.0; vVal[k] = 1u; }
    else if (tb == AGUA || tt == AGUA) { v[k] = v[k] - (q[i + j * nx] - q[i + (j - 1u) * nx]); vVal[k] = 1u; }
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
  let vmax = 0.95 * P.h / P.dt;
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
  x += vel * P.dt;
  let d = sdfEm(x);
  if (d < P.rPart) {
    if (d < -P.sdfMax + 0.5) { pIn[k].pos = vec2f(-1.0, -1.0); return; }
    let n = normalEm(x);
    x += n * (P.rPart - d) * P.h;
    let vn = dot(vel, n);
    if (vn < 0.0) { vel -= vn * n; }
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
    this.pressao = new PressaoPCG(device, { nx: this.nx, ny: this.ny, tipo: this.buffers.tipo, div: this.buffers.div, q: this.buffers.q },
      { iteracoes: ITERACOES_PCG });
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
    for (const n of ['tipo', 'div', 'q']) this.#buffer(n, nc * 4);
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
    this.leitura = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.lendo = false;
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
  }

  #escreverParams(dt, ferramenta, nNovas, checar, medir) {
    const f = ferramenta ?? { tipo: 'nenhuma' };
    const tipos = { nenhuma: 0, esponja: 1, mao: 2, corante: 3 };
    const buf = new ArrayBuffer(112);
    const u = new Uint32Array(buf), fl = new Float32Array(buf);
    const c = this.cena;
    u[0] = this.nx; u[1] = this.ny; u[2] = c.mw; u[3] = c.mh;
    fl[4] = this.h; fl[5] = dt; fl[6] = this.gravidade[0]; fl[7] = this.gravidade[1];
    fl[8] = this.tauDensidade ?? TAU_DENSIDADE; fl[9] = 0; fl[10] = RAIO_PARTICULA; fl[11] = SDF_MAX;
    fl[12] = 0; fl[13] = Math.exp(-dt / ESPUMA_VIDA); u[14] = this.maxParticulas; u[15] = nNovas;
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
    for (let s = 0; s < subpassos; s++) {
      enc.clearBuffer(this.buffers.uAcc);
      enc.clearBuffer(this.buffers.vAcc);
      enc.clearBuffer(this.buffers.celAcc);
      enc.clearBuffer(this.buffers.corAcc);
      pass = enc.beginComputePass({ label: 'subpasso' });
      this.#despacharParticulas(pass, 'p2g', tp);
      this.#despachar(pass, 'faces', nf);
      this.#despachar(pass, 'celulas', nc);
      this.pressao.gravar(pass);
      this.#despachar(pass, 'gradiente', nf);
      for (let c = 0; c < CAMADAS_EXTRAPOLACAO; c++) {
        const ida = c % 2 === 0;
        this.#despachar(pass, 'extrapolar', nf, ida
          ? { uI: 'u', vI: 'v', uValI: 'uVal', vValI: 'vVal', uO: 'uTmp', vO: 'vTmp', uValO: 'uValTmp', vValO: 'vValTmp' }
          : { uI: 'uTmp', vI: 'vTmp', uValI: 'uValTmp', vValI: 'vValTmp', uO: 'u', vO: 'v', uValO: 'uVal', vValO: 'vVal' });
      }
      if (CAMADAS_EXTRAPOLACAO % 2 === 1) throw new Error('CAMADAS_EXTRAPOLACAO precisa ser par (o resultado volta para u e v).');
      this.#despacharParticulas(pass, 'g2p', tp);
      pass.end();
    }
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
    this.pressao.destruir();
    for (const b of Object.values(this.buffers)) b.destroy();
    this.leitura.destroy();
  }
}
