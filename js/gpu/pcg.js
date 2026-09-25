// Pressão na GPU: gradiente conjugado pré-condicionado com multigrid (MGPCG,
// McAdams, Sifakis e Teran). Espelho de ../pressao-cpu.js — mesma equação,
// mesmos passos:
//     n·q − Σ q_vizinhos = −div   (nas células com água; parede fora da soma,
//                                  ar da atmosfera valendo zero)
// Cada bolha de ar fechada é uma incógnita a mais (y, a pressão dela), ligada
// a todas as células de água que encostam nela:
//     Σ_faces (y − q_água) = b_bolha
// Nas linhas da água, a vizinha de bolha soma o y da bolha; as linhas das
// bolhas são somadas face a face com soma atômica de float (troca-e-compara).
// Pré-condicionador: ciclo em V em que as bolhas contam como água (o resíduo
// da bolha é espalhado pelas células dela antes, e a bolha pega a média da
// correção depois — uma coisa é a transposta da outra). Beta de Polak-Ribière.

import { Multigrid } from './multigrid.js';
import { MAX_BOLHAS } from '../constantes.js';

const GRUPO = 256;
const GRUPOS_BOLHAS = Math.ceil(MAX_BOLHAS / GRUPO);

const CABECALHO = /* wgsl */ `
struct Tam { nx: u32, ny: u32, nGrupos: u32, modo: u32 };
const SOLIDO: u32 = 0u;
const AGUA: u32 = 1u;
const AR: u32 = 2u;
const RZ: u32 = 0u;
const ALFA: u32 = 1u;
const BETA: u32 = 2u;
const MAXB: u32 = ${MAX_BOLHAS}u;
`;

// A·v na célula de água c. A vizinha de água soma v; a de bolha, o valor da
// bolha (vb); a de ar da atmosfera, zero. ("tipo" e "rotulo" são globais.)
const aplicarA = (v, vb) => /* wgsl */ `
fn valor(n: u32) -> f32 {
  let t = tipo[n];
  if (t == AGUA) { return ${v}[n]; }
  if (t == AR && rotulo[n] >= 2u) { return ${vb}[rotulo[n]]; }
  return 0.0;
}
fn aplicarA(c: u32) -> f32 {
  let i = c % T.nx; let j = c / T.nx;
  var soma = 0.0; var n = 0.0;
  if (i > 0u && tipo[c - 1u] != SOLIDO) { n += 1.0; soma += valor(c - 1u); }
  if (i < T.nx - 1u && tipo[c + 1u] != SOLIDO) { n += 1.0; soma += valor(c + 1u); }
  if (j > 0u && tipo[c - T.nx] != SOLIDO) { n += 1.0; soma += valor(c - T.nx); }
  if (j < T.ny - 1u && tipo[c + T.nx] != SOLIDO) { n += 1.0; soma += valor(c + T.nx); }
  return n * ${v}[c] - soma;
}`;

// soma atômica de float por troca-e-compara (a GPU só tem soma atômica de inteiro)
const SOMA_ATOMICA = /* wgsl */ `
fn somarF(i: u32, v: f32) {
  var velho = atomicLoad(&acc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&acc[i], velho, bitcast<u32>(bitcast<f32>(velho) + v));
    if (r.exchanged) { break; }
    velho = r.old_value;
  }
}`;

// Soma v na bolha b com uma troca-e-compara por trecho contíguo de células da
// mesma bolha dentro do grupo, e não uma por célula: milhares de células
// somando no mesmo endereço viram fila (a pressão ficava ~8× mais lenta com
// uma bolha grande). Tem que ser chamada por todas as threads do grupo.
const SOMA_NO_GRUPO = /* wgsl */ `
${SOMA_ATOMICA}
const NENHUMA: u32 = 0xffffffffu;
var<workgroup> gLab: array<u32, ${GRUPO}>;
var<workgroup> gVal: array<f32, ${GRUPO}>;
fn somarNoGrupo(t: u32, b: u32, v: f32) {
  gLab[t] = b;
  gVal[t] = v;
  workgroupBarrier();
  if (b != NENHUMA && (t == 0u || gLab[t - 1u] != b)) {
    var s = v;
    var k = t + 1u;
    loop {
      if (k >= ${GRUPO}u) { break; }
      if (gLab[k] != b) { break; }
      s += gVal[k];
      k++;
    }
    somarF(b, s);
  }
}`;

// r = −div − A·q ; z = 0 (o ciclo em V parte de zero)
const INICIO = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> r: array<f32>;
@group(0) @binding(5) var<storage, read_write> z: array<f32>;
@group(0) @binding(6) var<storage, read_write> rotulo: array<u32>;
@group(0) @binding(7) var<storage, read_write> yb: array<f32>;
${aplicarA('q', 'yb')}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  z[c] = 0.0;
  if (tipo[c] != AGUA) { r[c] = 0.0; return; }
  r[c] = -div[c] - aplicarA(c);
}`;

// p = z
const COPIAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  p[c] = z[c];
}`;

// Ap = A·p (linhas da água)
const APLICAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> ap: array<f32>;
@group(0) @binding(4) var<storage, read_write> rotulo: array<u32>;
@group(0) @binding(5) var<storage, read_write> pb: array<f32>;
${aplicarA('p', 'pb')}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  if (tipo[c] != AGUA) { ap[c] = 0.0; return; }
  ap[c] = aplicarA(c);
}`;

// linhas das bolhas: acc[b] = Σ_faces (vb[b] − v[água])
const APLICAR_BOLHAS = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> rotulo: array<u32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> vb: array<f32>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<u32>>;
${SOMA_NO_GRUPO}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) t: u32) {
  let c = g.x;
  var alvo = NENHUMA; var s = 0.0;
  if (c < T.nx * T.ny) {
    let b = rotulo[c];
    let i = c % T.nx; let j = c / T.nx;
    if (tipo[c] == AR && b >= 2u && b < MAXB && i >= 1u && j >= 1u && i < T.nx - 1u && j < T.ny - 1u) {
      let y = vb[b];
      var k = 0u;
      if (tipo[c - 1u] == AGUA) { s += y - v[c - 1u]; k++; }
      if (tipo[c + 1u] == AGUA) { s += y - v[c + 1u]; k++; }
      if (tipo[c - T.nx] == AGUA) { s += y - v[c - T.nx]; k++; }
      if (tipo[c + T.nx] == AGUA) { s += y - v[c + T.nx]; k++; }
      if (k > 0u) { alvo = b; }
    }
  }
  somarNoGrupo(t, alvo, s);
}`;

// zera o acumulador das bolhas
const ZERAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<storage, read_write> acc: array<u32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x < MAXB) { acc[g.x] = 0u; }
}`;

// resíduo inicial das bolhas: rb = rhs − (A·y)_b
const BOLHAS_RESIDUO = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<storage, read_write> faces: array<u32>;
@group(0) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;
@group(0) @binding(3) var<storage, read_write> rb: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b >= MAXB) { return; }
  var r = 0.0;
  if (faces[b] > 0u) { r = rhs[b] - acc[b]; }
  rb[b] = r;
}`;

// tipos que o multigrid vê: bolha ativa conta como água
const TIPOS_MG = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> rotulo: array<u32>;
@group(0) @binding(3) var<storage, read_write> faces: array<u32>;
@group(0) @binding(4) var<storage, read_write> tipoMG: array<u32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  var t = tipo[c];
  let b = rotulo[c];
  if (t == AR && b >= 2u && b < MAXB && faces[b] > 0u) { t = AGUA; }
  tipoMG[c] = t;
}`;

// antes do ciclo em V: o resíduo da bolha espalhado igual pelas células dela
const ESPALHAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> rotulo: array<u32>;
@group(0) @binding(3) var<storage, read_write> faces: array<u32>;
@group(0) @binding(4) var<storage, read_write> vol: array<u32>;
@group(0) @binding(5) var<storage, read_write> rb: array<f32>;
@group(0) @binding(6) var<storage, read_write> r: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  let b = rotulo[c];
  if (tipo[c] != AR || b < 2u || b >= MAXB || faces[b] == 0u || vol[b] == 0u) { return; }
  r[c] = rb[b] / f32(vol[b]);
}`;

// depois do ciclo em V: a bolha junta a média da correção; a grade volta a zero ali
const JUNTAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> rotulo: array<u32>;
@group(0) @binding(3) var<storage, read_write> faces: array<u32>;
@group(0) @binding(4) var<storage, read_write> vol: array<u32>;
@group(0) @binding(5) var<storage, read_write> z: array<f32>;
@group(0) @binding(6) var<storage, read_write> r: array<f32>;
@group(0) @binding(7) var<storage, read_write> acc: array<atomic<u32>>;
${SOMA_NO_GRUPO}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) t: u32) {
  let c = g.x;
  var alvo = NENHUMA; var s = 0.0;
  if (c < T.nx * T.ny) {
    let b = rotulo[c];
    if (tipo[c] == AR && b >= 2u && b < MAXB && faces[b] > 0u && vol[b] > 0u) {
      alvo = b;
      s = z[c] / f32(vol[b]);
      z[c] = 0.0;
      r[c] = 0.0;
    }
  }
  somarNoGrupo(t, alvo, s);
}`;

// z das bolhas: zVelho = z ; z = média juntada (modo 0: também p = z)
const BOLHAS_Z = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> faces: array<u32>;
@group(0) @binding(2) var<storage, read_write> accZ: array<f32>;
@group(0) @binding(3) var<storage, read_write> zb: array<f32>;
@group(0) @binding(4) var<storage, read_write> zvb: array<f32>;
@group(0) @binding(5) var<storage, read_write> pb: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b >= MAXB) { return; }
  zvb[b] = zb[b];
  var z = 0.0;
  if (faces[b] > 0u) { z = accZ[b]; }
  zb[b] = z;
  if (T.modo == 0u) { pb[b] = z; }
}`;

// y += α·p ; r −= α·Ap (nas bolhas; Ap está no acumulador)
const BOLHAS_ATUALIZAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<storage, read_write> esc: array<f32>;
@group(0) @binding(1) var<storage, read_write> yb: array<f32>;
@group(0) @binding(2) var<storage, read_write> rb: array<f32>;
@group(0) @binding(3) var<storage, read_write> pb: array<f32>;
@group(0) @binding(4) var<storage, read_write> acc: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b >= MAXB) { return; }
  let a = esc[ALFA];
  yb[b] = yb[b] + a * pb[b];
  rb[b] = rb[b] - a * acc[b];
}`;

// p = z + β·p (nas bolhas)
const BOLHAS_DIRECAO = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<storage, read_write> esc: array<f32>;
@group(0) @binding(1) var<storage, read_write> zb: array<f32>;
@group(0) @binding(2) var<storage, read_write> pb: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b >= MAXB) { return; }
  pb[b] = zb[b] + esc[BETA] * pb[b];
}`;

// q += α·p ; r −= α·Ap ; zVelho = z ; z = 0
const ATUALIZAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> esc: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> p: array<f32>;
@group(0) @binding(5) var<storage, read_write> ap: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
@group(0) @binding(7) var<storage, read_write> zVelho: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  let a = esc[ALFA];
  q[c] = q[c] + a * p[c];
  r[c] = r[c] - a * ap[c];
  zVelho[c] = z[c];
  z[c] = 0.0;
}`;

// p = z + β·p
const DIRECAO = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> esc: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  p[c] = z[c] + esc[BETA] * p[c];
}`;

// Produto interno em dois canais (a·b e a·b2), somado por grupo.
const REDUZIR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> b2: array<f32>;
@group(0) @binding(4) var<storage, read_write> parciais: array<vec2f>;
var<workgroup> soma: array<vec2f, ${GRUPO}>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3u) {
  let c = g.x;
  var v = vec2f(0.0);
  if (c < T.nx * T.ny) { v = vec2f(a[c] * b[c], a[c] * b2[c]); }
  soma[l] = v;
  workgroupBarrier();
  for (var s = ${GRUPO / 2}u; s > 0u; s >>= 1u) {
    if (l < s) { soma[l] += soma[l + s]; }
    workgroupBarrier();
  }
  if (l == 0u) { parciais[w.x] = soma[0]; }
}`;

// Soma as parciais da água, soma a parte das bolhas e faz a conta escalar.
//   modo 0: rz inicial (Σ r·z)
//   modo 1: α = rz / Σ p·Ap
//   modo 2: β de Polak-Ribière e rz novo
const ESCALAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> parciais: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> esc: array<f32>;
@group(0) @binding(3) var<storage, read_write> rb: array<f32>;
@group(0) @binding(4) var<storage, read_write> zb: array<f32>;
@group(0) @binding(5) var<storage, read_write> zvb: array<f32>;
@group(0) @binding(6) var<storage, read_write> pb: array<f32>;
@group(0) @binding(7) var<storage, read_write> acc: array<f32>;
var<workgroup> soma: array<vec2f, ${GRUPO}>;
fn dividir(a: f32, b: f32) -> f32 { if (abs(b) < 1e-30) { return 0.0; } return a / b; }
@compute @workgroup_size(${GRUPO}) fn main(@builtin(local_invocation_index) l: u32) {
  var v = vec2f(0.0);
  for (var k = l; k < T.nGrupos; k += ${GRUPO}u) { v += parciais[k]; }
  for (var b = l; b < MAXB; b += ${GRUPO}u) {
    if (T.modo == 1u) { v.x += pb[b] * acc[b]; }
    else { v += vec2f(rb[b] * zb[b], rb[b] * zvb[b]); }
  }
  soma[l] = v;
  workgroupBarrier();
  for (var s = ${GRUPO / 2}u; s > 0u; s >>= 1u) {
    if (l < s) { soma[l] += soma[l + s]; }
    workgroupBarrier();
  }
  if (l != 0u) { return; }
  let t = soma[0];
  if (T.modo == 0u) { esc[RZ] = t.x; esc[BETA] = 0.0; }
  else if (T.modo == 1u) { esc[ALFA] = dividir(esc[RZ], t.x); }
  else {
    esc[BETA] = max(0.0, dividir(t.x - t.y, esc[RZ]));
    esc[RZ] = t.x;
  }
}`;

export class PressaoPCG {
  /**
   * @param {GPUDevice} device
   * @param {{nx:number, ny:number, tipo:GPUBuffer, div:GPUBuffer, q:GPUBuffer,
   *          rotulo:GPUBuffer, facesBolha:GPUBuffer, rhsBolha:GPUBuffer, yBolha:GPUBuffer, volBolha:GPUBuffer}} grade
   */
  constructor(device, grade, opcoes = {}) {
    this.device = device;
    this.nx = grade.nx;
    this.ny = grade.ny;
    this.iteracoes = opcoes.iteracoes ?? 5;
    const n = this.nx * this.ny;
    this.nGrupos = Math.ceil(n / GRUPO);
    const uso = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const vetor = (label, k = n) => device.createBuffer({ label, size: Math.max(16, k * 4), usage: uso });
    this.b = {
      tipo: grade.tipo, div: grade.div, q: grade.q, rotulo: grade.rotulo,
      faces: grade.facesBolha, rhs: grade.rhsBolha, yb: grade.yBolha,
      r: vetor('r'), z: vetor('z'), zTmp: vetor('zTmp'), zVelho: vetor('zVelho'), p: vetor('p'), ap: vetor('ap'),
      rb: vetor('rb', MAX_BOLHAS), zb: vetor('zb', MAX_BOLHAS), zvb: vetor('zvb', MAX_BOLHAS),
      pb: vetor('pb', MAX_BOLHAS), acc: vetor('acc', MAX_BOLHAS), accZ: vetor('accZ', MAX_BOLHAS),
      vol: grade.volBolha, tipoMG: vetor('tipoMG'),
      parciais: device.createBuffer({ size: Math.max(16, this.nGrupos * 8), usage: uso }),
      esc: device.createBuffer({ size: 16, usage: uso }),
    };
    this.tams = [0, 1, 2].map((modo) => {
      const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(u, 0, new Uint32Array([this.nx, this.ny, this.nGrupos, modo]));
      return u;
    });
    // o pré-condicionador: um ciclo em V sobre o resíduo (b = r, x = z), com as bolhas contando como água
    this.mg = new Multigrid(device, { nx: this.nx, ny: this.ny, tipo: this.b.tipoMG, b: this.b.r, x: this.b.z, tmp: this.b.zTmp },
      { varreduras: opcoes.varreduras });

    const pipe = (code, label) => device.createComputePipeline({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    const grupo = (p, recursos) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: recursos.map((r, i) => ({ binding: i, resource: { buffer: r } })),
    });
    const B = this.b, T0 = this.tams[0];
    this.pInicio = pipe(INICIO, 'pcg-inicio');
    this.gInicio = grupo(this.pInicio, [T0, B.tipo, B.div, B.q, B.r, B.z, B.rotulo, B.yb]);
    this.pCopiar = pipe(COPIAR, 'pcg-copiar');
    this.gCopiar = grupo(this.pCopiar, [T0, B.z, B.p]);
    this.pAplicar = pipe(APLICAR, 'pcg-aplicar');
    this.gAplicar = grupo(this.pAplicar, [T0, B.tipo, B.p, B.ap, B.rotulo, B.pb]);
    this.pAtualizar = pipe(ATUALIZAR, 'pcg-atualizar');
    this.gAtualizar = grupo(this.pAtualizar, [T0, B.esc, B.q, B.r, B.p, B.ap, B.z, B.zVelho]);
    this.pDirecao = pipe(DIRECAO, 'pcg-direcao');
    this.gDirecao = grupo(this.pDirecao, [T0, B.esc, B.z, B.p]);
    this.pReduzir = pipe(REDUZIR, 'pcg-reduzir');
    this.gReduzirRZ = grupo(this.pReduzir, [T0, B.r, B.z, B.zVelho, B.parciais]); // (r·z, r·zVelho)
    this.gReduzirPAP = grupo(this.pReduzir, [T0, B.p, B.ap, B.zVelho, B.parciais]); // (p·Ap, canal 2 ignorado)
    this.pEscalar = pipe(ESCALAR, 'pcg-escalar');
    this.gEscalar = this.tams.map((t) => grupo(this.pEscalar, [t, B.parciais, B.esc, B.rb, B.zb, B.zvb, B.pb, B.acc]));
    // bolhas
    this.pAplicarBolhas = pipe(APLICAR_BOLHAS, 'pcg-aplicar-bolhas');
    this.gAplicarBolhasQ = grupo(this.pAplicarBolhas, [T0, B.tipo, B.rotulo, B.q, B.yb, B.acc]);
    this.gAplicarBolhasP = grupo(this.pAplicarBolhas, [T0, B.tipo, B.rotulo, B.p, B.pb, B.acc]);
    this.pZerar = pipe(ZERAR, 'pcg-zerar');
    this.gZerar = grupo(this.pZerar, [B.acc]);
    this.gZerarZ = grupo(this.pZerar, [B.accZ]);
    this.pBolhasResiduo = pipe(BOLHAS_RESIDUO, 'pcg-bolhas-residuo');
    this.gBolhasResiduo = grupo(this.pBolhasResiduo, [B.faces, B.rhs, B.acc, B.rb]);
    this.pTiposMG = pipe(TIPOS_MG, 'pcg-tipos-mg');
    this.gTiposMG = grupo(this.pTiposMG, [T0, B.tipo, B.rotulo, B.faces, B.tipoMG]);
    this.pEspalhar = pipe(ESPALHAR, 'pcg-espalhar');
    this.gEspalhar = grupo(this.pEspalhar, [T0, B.tipo, B.rotulo, B.faces, B.vol, B.rb, B.r]);
    this.pJuntar = pipe(JUNTAR, 'pcg-juntar');
    this.gJuntar = grupo(this.pJuntar, [T0, B.tipo, B.rotulo, B.faces, B.vol, B.z, B.r, B.accZ]);
    this.pBolhasZ = pipe(BOLHAS_Z, 'pcg-bolhas-z');
    this.gBolhasZ = [0, 2].map((m) => grupo(this.pBolhasZ, [this.tams[m], B.faces, B.accZ, B.zb, B.zvb, B.pb]));
    this.pBolhasAtualizar = pipe(BOLHAS_ATUALIZAR, 'pcg-bolhas-atualizar');
    this.gBolhasAtualizar = grupo(this.pBolhasAtualizar, [B.esc, B.yb, B.rb, B.pb, B.acc]);
    this.pBolhasDirecao = pipe(BOLHAS_DIRECAO, 'pcg-bolhas-direcao');
    this.gBolhasDirecao = grupo(this.pBolhasDirecao, [B.esc, B.zb, B.pb]);
  }

  #celulas(pass, p, g) {
    pass.setPipeline(p);
    pass.setBindGroup(0, g);
    pass.dispatchWorkgroups(this.nGrupos);
  }

  #bolhas(pass, p, g) {
    pass.setPipeline(p);
    pass.setBindGroup(0, g);
    pass.dispatchWorkgroups(GRUPOS_BOLHAS);
  }

  #produto(pass, gReduzir, modo) {
    this.#celulas(pass, this.pReduzir, gReduzir);
    pass.setPipeline(this.pEscalar);
    pass.setBindGroup(0, this.gEscalar[modo]);
    pass.dispatchWorkgroups(1);
  }

  /** Ciclo em V com as bolhas: espalha o resíduo delas, suaviza, e junta a média (modo 0: também p = z). */
  #precondicionar(pass, modo, bolhas) {
    if (bolhas) this.#celulas(pass, this.pEspalhar, this.gEspalhar);
    this.mg.gravar(pass);
    if (!bolhas) return;
    this.#bolhas(pass, this.pZerar, this.gZerarZ);
    this.#celulas(pass, this.pJuntar, this.gJuntar);
    this.#bolhas(pass, this.pBolhasZ, this.gBolhasZ[modo === 0 ? 0 : 1]);
  }

  /** Zera os vetores das bolhas (chamar fora de um passe quando as bolhas acabam). */
  zerarBolhas(enc) {
    for (const k of ['rb', 'zb', 'zvb', 'pb', 'acc', 'accZ']) enc.clearBuffer(this.b[k]);
  }

  /**
   * Grava a solução no passe. Os tipos, a divergência, os rótulos e o lado
   * direito das bolhas já estão prontos; q e y guardam o chute inicial.
   * Sem bolha nenhuma (o caso comum), os passos delas são pulados — os
   * vetores delas precisam estar zerados (zerarBolhas).
   */
  gravar(pass, bolhas = true) {
    this.#celulas(pass, this.pTiposMG, this.gTiposMG);
    this.mg.gravarTipos(pass);
    if (bolhas) {
      this.#bolhas(pass, this.pZerar, this.gZerar);
      this.#celulas(pass, this.pAplicarBolhas, this.gAplicarBolhasQ);
    }
    this.#celulas(pass, this.pInicio, this.gInicio);
    if (bolhas) this.#bolhas(pass, this.pBolhasResiduo, this.gBolhasResiduo);
    this.#precondicionar(pass, 0, bolhas);
    this.#celulas(pass, this.pCopiar, this.gCopiar);
    this.#produto(pass, this.gReduzirRZ, 0);
    for (let k = 0; k < this.iteracoes; k++) {
      this.#celulas(pass, this.pAplicar, this.gAplicar);
      if (bolhas) {
        this.#bolhas(pass, this.pZerar, this.gZerar);
        this.#celulas(pass, this.pAplicarBolhas, this.gAplicarBolhasP);
      }
      this.#produto(pass, this.gReduzirPAP, 1);
      this.#celulas(pass, this.pAtualizar, this.gAtualizar);
      if (bolhas) this.#bolhas(pass, this.pBolhasAtualizar, this.gBolhasAtualizar);
      if (k === this.iteracoes - 1) break; // a última atualização de q e y já basta
      this.#precondicionar(pass, 2, bolhas);
      this.#produto(pass, this.gReduzirRZ, 2);
      this.#celulas(pass, this.pDirecao, this.gDirecao);
      if (bolhas) this.#bolhas(pass, this.pBolhasDirecao, this.gBolhasDirecao);
    }
  }

  destruir() {
    this.mg.destruir();
    for (const k of ['r', 'z', 'zTmp', 'zVelho', 'p', 'ap', 'rb', 'zb', 'zvb', 'pb', 'acc', 'accZ', 'tipoMG', 'parciais', 'esc']) this.b[k].destroy();
    for (const t of this.tams) t.destroy();
  }
}
