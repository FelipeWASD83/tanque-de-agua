// Pressão na GPU por gradiente conjugado pré-condicionado com multigrid
// (MGPCG, o método de McAdams, Sifakis e Teran). O multigrid sozinho é rápido,
// mas numa grade grossa uma célula pode juntar a água dos dois lados de uma
// parede fina e a correção sai errada — a água explode. Dentro do gradiente
// conjugado ele só precisa apontar mais ou menos a direção certa, e a
// convergência fica garantida.
//
// Mesma equação do motor da CPU: n·q − Σ q_vizinhos = −div nas células com
// água (parede fora da soma, ar valendo zero). Guardamos o resíduo com o sinal
// trocado ("nr" = −r = A·q + div), que é a convenção que o multigrid espera.
// Beta de Polak-Ribière (gradiente conjugado flexível): o suavizador em blocos
// da GPU não é exatamente o mesmo operador a cada chamada.

import { Multigrid } from './multigrid.js';

const GRUPO = 256;

// A·v na célula c (v é o nome do vetor global; "tipo" também é global).
const aplicarA = (v) => /* wgsl */ `
fn aplicarA(c: u32) -> f32 {
  let i = c % T.nx; let j = c / T.nx;
  var soma = 0.0; var n = 0.0;
  if (i > 0u && tipo[c - 1u] != SOLIDO) { n += 1.0; soma += ${v}[c - 1u]; }
  if (i < T.nx - 1u && tipo[c + 1u] != SOLIDO) { n += 1.0; soma += ${v}[c + 1u]; }
  if (j > 0u && tipo[c - T.nx] != SOLIDO) { n += 1.0; soma += ${v}[c - T.nx]; }
  if (j < T.ny - 1u && tipo[c + T.nx] != SOLIDO) { n += 1.0; soma += ${v}[c + T.nx]; }
  return n * ${v}[c] - soma;
}`;

const CABECALHO = /* wgsl */ `
struct Tam { nx: u32, ny: u32, nGrupos: u32, modo: u32 };
const SOLIDO: u32 = 0u;
const AGUA: u32 = 1u;
// índices do vetor de escalares
const RZ: u32 = 0u;
const ALFA: u32 = 1u;
const BETA: u32 = 2u;
`;

// nr = A·q + div ; z = 0
const INICIO = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> nr: array<f32>;
@group(0) @binding(5) var<storage, read_write> z: array<f32>;
${aplicarA('q')}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  z[c] = 0.0;
  if (tipo[c] != AGUA) { nr[c] = 0.0; q[c] = 0.0; return; }
  nr[c] = aplicarA(c) + div[c];
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

// Ap = A·p
const APLICAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> ap: array<f32>;
${aplicarA('p')}
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  if (tipo[c] != AGUA) { ap[c] = 0.0; return; }
  ap[c] = aplicarA(c);
}`;

// q += α·p ; nr += α·Ap ; zVelho = z ; z = 0
const ATUALIZAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> esc: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> nr: array<f32>;
@group(0) @binding(4) var<storage, read_write> p: array<f32>;
@group(0) @binding(5) var<storage, read_write> ap: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
@group(0) @binding(7) var<storage, read_write> zVelho: array<f32>;
@compute @workgroup_size(${GRUPO}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= T.nx * T.ny) { return; }
  let a = esc[ALFA];
  q[c] = q[c] + a * p[c];
  nr[c] = nr[c] + a * ap[c];
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

// Produto interno em dois canais: soma de a·b e de a·b2, por grupo.
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

// Soma as parciais num grupo só e faz a conta escalar do gradiente conjugado.
//   modo 0: rz inicial     (rz = −Σ nr·z)
//   modo 1: α = rz / Σ p·Ap
//   modo 2: β de Polak-Ribière e rz novo
const ESCALAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> T: Tam;
@group(0) @binding(1) var<storage, read_write> parciais: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> esc: array<f32>;
var<workgroup> soma: array<vec2f, ${GRUPO}>;
fn dividir(a: f32, b: f32) -> f32 { if (abs(b) < 1e-30) { return 0.0; } return a / b; }
@compute @workgroup_size(${GRUPO}) fn main(@builtin(local_invocation_index) l: u32) {
  var v = vec2f(0.0);
  for (var k = l; k < T.nGrupos; k += ${GRUPO}u) { v += parciais[k]; }
  soma[l] = v;
  workgroupBarrier();
  for (var s = ${GRUPO / 2}u; s > 0u; s >>= 1u) {
    if (l < s) { soma[l] += soma[l + s]; }
    workgroupBarrier();
  }
  if (l != 0u) { return; }
  let t = soma[0];
  if (T.modo == 0u) { esc[RZ] = -t.x; esc[BETA] = 0.0; }
  else if (T.modo == 1u) { esc[ALFA] = dividir(esc[RZ], t.x); }
  else {
    let rzNovo = -t.x;
    let cruzado = -t.y;
    esc[BETA] = max(0.0, dividir(rzNovo - cruzado, esc[RZ]));
    esc[RZ] = rzNovo;
  }
}`;

export class PressaoPCG {
  /**
   * @param {GPUDevice} device
   * @param {{nx:number, ny:number, tipo:GPUBuffer, div:GPUBuffer, q:GPUBuffer}} grade
   */
  constructor(device, grade, opcoes = {}) {
    this.device = device;
    this.nx = grade.nx;
    this.ny = grade.ny;
    this.iteracoes = opcoes.iteracoes ?? 3;
    const n = this.nx * this.ny;
    this.nGrupos = Math.ceil(n / GRUPO);
    const vetor = (label) => device.createBuffer({ label, size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE });
    this.b = {
      tipo: grade.tipo, div: grade.div, q: grade.q,
      nr: vetor('nr'), z: vetor('z'), zVelho: vetor('zVelho'), p: vetor('p'), ap: vetor('ap'),
      parciais: device.createBuffer({ size: Math.max(16, this.nGrupos * 8), usage: GPUBufferUsage.STORAGE }),
      esc: device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE }),
    };
    this.tams = [0, 1, 2].map((modo) => {
      const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(u, 0, new Uint32Array([this.nx, this.ny, this.nGrupos, modo]));
      return u;
    });
    // o pré-condicionador: um ciclo em V sobre o resíduo, partindo de zero
    this.mg = new Multigrid(device, { nx: this.nx, ny: this.ny, tipo: grade.tipo, div: this.b.nr, q: this.b.z },
      { ciclos: 1, omega: 1.0, suavizarFino: 1, suavizarGrosso: 6 });

    const pipe = (code, label) => device.createComputePipeline({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    const grupo = (p, recursos) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: recursos.map((r, i) => ({ binding: i, resource: { buffer: r } })),
    });
    const B = this.b, T0 = this.tams[0];
    this.pInicio = pipe(INICIO, 'pcg-inicio');
    this.gInicio = grupo(this.pInicio, [T0, B.tipo, B.div, B.q, B.nr, B.z]);
    this.pCopiar = pipe(COPIAR, 'pcg-copiar');
    this.gCopiar = grupo(this.pCopiar, [T0, B.z, B.p]);
    this.pAplicar = pipe(APLICAR, 'pcg-aplicar');
    this.gAplicar = grupo(this.pAplicar, [T0, B.tipo, B.p, B.ap]);
    this.pAtualizar = pipe(ATUALIZAR, 'pcg-atualizar');
    this.gAtualizar = grupo(this.pAtualizar, [T0, B.esc, B.q, B.nr, B.p, B.ap, B.z, B.zVelho]);
    this.pDirecao = pipe(DIRECAO, 'pcg-direcao');
    this.gDirecao = grupo(this.pDirecao, [T0, B.esc, B.z, B.p]);
    this.pReduzir = pipe(REDUZIR, 'pcg-reduzir');
    this.gReduzirRZ = grupo(this.pReduzir, [T0, B.nr, B.z, B.zVelho, B.parciais]); // (nr·z, nr·zVelho)
    this.gReduzirPAP = grupo(this.pReduzir, [T0, B.p, B.ap, B.zVelho, B.parciais]); // (p·Ap, canal 2 ignorado)
    this.pEscalar = pipe(ESCALAR, 'pcg-escalar');
    this.gEscalar = this.tams.map((t) => grupo(this.pEscalar, [t, B.parciais, B.esc]));
  }

  #celulas(pass, p, g) {
    pass.setPipeline(p);
    pass.setBindGroup(0, g);
    pass.dispatchWorkgroups(this.nGrupos);
  }

  #produto(pass, gReduzir, modo) {
    this.#celulas(pass, this.pReduzir, gReduzir);
    pass.setPipeline(this.pEscalar);
    pass.setBindGroup(0, this.gEscalar[modo]);
    pass.dispatchWorkgroups(1);
  }

  /** Grava a solução no passe (os tipos e a divergência já estão prontos; q guarda o chute inicial). */
  gravar(pass) {
    this.#celulas(pass, this.pInicio, this.gInicio);
    this.mg.gravar(pass);
    this.#celulas(pass, this.pCopiar, this.gCopiar);
    this.#produto(pass, this.gReduzirRZ, 0);
    for (let k = 0; k < this.iteracoes; k++) {
      this.#celulas(pass, this.pAplicar, this.gAplicar);
      this.#produto(pass, this.gReduzirPAP, 1);
      this.#celulas(pass, this.pAtualizar, this.gAtualizar);
      if (k === this.iteracoes - 1) break; // a última atualização de q já basta
      this.mg.gravar(pass);
      this.#produto(pass, this.gReduzirRZ, 2);
      this.#celulas(pass, this.pDirecao, this.gDirecao);
    }
  }

  destruir() {
    this.mg.destruir();
    for (const k of ['nr', 'z', 'zVelho', 'p', 'ap', 'parciais', 'esc']) this.b[k].destroy();
    for (const t of this.tams) t.destroy();
  }
}
