// Pré-condicionador multigrid da GPU: um ciclo em V sobre A·x = b, partindo
// de x = 0. É o espelho de ../pressao-cpu.js (lá está a explicação e o teste
// de convergência). Resumo:
//   - célula grossa: ar se algum filho é ar; senão água se algum é água;
//     senão parede;
//   - restrição = transposta da interpolação bilinear (pesos 1/4, 3/4, 3/4, 1/4);
//   - suavizador em blocos de 16×16 na memória do grupo, lendo um retrato
//     (x) e gravando em outro buffer (tmp): sem corrida entre blocos, o
//     resultado é o mesmo toda vez. As passadas vêm em pares, então o
//     resultado sempre volta para x.

const SOLIDO = 0, AGUA = 1, AR = 2;
/** Varreduras (ida e volta alternadas) na grade mais grossa, num despacho só. */
export const VARREDURAS_GROSSO = 64;


const CABECALHO = /* wgsl */ `
struct Nivel { nx: u32, ny: u32, nxF: u32, nyF: u32, omega: f32, _a: f32, _b: f32, _c: f32 };
struct Passada { desl: u32, ordem: u32, _b: u32, _c: u32 };
const SOLIDO: u32 = ${SOLIDO}u;
const AGUA: u32 = ${AGUA}u;
const AR: u32 = ${AR}u;
`;

// ordem 0: vermelho depois preto; ordem 1: preto depois vermelho (a volta simétrica)
const SUAVIZAR = (VARREDURAS) => /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<uniform> F: Passada;
@group(0) @binding(2) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(3) var<storage, read_write> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> tmp: array<f32>;
var<workgroup> qs: array<f32, 324>;
var<workgroup> ts: array<u32, 324>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let nx = i32(N.nx); let ny = i32(N.ny);
  let desl = i32(F.desl);
  let ox = i32(wg.x) * 16 - desl; let oy = i32(wg.y) * 16 - desl;
  let lid = l.y * 16u + l.x;
  for (var s = lid; s < 324u; s += 256u) {
    let sx = i32(s % 18u) - 1 + ox; let sy = i32(s / 18u) - 1 + oy;
    var qv = 0.0; var tv = SOLIDO;
    if (sx >= 0 && sy >= 0 && sx < nx && sy < ny) { let c = u32(sx + sy * nx); qv = x[c]; tv = tipo[c]; }
    qs[s] = qv; ts[s] = tv;
  }
  let gx = ox + i32(l.x); let gy = oy + i32(l.y);
  let dentro = gx >= 0 && gy >= 0 && gx < nx && gy < ny;
  let li = (l.y + 1u) * 18u + l.x + 1u;
  var bv = 0.0;
  if (dentro) { bv = b[u32(gx + gy * nx)]; }
  workgroupBarrier();
  let agua = dentro && ts[li] == AGUA;
  var n = 0.0;
  if (agua) {
    n = f32(ts[li - 1u] != SOLIDO) + f32(ts[li + 1u] != SOLIDO) + f32(ts[li - 18u] != SOLIDO) + f32(ts[li + 18u] != SOLIDO);
  }
  let minhaCor = u32(gx + gy + 1) & 1u;
  for (var it = 0; it < ${VARREDURAS}; it++) {
    for (var k = 0u; k < 2u; k++) {
      let cor = k ^ F.ordem;
      if (agua && n > 0.0 && minhaCor == cor) {
        var soma = 0.0;
        if (ts[li - 1u] != SOLIDO) { soma += qs[li - 1u]; }
        if (ts[li + 1u] != SOLIDO) { soma += qs[li + 1u]; }
        if (ts[li - 18u] != SOLIDO) { soma += qs[li - 18u]; }
        if (ts[li + 18u] != SOLIDO) { soma += qs[li + 18u]; }
        qs[li] = qs[li] + N.omega * ((soma + bv) / n - qs[li]);
      }
      workgroupBarrier();
    }
  }
  if (dentro) {
    var v = 0.0;
    if (agua) { v = qs[li]; }
    tmp[u32(gx + gy * nx)] = v;
  }
}`;

// Tipo da célula grossa a partir dos 4 filhos.
const TIPOS = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<storage, read_write> tipoF: array<u32>;
@group(0) @binding(2) var<storage, read_write> tipoC: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let C = g.x;
  if (C >= N.nx * N.ny) { return; }
  let I = C % N.nx; let J = C / N.nx;
  var temAr = false; var temAgua = false;
  for (var b = 0u; b < 2u; b++) { for (var a = 0u; a < 2u; a++) {
    let i = 2u * I + a; let j = 2u * J + b;
    if (i < N.nxF && j < N.nyF) {
      let t = tipoF[i + j * N.nxF];
      if (t == AR) { temAr = true; } else if (t == AGUA) { temAgua = true; }
    }
  } }
  var t = SOLIDO;
  if (temAr) { t = AR; } else if (temAgua) { t = AGUA; }
  tipoC[C] = t;
}`;

// b grosso = transposta da bilinear aplicada ao resíduo fino (r = b − A·x,
// calculado aqui mesmo); x grosso = 0.
const RESTRINGIR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<storage, read_write> tipoF: array<u32>;
@group(0) @binding(2) var<storage, read_write> bF: array<f32>;
@group(0) @binding(3) var<storage, read_write> xF: array<f32>;
@group(0) @binding(4) var<storage, read_write> tipoC: array<u32>;
@group(0) @binding(5) var<storage, read_write> bC: array<f32>;
@group(0) @binding(6) var<storage, read_write> xC: array<f32>;
fn livre(i: i32, j: i32) -> bool {
  return i >= 0 && j >= 0 && i < i32(N.nxF) && j < i32(N.nyF) && tipoF[u32(i + j * i32(N.nxF))] != SOLIDO;
}
fn residuo(i: i32, j: i32) -> f32 {
  let c = u32(i + j * i32(N.nxF));
  if (tipoF[c] != AGUA) { return 0.0; }
  var s = 0.0; var k = 0.0;
  if (livre(i - 1, j)) { k += 1.0; s += xF[c - 1u]; }
  if (livre(i + 1, j)) { k += 1.0; s += xF[c + 1u]; }
  if (livre(i, j - 1)) { k += 1.0; s += xF[c - N.nxF]; }
  if (livre(i, j + 1)) { k += 1.0; s += xF[c + N.nxF]; }
  return bF[c] - (k * xF[c] - s);
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let C = g.x;
  if (C >= N.nx * N.ny) { return; }
  xC[C] = 0.0;
  if (tipoC[C] != AGUA) { bC[C] = 0.0; return; }
  let I = i32(C % N.nx); let J = i32(C / N.nx);
  var pesos = array(0.25, 0.75, 0.75, 0.25);
  var s = 0.0;
  for (var b = 0; b < 4; b++) {
    let j = 2 * J - 1 + b;
    if (j < 0 || j >= i32(N.nyF)) { continue; }
    for (var a = 0; a < 4; a++) {
      let i = 2 * I - 1 + a;
      if (i < 0 || i >= i32(N.nxF)) { continue; }
      s += residuo(i, j) * pesos[a] * pesos[b];
    }
  }
  bC[C] = s;
}`;

// O nível mais grosso inteiro num grupo só: várias varreduras vermelho-preto
// no lugar (ida e volta, para o pré-condicionador continuar simétrico).
const GROSSO = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
fn livre(i: i32, j: i32) -> bool {
  return i >= 0 && j >= 0 && i < i32(N.nx) && j < i32(N.ny) && tipo[u32(i + j * i32(N.nx))] != SOLIDO;
}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) l: u32) {
  let n = N.nx * N.ny;
  for (var k = 0u; k < ${VARREDURAS_GROSSO}u; k++) {
    for (var o = 0u; o < 2u; o++) {
      let cor = select(o, 1u - o, (k & 1u) == 1u);
      for (var c = l; c < n; c += 256u) {
        let i = i32(c % N.nx); let j = i32(c / N.nx);
        if (tipo[c] != AGUA || (u32(i + j + 1) & 1u) != cor) { continue; }
        var s = 0.0; var q = 0.0;
        if (livre(i - 1, j)) { q += 1.0; s += x[c - 1u]; }
        if (livre(i + 1, j)) { q += 1.0; s += x[c + 1u]; }
        if (livre(i, j - 1)) { q += 1.0; s += x[c - N.nx]; }
        if (livre(i, j + 1)) { q += 1.0; s += x[c + N.nx]; }
        if (q > 0.0) { x[c] = (s + b[c]) / q; }
      }
      storageBarrier();
      workgroupBarrier();
    }
  }
}`;

// x fino += bilinear da correção grossa (só células grossas com água contam).
const PROLONGAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<storage, read_write> tipoF: array<u32>;
@group(0) @binding(2) var<storage, read_write> xF: array<f32>;
@group(0) @binding(3) var<storage, read_write> tipoC: array<u32>;
@group(0) @binding(4) var<storage, read_write> xC: array<f32>;
fn valor(I: i32, J: i32) -> f32 {
  if (I < 0 || J < 0 || I >= i32(N.nx) || J >= i32(N.ny)) { return 0.0; }
  let C = u32(I + J * i32(N.nx));
  if (tipoC[C] != AGUA) { return 0.0; }
  return xC[C];
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= N.nxF * N.nyF) { return; }
  if (tipoF[c] != AGUA) { return; }
  let i = i32(c % N.nxF); let j = i32(c / N.nxF);
  let f = vec2f((f32(i) + 0.5) * 0.5 - 0.5, (f32(j) + 0.5) * 0.5 - 0.5);
  let I0 = i32(floor(f.x)); let J0 = i32(floor(f.y));
  let t = f - vec2f(f32(I0), f32(J0));
  let e = mix(mix(valor(I0, J0), valor(I0 + 1, J0), t.x), mix(valor(I0, J0 + 1), valor(I0 + 1, J0 + 1), t.x), t.y);
  xF[c] = xF[c] + e;
}`;

export class Multigrid {
  /**
   * @param {GPUDevice} device
   * @param {{nx:number, ny:number, tipo:GPUBuffer, b:GPUBuffer, x:GPUBuffer, tmp:GPUBuffer}} fino
   */
  constructor(device, fino, opcoes = {}) {
    this.device = device;
    const omega = opcoes.omega ?? 1.0;
    const varreduras = opcoes.varreduras ?? 4;
    const buf = (n) => device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.niveis = [{ nx: fino.nx, ny: fino.ny, tipo: fino.tipo, b: fino.b, x: fino.x, tmp: fino.tmp, proprios: [] }];
    while (Math.min(this.niveis.at(-1).nx, this.niveis.at(-1).ny) > 12) {
      const a = this.niveis.at(-1);
      const nx = Math.ceil(a.nx / 2), ny = Math.ceil(a.ny / 2);
      const k = nx * ny;
      this.niveis.push({ nx, ny, tipo: buf(k), b: buf(k), x: buf(k), tmp: buf(k), proprios: ['tipo', 'b', 'x', 'tmp'] });
    }
    for (let l = 0; l < this.niveis.length; l++) {
      const n = this.niveis[l];
      const fin = this.niveis[l - 1];
      n.uniforme = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const d = new ArrayBuffer(32);
      const u = new Uint32Array(d), f = new Float32Array(d);
      u[0] = n.nx; u[1] = n.ny; u[2] = fin?.nx ?? 0; u[3] = fin?.ny ?? 0; f[4] = omega;
      device.queue.writeBuffer(n.uniforme, 0, d);
    }
    // passadas: (deslocamento, ordem das cores)
    this.passadas = [[0, 0], [8, 0], [8, 1], [0, 1]].map(([desl, ordem]) => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([desl, ordem, 0, 0]));
      return b;
    });

    const pipe = (code, label) => device.createComputePipeline({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    this.pSuavizar = pipe(SUAVIZAR(varreduras), 'mg-suavizar');
    this.pTipos = pipe(TIPOS, 'mg-tipos');
    this.pGrosso = pipe(GROSSO, 'mg-grosso');
    this.pRestringir = pipe(RESTRINGIR, 'mg-restringir');
    this.pProlongar = pipe(PROLONGAR, 'mg-prolongar');
    const grupo = (p, recursos) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: recursos.map((r, i) => ({ binding: i, resource: { buffer: r } })),
    });
    for (let l = 0; l < this.niveis.length; l++) {
      const n = this.niveis[l];
      // ida: x → tmp; volta: tmp → x (cada par de passadas devolve o resultado para x)
      n.gSuavizar = this.passadas.map((pb, k) => grupo(this.pSuavizar,
        k % 2 === 0 ? [n.uniforme, pb, n.tipo, n.b, n.x, n.tmp] : [n.uniforme, pb, n.tipo, n.b, n.tmp, n.x]));
      if (l > 0) {
        const f = this.niveis[l - 1];
        n.gTipos = grupo(this.pTipos, [n.uniforme, f.tipo, n.tipo]);
        n.gRestringir = grupo(this.pRestringir, [n.uniforme, f.tipo, f.b, f.x, n.tipo, n.b, n.x]);
        n.gProlongar = grupo(this.pProlongar, [n.uniforme, f.tipo, f.x, n.tipo, n.x]);
      }
      if (l === this.niveis.length - 1) n.gGrosso = grupo(this.pGrosso, [n.uniforme, n.tipo, n.b, n.x]);
    }
  }

  #grupos(pass, p, g, n) {
    pass.setPipeline(p);
    pass.setBindGroup(0, g);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
  }

  /** Tipos dos níveis grossos (uma vez por subpasso, depois que os tipos finos estão prontos). */
  gravarTipos(pass) {
    for (let l = 1; l < this.niveis.length; l++) {
      const n = this.niveis[l];
      this.#grupos(pass, this.pTipos, n.gTipos, n.nx * n.ny);
    }
  }

  /** Duas passadas do suavizador (ida: passadas 0 e 1; volta: 2 e 3); o resultado volta para x. */
  #suavizar(pass, l, volta) {
    const n = this.niveis[l];
    pass.setPipeline(this.pSuavizar);
    const wx = Math.ceil((n.nx + 8) / 16), wy = Math.ceil((n.ny + 8) / 16);
    for (const k of volta ? [2, 3] : [0, 1]) {
      pass.setBindGroup(0, n.gSuavizar[k]);
      pass.dispatchWorkgroups(wx, wy);
    }
  }

  /** Um ciclo em V (o x do nível fino tem que estar zerado antes). */
  gravar(pass) {
    const L = this.niveis.length;
    for (let l = 0; l < L - 1; l++) {
      this.#suavizar(pass, l, false);
      const g = this.niveis[l + 1];
      this.#grupos(pass, this.pRestringir, g.gRestringir, g.nx * g.ny);
    }
    const grosso = this.niveis[L - 1];
    pass.setPipeline(this.pGrosso);
    pass.setBindGroup(0, grosso.gGrosso);
    pass.dispatchWorkgroups(1);
    for (let l = L - 2; l >= 0; l--) {
      const g = this.niveis[l + 1];
      this.#grupos(pass, this.pProlongar, g.gProlongar, this.niveis[l].nx * this.niveis[l].ny);
      this.#suavizar(pass, l, true);
    }
  }

  destruir() {
    for (const n of this.niveis) {
      n.uniforme.destroy();
      for (const k of n.proprios) n[k].destroy();
    }
    for (const b of this.passadas) b.destroy();
  }
}
