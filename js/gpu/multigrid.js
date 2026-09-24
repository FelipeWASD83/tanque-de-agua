// Solver de pressão multigrid para a GPU.
//
// A equação é a mesma do motor da CPU (para cada célula com água:
// n·q − Σ q_vizinhos = −div, com parede fora da soma e ar valendo zero). O
// Gauss-Seidel sozinho apaga rápido o erro "miúdo", mas o erro que se
// espalha pelo tanque inteiro (o peso da coluna d'água, por exemplo) leva
// centenas de iterações — e a água sem pressão certa treme e não assenta.
// O multigrid resolve isso: suaviza, leva o resto para uma grade com metade
// da resolução (onde o erro grande vira erro pequeno), resolve lá, traz a
// correção de volta e suaviza de novo. Um ciclo em V por nível.
//
// O suavizador trabalha em blocos de 16×16 células na memória rápida da placa
// (várias varreduras vermelho-preto por despacho, com os blocos deslocados
// meio bloco a cada despacho para as emendas não ficarem paradas).

const CABECALHO = /* wgsl */ `
struct Nivel { nx: u32, ny: u32, nxF: u32, nyF: u32, omega: f32, _a: f32, _b: f32, _c: f32 };
struct Fase { desl: u32, _a: u32, _b: u32, _c: u32 };
const SOLIDO: u32 = 0u;
const AGUA: u32 = 1u;
const AR: u32 = 2u;
`;

function suavizador(varreduras) {
  return /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<uniform> F: Fase;
@group(0) @binding(2) var<storage, read_write> tipo: array<u32>;
@group(0) @binding(3) var<storage, read_write> div: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
var<workgroup> qs: array<f32, 324>;
var<workgroup> ts: array<u32, 324>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let nx = i32(N.nx); let ny = i32(N.ny);
  let desl = i32(F.desl) * 8;
  let ox = i32(wg.x) * 16 - desl; let oy = i32(wg.y) * 16 - desl;
  let lid = l.y * 16u + l.x;
  for (var s = lid; s < 324u; s += 256u) {
    let sx = i32(s % 18u) - 1 + ox; let sy = i32(s / 18u) - 1 + oy;
    var qv = 0.0; var tv = SOLIDO;
    if (sx >= 0 && sy >= 0 && sx < nx && sy < ny) { let c = u32(sx + sy * nx); qv = q[c]; tv = tipo[c]; }
    qs[s] = qv; ts[s] = tv;
  }
  let gx = ox + i32(l.x); let gy = oy + i32(l.y);
  let dentro = gx >= 0 && gy >= 0 && gx < nx && gy < ny;
  let li = (l.y + 1u) * 18u + l.x + 1u;
  var dv = 0.0;
  if (dentro) { dv = div[u32(gx + gy * nx)]; }
  workgroupBarrier();
  let agua = dentro && ts[li] == AGUA;
  var n = 0.0;
  if (agua) {
    n = f32(ts[li - 1u] != SOLIDO) + f32(ts[li + 1u] != SOLIDO) + f32(ts[li - 18u] != SOLIDO) + f32(ts[li + 18u] != SOLIDO);
  }
  let minhaCor = u32(gx + gy + 1) & 1u;
  for (var it = 0; it < ${varreduras}; it++) {
    for (var cor = 0u; cor < 2u; cor++) {
      if (agua && n > 0.0 && minhaCor == cor) {
        var soma = 0.0;
        if (ts[li - 1u] != SOLIDO) { soma += qs[li - 1u]; }
        if (ts[li + 1u] != SOLIDO) { soma += qs[li + 1u]; }
        if (ts[li - 18u] != SOLIDO) { soma += qs[li - 18u]; }
        if (ts[li + 18u] != SOLIDO) { soma += qs[li + 18u]; }
        let novo = (soma - dv) / n;
        qs[li] = qs[li] + N.omega * (novo - qs[li]);
      }
      workgroupBarrier();
    }
  }
  if (agua) { q[u32(gx + gy * nx)] = qs[li]; }
}`;
}

// Resíduo da grade fina, somado nos 4 filhos de cada célula grossa. A célula
// grossa tem água se algum filho tem; é parede se todos são parede.
const RESTRINGIR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<storage, read_write> tipoF: array<u32>;
@group(0) @binding(2) var<storage, read_write> divF: array<f32>;
@group(0) @binding(3) var<storage, read_write> qF: array<f32>;
@group(0) @binding(4) var<storage, read_write> tipoC: array<u32>;
@group(0) @binding(5) var<storage, read_write> divC: array<f32>;
@group(0) @binding(6) var<storage, read_write> qC: array<f32>;
fn tipoFino(i: i32, j: i32) -> u32 {
  if (i < 0 || j < 0 || i >= i32(N.nxF) || j >= i32(N.nyF)) { return SOLIDO; }
  return tipoF[u32(i + j * i32(N.nxF))];
}
fn residuo(i: i32, j: i32) -> f32 {
  let c = u32(i + j * i32(N.nxF));
  var soma = 0.0; var n = 0.0;
  if (tipoFino(i - 1, j) != SOLIDO) { n += 1.0; soma += qF[c - 1u]; }
  if (tipoFino(i + 1, j) != SOLIDO) { n += 1.0; soma += qF[c + 1u]; }
  if (tipoFino(i, j - 1) != SOLIDO) { n += 1.0; soma += qF[c - N.nxF]; }
  if (tipoFino(i, j + 1) != SOLIDO) { n += 1.0; soma += qF[c + N.nxF]; }
  return -divF[c] - (n * qF[c] - soma);
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let C = g.x;
  if (C >= N.nx * N.ny) { return; }
  let I = i32(C % N.nx); let J = i32(C / N.nx);
  var temAgua = false; var todoSolido = true; var R = 0.0;
  for (var b = 0; b < 2; b++) { for (var a = 0; a < 2; a++) {
    let i = 2 * I + a; let j = 2 * J + b;
    let t = tipoFino(i, j);
    if (t == AGUA) { temAgua = true; R += residuo(i, j); }
    if (t != SOLIDO) { todoSolido = false; }
  } }
  var t = AR;
  if (temAgua) { t = AGUA; } else if (todoSolido) { t = SOLIDO; }
  tipoC[C] = t;
  divC[C] = -R;
  qC[C] = 0.0;
}`;

// Correção da grade grossa interpolada (bilinear) de volta na fina.
const PROLONGAR = /* wgsl */ `
${CABECALHO}
@group(0) @binding(0) var<uniform> N: Nivel;
@group(0) @binding(1) var<storage, read_write> tipoF: array<u32>;
@group(0) @binding(2) var<storage, read_write> qF: array<f32>;
@group(0) @binding(3) var<storage, read_write> tipoC: array<u32>;
@group(0) @binding(4) var<storage, read_write> qC: array<f32>;
fn valor(I: i32, J: i32, pai: f32) -> f32 {
  if (I < 0 || J < 0 || I >= i32(N.nx) || J >= i32(N.ny)) { return pai; }
  let c = u32(I + J * i32(N.nx));
  if (tipoC[c] == SOLIDO) { return pai; }
  return qC[c];
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= N.nxF * N.nyF) { return; }
  if (tipoF[c] != AGUA) { return; }
  let i = i32(c % N.nxF); let j = i32(c / N.nxF);
  let pai = qC[u32(i / 2 + (j / 2) * i32(N.nx))];
  let f = vec2f((f32(i) + 0.5) * 0.5 - 0.5, (f32(j) + 0.5) * 0.5 - 0.5);
  let I0 = i32(floor(f.x)); let J0 = i32(floor(f.y));
  let t = f - vec2f(f32(I0), f32(J0));
  let e = mix(mix(valor(I0, J0, pai), valor(I0 + 1, J0, pai), t.x), mix(valor(I0, J0 + 1, pai), valor(I0 + 1, J0 + 1, pai), t.x), t.y);
  qF[c] = qF[c] + e;
}`;

export class Multigrid {
  /**
   * @param {GPUDevice} device
   * @param {{nx:number, ny:number, tipo:GPUBuffer, div:GPUBuffer, q:GPUBuffer}} fino
   */
  constructor(device, fino, opcoes = {}) {
    this.device = device;
    this.ciclos = opcoes.ciclos ?? 2;
    this.suavizarFino = opcoes.suavizarFino ?? 1; // despachos antes e depois, na grade fina
    this.suavizarGrosso = opcoes.suavizarGrosso ?? 4; // despachos na grade mais grossa
    const omega = opcoes.omega ?? 1.5;
    const varreduras = opcoes.varreduras ?? 4;

    this.niveis = [{ nx: fino.nx, ny: fino.ny, tipo: fino.tipo, div: fino.div, q: fino.q }];
    while (Math.min(this.niveis.at(-1).nx, this.niveis.at(-1).ny) > 12) {
      const a = this.niveis.at(-1);
      const nx = Math.ceil(a.nx / 2), ny = Math.ceil(a.ny / 2);
      const uso = GPUBufferUsage.STORAGE;
      const t = (n) => device.createBuffer({ size: Math.max(16, n * 4), usage: uso });
      this.niveis.push({ nx, ny, tipo: t(nx * ny), div: t(nx * ny), q: t(nx * ny), proprios: true });
    }
    for (const n of this.niveis) n.uniforme = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.relaxacao = omega;
    this.fases = [0, 1].map((d) => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([d, 0, 0, 0]));
      return b;
    });

    const pipe = (code, label) => device.createComputePipeline({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    this.pSuavizar = pipe(suavizador(varreduras), 'mg-suavizar');
    this.pRestringir = pipe(RESTRINGIR, 'mg-restringir');
    this.pProlongar = pipe(PROLONGAR, 'mg-prolongar');

    const grupo = (p, recursos) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: recursos.map((r, i) => ({ binding: i, resource: { buffer: r } })),
    });
    for (let l = 0; l < this.niveis.length; l++) {
      const n = this.niveis[l];
      n.gSuavizar = this.fases.map((fb) => grupo(this.pSuavizar, [n.uniforme, fb, n.tipo, n.div, n.q]));
      if (l > 0) {
        const f = this.niveis[l - 1];
        n.gRestringir = grupo(this.pRestringir, [n.uniforme, f.tipo, f.div, f.q, n.tipo, n.div, n.q]);
        n.gProlongar = grupo(this.pProlongar, [n.uniforme, f.tipo, f.q, n.tipo, n.q]);
      }
    }
    this.contador = 0;
  }

  /** Fator de relaxação do suavizador (1 = Gauss-Seidel puro). */
  set relaxacao(omega) {
    for (let l = 0; l < this.niveis.length; l++) {
      const n = this.niveis[l];
      const fin = this.niveis[l - 1];
      const buf = new ArrayBuffer(32);
      const u = new Uint32Array(buf), f = new Float32Array(buf);
      u[0] = n.nx; u[1] = n.ny; u[2] = fin?.nx ?? 0; u[3] = fin?.ny ?? 0; f[4] = omega;
      this.device.queue.writeBuffer(n.uniforme, 0, buf);
    }
  }

  #suavizar(pass, l, vezes) {
    const n = this.niveis[l];
    pass.setPipeline(this.pSuavizar);
    for (let k = 0; k < vezes; k++) {
      pass.setBindGroup(0, n.gSuavizar[this.contador++ & 1]);
      pass.dispatchWorkgroups(Math.ceil((n.nx + 8) / 16), Math.ceil((n.ny + 8) / 16));
    }
  }

  /** Grava os ciclos em V no passe de compute (os tipos e a divergência da grade fina já estão prontos). */
  gravar(pass) {
    const L = this.niveis.length;
    for (let c = 0; c < this.ciclos; c++) {
      for (let l = 0; l < L - 1; l++) {
        this.#suavizar(pass, l, this.suavizarFino);
        const g = this.niveis[l + 1];
        pass.setPipeline(this.pRestringir);
        pass.setBindGroup(0, g.gRestringir);
        pass.dispatchWorkgroups(Math.ceil((g.nx * g.ny) / 256));
      }
      this.#suavizar(pass, L - 1, this.suavizarGrosso);
      for (let l = L - 2; l >= 0; l--) {
        const g = this.niveis[l + 1];
        const f = this.niveis[l];
        pass.setPipeline(this.pProlongar);
        pass.setBindGroup(0, g.gProlongar);
        pass.dispatchWorkgroups(Math.ceil((f.nx * f.ny) / 256));
        this.#suavizar(pass, l, this.suavizarFino);
      }
    }
  }

  destruir() {
    for (const n of this.niveis) {
      n.uniforme.destroy();
      if (n.proprios) { n.tipo.destroy(); n.div.destroy(); n.q.destroy(); }
    }
    for (const b of this.fases) b.destroy();
  }
}
