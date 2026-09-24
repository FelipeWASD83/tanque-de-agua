// Desenho na GPU. Duas etapas:
//   1. cada partícula vira uma mancha suave num campo (meia resolução):
//      quanta água há ali, quanta espuma e quanto corante;
//   2. um shader de tela inteira transforma o campo em água: o painel de
//      calibração atrás, desviado pela água onde a superfície é curva
//      (refração), a luz atravessando e perdendo cor (Beer-Lambert), a espuma
//      branca, o menisco da superfície e as paredes com sombra no painel.

import { PARTICULAS_POR_CELULA, PROFUNDIDADE, ABSORCAO_AGUA, MOLDURA, RES_MASCARA, RAIO_MANCHA } from '../constantes.js';

const WGSL_MANCHA = /* wgsl */ `
struct Particula { pos: vec2f, vel: vec2f, c: vec4f, extra: vec4f };
struct RP { W: f32, H: f32, raio: f32, _p: f32 };
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> part: array<Particula>;
struct VS {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) extra: vec4f,
};
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VS {
  var cantos = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = cantos[vi];
  let p = part[ii];
  var o: VS;
  o.uv = c;
  o.extra = p.extra;
  if (p.pos.x < 0.0) { o.pos = vec4f(9.0, 9.0, 0.0, 1.0); return o; }
  let w = p.pos + c * R.raio;
  o.pos = vec4f(w.x / R.W * 2.0 - 1.0, w.y / R.H * 2.0 - 1.0, 0.0, 1.0);
  return o;
}
struct FS { @location(0) a: vec4f, @location(1) b: vec4f };
@fragment fn fs(i: VS) -> FS {
  let r2 = dot(i.uv, i.uv);
  if (r2 >= 1.0) { discard; }
  let w = (1.0 - r2) * (1.0 - r2);
  var o: FS;
  o.a = vec4f(w, w * i.extra.x, 0.0, 0.0);
  o.b = vec4f(w * i.extra.yzw, 0.0);
  return o;
}
`;

const WGSL_TELA = /* wgsl */ `
struct CP {
  W: f32, H: f32, h: f32, phi0: f32,
  mw: u32, mh: u32, px: f32, refr: f32,
  texel: vec2f, moldura: f32, prof: f32,
  absAgua: vec3f, pxY: f32,
};
@group(0) @binding(0) var<uniform> C: CP;
@group(0) @binding(1) var campoA: texture_2d<f32>;
@group(0) @binding(2) var campoB: texture_2d<f32>;
@group(0) @binding(3) var amostra: sampler;
@group(0) @binding(4) var<storage, read> sdf: array<f32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}

fn sdfPx(x: i32, y: i32) -> f32 {
  let xx = clamp(x, 0, i32(C.mw) - 1);
  let yy = clamp(y, 0, i32(C.mh) - 1);
  return sdf[u32(yy) * C.mw + u32(xx)];
}
// distância até a parede em metros (positiva fora dela)
fn parede(p: vec2f) -> f32 {
  let f = clamp(p / C.h * ${RES_MASCARA}.0 - 0.5, vec2f(0.0), vec2f(f32(C.mw) - 1.0, f32(C.mh) - 1.0));
  let x0 = min(i32(floor(f.x)), i32(C.mw) - 2);
  let y0 = min(i32(floor(f.y)), i32(C.mh) - 2);
  let t = f - vec2f(f32(x0), f32(y0));
  return mix(mix(sdfPx(x0, y0), sdfPx(x0 + 1, y0), t.x), mix(sdfPx(x0, y0 + 1), sdfPx(x0 + 1, y0 + 1), t.x), t.y) * C.h;
}

// marca de calibração: pontos a cada 1 cm, maiores a cada 5, linhas finas a cada 10
fn painel(p: vec2f, px: f32) -> vec3f {
  let origem = vec2f(C.moldura, C.moldura);
  let q = p - origem;
  let centro = vec2f(C.W * 0.5, C.H * 0.62);
  let r = (p - centro) / vec2f(max(C.W, C.H) * 0.75);
  var cor = vec3f(0.936, 0.947, 0.95) * (1.02 - 0.075 * dot(r, r));
  let cm = abs(fract(q / 0.01 + 0.5) - 0.5) * 0.01;
  let dPonto = length(cm);
  let raioPonto = max(0.00045, px * 0.55);
  cor = mix(cor, vec3f(0.62, 0.67, 0.7), (1.0 - smoothstep(raioPonto - px * 0.5, raioPonto + px * 0.5, dPonto)) * 0.8);
  let cinco = abs(fract(q / 0.05 + 0.5) - 0.5) * 0.05;
  let raio5 = max(0.0009, px * 0.9);
  cor = mix(cor, vec3f(0.47, 0.53, 0.57), 1.0 - smoothstep(raio5 - px * 0.5, raio5 + px * 0.5, length(cinco)));
  let dez = abs(fract(q / 0.1 + 0.5) - 0.5) * 0.1;
  let linha = min(dez.x, dez.y);
  cor = mix(cor, vec3f(0.6, 0.66, 0.7), (1.0 - smoothstep(0.0, px * 1.1, linha - px * 0.15)) * 0.42);
  return cor;
}

// régua na borda esquerda, por cima de tudo que é painel
fn regua(p: vec2f, px: f32) -> f32 {
  let x = p.x - C.moldura;
  let y = p.y - C.moldura;
  if (x < 0.0 || y < 0.0) { return 0.0; }
  let cm = abs(fract(y / 0.01 + 0.5) - 0.5) * 0.01;
  let idx = round(y / 0.01);
  var comp = 0.004;
  if (abs(idx - round(idx / 5.0) * 5.0) < 0.5) { comp = 0.008; }
  if (abs(idx - round(idx / 10.0) * 10.0) < 0.5) { comp = 0.014; }
  let tracinho = (1.0 - smoothstep(px * 0.35, px * 1.15, cm)) * (1.0 - smoothstep(comp - px, comp, x));
  return tracinho;
}

fn T(phi: f32) -> f32 { return smoothstep(0.26, 0.5, phi); }
fn TL(phi: f32) -> f32 { return smoothstep(0.15, 0.95, phi); }

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let px = C.px;
  let p = vec2f(fc.x * px, C.H - fc.y * C.pxY);
  let uv = vec2f(fc.x * px / C.W, fc.y * C.pxY / C.H);

  let A = textureSampleLevel(campoA, amostra, uv, 0.0);
  let B = textureSampleLevel(campoB, amostra, uv, 0.0);
  let phi = A.x / C.phi0;
  let e = C.texel * 1.5;
  let lR = TL(textureSampleLevel(campoA, amostra, uv + vec2f(e.x, 0.0), 0.0).x / C.phi0);
  let lL = TL(textureSampleLevel(campoA, amostra, uv - vec2f(e.x, 0.0), 0.0).x / C.phi0);
  let lD = TL(textureSampleLevel(campoA, amostra, uv + vec2f(0.0, e.y), 0.0).x / C.phi0);
  let lU = TL(textureSampleLevel(campoA, amostra, uv - vec2f(0.0, e.y), 0.0).x / C.phi0);
  // inclinação da "lente" de água, em coordenadas do mundo (y para cima)
  let grad = vec2f(lR - lL, lU - lD);

  let cobre = T(phi);
  let espessura = TL(phi);

  // paredes
  let dParede = parede(p);
  let pxParede = dParede / px;
  let naParede = clamp(0.5 - pxParede, 0.0, 1.0);

  // painel visto através da água: desvio onde a superfície é curva
  let desvio = grad * C.refr * cobre;
  var fundo = painel(p + desvio, px);
  fundo = mix(fundo, vec3f(0.36, 0.41, 0.44), regua(p + desvio, px));
  // sombra das paredes no painel (luz vem de cima, à esquerda)
  let dSombra = parede(p + vec2f(-0.6, 0.9) * C.h * 1.6);
  fundo *= 1.0 - 0.2 * (1.0 - smoothstep(0.0, C.h * 3.5, dSombra));

  var cor = fundo;
  if (cobre > 0.001) {
    // corante: absorção média das partículas que cobrem o ponto
    let corante = B.xyz / max(A.x, 1e-4);
    let absorcao = C.absAgua + corante;
    let transm = exp(-absorcao * C.prof * espessura);
    var agua = fundo * transm;
    // luz espalhada: a água nunca fica totalmente escura, ganha o tom do corante
    agua += (vec3f(1.0) - transm) * vec3f(0.05, 0.075, 0.085);
    // espuma e bolhas: branco leitoso
    let esp = clamp(A.y / max(A.x, 1e-4), 0.0, 1.0);
    agua = mix(agua, vec3f(0.97, 0.985, 0.99), smoothstep(0.05, 0.75, esp) * 0.85);
    // superfície: normal da lente, brilho de cima e uma linha escura no menisco
    let n = normalize(vec3f(-grad * 5.0, 1.0));
    let luz = normalize(vec3f(-0.35, 0.8, 0.5));
    let brilho = pow(max(dot(reflect(-luz, n), vec3f(0.0, 0.0, 1.0)), 0.0), 24.0);
    agua += vec3f(brilho) * 0.55 * smoothstep(0.02, 0.2, length(grad));
    let borda = cobre * (1.0 - cobre) * 4.0;
    agua = mix(agua, agua * vec3f(0.5, 0.6, 0.64), borda * 0.55);
    cor = mix(fundo, agua, cobre);
  }

  if (naParede > 0.0) {
    // parede de acrílico grafite com um chanfro iluminado
    let e2 = C.h * 0.5;
    let g = vec2f(parede(p + vec2f(e2, 0.0)) - parede(p - vec2f(e2, 0.0)), parede(p + vec2f(0.0, e2)) - parede(p - vec2f(0.0, e2)));
    let nn = select(vec2f(0.0, 1.0), normalize(g), length(g) > 1e-7);
    let chanfro = 1.0 - smoothstep(0.0, C.h * 1.1, -dParede);
    var pc = vec3f(0.2, 0.225, 0.24);
    pc += vec3f(0.2, 0.21, 0.2) * chanfro * max(dot(nn, normalize(vec2f(-0.6, 0.8))), 0.0);
    pc -= vec3f(0.07) * chanfro * max(dot(nn, normalize(vec2f(0.6, -0.8))), 0.0);
    cor = mix(cor, pc, naParede);
  }
  return vec4f(cor, 1.0);
}
`;

export class RenderGPU {
  constructor(device, contexto, formato) {
    this.device = device;
    this.contexto = contexto;
    this.formato = formato;
    this.amostra = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.rpMancha = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rpTela = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const alvoCampo = { format: 'rgba16float', blend: { color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' } } };
    const mMancha = device.createShaderModule({ label: 'mancha', code: WGSL_MANCHA });
    this.pipeMancha = device.createRenderPipeline({
      label: 'mancha', layout: 'auto',
      vertex: { module: mMancha, entryPoint: 'vs' },
      fragment: { module: mMancha, entryPoint: 'fs', targets: [alvoCampo, alvoCampo] },
      primitive: { topology: 'triangle-list' },
    });
    const mTela = device.createShaderModule({ label: 'tela', code: WGSL_TELA });
    this.pipeTela = device.createRenderPipeline({
      label: 'tela', layout: 'auto',
      vertex: { module: mTela, entryPoint: 'vs' },
      fragment: { module: mTela, entryPoint: 'fs', targets: [{ format: formato }] },
      primitive: { topology: 'triangle-list' },
    });
    this.largura = 0;
    this.altura = 0;
    this.grupos = new Map();
  }

  /** Ajusta as texturas do campo ao tamanho da tela (em pixels reais). */
  #garantirCampo(larg, alt) {
    const w = Math.max(8, Math.round(larg * 0.5)), h = Math.max(8, Math.round(alt * 0.5));
    if (this.campoA && this.campoA.width === w && this.campoA.height === h) return;
    this.campoA?.destroy();
    this.campoB?.destroy();
    const uso = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.campoA = this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: uso });
    this.campoB = this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: uso });
    this.vistaA = this.campoA.createView();
    this.vistaB = this.campoB.createView();
    this.grupos.clear();
  }

  #grupo(chave, fn) {
    let g = this.grupos.get(chave);
    if (!g) { g = fn(); this.grupos.set(chave, g); }
    return g;
  }

  /**
   * @param {GPUCommandEncoder} enc
   * @param {import('./fisica.js').FisicaGPU} fisica
   * @param {number} larg largura da tela em pixels reais
   * @param {number} alt altura da tela em pixels reais
   */
  desenhar(enc, fisica, larg, alt) {
    this.#garantirCampo(larg, alt);
    const cena = fisica.cena;
    const W = cena.largura, H = cena.altura, h = cena.h;
    this.device.queue.writeBuffer(this.rpMancha, 0, new Float32Array([W, H, RAIO_MANCHA * h, 0]));
    const phi0 = (PARTICULAS_POR_CELULA * Math.PI * RAIO_MANCHA * RAIO_MANCHA) / 3;
    const buf = new ArrayBuffer(64);
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    f[0] = W; f[1] = H; f[2] = h; f[3] = phi0;
    u[4] = cena.mw; u[5] = cena.mh; f[6] = W / larg; f[7] = h * 2.2;
    f[8] = 1 / this.campoA.width; f[9] = 1 / this.campoA.height; f[10] = MOLDURA * h; f[11] = PROFUNDIDADE;
    f[12] = ABSORCAO_AGUA[0]; f[13] = ABSORCAO_AGUA[1]; f[14] = ABSORCAO_AGUA[2]; f[15] = H / alt;
    this.device.queue.writeBuffer(this.rpTela, 0, buf);

    const particulas = fisica.particulas;
    const gM = this.#grupo('m' + fisica.paridade + fisica.idRender, () => this.device.createBindGroup({
      layout: this.pipeMancha.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.rpMancha } }, { binding: 1, resource: { buffer: particulas } }],
    }));
    let pass = enc.beginRenderPass({
      label: 'campo',
      colorAttachments: [
        { view: this.vistaA, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: this.vistaB, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
    });
    pass.setPipeline(this.pipeMancha);
    pass.setBindGroup(0, gM);
    pass.drawIndirect(fisica.buffers.desenho, 0);
    pass.end();

    const gT = this.#grupo('t' + fisica.idRender, () => this.device.createBindGroup({
      layout: this.pipeTela.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.rpTela } },
        { binding: 1, resource: this.vistaA },
        { binding: 2, resource: this.vistaB },
        { binding: 3, resource: this.amostra },
        { binding: 4, resource: { buffer: fisica.buffers.sdf } },
      ],
    }));
    pass = enc.beginRenderPass({
      label: 'tela',
      colorAttachments: [{ view: this.contexto.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [1, 1, 1, 1] }],
    });
    pass.setPipeline(this.pipeTela);
    pass.setBindGroup(0, gT);
    pass.draw(3);
    pass.end();
  }
}
