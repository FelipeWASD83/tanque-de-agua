// Os dois motores com a mesma cara para o resto do app: o da placa de vídeo
// (WebGPU) e o de reserva (CPU + canvas 2D). Quem escolhe é o principal.js.

import { FisicaGPU } from './gpu/fisica.js';
import { RenderGPU } from './gpu/render.js';
import { SimulacaoCPU, CAMPOS } from './fisica-cpu.js';
import { PARTICULAS_MAX_POR_CELULA } from './constantes.js';
import { Render2D } from './render-2d.js';

/** Tenta abrir a placa de vídeo. Devolve null (com o motivo) se não der. */
export async function abrirGPU() {
  if (!('gpu' in navigator)) return { erro: 'Este navegador não tem WebGPU.' };
  try {
    const adaptador = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adaptador) return { erro: 'O navegador não encontrou uma placa de vídeo compatível.' };
    const device = await adaptador.requestDevice({
      // carimbo de tempo da GPU: só serve para medir pelo console, mas não custa nada pedir
      requiredFeatures: adaptador.features.has('timestamp-query') ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adaptador.limits.maxStorageBufferBindingSize,
        maxBufferSize: adaptador.limits.maxBufferSize,
      },
    });
    device.addEventListener('uncapturederror', (ev) => console.error('WebGPU:', ev.error.message));
    const info = adaptador.info ?? {};
    const nome = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU';
    return { device, nome };
  } catch (e) {
    return { erro: 'Não deu para abrir a placa de vídeo: ' + e.message };
  }
}

export class MotorGPU {
  constructor(gpu, canvas, cena) {
    this.tipo = 'gpu';
    this.nome = gpu.nome;
    this.device = gpu.device;
    this.canvas = canvas;
    this.cena = cena;
    if (!MotorGPU.contextos.has(canvas)) {
      const ctx = canvas.getContext('webgpu');
      const formato = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: this.device, format: formato, alphaMode: 'opaque' });
      MotorGPU.contextos.set(canvas, { ctx, render: new RenderGPU(this.device, ctx, formato) });
    }
    this.render = MotorGPU.contextos.get(canvas).render;
    this.fisica = new FisicaGPU(this.device, cena);
  }

  static contextos = new Map();

  get h() { return this.cena.h; }
  get maxParticulas() { return this.fisica.maxParticulas; }
  /** Maior velocidade da água no último quadro medido (m/s), ou null. */
  get vMax() { return this.fisica.vMax; }
  paredesMudaram() { this.fisica.subirParedes(); }
  adicionar(lista, checar = true) { this.fisica.adicionar(lista, checar); }
  limparAgua() { this.fisica.limparAgua(); }
  set gravidade(g) { this.fisica.gravidade = g; }

  quadro(dt, subpassos, ferramenta) {
    const enc = this.device.createCommandEncoder();
    this.fisica.quadro(enc, dt, subpassos, ferramenta);
    this.render.desenhar(enc, this.fisica, this.canvas.width, this.canvas.height);
    this.device.queue.submit([enc.finish()]);
    this.fisica.aposEnviar();
  }

  /** Milissegundos de GPU por subpasso (null se a placa não mede). */
  get custoPorSubpasso() { return this.fisica.custoPorSubpasso; }

  contagem() { return this.fisica.lerContagem(); }
  particulas() { return this.fisica.lerParticulas(); }
  /** Espera a GPU terminar o que já foi pedido (útil para medir). */
  esperar() { return this.device.queue.onSubmittedWorkDone(); }
  destruir() { this.fisica.destruir(); }
}

export class MotorCPU {
  constructor(canvas, cena) {
    this.tipo = 'cpu';
    this.nome = 'CPU';
    this.canvas = canvas;
    this.cena = cena;
    this.sim = new SimulacaoCPU(cena);
    this.render = new Render2D(canvas);
    this.pendentes = [];
  }

  get h() { return this.cena.h; }
  get maxParticulas() { return Math.ceil(this.sim.nx * this.sim.ny * PARTICULAS_MAX_POR_CELULA); }
  get vMax() { return this.sim.vMax ?? null; }
  paredesMudaram() { this.sim.atualizarParedes(); }
  adicionar(lista, checar = true) { this.pendentes.push([lista, checar]); }
  limparAgua() { this.sim.n = 0; this.pendentes = []; this.sim.contagemCelula.fill(0); this.sim.reiniciarAr(); }
  set gravidade(g) { this.sim.gravidade = g; }

  quadro(dt, subpassos, ferramenta) {
    this.sim.aplicarFerramenta(ferramenta);
    const teto = this.maxParticulas;
    for (const [l, c] of this.pendentes) {
      const cabe = Math.max(0, teto - this.sim.n) * CAMPOS;
      if (cabe > 0) this.sim.adicionar(l.length > cabe ? l.subarray(0, cabe) : l, c);
    }
    this.pendentes = [];
    if (subpassos > 0) {
      const t0 = performance.now();
      this.sim.avancar(dt, subpassos);
      const ms = (performance.now() - t0) / subpassos;
      this.custoCpu = this.custoCpu == null ? ms : this.custoCpu + (ms - this.custoCpu) * 0.2;
    }
    this.render.desenhar(this.sim);
  }

  get custoPorSubpasso() { return this.custoCpu ?? null; }
  async contagem() { return this.sim.n; }
  async particulas() { return this.sim.p.slice(0, this.sim.n * CAMPOS); }
  async esperar() {}
  destruir() {}
}
