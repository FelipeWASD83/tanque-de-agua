// Desenho de reserva, sem WebGPU. O painel e as paredes são desenhados em
// vetor, na resolução da tela (ficam nítidos); só a água é calculada numa
// imagem menor (3 pixels por célula) e aplicada por cima como uma tinta
// multiplicada — a mesma absorção de cor do shader da GPU (Beer-Lambert), sem
// a refração.

import { PARTICULAS_POR_CELULA, PROFUNDIDADE, ABSORCAO_AGUA, MOLDURA, RAIO_MANCHA } from './constantes.js';
import { CAMPOS } from './fisica-cpu.js';

const PX_POR_CELULA = 3;
const COR_PAREDE = '#33393d';

function suave(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export class Render2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  #tela(cena) {
    const w = this.canvas.width, h = this.canvas.height;
    return { w, h, sx: w / cena.largura, sy: h / cena.altura };
  }

  #prepararPainel(cena) {
    const { w, h, sx, sy } = this.#tela(cena);
    if (this.painel && this.painel.width === w && this.painel.height === h && this.painelCena === cena) return;
    this.painelCena = cena;
    this.painel = new OffscreenCanvas(w, h);
    const c = this.painel.getContext('2d');
    const g = c.createRadialGradient(w / 2, h * 0.38, 0, w / 2, h * 0.38, Math.max(w, h) * 0.75);
    g.addColorStop(0, '#f1f4f5');
    g.addColorStop(1, '#e2e7e9');
    c.fillStyle = g;
    c.fillRect(0, 0, w, h);
    const m = MOLDURA * cena.h;
    const X = (x) => x * sx, Y = (y) => h - y * sy;
    // linhas a cada 10 cm
    c.strokeStyle = 'rgba(120, 136, 146, 0.35)';
    c.lineWidth = 1;
    for (let x = m; x < cena.largura; x += 0.1) { c.beginPath(); c.moveTo(Math.round(X(x)) + 0.5, 0); c.lineTo(Math.round(X(x)) + 0.5, h); c.stroke(); }
    for (let y = m; y < cena.altura; y += 0.1) { c.beginPath(); c.moveTo(0, Math.round(Y(y)) + 0.5); c.lineTo(w, Math.round(Y(y)) + 0.5); c.stroke(); }
    // pontos a cada 1 cm, maiores a cada 5
    for (let i = 0; m + i * 0.01 < cena.largura; i++) {
      for (let j = 0; m + j * 0.01 < cena.altura; j++) {
        const grande = i % 5 === 0 && j % 5 === 0;
        c.fillStyle = grande ? 'rgba(90, 104, 112, 0.9)' : 'rgba(140, 152, 160, 0.7)';
        const r = grande ? 1.3 : 0.7;
        c.fillRect(X(m + i * 0.01) - r, Y(m + j * 0.01) - r, r * 2, r * 2);
      }
    }
    // régua na borda esquerda
    c.fillStyle = 'rgba(80, 92, 99, 0.9)';
    for (let j = 0; m + j * 0.01 < cena.altura; j++) {
      const comp = j % 10 === 0 ? 0.014 : j % 5 === 0 ? 0.008 : 0.004;
      c.fillRect(X(m), Y(m + j * 0.01) - 0.5, comp * sx, 1);
    }
  }

  #prepararParedes(cena) {
    const { w, h, sx, sy } = this.#tela(cena);
    if (this.paredes && this.paredes.width === w && this.paredes.height === h && this.paredesRev === cena.revisao && this.paredesCena === cena) return;
    this.paredesCena = cena;
    this.paredesRev = cena.revisao;
    this.paredes = new OffscreenCanvas(w, h);
    const c = this.paredes.getContext('2d');
    c.lineCap = 'round';
    c.strokeStyle = COR_PAREDE;
    for (const t of cena.tracos) {
      c.globalCompositeOperation = t.apagar ? 'destination-out' : 'source-over';
      c.lineWidth = t.raio * 2 * sx;
      c.beginPath();
      c.moveTo(t.x0 * sx, h - t.y0 * sy);
      c.lineTo(t.x1 * sx, h - t.y1 * sy);
      c.stroke();
    }
    c.globalCompositeOperation = 'source-over';
    const m = MOLDURA * cena.h;
    c.fillStyle = COR_PAREDE;
    c.fillRect(0, 0, w, m * sy);
    c.fillRect(0, h - m * sy, w, m * sy);
    c.fillRect(0, 0, m * sx, h);
    c.fillRect(w - m * sx, 0, m * sx, h);
  }

  #prepararAgua(cena) {
    const w = cena.nx * PX_POR_CELULA, h = cena.ny * PX_POR_CELULA;
    if (this.tinta && this.tinta.width === w && this.tinta.height === h) return;
    this.tinta = new OffscreenCanvas(w, h);
    this.tctx = this.tinta.getContext('2d');
    this.imgTinta = new ImageData(w, h);
    this.espumaCanvas = new OffscreenCanvas(w, h);
    this.ectx = this.espumaCanvas.getContext('2d');
    this.imgEspuma = new ImageData(w, h);
    this.campo = new Float32Array(w * h);
    this.espuma = new Float32Array(w * h);
    this.corante = new Float32Array(w * h * 3);
  }

  desenhar(sim) {
    const cena = sim.cena;
    this.#prepararPainel(cena);
    this.#prepararParedes(cena);
    this.#prepararAgua(cena);
    const ctx = this.ctx;
    const { w: W, h: H } = this.#tela(cena);
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(this.painel, 0, 0);

    // campo de água (y para baixo, como a imagem)
    const w = this.tinta.width, hh = this.tinta.height;
    const campo = this.campo, espuma = this.espuma, corante = this.corante;
    campo.fill(0); espuma.fill(0); corante.fill(0);
    const escala = PX_POR_CELULA / cena.h;
    const R = RAIO_MANCHA * PX_POR_CELULA, R2 = R * R;
    const p = sim.p;
    for (let k = 0; k < sim.n; k++) {
      const o = k * CAMPOS;
      const cx = p[o] * escala, cy = hh - p[o + 1] * escala;
      const x0 = Math.max(0, Math.ceil(cx - R - 0.5)), x1 = Math.min(w - 1, Math.floor(cx + R - 0.5));
      const y0 = Math.max(0, Math.ceil(cy - R - 0.5)), y1 = Math.min(hh - 1, Math.floor(cy + R - 0.5));
      const esp = p[o + 8], a0 = p[o + 9], a1 = p[o + 10], a2 = p[o + 11];
      for (let y = y0; y <= y1; y++) {
        const dy = y + 0.5 - cy;
        for (let x = x0; x <= x1; x++) {
          const dx = x + 0.5 - cx;
          const r2 = (dx * dx + dy * dy) / R2;
          if (r2 >= 1) continue;
          const peso = (1 - r2) * (1 - r2);
          const i = y * w + x;
          campo[i] += peso;
          espuma[i] += peso * esp;
          corante[i * 3] += peso * a0; corante[i * 3 + 1] += peso * a1; corante[i * 3 + 2] += peso * a2;
        }
      }
    }
    const phi0 = (PARTICULAS_POR_CELULA * Math.PI * RAIO_MANCHA * RAIO_MANCHA) / 3;
    const dt = this.imgTinta.data, de = this.imgEspuma.data;
    for (let i = 0; i < w * hh; i++) {
      const phi = campo[i] / phi0;
      const cobre = suave(0.26, 0.5, phi);
      const o = i * 4;
      if (cobre < 0.002) {
        dt[o] = dt[o + 1] = dt[o + 2] = 255; dt[o + 3] = 255;
        de[o + 3] = 0;
        continue;
      }
      const esp = suave(0.15, 0.95, phi) * PROFUNDIDADE;
      const inv = 1 / Math.max(campo[i], 1e-4);
      const borda = 1 - cobre * (1 - cobre) * 4 * 0.45;
      const tr = Math.exp(-(ABSORCAO_AGUA[0] + corante[i * 3] * inv) * esp) * borda;
      const tg = Math.exp(-(ABSORCAO_AGUA[1] + corante[i * 3 + 1] * inv) * esp) * borda;
      const tb = Math.exp(-(ABSORCAO_AGUA[2] + corante[i * 3 + 2] * inv) * esp) * borda;
      dt[o] = (1 - cobre + cobre * tr) * 255;
      dt[o + 1] = (1 - cobre + cobre * tg) * 255;
      dt[o + 2] = (1 - cobre + cobre * tb) * 255;
      dt[o + 3] = 255;
      const e = suave(0.05, 0.75, Math.min(1, espuma[i] * inv)) * 0.85 * cobre;
      de[o] = 248; de[o + 1] = 251; de[o + 2] = 252; de[o + 3] = e * 255;
    }
    this.tctx.putImageData(this.imgTinta, 0, 0);
    this.ectx.putImageData(this.imgEspuma, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.globalCompositeOperation = 'multiply';
    ctx.drawImage(this.tinta, 0, 0, W, H);
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(this.espumaCanvas, 0, 0, W, H);

    // paredes com sombra no painel (luz de cima, à esquerda)
    ctx.save();
    ctx.shadowColor = 'rgba(20, 28, 32, 0.28)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 3;
    ctx.shadowOffsetY = 5;
    ctx.drawImage(this.paredes, 0, 0);
    ctx.restore();
  }
}
