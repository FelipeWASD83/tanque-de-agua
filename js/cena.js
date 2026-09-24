// As paredes do tanque, guardadas como um campo de distância com sinal
// (sdf) numa grade de 2×2 pontos por célula da simulação. Cada traço é um
// segmento com pontas redondas (uma "cápsula"), e a distância até ele é
// calculada pela geometria, não por uma máscara de pixels: por isso a borda
// da parede sai lisa em qualquer ângulo. Sem DOM: roda no navegador e nos testes.
//
// Coordenadas em metros, com y para cima (y = 0 é o fundo do tanque).
// O sdf é medido em CÉLULAS: positivo na água/ar, negativo dentro da parede,
// limitado a ±SDF_MAX. Desenhar une (mínimo); apagar subtrai (máximo com o
// negativo). A moldura do tanque é reposta depois de cada traço.

import { RES_MASCARA, SDF_MAX, MOLDURA } from './constantes.js';

export class Cena {
  constructor(nx, ny, h) {
    this.nx = nx;
    this.ny = ny;
    this.h = h;
    this.r = RES_MASCARA;
    this.mw = nx * this.r;
    this.mh = ny * this.r;
    this.largura = nx * h;
    this.altura = ny * h;
    this.sdf = new Float32Array(this.mw * this.mh);
    /** Traços desde o início, em metros — é com eles que o desenho sobrevive a um redimensionamento. */
    this.tracos = [];
    /** Faixa de linhas do sdf que mudou desde a última leitura (para subir à GPU). */
    this.linhasAlteradas = null;
    this.mudou = false;
    /** Conta as mudanças nas paredes (o desenho de reserva usa para saber quando redesenhar). */
    this.revisao = 0;
    this.#zerar();
  }

  /** Pixel sólido? (derivado do sdf) */
  get solido() {
    const s = new Uint8Array(this.sdf.length);
    for (let k = 0; k < s.length; k++) s[k] = this.sdf[k] < 0 ? 1 : 0;
    return s;
  }

  /** Desenha (ou apaga) um segmento de parede com pontas redondas. Raio em metros. */
  tracar(x0, y0, x1, y1, raio, apagar = false, registrar = true) {
    if (registrar) this.tracos.push({ x0, y0, x1, y1, raio, apagar });
    const r = this.r;
    const s = r / this.h; // pontos de grade por metro
    const ax = x0 * s - 0.5, ay = y0 * s - 0.5, bx = x1 * s - 0.5, by = y1 * s - 0.5;
    const rp = raio * s;
    const alc = rp + SDF_MAX * r + 1;
    const xmin = Math.max(0, Math.floor(Math.min(ax, bx) - alc));
    const xmax = Math.min(this.mw - 1, Math.ceil(Math.max(ax, bx) + alc));
    const ymin = Math.max(0, Math.floor(Math.min(ay, by) - alc));
    const ymax = Math.min(this.mh - 1, Math.ceil(Math.max(ay, by) + alc));
    if (xmin > xmax || ymin > ymax) return;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const w = this.mw, sdf = this.sdf;
    for (let y = ymin; y <= ymax; y++) {
      for (let x = xmin; x <= xmax; x++) {
        let t = len2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / len2 : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const px = ax + t * dx - x, py = ay + t * dy - y;
        let d = (Math.sqrt(px * px + py * py) - rp) / r;
        if (d > SDF_MAX) d = SDF_MAX; else if (d < -SDF_MAX) d = -SDF_MAX;
        const k = y * w + x;
        let v = apagar ? Math.max(sdf[k], -d) : Math.min(sdf[k], d);
        if (apagar) v = Math.min(v, this.#moldura(x, y));
        sdf[k] = v;
      }
    }
    this.#marcar(ymin, ymax);
  }

  /** Apaga todas as paredes (a moldura fica). */
  limpar() {
    this.tracos = [];
    this.#zerar();
  }

  /** Refaz o desenho a partir de uma lista de traços (desfazer, redimensionar). */
  refazer(tracos) {
    this.tracos = [];
    this.#zerar();
    for (const t of tracos) this.tracar(t.x0, t.y0, t.x1, t.y1, t.raio, t.apagar);
  }

  /** Devolve true (uma vez) se as paredes mudaram desde a última chamada. */
  atualizar() {
    const m = this.mudou;
    this.mudou = false;
    return m;
  }

  /** Lê e zera a faixa de linhas do sdf que mudou. */
  pegarLinhasAlteradas() {
    const l = this.linhasAlteradas;
    this.linhasAlteradas = null;
    return l;
  }

  /** Distância (em células) da posição (metros) até a parede, interpolada. */
  distancia(x, y) {
    const fx = (x / this.h) * this.r - 0.5;
    const fy = (y / this.h) * this.r - 0.5;
    return this.#bilinear(fx, fy);
  }

  /** Normal (unitária, apontando para fora da parede) na posição (metros). */
  normal(x, y) {
    const fx = (x / this.h) * this.r - 0.5;
    const fy = (y / this.h) * this.r - 0.5;
    const gx = this.#bilinear(fx + 1, fy) - this.#bilinear(fx - 1, fy);
    const gy = this.#bilinear(fx, fy + 1) - this.#bilinear(fx, fy - 1);
    const n = Math.hypot(gx, gy);
    return n > 1e-9 ? [gx / n, gy / n] : [0, 1];
  }

  /** A célula (i, j) da grade é parede? (Centro da célula dentro da parede.) */
  celulaSolida(i, j) {
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return true;
    const w = this.mw, r = this.r;
    const a = i * r, b = j * r;
    // Com r = 2 o centro da célula cai no meio dos 4 pontos dela.
    const s = this.sdf[b * w + a] + this.sdf[b * w + a + 1] +
      this.sdf[(b + 1) * w + a] + this.sdf[(b + 1) * w + a + 1];
    return s < 0;
  }

  // ---------------------------------------------------------------- internos

  /** Distância (células) até a borda interna da moldura: positiva dentro do tanque. */
  #moldura(x, y) {
    const r = this.r;
    const m = MOLDURA * r - 0.5; // borda interna, em coordenadas de ponto
    const d = Math.min(x - m, this.mw - 1 - m - x, y - m, this.mh - 1 - m - y) / r;
    return d > SDF_MAX ? SDF_MAX : d < -SDF_MAX ? -SDF_MAX : d;
  }

  #zerar() {
    for (let y = 0; y < this.mh; y++) {
      for (let x = 0; x < this.mw; x++) this.sdf[y * this.mw + x] = this.#moldura(x, y);
    }
    this.#marcar(0, this.mh - 1);
  }

  #marcar(y0, y1) {
    this.mudou = true;
    this.revisao++;
    if (!this.linhasAlteradas) this.linhasAlteradas = [y0, y1];
    else {
      this.linhasAlteradas[0] = Math.min(this.linhasAlteradas[0], y0);
      this.linhasAlteradas[1] = Math.max(this.linhasAlteradas[1], y1);
    }
  }

  #bilinear(fx, fy) {
    const w = this.mw, hh = this.mh;
    if (fx < 0) fx = 0; else if (fx > w - 1) fx = w - 1;
    if (fy < 0) fy = 0; else if (fy > hh - 1) fy = hh - 1;
    const x0 = Math.min(Math.floor(fx), w - 2), y0 = Math.min(Math.floor(fy), hh - 2);
    const tx = fx - x0, ty = fy - y0;
    const k = y0 * w + x0;
    const d = this.sdf;
    return (d[k] * (1 - tx) + d[k + 1] * tx) * (1 - ty) + (d[k + w] * (1 - tx) + d[k + w + 1] * tx) * ty;
  }
}
