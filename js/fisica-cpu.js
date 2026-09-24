// A física da água na CPU: FLIP/APIC (partículas carregam a água, uma grade
// resolve a pressão). É a versão de referência — os testes rodam nela — e o
// motor de reserva quando o navegador não tem WebGPU. O motor da GPU
// (gpu/fisica.js) faz exatamente os mesmos passos, na mesma ordem.
//
// Um subpasso:
//   1. partículas → grade (APIC): velocidade nas faces, densidade nas células
//   2. gravidade nas faces com água
//   3. classifica as células (parede / água / ar) e zera as faces de parede
//   4. pressão: deixa a água incompressível (e desfaz compressão acumulada)
//   5. aplica o gradiente de pressão e estende a velocidade para o ar
//   6. grade → partículas (APIC), anda, bate na parede

import {
  GRAVIDADE, PARTICULAS_POR_CELULA, TAU_DENSIDADE, ITERACOES_PRESSAO, OMEGA,
  CAMADAS_EXTRAPOLACAO, RAIO_PARTICULA, SDF_MAX, ESPUMA_V0, ESPUMA_V1, ESPUMA_TAXA, ESPUMA_VIDA, DIFUSAO_CORANTE,
} from './constantes.js';

const SOLIDO = 0, AGUA = 1, AR = 2;
/** Campos guardados por partícula (mesma ordem da struct da GPU). */
export const CAMPOS = 12; // x y vx vy cux cuy cvx cvy espuma absR absG absB

export class SimulacaoCPU {
  constructor(cena, opcoes = {}) {
    this.cena = cena;
    this.nx = cena.nx;
    this.ny = cena.ny;
    this.h = cena.h;
    this.gravidade = opcoes.gravidade ?? [0, -GRAVIDADE];
    this.iteracoes = opcoes.iteracoes ?? ITERACOES_PRESSAO;
    this.tauDensidade = opcoes.tauDensidade ?? TAU_DENSIDADE;
    const nx = this.nx, ny = this.ny;
    this.nu = (nx + 1) * ny;
    this.nv = nx * (ny + 1);
    this.u = new Float32Array(this.nu);
    this.v = new Float32Array(this.nv);
    this.uPeso = new Float32Array(this.nu);
    this.vPeso = new Float32Array(this.nv);
    this.uValida = new Uint8Array(this.nu);
    this.vValida = new Uint8Array(this.nv);
    this.uTmp = new Float32Array(this.nu);
    this.vTmp = new Float32Array(this.nv);
    this.densidade = new Float32Array(nx * ny);
    this.contagemCelula = new Uint32Array(nx * ny);
    this.corCelula = new Float32Array(nx * ny * 3);
    this.tipo = new Uint8Array(nx * ny);
    this.solidoCelula = new Uint8Array(nx * ny);
    this.div = new Float32Array(nx * ny);
    this.q = new Float32Array(nx * ny);
    this.capacidade = 0;
    this.n = 0;
    this.p = new Float32Array(0);
    this.#garantir(1024);
    this.atualizarParedes();
  }

  /** Recalcula quais células são parede (chamar depois de cena.atualizar()). */
  atualizarParedes() {
    const { nx, ny, cena } = this;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) this.solidoCelula[i + j * nx] = cena.celulaSolida(i, j) ? 1 : 0;
    }
  }

  /**
   * Acrescenta partículas. `lista` é um Float32Array com CAMPOS valores por
   * partícula. Partículas dentro da parede ou em célula já cheia são puladas
   * (como no motor da GPU), para não comprimir a água que já está lá.
   */
  adicionar(lista, checarCheia = true) {
    const total = lista.length / CAMPOS;
    this.#garantir(this.n + total);
    const { h, nx, ny } = this;
    let aceitas = 0;
    for (let k = 0; k < total; k++) {
      const o = k * CAMPOS;
      const x = lista[o], y = lista[o + 1];
      if (!(this.cena.distancia(x, y) > RAIO_PARTICULA + 0.05)) continue;
      if (checarCheia) {
        const i = Math.floor(x / h), j = Math.floor(y / h);
        if (i < 0 || j < 0 || i >= nx || j >= ny) continue;
        if (this.contagemCelula[i + j * nx] >= PARTICULAS_POR_CELULA - 1) continue;
      }
      this.p.set(lista.subarray(o, o + CAMPOS), this.n * CAMPOS);
      this.n++;
      aceitas++;
    }
    return aceitas;
  }

  /** Remove as partículas para as quais `fn(x, y, indice)` devolve true. */
  remover(fn) {
    const p = this.p;
    let w = 0;
    for (let k = 0; k < this.n; k++) {
      const o = k * CAMPOS;
      if (fn(p[o], p[o + 1], k)) continue;
      if (w !== k) p.copyWithin(w * CAMPOS, o, o + CAMPOS);
      w++;
    }
    const removidas = this.n - w;
    this.n = w;
    return removidas;
  }

  /** Ferramentas que agem sobre a água num círculo (esponja, mão, corante). */
  aplicarFerramenta(f) {
    if (!f || f.tipo === 'nenhuma') return;
    const r2 = f.raio * f.raio;
    if (f.tipo === 'esponja') {
      this.remover((x, y) => (x - f.x) ** 2 + (y - f.y) ** 2 < r2);
      return;
    }
    const p = this.p;
    for (let k = 0; k < this.n; k++) {
      const o = k * CAMPOS;
      const d2 = (p[o] - f.x) ** 2 + (p[o + 1] - f.y) ** 2;
      if (d2 >= r2) continue;
      const peso = 1 - Math.sqrt(d2 / r2);
      if (f.tipo === 'mao') {
        const a = Math.min(1, peso * 0.9);
        p[o + 2] += (f.vx - p[o + 2]) * a;
        p[o + 3] += (f.vy - p[o + 3]) * a;
      } else if (f.tipo === 'corante') {
        p[o + 9] = f.absorcao[0];
        p[o + 10] = f.absorcao[1];
        p[o + 11] = f.absorcao[2];
      }
    }
  }

  /** Avança dt segundos em n subpassos iguais. */
  avancar(dt, subpassos = 1) {
    const d = dt / subpassos;
    for (let s = 0; s < subpassos; s++) this.subpasso(d);
  }

  subpasso(dt) {
    this.#paraGrade();
    this.#gravidadeEFronteira(dt);
    this.#classificar(dt);
    this.#pressao();
    this.#gradiente();
    this.#extrapolar();
    this.#paraParticulas(dt);
  }

  /** Volume de água em m² (área no plano do tanque), pela contagem de partículas. */
  area() {
    return (this.n / PARTICULAS_POR_CELULA) * this.h * this.h;
  }

  // ------------------------------------------------------------ 1. P2G

  #paraGrade() {
    const { nx, ny, h, p } = this;
    const u = this.u, v = this.v, uw = this.uPeso, vw = this.vPeso;
    u.fill(0); v.fill(0); uw.fill(0); vw.fill(0);
    this.densidade.fill(0);
    this.contagemCelula.fill(0);
    this.corCelula.fill(0);
    const inv = 1 / h;
    for (let k = 0; k < this.n; k++) {
      const o = k * CAMPOS;
      const x = p[o], y = p[o + 1];
      const vx = p[o + 2], vy = p[o + 3];
      const cux = p[o + 4], cuy = p[o + 5], cvx = p[o + 6], cvy = p[o + 7];

      // faces u: nós em (i·h, (j+½)·h)
      {
        const fx = x * inv, fy = y * inv - 0.5;
        const i0 = clampi(Math.floor(fx), 0, nx - 1), j0 = clampi(Math.floor(fy), 0, ny - 2);
        const tx = fx - i0, ty = fy - j0;
        for (let b = 0; b < 2; b++) {
          const wy = b ? ty : 1 - ty;
          const dy = (j0 + b + 0.5) * h - y;
          for (let a = 0; a < 2; a++) {
            const w = (a ? tx : 1 - tx) * wy;
            const dx = (i0 + a) * h - x;
            const idx = i0 + a + (j0 + b) * (nx + 1);
            u[idx] += w * (vx + cux * dx + cuy * dy);
            uw[idx] += w;
          }
        }
      }
      // faces v: nós em ((i+½)·h, j·h)
      {
        const fx = x * inv - 0.5, fy = y * inv;
        const i0 = clampi(Math.floor(fx), 0, nx - 2), j0 = clampi(Math.floor(fy), 0, ny - 1);
        const tx = fx - i0, ty = fy - j0;
        for (let b = 0; b < 2; b++) {
          const wy = b ? ty : 1 - ty;
          const dy = (j0 + b) * h - y;
          for (let a = 0; a < 2; a++) {
            const w = (a ? tx : 1 - tx) * wy;
            const dx = (i0 + a + 0.5) * h - x;
            const idx = i0 + a + (j0 + b) * nx;
            v[idx] += w * (vy + cvx * dx + cvy * dy);
            vw[idx] += w;
          }
        }
      }
      // densidade nos centros das células
      {
        const fx = x * inv - 0.5, fy = y * inv - 0.5;
        const i0 = clampi(Math.floor(fx), 0, nx - 2), j0 = clampi(Math.floor(fy), 0, ny - 2);
        const tx = fx - i0, ty = fy - j0;
        const d = this.densidade;
        const c = i0 + j0 * nx;
        d[c] += (1 - tx) * (1 - ty);
        d[c + 1] += tx * (1 - ty);
        d[c + nx] += (1 - tx) * ty;
        d[c + nx + 1] += tx * ty;
      }
      const ci = clampi(Math.floor(x * inv), 0, nx - 1), cj = clampi(Math.floor(y * inv), 0, ny - 1);
      const cc = ci + cj * nx;
      this.contagemCelula[cc]++;
      this.corCelula[cc * 3] += p[o + 9];
      this.corCelula[cc * 3 + 1] += p[o + 10];
      this.corCelula[cc * 3 + 2] += p[o + 11];
    }
    for (let k = 0; k < this.nu; k++) if (uw[k] > 0) u[k] /= uw[k];
    for (let k = 0; k < this.nv; k++) if (vw[k] > 0) v[k] /= vw[k];
  }

  // ------------------------------------------------------------ 2 e 3

  #gravidadeEFronteira(dt) {
    const { nx, ny } = this;
    const [gx, gy] = this.gravidade;
    const sol = this.solidoCelula;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const k = i + j * (nx + 1);
        if (this.uPeso[k] > 0) this.u[k] += gx * dt;
        const s = i === 0 || i === nx || sol[i - 1 + j * nx] || sol[i + j * nx];
        if (s) this.u[k] = 0;
      }
    }
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = i + j * nx;
        if (this.vPeso[k] > 0) this.v[k] += gy * dt;
        const s = j === 0 || j === ny || sol[i + (j - 1) * nx] || sol[i + j * nx];
        if (s) this.v[k] = 0;
      }
    }
  }

  #classificar(dt) {
    const { nx, ny, h } = this;
    const alvo = h / Math.max(this.tauDensidade, dt); // velocidade que desfaz a compressão em tau segundos
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = i + j * nx;
        let t = AR;
        if (this.solidoCelula[c]) t = SOLIDO;
        else if (this.contagemCelula[c] > 0) t = AGUA;
        this.tipo[c] = t;
        if (t !== AGUA) { this.div[c] = 0; this.q[c] = 0; continue; }
        const d = this.u[i + 1 + j * (nx + 1)] - this.u[i + j * (nx + 1)] +
          this.v[i + (j + 1) * nx] - this.v[i + j * nx];
        let excesso = this.densidade[c] / PARTICULAS_POR_CELULA - 1;
        // Rarefeita só conta no miolo da água: perto do ar e da parede a
        // densidade medida é naturalmente menor (falta vizinho de um lado).
        if (excesso < 0 && !this.#miolo(i, j)) excesso = 0;
        // O que a pressão precisa desfazer: a divergência atual menos a expansão desejada.
        this.div[c] = d - alvo * excesso;
      }
    }
  }

  #miolo(i, j) {
    const { nx, ny } = this;
    if (i < 1 || j < 1 || i >= nx - 1 || j >= ny - 1) return false;
    for (let b = -1; b <= 1; b++) {
      for (let a = -1; a <= 1; a++) {
        const c = i + a + (j + b) * nx;
        if (this.solidoCelula[c] || this.contagemCelula[c] === 0) return false;
      }
    }
    return true;
  }

  // ------------------------------------------------------------ 4. pressão

  #pressao() {
    const { nx, ny, tipo, q, div } = this;
    for (let it = 0; it < this.iteracoes; it++) {
      for (let cor = 0; cor < 2; cor++) {
        for (let j = 1; j < ny - 1; j++) {
          for (let i = 1 + ((j + cor) & 1); i < nx - 1; i += 2) {
            const c = i + j * nx;
            if (tipo[c] !== AGUA) continue;
            let soma = 0, n = 0;
            let t = tipo[c - 1]; if (t !== SOLIDO) { n++; soma += q[c - 1]; }
            t = tipo[c + 1]; if (t !== SOLIDO) { n++; soma += q[c + 1]; }
            t = tipo[c - nx]; if (t !== SOLIDO) { n++; soma += q[c - nx]; }
            t = tipo[c + nx]; if (t !== SOLIDO) { n++; soma += q[c + nx]; }
            const novo = n > 0 ? (soma - div[c]) / n : 0;
            q[c] += OMEGA * (novo - q[c]);
          }
        }
      }
    }
  }

  // ------------------------------------------------------------ 5

  #gradiente() {
    const { nx, ny, tipo, q } = this;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const k = i + j * (nx + 1);
        const tl = i > 0 ? tipo[i - 1 + j * nx] : SOLIDO;
        const tr = i < nx ? tipo[i + j * nx] : SOLIDO;
        if (tl === SOLIDO || tr === SOLIDO) { this.u[k] = 0; this.uValida[k] = 1; continue; }
        if (tl === AGUA || tr === AGUA) {
          this.u[k] -= q[i + j * nx] - q[i - 1 + j * nx];
          this.uValida[k] = 1;
        } else this.uValida[k] = 0;
      }
    }
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = i + j * nx;
        const tb = j > 0 ? tipo[i + (j - 1) * nx] : SOLIDO;
        const tt = j < ny ? tipo[i + j * nx] : SOLIDO;
        if (tb === SOLIDO || tt === SOLIDO) { this.v[k] = 0; this.vValida[k] = 1; continue; }
        if (tb === AGUA || tt === AGUA) {
          this.v[k] -= q[i + j * nx] - q[i + (j - 1) * nx];
          this.vValida[k] = 1;
        } else this.vValida[k] = 0;
      }
    }
  }

  #extrapolar() {
    for (let camada = 0; camada < CAMADAS_EXTRAPOLACAO; camada++) {
      extrapolarComponente(this.u, this.uTmp, this.uValida, this.nx + 1, this.ny);
      extrapolarComponente(this.v, this.vTmp, this.vValida, this.nx, this.ny + 1);
    }
  }

  // ------------------------------------------------------------ 6. G2P

  #paraParticulas(dt) {
    const { nx, ny, h, p, cena } = this;
    const inv = 1 / h;
    const vmax = (0.95 * h) / dt; // não deixa andar mais de uma célula por subpasso
    const decai = Math.exp(-dt / ESPUMA_VIDA);
    const mistura = Math.min(1, dt / DIFUSAO_CORANTE);
    let mortas = false;
    let maior = 0;
    for (let k = 0; k < this.n; k++) {
      const o = k * CAMPOS;
      let x = p[o], y = p[o + 1];
      let vx = 0, vy = 0, cux = 0, cuy = 0, cvx = 0, cvy = 0;
      {
        const fx = x * inv, fy = y * inv - 0.5;
        const i0 = clampi(Math.floor(fx), 0, nx - 1), j0 = clampi(Math.floor(fy), 0, ny - 2);
        const tx = fx - i0, ty = fy - j0;
        for (let b = 0; b < 2; b++) {
          for (let a = 0; a < 2; a++) {
            const val = this.u[i0 + a + (j0 + b) * (nx + 1)];
            const wx = a ? tx : 1 - tx, wy = b ? ty : 1 - ty;
            vx += wx * wy * val;
            cux += (a ? 1 : -1) * wy * inv * val;
            cuy += wx * (b ? 1 : -1) * inv * val;
          }
        }
      }
      {
        const fx = x * inv - 0.5, fy = y * inv;
        const i0 = clampi(Math.floor(fx), 0, nx - 2), j0 = clampi(Math.floor(fy), 0, ny - 1);
        const tx = fx - i0, ty = fy - j0;
        for (let b = 0; b < 2; b++) {
          for (let a = 0; a < 2; a++) {
            const val = this.v[i0 + a + (j0 + b) * nx];
            const wx = a ? tx : 1 - tx, wy = b ? ty : 1 - ty;
            vy += wx * wy * val;
            cvx += (a ? 1 : -1) * wy * inv * val;
            cvy += wx * (b ? 1 : -1) * inv * val;
          }
        }
      }
      const vel = Math.hypot(vx, vy);
      // Espuma: rápida e rarefeita (densidade da célula abaixo da metade da água).
      const ci = clampi(Math.floor(x * inv), 0, nx - 1), cj = clampi(Math.floor(y * inv), 0, ny - 1);
      const rarefeita = suave(0.95, 0.45, this.densidade[ci + cj * nx] / PARTICULAS_POR_CELULA);
      const fonte = suave(ESPUMA_V0, ESPUMA_V1, vel) * rarefeita;
      const espuma = Math.min(1, p[o + 8] * decai + fonte * ESPUMA_TAXA * dt);
      // corante: difunde puxando a cor para a média da célula
      const nCel = this.contagemCelula[ci + cj * nx];
      if (nCel > 1) {
        const cc = (ci + cj * nx) * 3;
        for (let q = 0; q < 3; q++) p[o + 9 + q] += (this.corCelula[cc + q] / nCel - p[o + 9 + q]) * mistura;
      }

      if (vel > vmax) { vx *= vmax / vel; vy *= vmax / vel; }
      if (vel > maior) maior = vel;
      x += vx * dt;
      y += vy * dt;

      const d = cena.distancia(x, y);
      if (d < RAIO_PARTICULA) {
        if (d < -SDF_MAX + 0.5) { p[o] = NaN; mortas = true; continue; }
        const [nX, nY] = cena.normal(x, y);
        const empurra = (RAIO_PARTICULA - d) * h;
        x += nX * empurra;
        y += nY * empurra;
        const vn = vx * nX + vy * nY;
        if (vn < 0) { vx -= vn * nX; vy -= vn * nY; }
      }
      p[o] = x; p[o + 1] = y; p[o + 2] = vx; p[o + 3] = vy;
      p[o + 4] = cux; p[o + 5] = cuy; p[o + 6] = cvx; p[o + 7] = cvy;
      p[o + 8] = espuma;
    }
    if (mortas) this.remover((x) => Number.isNaN(x));
    this.vMax = maior;
  }

  #garantir(n) {
    if (n <= this.capacidade) return;
    const cap = Math.max(n, this.capacidade * 2);
    const novo = new Float32Array(cap * CAMPOS);
    novo.set(this.p.subarray(0, this.n * CAMPOS));
    this.p = novo;
    this.capacidade = cap;
  }
}

function extrapolarComponente(a, tmp, valida, w, hh) {
  tmp.set(a);
  const novas = [];
  for (let j = 0; j < hh; j++) {
    for (let i = 0; i < w; i++) {
      const k = i + j * w;
      if (valida[k]) continue;
      let soma = 0, n = 0;
      if (i > 0 && valida[k - 1]) { soma += a[k - 1]; n++; }
      if (i < w - 1 && valida[k + 1]) { soma += a[k + 1]; n++; }
      if (j > 0 && valida[k - w]) { soma += a[k - w]; n++; }
      if (j < hh - 1 && valida[k + w]) { soma += a[k + w]; n++; }
      if (n > 0) { tmp[k] = soma / n; novas.push(k); }
    }
  }
  a.set(tmp);
  for (const k of novas) valida[k] = 1;
}

function suave(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function clampi(v, a, b) {
  return v < a ? a : v > b ? b : v;
}

/**
 * Semeia água num retângulo (metros): 2×2 partículas por célula, com um
 * tremor pequeno para não formar uma grade perfeita.
 */
export function semearRetangulo(x0, y0, x1, y1, h, absorcao = [0, 0, 0], aleatorio = Math.random) {
  const lado = Math.round(Math.sqrt(PARTICULAS_POR_CELULA));
  const passo = h / lado;
  const xa = Math.min(x0, x1), xb = Math.max(x0, x1);
  const ya = Math.min(y0, y1), yb = Math.max(y0, y1);
  const ia = Math.ceil(xa / passo - 0.5), ib = Math.floor(xb / passo - 0.5);
  const ja = Math.ceil(ya / passo - 0.5), jb = Math.floor(yb / passo - 0.5);
  const n = Math.max(0, ib - ia + 1) * Math.max(0, jb - ja + 1);
  const lista = new Float32Array(n * CAMPOS);
  let k = 0;
  for (let j = ja; j <= jb; j++) {
    for (let i = ia; i <= ib; i++) {
      const o = k * CAMPOS;
      lista[o] = (i + 0.5 + (aleatorio() - 0.5) * 0.5) * passo;
      lista[o + 1] = (j + 0.5 + (aleatorio() - 0.5) * 0.5) * passo;
      lista[o + 9] = absorcao[0];
      lista[o + 10] = absorcao[1];
      lista[o + 11] = absorcao[2];
      k++;
    }
  }
  return lista;
}
