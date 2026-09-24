// Cenários prontos: cada um devolve paredes (traços) e água inicial, em metros,
// ajustados ao tamanho do tanque. Depois de carregado, tudo pode ser apagado
// e redesenhado como qualquer traço feito à mão.

import { MOLDURA } from './constantes.js';
import { semearRetangulo, CAMPOS } from './fisica-cpu.js';

/** Espessura (raio) padrão de parede dos modelos, em metros. */
export function raioDeParede(h) {
  return Math.max(1.6 * h, 0.006);
}

function linha(pts, raio, fechar = false) {
  const t = [];
  const n = fechar ? pts.length : pts.length - 1;
  for (let k = 0; k < n; k++) {
    const a = pts[k], b = pts[(k + 1) % pts.length];
    t.push({ x0: a[0], y0: a[1], x1: b[0], y1: b[1], raio });
  }
  return t;
}

/** Palco: a faixa central onde o modelo é montado (não estica no ultrawide). */
function palco(W, H, h) {
  const m = MOLDURA * h;
  const largura = Math.min(W - 2 * m, 1.7 * H);
  return {
    x0: (W - largura) / 2, largura,
    chao: m, teto: H - m, altura: H - 2 * m,
    // x e y em frações do palco → metros
    X: (f) => (W - largura) / 2 + f * largura,
    Y: (f) => m + f * (H - 2 * m),
  };
}

export const MODELOS = {
  vazio: {
    nome: 'Vazio',
    montar: () => ({ tracos: [], agua: [] }),
  },

  copo: {
    nome: 'Copo',
    montar(W, H, h) {
      const p = palco(W, H, h), r = raioDeParede(h);
      const cx = W / 2;
      const meia = Math.min(p.largura * 0.14, p.altura * 0.2);
      const base = p.chao + r;
      const topo = p.Y(0.46);
      return {
        tracos: linha([[cx - meia * 1.18, topo], [cx - meia, base], [cx + meia, base], [cx + meia * 1.18, topo]], r),
        agua: [{ x0: cx - meia * 0.75, y0: p.Y(0.66), x1: cx + meia * 0.75, y1: p.Y(0.97) }],
      };
    },
  },

  funil: {
    nome: 'Funil',
    montar(W, H, h) {
      const p = palco(W, H, h), r = raioDeParede(h);
      const cx = W / 2;
      const boca = Math.min(p.largura * 0.24, p.altura * 0.36);
      const bico = r + 2.2 * h; // meia abertura do bico, contando a espessura
      const yBoca = p.Y(0.8), yGarganta = p.Y(0.54), yBico = p.Y(0.45);
      const copoMeia = Math.min(p.largura * 0.1, p.altura * 0.15);
      const tracos = [
        ...linha([[cx - boca, yBoca], [cx - bico, yGarganta], [cx - bico, yBico]], r),
        ...linha([[cx + boca, yBoca], [cx + bico, yGarganta], [cx + bico, yBico]], r),
        ...linha([[cx - copoMeia * 1.1, p.Y(0.3)], [cx - copoMeia, p.chao + r], [cx + copoMeia, p.chao + r], [cx + copoMeia * 1.1, p.Y(0.3)]], r),
      ];
      return {
        tracos,
        // começa acima da boca: nenhum canto do bloco pode ficar do lado de fora das paredes
        agua: [{ x0: cx - boca * 0.62, y0: p.Y(0.83), x1: cx + boca * 0.62, y1: p.Y(0.99) }],
      };
    },
  },

  ampulheta: {
    nome: 'Ampulheta',
    montar(W, H, h) {
      const p = palco(W, H, h), r = raioDeParede(h);
      const cx = W / 2;
      const meia = Math.min(p.largura * 0.2, p.altura * 0.3);
      const pescoco = r + 1.8 * h;
      const topo = p.Y(0.94), meio = p.Y(0.5), base = p.Y(0.06);
      const contorno = [
        [cx - meia, topo], [cx + meia, topo], [cx + meia, topo - (topo - meio) * 0.25],
        [cx + pescoco, meio],
        [cx + meia, base + (meio - base) * 0.25], [cx + meia, base], [cx - meia, base],
        [cx - meia, base + (meio - base) * 0.25], [cx - pescoco, meio],
        [cx - meia, topo - (topo - meio) * 0.25],
      ];
      // Água: a metade de cima, por dentro do vidro.
      const folga = r + h;
      const dentro = [
        [cx - meia + folga, topo - folga], [cx + meia - folga, topo - folga],
        [cx + meia - folga, topo - (topo - meio) * 0.25],
        [cx + pescoco * 0.3, meio + (topo - meio) * 0.12],
        [cx - pescoco * 0.3, meio + (topo - meio) * 0.12],
        [cx - meia + folga, topo - (topo - meio) * 0.25],
      ];
      return {
        tracos: linha(contorno, r, true),
        agua: [{ poligono: dentro, limiteY: meio + (topo - meio) * 0.72 }],
      };
    },
  },

  cascata: {
    nome: 'Cascata',
    montar(W, H, h) {
      const p = palco(W, H, h), r = raioDeParede(h);
      const tracos = [];
      const n = 5;
      const passoX = 0.17, passoY = 0.17;
      for (let k = 0; k < n; k++) {
        const xa = 0.06 + k * passoX, xb = xa + passoX * 1.15;
        const y = 0.82 - k * passoY;
        const borda = p.altura * 0.035;
        // bandeja: parede de trás alta, fundo e uma borda baixa por onde a água transborda
        tracos.push(...linha([
          [p.X(xa), p.Y(y) + borda * 4], [p.X(xa), p.Y(y)], [p.X(xb), p.Y(y)], [p.X(xb), p.Y(y) + borda],
        ], r));
      }
      return {
        tracos,
        agua: [{ x0: p.X(0.075), y0: p.Y(0.85), x1: p.X(0.245), y1: p.Y(0.99) }],
      };
    },
  },

  labirinto: {
    nome: 'Labirinto',
    // Um labirinto novo a cada vez (backtracker recursivo): corredores numa
    // caixa, entrada em cima à esquerda e saída embaixo à direita. A água
    // enche os bolsões sem saída e transborda até achar o caminho.
    montar(W, H, h, aleatorio = Math.random) {
      const p = palco(W, H, h), r = raioDeParede(h);
      const linhas = 6;
      const altura = p.altura * 0.78;
      const tam = altura / linhas;
      const colunas = Math.max(3, Math.min(10, Math.floor((p.largura * 0.9) / tam)));
      const largura = colunas * tam;
      const x0 = W / 2 - largura / 2, y0 = p.chao + p.altura * 0.03;
      // paredes internas: dir[c][l] (parede à direita da casa) e baixo[c][l] (parede embaixo)
      const dir = Array.from({ length: colunas }, () => Array(linhas).fill(true));
      const baixo = Array.from({ length: colunas }, () => Array(linhas).fill(true));
      const visto = Array.from({ length: colunas }, () => Array(linhas).fill(false));
      const pilha = [[0, linhas - 1]];
      visto[0][linhas - 1] = true;
      while (pilha.length) {
        const [c, l] = pilha.at(-1);
        const viz = [[c + 1, l], [c - 1, l], [c, l + 1], [c, l - 1]]
          .filter(([a, b]) => a >= 0 && b >= 0 && a < colunas && b < linhas && !visto[a][b]);
        if (!viz.length) { pilha.pop(); continue; }
        const [a, b] = viz[Math.floor(aleatorio() * viz.length)];
        if (a === c + 1) dir[c][l] = false;
        else if (a === c - 1) dir[a][l] = false;
        else if (b === l + 1) baixo[c][b] = false; // a casa de cima (l+1) perde o chão
        else baixo[c][l] = false;
        visto[a][b] = true;
        pilha.push([a, b]);
      }
      const X = (c) => x0 + c * tam, Y = (l) => y0 + l * tam; // l = 0 é a linha de baixo
      const tracos = [];
      const seg = (xa, ya, xb, yb) => tracos.push({ x0: xa, y0: ya, x1: xb, y1: yb, raio: r });
      for (let c = 0; c < colunas; c++) {
        for (let l = 0; l < linhas; l++) {
          if (dir[c][l] && c < colunas - 1) seg(X(c + 1), Y(l), X(c + 1), Y(l + 1));
          if (baixo[c][l] && l > 0) seg(X(c), Y(l), X(c + 1), Y(l));
        }
      }
      // contorno com entrada (em cima, 1ª coluna) e saída (embaixo, última coluna)
      seg(X(0), Y(0), X(0), Y(linhas));
      seg(X(colunas), Y(0), X(colunas), Y(linhas));
      seg(X(1), Y(linhas), X(colunas), Y(linhas));
      seg(X(0), Y(0), X(colunas - 1), Y(0));
      const folga = r + h;
      return {
        tracos,
        agua: [{ x0: X(0) + folga, y0: Y(linhas) + p.altura * 0.04, x1: X(1) - folga, y1: p.teto - 2 * h }],
      };
    },
  },

  vasos: {
    nome: 'Vasos comunicantes',
    montar(W, H, h) {
      const p = palco(W, H, h), r = raioDeParede(h);
      const xa = p.X(0.18), xb = p.X(0.82);
      const topo = p.Y(0.82);
      const vao = p.Y(0.1); // altura do canal de baixo que liga os vasos
      const d1 = xa + (xb - xa) * 0.3, d2 = xa + (xb - xa) * 0.52;
      const tracos = [
        ...linha([[xa, topo], [xa, p.chao]], r),
        ...linha([[xb, topo], [xb, p.chao]], r),
        ...linha([[d1, topo], [d1, vao]], r),
        // o do meio é estreito e o da direita é largo e inclinado: a água sobe igual
        ...linha([[d2, topo], [d2 + (xb - d2) * 0.18, vao]], r),
      ];
      return {
        tracos,
        // coluna só no vaso da esquerda; baixa o bastante para a primeira onda não passar por cima das paredes
        agua: [{ x0: xa, y0: p.Y(0.02), x1: d1, y1: p.Y(0.62) }],
      };
    },
  },
};

export const ORDEM_MODELOS = ['copo', 'funil', 'ampulheta', 'cascata', 'labirinto', 'vasos', 'vazio'];

/** O ponto (x, y) está dentro do polígono? */
export function dentroDoPoligono(x, y, pts) {
  let dentro = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) dentro = !dentro;
  }
  return dentro;
}

/**
 * Transforma uma entrada de água de um modelo (retângulo ou polígono) em
 * partículas (mesmo formato de semearRetangulo).
 */
export function semearAgua(entrada, h, absorcao, aleatorio = Math.random) {
  if (!entrada.poligono) return semearRetangulo(entrada.x0, entrada.y0, entrada.x1, entrada.y1, h, absorcao, aleatorio);
  const xs = entrada.poligono.map((q) => q[0]), ys = entrada.poligono.map((q) => q[1]);
  const topo = Math.min(Math.max(...ys), entrada.limiteY ?? Infinity);
  const bruto = semearRetangulo(Math.min(...xs), Math.min(...ys), Math.max(...xs), topo, h, absorcao, aleatorio);
  const n = bruto.length / CAMPOS;
  const saida = new Float32Array(bruto.length);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const o = i * CAMPOS;
    if (!dentroDoPoligono(bruto[o], bruto[o + 1], entrada.poligono)) continue;
    saida.set(bruto.subarray(o, o + CAMPOS), k * CAMPOS);
    k++;
  }
  return saida.slice(0, k * CAMPOS);
}
