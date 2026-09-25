// Solver de pressão MGPCG na CPU — o espelho exato do da GPU (gpu/pcg.js e
// gpu/multigrid.js). Resolve, nas células com água:
//     n·q − Σ q_vizinhos = b        (b = −div)
// com parede fora da soma (Neumann) e ar valendo zero (Dirichlet).
//
// Gradiente conjugado pré-condicionado por um ciclo em V de multigrid
// (McAdams, Sifakis e Teran, 2010):
//   - célula grossa: ar se algum filho é ar; senão água se algum filho é água;
//     senão parede;
//   - restrição = transposta da interpolação bilinear (pesos 1/4 e 3/4);
//   - suavizador em blocos de 16×16: cada bloco lê um retrato fixo da passada
//     anterior (sem corrida entre blocos), faz VARREDURAS vermelho-preto por
//     dentro e grava; duas passadas deslocadas meio bloco antes e depois;
//   - beta de Polak-Ribière.

export const SOLIDO = 0, AGUA = 1, AR = 2;
export const BLOCO = 16;
export const VARREDURAS = 4;
/** Varreduras (ida e volta alternadas) na grade mais grossa, num despacho só (igual à GPU). */
export const VARREDURAS_GROSSO = 64;
const LOCAL = new Float32Array((BLOCO + 2) * (BLOCO + 2));

/** Monta os níveis do multigrid (só os tamanhos; os tipos são refeitos a cada solução). */
export function montarNiveis(nx, ny) {
  const niveis = [{ nx, ny }];
  while (Math.min(niveis.at(-1).nx, niveis.at(-1).ny) > 12) {
    const a = niveis.at(-1);
    niveis.push({ nx: Math.ceil(a.nx / 2), ny: Math.ceil(a.ny / 2) });
  }
  for (const n of niveis) {
    const k = n.nx * n.ny;
    n.tipo = new Uint8Array(k);
    n.b = new Float32Array(k);
    n.x = new Float32Array(k);
    n.tmp = new Float32Array(k);
    n.r = new Float32Array(k);
    n.viz = new Uint8Array(k); // bits dos vizinhos livres: 1 esquerda, 2 direita, 4 baixo, 8 cima
    n.cnt = new Uint8Array(k); // quantos vizinhos livres
  }
  return niveis;
}

/** Máscara de vizinhos livres de cada nível (os tipos não mudam durante uma solução). */
function prepararVizinhos(niveis) {
  for (const n of niveis) {
    const { nx, ny, tipo, viz, cnt } = n;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = i + j * nx;
        let m = 0;
        if (i > 0 && tipo[c - 1] !== SOLIDO) m |= 1;
        if (i < nx - 1 && tipo[c + 1] !== SOLIDO) m |= 2;
        if (j > 0 && tipo[c - nx] !== SOLIDO) m |= 4;
        if (j < ny - 1 && tipo[c + nx] !== SOLIDO) m |= 8;
        viz[c] = m;
        cnt[c] = (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1) + ((m >> 3) & 1);
      }
    }
  }
}

/** Tipo das células grossas a partir das finas. */
export function tiposGrossos(niveis) {
  for (let l = 1; l < niveis.length; l++) {
    const f = niveis[l - 1], g = niveis[l];
    for (let J = 0; J < g.ny; J++) {
      for (let I = 0; I < g.nx; I++) {
        let temAr = false, temAgua = false;
        for (let b = 0; b < 2; b++) {
          for (let a = 0; a < 2; a++) {
            const i = 2 * I + a, j = 2 * J + b;
            if (i >= f.nx || j >= f.ny) continue;
            const t = f.tipo[i + j * f.nx];
            if (t === AR) temAr = true; else if (t === AGUA) temAgua = true;
          }
        }
        g.tipo[I + J * g.nx] = temAr ? AR : temAgua ? AGUA : SOLIDO;
      }
    }
  }
}

/** A·v na célula (i, j) do nível n. */
function aplicar(n, v, i, j) {
  const c = i + j * n.nx;
  const m = n.viz[c];
  let s = 0;
  if (m & 1) s += v[c - 1];
  if (m & 2) s += v[c + 1];
  if (m & 4) s += v[c - n.nx];
  if (m & 8) s += v[c + n.nx];
  return n.cnt[c] * v[c] - s;
}

/**
 * Uma passada do suavizador em blocos: lê x (retrato), grava em tmp e troca.
 * `desl` = 0 ou 8 (meio bloco); `ordem` = [0, 1] (vermelho, preto) ou [1, 0].
 */
function suavizar(n, desl, ordem, omega) {
  const { nx, ny, tipo, b } = n;
  const src = n.x, dst = n.tmp;
  dst.set(src);
  const L = BLOCO + 2;
  const local = LOCAL;
  for (let oy = -desl; oy < ny; oy += BLOCO) {
    for (let ox = -desl; ox < nx; ox += BLOCO) {
      // retrato do bloco com a borda
      for (let ly = 0; ly < L; ly++) {
        for (let lx = 0; lx < L; lx++) {
          const i = ox + lx - 1, j = oy + ly - 1;
          local[lx + ly * L] = i >= 0 && j >= 0 && i < nx && j < ny ? src[i + j * nx] : 0;
        }
      }
      for (let it = 0; it < VARREDURAS; it++) {
        for (let o = 0; o < 2; o++) {
          const cor = ordem[o];
          for (let ly = 1; ly <= BLOCO; ly++) {
            for (let lx = 1; lx <= BLOCO; lx++) {
              const i = ox + lx - 1, j = oy + ly - 1;
              if (i < 0 || j < 0 || i >= nx || j >= ny) continue;
              if (((i + j + 1) & 1) !== cor) continue;
              const c = i + j * nx;
              if (tipo[c] !== AGUA) continue;
              const k = n.cnt[c];
              if (k === 0) continue;
              const m = n.viz[c], e = lx + ly * L;
              let s = 0;
              if (m & 1) s += local[e - 1];
              if (m & 2) s += local[e + 1];
              if (m & 4) s += local[e - L];
              if (m & 8) s += local[e + L];
              const q = local[lx + ly * L];
              local[lx + ly * L] = q + omega * ((s + b[c]) / k - q);
            }
          }
        }
      }
      for (let ly = 1; ly <= BLOCO; ly++) {
        for (let lx = 1; lx <= BLOCO; lx++) {
          const i = ox + lx - 1, j = oy + ly - 1;
          if (i < 0 || j < 0 || i >= nx || j >= ny) continue;
          const c = i + j * nx;
          dst[c] = tipo[c] === AGUA ? local[lx + ly * L] : 0;
        }
      }
    }
  }
  n.x = dst;
  n.tmp = src;
}

// Interpolação bilinear entre níveis: o centro da célula fina i cai em
// (i + 0.5)/2 − 0.5 na grade grossa. Para i par, pesos 1/4 em i/2 − 1 e 3/4 em
// i/2; para i ímpar, 3/4 em (i−1)/2 e 1/4 em (i+1)/2.
function grossoBase(i) { return (i & 1) === 0 ? (i >> 1) - 1 : (i - 1) >> 1; }
function pesoBase(i) { return (i & 1) === 0 ? 0.25 : 0.75; }

/** Resíduo do nível fino, restrito (transposta da bilinear) para o grosso: g.b = Pᵀ r. */
function restringir(f, g) {
  const fnx = f.nx, fny = f.ny, gnx = g.nx, gny = g.ny;
  for (let j = 0; j < fny; j++) {
    for (let i = 0; i < fnx; i++) {
      const c = i + j * fnx;
      f.r[c] = f.tipo[c] === AGUA ? f.b[c] - aplicar(f, f.x, i, j) : 0;
    }
  }
  g.b.fill(0);
  g.x.fill(0);
  for (let j = 0; j < fny; j++) {
    const J0 = grossoBase(j), wy0 = pesoBase(j), wy1 = 1 - wy0;
    for (let i = 0; i < fnx; i++) {
      const r = f.r[i + j * fnx];
      if (r === 0) continue;
      const I0 = grossoBase(i), wx0 = pesoBase(i), wx1 = 1 - wx0;
      for (let b = 0; b < 2; b++) {
        const J = J0 + b;
        if (J < 0 || J >= gny) continue;
        const wy = b ? wy1 : wy0;
        for (let a = 0; a < 2; a++) {
          const I = I0 + a;
          if (I < 0 || I >= gnx) continue;
          const C = I + J * gnx;
          if (g.tipo[C] === AGUA) g.b[C] += r * (a ? wx1 : wx0) * wy;
        }
      }
    }
  }
}

/** Correção do grosso interpolada (bilinear) e somada no fino: f.x += P e. */
function prolongar(g, f) {
  const fnx = f.nx, fny = f.ny, gnx = g.nx, gny = g.ny;
  for (let j = 0; j < fny; j++) {
    const J0 = grossoBase(j), wy0 = pesoBase(j), wy1 = 1 - wy0;
    for (let i = 0; i < fnx; i++) {
      const c = i + j * fnx;
      if (f.tipo[c] !== AGUA) continue;
      const I0 = grossoBase(i), wx0 = pesoBase(i), wx1 = 1 - wx0;
      let e = 0;
      for (let b = 0; b < 2; b++) {
        const J = J0 + b;
        if (J < 0 || J >= gny) continue;
        const wy = b ? wy1 : wy0;
        for (let a = 0; a < 2; a++) {
          const I = I0 + a;
          if (I < 0 || I >= gnx) continue;
          const C = I + J * gnx;
          if (g.tipo[C] === AGUA) e += g.x[C] * (a ? wx1 : wx0) * wy;
        }
      }
      f.x[c] += e;
    }
  }
}

/** A grade mais grossa: varreduras vermelho-preto no lugar (ida e volta alternadas). */
function resolverGrosso(n) {
  const { nx, ny, tipo, b } = n;
  const x = n.x;
  for (let k = 0; k < VARREDURAS_GROSSO; k++) {
    for (let o = 0; o < 2; o++) {
      const cor = k & 1 ? 1 - o : o;
      for (let c = 0; c < nx * ny; c++) {
        const i = c % nx, j = (c / nx) | 0;
        if (tipo[c] !== AGUA || ((i + j + 1) & 1) !== cor) continue;
        const q = n.cnt[c];
        if (!q) continue;
        const m = n.viz[c];
        let s = 0;
        if (m & 1) s += x[c - 1];
        if (m & 2) s += x[c + 1];
        if (m & 4) s += x[c - nx];
        if (m & 8) s += x[c + nx];
        x[c] = (s + b[c]) / q;
      }
    }
  }
}

/** Um ciclo em V: resolve aproximadamente A·x = b no nível 0, partindo de x = 0. */
export function cicloV(niveis, omega = 1) {
  const L = niveis.length;
  niveis[0].x.fill(0);
  for (let l = 0; l < L - 1; l++) {
    suavizar(niveis[l], 0, [0, 1], omega);
    suavizar(niveis[l], 8, [0, 1], omega);
    restringir(niveis[l], niveis[l + 1]);
  }
  resolverGrosso(niveis[L - 1]);
  for (let l = L - 2; l >= 0; l--) {
    prolongar(niveis[l + 1], niveis[l]);
    suavizar(niveis[l], 8, [1, 0], omega);
    suavizar(niveis[l], 0, [1, 0], omega);
  }
}

function produto(a, b, tipo) {
  let s = 0;
  for (let k = 0; k < a.length; k++) if (tipo[k] === AGUA) s += a[k] * b[k];
  return s;
}

/**
 * Bolhas de ar fechadas entram no sistema como "supercélulas": cada bolha é
 * uma incógnita só (a pressão dela, y), ligada a todas as células de água que
 * encostam nela. A linha da bolha diz que a água que sai dela é a pedida:
 *     Σ_faces (y − q_água) = b_bolha
 * Para as linhas da água, basta copiar y para todas as células da bolha antes
 * de aplicar A: a vizinha de água já soma o valor certo.
 */
function montarBolhas(bolhas, tipo, nx, ny) {
  const { rotulo } = bolhas;
  const pares = []; // [rótulo, célula de água] por face
  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const c = i + j * nx;
      const b = rotulo[c];
      if (b < 2 || tipo[c] === AGUA || tipo[c] === SOLIDO) continue;
      if (tipo[c - 1] === AGUA) pares.push(b, c - 1);
      if (tipo[c + 1] === AGUA) pares.push(b, c + 1);
      if (tipo[c - nx] === AGUA) pares.push(b, c - nx);
      if (tipo[c + nx] === AGUA) pares.push(b, c + nx);
    }
  }
  const faces = new Map();
  for (let k = 0; k < pares.length; k += 2) faces.set(pares[k], (faces.get(pares[k]) ?? 0) + 1);
  return { pares, faces, ativos: [...faces.keys()] };
}

/**
 * Resolve n·q − Σ q_viz = −div (nas células AGUA) partindo do q atual.
 * `bolhas` (opcional): { rotulo, rhs, y } — rótulo por célula, lado direito
 * de cada bolha e a pressão dela (entra como chute inicial, sai resolvida).
 * Devolve as normas do resíduo (inicial e depois de cada iteração).
 */
export function resolverPCG(niveis, tipo, div, q, iteracoes, bolhas = null) {
  const n0 = niveis[0];
  const { nx, ny } = n0;
  const k = nx * ny;
  // o operador usa os tipos de verdade; o multigrid (pré-condicionador) vê as
  // bolhas como água: a bolha vira uma região de pressão ligada à água em volta
  const op = niveis.operador ??= { nx, ny, tipo: new Uint8Array(k), viz: new Uint8Array(k), cnt: new Uint8Array(k) };
  op.tipo.set(tipo);
  prepararVizinhos([op]);
  n0.tipo.set(tipo);
  if (bolhas) {
    for (let c = 0; c < k; c++) if (tipo[c] === AR && bolhas.rotulo[c] >= 2) n0.tipo[c] = AGUA;
  }
  tiposGrossos(niveis);
  prepararVizinhos(niveis);
  n0.cg ??= { r: new Float32Array(k), z: new Float32Array(k), zVelho: new Float32Array(k), p: new Float32Array(k), ap: new Float32Array(k) };
  const { r, z, zVelho, p, ap } = n0.cg;
  const B = bolhas ? montarBolhas(bolhas, tipo, nx, ny) : { pares: [], faces: new Map(), ativos: [] };
  const rot = bolhas?.rotulo;
  // vetores das bolhas (por rótulo)
  const yb = new Map(), rb = new Map(), zb = new Map(), zvb = new Map(), pb = new Map(), apb = new Map();
  for (const b of B.ativos) { yb.set(b, bolhas.y[b]); }
  const difundir = (grade, valores) => {
    if (!rot) return;
    for (let c = 0; c < k; c++) {
      const b = rot[c];
      if (b >= 2 && tipo[c] === AR) grade[c] = valores.get(b) ?? 0;
    }
  };
  const aplicarBolhas = (grade, valores, saida) => {
    for (const b of B.ativos) saida.set(b, B.faces.get(b) * valores.get(b));
    for (let m = 0; m < B.pares.length; m += 2) saida.set(B.pares[m], saida.get(B.pares[m]) - grade[B.pares[m + 1]]);
  };
  const produtoB = (a, b) => { let s = 0; for (const x of B.ativos) s += a.get(x) * b.get(x); return s; };

  for (let c = 0; c < k; c++) if (tipo[c] !== AGUA) q[c] = 0;
  difundir(q, yb);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const c = i + j * nx;
      r[c] = tipo[c] === AGUA ? -div[c] - aplicar(op, q, i, j) : 0;
    }
  }
  const aq = new Map();
  aplicarBolhas(q, yb, aq);
  for (const b of B.ativos) rb.set(b, bolhas.rhs[b] - aq.get(b));
  const norma = () => Math.sqrt(produto(r, r, tipo) + produtoB(rb, rb));
  const normas = [norma()];
  // células de cada bolha ativa (para espalhar o resíduo e tirar a média)
  const celulasDe = new Map();
  if (rot) {
    for (let c = 0; c < k; c++) {
      const b = rot[c];
      if (b >= 2 && tipo[c] === AR && B.faces.has(b)) celulasDe.set(b, (celulasDe.get(b) ?? 0) + 1);
    }
  }
  const precondicionar = (destino, destinoB) => {
    n0.b.set(r);
    // o resíduo da bolha, espalhado igual pelas células dela
    if (rot) for (let c = 0; c < k; c++) { const b = rot[c]; if (celulasDe.has(b) && tipo[c] === AR) n0.b[c] = rb.get(b) / celulasDe.get(b); }
    cicloV(niveis);
    destino.set(n0.x);
    // a bolha pega a média da correção nas células dela (transposta de espalhar)
    for (const b of B.ativos) destinoB.set(b, 0);
    if (rot) for (let c = 0; c < k; c++) { const b = rot[c]; if (celulasDe.has(b) && tipo[c] === AR) { destinoB.set(b, destinoB.get(b) + n0.x[c] / celulasDe.get(b)); destino[c] = 0; } }
  };
  precondicionar(z, zb);
  p.set(z);
  for (const b of B.ativos) pb.set(b, zb.get(b));
  let rz = produto(r, z, tipo) + produtoB(rb, zb);
  for (let it = 0; it < iteracoes; it++) {
    difundir(p, pb);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = i + j * nx;
        ap[c] = tipo[c] === AGUA ? aplicar(op, p, i, j) : 0;
      }
    }
    aplicarBolhas(p, pb, apb);
    const pap = produto(p, ap, tipo) + produtoB(pb, apb);
    const alfa = Math.abs(pap) < 1e-30 ? 0 : rz / pap;
    for (let c = 0; c < k; c++) {
      if (tipo[c] !== AGUA) continue;
      q[c] += alfa * p[c];
      r[c] -= alfa * ap[c];
    }
    for (const b of B.ativos) {
      yb.set(b, yb.get(b) + alfa * pb.get(b));
      rb.set(b, rb.get(b) - alfa * apb.get(b));
    }
    normas.push(norma());
    if (it === iteracoes - 1) break;
    zVelho.set(z);
    for (const b of B.ativos) zvb.set(b, zb.get(b));
    precondicionar(z, zb);
    const rzNovo = produto(r, z, tipo) + produtoB(rb, zb);
    const cruzado = produto(r, zVelho, tipo) + produtoB(rb, zvb);
    const beta = Math.abs(rz) < 1e-30 ? 0 : Math.max(0, (rzNovo - cruzado) / rz);
    rz = rzNovo;
    for (let c = 0; c < k; c++) p[c] = tipo[c] === AGUA ? z[c] + beta * p[c] : 0;
    for (const b of B.ativos) pb.set(b, zb.get(b) + beta * pb.get(b));
  }
  // q das células de ar volta a zero (quem soma a pressão do ar é quem chamou)
  for (let c = 0; c < k; c++) if (tipo[c] !== AGUA) q[c] = 0;
  if (bolhas) for (const b of B.ativos) bolhas.y[b] = yb.get(b);
  return normas;
}
