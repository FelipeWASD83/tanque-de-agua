import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cena } from '../js/cena.js';
import { SimulacaoCPU, semearRetangulo } from '../js/fisica-cpu.js';
import { MODELOS, semearAgua } from '../js/modelos.js';
import { MOLDURA, PINCEL_MIN_CELULAS, PARTICULAS_POR_CELULA, passoMaximo } from '../js/constantes.js';
import { semente, particulas, velocidadeMaxima, menorDistanciaAParede } from './apoio.js';

const QUADRO = 1 / 60;

// Mesmo passo adaptativo do app: segue a maior velocidade do quadro anterior.
function rodar(sim, segundos, aCada) {
  const quadros = Math.round(segundos / QUADRO);
  for (let f = 0; f < quadros; f++) {
    const sub = Math.ceil(QUADRO / passoMaximo(sim.h, sim.cena.altura, sim.vMax ?? null, QUADRO));
    sim.avancar(QUADRO, sub);
    if (aCada) aCada(f * QUADRO);
  }
}

function montarModelo(nome, nx, ny, h, aleatorio) {
  const cena = new Cena(nx, ny, h);
  const m = MODELOS[nome].montar(cena.largura, cena.altura, h);
  for (const t of m.tracos) cena.tracar(t.x0, t.y0, t.x1, t.y1, t.raio);
  cena.atualizar();
  const sim = new SimulacaoCPU(cena);
  for (const a of m.agua) sim.adicionar(semearAgua(a, h, [0, 0, 0], aleatorio), false);
  return { cena, sim, modelo: m };
}

for (const [nome, tensao, limite] of [['sem tensão superficial', 0, 0.01], ['com tensão superficial', undefined, 0.02]]) {
  // Com tensão, a curvatura medida nas partículas faz "correntes parasitas" de
  // ~1 cm/s nesta grade de 1 cm (conhecidas em qualquer método desse tipo).
  test(`água parada fica parada (${nome})`, () => {
    const h = 0.01;
    const cena = new Cena(60, 40, h);
    const sim = new SimulacaoCPU(cena, { tensao });
    const m = MOLDURA * h;
    sim.adicionar(semearRetangulo(m, m, 0.6 - m, 0.2, h, [0, 0, 0], semente(3)), false);
    rodar(sim, 3);
    assert.ok(velocidadeMaxima(sim) < limite, `velocidade máxima ${velocidadeMaxima(sim)} m/s`);
  });
}

test('sem gravidade, a tensão superficial arredonda uma gota quadrada', () => {
  const h = 0.002, n = 70;
  const lado = 32 * h;
  const centro = (n * h) / 2;
  const medir = (tensao) => {
    const cena = new Cena(n, n, h);
    const sim = new SimulacaoCPU(cena, { gravidade: [0, 0], tensao });
    sim.adicionar(semearRetangulo(centro - lado / 2, centro - lado / 2, centro + lado / 2, centro + lado / 2, h, [0, 0, 0], semente(8)), false);
    rodar(sim, 1.2);
    // num quadrado, o ponto mais longe do centro na diagonal (|dx|+|dy|) vale o
    // lado; num círculo de mesma área, 0,80 do lado
    let cx = 0, cy = 0;
    for (const q of particulas(sim)) { cx += q.x; cy += q.y; }
    cx /= sim.n; cy /= sim.n;
    let diagonal = 0;
    for (const q of particulas(sim)) diagonal = Math.max(diagonal, Math.abs(q.x - cx) + Math.abs(q.y - cy));
    return diagonal / lado;
  };
  const com = medir(undefined), sem = medir(0);
  assert.ok(sem > 0.93, `sem tensão a gota não devia mudar de forma (${sem.toFixed(2)})`);
  assert.ok(com < 0.88, `com tensão a gota devia ficar redonda (${com.toFixed(2)}; círculo = 0,80)`);
});

test('a água despejada no copo não atravessa parede nem some', () => {
  const { sim } = montarModelo('copo', 90, 60, 0.01, semente(11));
  const n0 = sim.n;
  assert.ok(n0 > 1000);
  let pior = Infinity;
  rodar(sim, 4, () => { pior = Math.min(pior, menorDistanciaAParede(sim)); });
  assert.equal(sim.n, n0, 'partículas sumiram');
  assert.ok(pior > -0.05, `partícula entrou na parede (distância ${pior} células)`);
});

test('uma parede do pincel mais fino segura um jato forte', () => {
  const h = 0.01;
  const cena = new Cena(80, 60, h);
  // parede diagonal de lado a lado: divide o tanque em dois
  cena.tracar(0.2, 0, 0.62, 0.6, PINCEL_MIN_CELULAS * h);
  cena.atualizar();
  const sim = new SimulacaoCPU(cena);
  const lado = (x, y) => (x - 0.2) * 0.6 - y * 0.42; // > 0: à direita da parede
  const jato = semearRetangulo(0.03, 0.35, 0.12, 0.55, h, [0, 0, 0], semente(5));
  for (let k = 0; k < jato.length; k += 12) { jato[k + 2] = 3.5; jato[k + 3] = -1; }
  sim.adicionar(jato, false);
  let vazou = 0;
  rodar(sim, 3, () => {
    for (const q of particulas(sim)) if (lado(q.x, q.y) > 0) vazou++;
  });
  assert.equal(vazou, 0, `${vazou} partículas passaram para o outro lado`);
});

test('o volume se conserva depois de um rompimento de barragem', () => {
  const h = 0.01, nx = 80, ny = 60;
  const cena = new Cena(nx, ny, h);
  const sim = new SimulacaoCPU(cena);
  const m = MOLDURA * h;
  sim.adicionar(semearRetangulo(m, m, 0.3, 0.42, h, [0, 0, 0], semente(9)), false);
  const area = sim.area();
  rodar(sim, 10);
  // nível médio da superfície (a partícula mais alta de cada coluna, longe das paredes)
  const topo = new Float32Array(nx);
  for (const q of particulas(sim)) {
    const i = Math.floor(q.x / h);
    topo[i] = Math.max(topo[i], q.y);
  }
  let soma = 0, n = 0;
  for (let i = MOLDURA + 3; i < nx - MOLDURA - 3; i++) { soma += topo[i]; n++; }
  const nivel = soma / n - m;
  const esperado = area / (nx * h - 2 * m);
  // meia célula de folga: a partícula mais alta fica acima da superfície média
  assert.ok(Math.abs(nivel - esperado) < 0.06 * esperado + 0.5 * h, `nível ${nivel.toFixed(4)} m, esperado ${esperado.toFixed(4)} m`);
});

test('vasos comunicantes: a água sobe até a mesma altura nos três', () => {
  const h = 0.01;
  const { sim, cena } = montarModelo('vasos', 100, 64, h, semente(21));
  const m = MODELOS.vasos.montar(cena.largura, cena.altura, h);
  // os três vasos, pelas x das paredes verticais
  const xs = [m.tracos[0].x0, m.tracos[2].x0, (m.tracos[3].x0 + m.tracos[3].x1) / 2, m.tracos[1].x0];
  const niveis = [[], [], []];
  rodar(sim, 16, (t) => {
    if (t < 12) return;
    const topo = [0, 0, 0];
    for (const q of particulas(sim)) {
      for (let k = 0; k < 3; k++) {
        if (q.x > xs[k] + 2.5 * h && q.x < xs[k + 1] - 2.5 * h) topo[k] = Math.max(topo[k], q.y);
      }
    }
    topo.forEach((y, k) => niveis[k].push(y));
  });
  const media = niveis.map((l) => l.reduce((a, b) => a + b, 0) / l.length);
  const dif = Math.max(...media) - Math.min(...media);
  assert.ok(dif < 2 * h, `níveis ${media.map((y) => y.toFixed(3)).join(' / ')} m (diferença ${(dif / h).toFixed(1)} células)`);
});

test('a densidade das partículas continua perto da de repouso', () => {
  const { sim } = montarModelo('copo', 90, 60, 0.01, semente(4));
  rodar(sim, 5);
  // no miolo da água, cada célula tem que ter perto de 4 partículas
  let soma = 0, n = 0;
  const { nx, ny } = sim;
  for (let j = 2; j < ny - 2; j++) {
    for (let i = 2; i < nx - 2; i++) {
      let cheio = true;
      for (let b = -1; b <= 1 && cheio; b++) for (let a = -1; a <= 1; a++) {
        const c = i + a + (j + b) * nx;
        if (sim.contagemCelula[c] === 0 || sim.solidoCelula[c]) { cheio = false; break; }
      }
      if (cheio) { soma += sim.contagemCelula[i + j * nx]; n++; }
    }
  }
  assert.ok(n > 50, 'quase não sobrou água no miolo');
  const media = soma / n;
  assert.ok(Math.abs(media - PARTICULAS_POR_CELULA) < 0.4, `média de ${media.toFixed(2)} partículas por célula`);
});

// ------------------------------------------------------------------- ar

function tanqueComBolha(ar) {
  const h = 0.01;
  const cena = new Cena(60, 60, h);
  const sim = new SimulacaoCPU(cena, { ar });
  const m = MOLDURA * h;
  const agua = semearRetangulo(m, m, 0.6 - m, 0.45, h, [0, 0, 0], semente(5));
  const fora = [];
  for (let k = 0; k < agua.length / 12; k++) {
    const x = agua[k * 12], y = agua[k * 12 + 1];
    if ((x - 0.3) ** 2 + (y - 0.12) ** 2 < 0.05 ** 2) continue; // bolha de 5 cm de raio
    fora.push(...agua.subarray(k * 12, (k + 1) * 12));
  }
  sim.adicionar(new Float32Array(fora), false);
  return sim;
}

function medirBolhas(sim) {
  let n = 0, sy = 0;
  for (let j = 0; j < sim.ny; j++) {
    for (let i = 0; i < sim.nx; i++) {
      const c = i + j * sim.nx;
      if (sim.tipo[c] === 2 && sim.rotulo[c] >= 2) { n++; sy += (j + 0.5) * sim.h; }
    }
  }
  return { celulas: n, altura: n ? sy / n : null };
}

test('uma bolha de ar no fundo não some: sobe mantendo o volume', () => {
  const sim = tanqueComBolha(true);
  rodar(sim, 0.1);
  const inicio = medirBolhas(sim);
  rodar(sim, 0.5);
  const depois = medirBolhas(sim);
  const v0 = Math.PI * 25; // células (raio de 5 células)
  assert.ok(inicio.celulas > 0.8 * v0, `bolha encolheu no começo: ${inicio.celulas} de ${v0.toFixed(0)} células`);
  assert.ok(depois.celulas > 0.75 * v0, `bolha perdeu ar subindo: ${depois.celulas} de ${v0.toFixed(0)} células`);
  assert.ok(depois.altura - inicio.altura > 0.1, `bolha subiu só ${((depois.altura - inicio.altura) * 100).toFixed(1)} cm em 0,5 s`);
});

test('montar um cenário por cima de água que já estava lá não perde a bolha', () => {
  // o mesmo que o botão Montar faz: tira a água, zera o ar e põe a nova
  const sim = tanqueComBolha(true);
  const m = MOLDURA * sim.h;
  const cheio = semearRetangulo(m, m, 0.6 - m, 0.45, sim.h, [0, 0, 0], semente(6));
  sim.n = 0;
  sim.contagemCelula.fill(0);
  sim.reiniciarAr();
  sim.adicionar(cheio, false);
  rodar(sim, 0.1); // o tanque todo com água, sem bolha
  const comBolha = tanqueComBolha(true);
  const nova = new Float32Array(comBolha.p.subarray(0, comBolha.n * 12));
  sim.n = 0;
  sim.contagemCelula.fill(0);
  sim.reiniciarAr();
  sim.adicionar(nova, false);
  rodar(sim, 0.1);
  const v0 = Math.PI * 25;
  const { celulas } = medirBolhas(sim);
  assert.ok(celulas > 0.8 * v0, `a bolha do cenário novo virou vácuo: ${celulas} de ${v0.toFixed(0)} células`);
});

test('sem o ar, o mesmo buraco vira vácuo e fecha', () => {
  const sim = tanqueComBolha(false);
  rodar(sim, 0.3);
  let vazias = 0;
  for (let j = 3; j < 25; j++) for (let i = 20; i < 40; i++) if (sim.tipo[i + j * sim.nx] === 2) vazias++;
  assert.ok(vazias < 10, `${vazias} células vazias onde era o buraco`);
});

test('a ampulheta com ar esvazia bem mais devagar do que sem ar', () => {
  const emCima = (ar) => {
    const h = 0.01;
    const cena = new Cena(60, 90, h);
    const m = MODELOS.ampulheta.montar(cena.largura, cena.altura, h);
    for (const t of m.tracos) cena.tracar(t.x0, t.y0, t.x1, t.y1, t.raio);
    cena.atualizar();
    const sim = new SimulacaoCPU(cena, { ar });
    for (const a of m.agua) sim.adicionar(semearAgua(a, h, [0, 0, 0], semente(2)), false);
    const n0 = sim.n;
    rodar(sim, 1);
    let cima = 0;
    for (const q of particulas(sim)) if (q.y > cena.altura / 2) cima++;
    return cima / n0;
  };
  const com = emCima(true), sem = emCima(false);
  assert.ok(sem < 0.1, `sem ar, depois de 1 s ainda sobrava ${(sem * 100).toFixed(0)}% em cima`);
  assert.ok(com > 0.4, `com ar, depois de 1 s só sobrava ${(com * 100).toFixed(0)}% em cima (o ar de baixo tinha que segurar)`);
});
