import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cena } from '../js/cena.js';
import { SimulacaoCPU, semearRetangulo } from '../js/fisica-cpu.js';
import { MODELOS, semearAgua } from '../js/modelos.js';
import { MOLDURA, PINCEL_MIN_CELULAS, PARTICULAS_POR_CELULA, passoMaximo } from '../js/constantes.js';
import { semente, particulas, velocidadeMaxima, menorDistanciaAParede } from './apoio.js';

const QUADRO = 1 / 60;

function rodar(sim, segundos, aCada) {
  const sub = Math.ceil(QUADRO / passoMaximo(sim.h, sim.cena.altura));
  const quadros = Math.round(segundos / QUADRO);
  for (let f = 0; f < quadros; f++) {
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

test('água parada fica parada', () => {
  const h = 0.01;
  const cena = new Cena(60, 40, h);
  const sim = new SimulacaoCPU(cena);
  const m = MOLDURA * h;
  sim.adicionar(semearRetangulo(m, m, 0.6 - m, 0.2, h, [0, 0, 0], semente(3)), false);
  rodar(sim, 3);
  assert.ok(velocidadeMaxima(sim) < 0.01, `velocidade máxima ${velocidadeMaxima(sim)} m/s`);
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
