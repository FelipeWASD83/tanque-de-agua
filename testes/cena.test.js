import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cena } from '../js/cena.js';
import { SDF_MAX, MOLDURA } from '../js/constantes.js';
import { semente } from './apoio.js';

const h = 0.01;

test('a moldura é parede e o meio do tanque é livre', () => {
  const c = new Cena(40, 30, h);
  assert.ok(c.celulaSolida(0, 0));
  assert.ok(c.celulaSolida(MOLDURA - 1, 15));
  assert.ok(!c.celulaSolida(MOLDURA, 15));
  assert.ok(!c.celulaSolida(20, 15));
  assert.equal(c.distancia(0.2, 0.15), SDF_MAX);
  // a borda interna da moldura fica em MOLDURA células
  assert.ok(Math.abs(c.distancia(MOLDURA * h + 3 * h, 0.15) - 3) < 0.05);
});

test('a distância até um traço bate com a geometria', () => {
  const c = new Cena(60, 60, h);
  c.tracar(0.3, 0.3, 0.3, 0.3, 0.02);
  c.atualizar();
  for (const d of [0.03, 0.045, 0.06, 0.08]) {
    const esperado = (d - 0.02) / h;
    assert.ok(Math.abs(c.distancia(0.3 + d, 0.3) - esperado) < 0.35, `d=${d}: ${c.distancia(0.3 + d, 0.3)} vs ${esperado}`);
  }
  assert.ok(c.distancia(0.3, 0.3) < -1.5);
  assert.ok(c.celulaSolida(30, 30));
  const [nx, ny] = c.normal(0.35, 0.3);
  assert.ok(nx > 0.99 && Math.abs(ny) < 0.05);
});

test('apagar devolve o espaço, mas nunca a moldura', () => {
  const c = new Cena(40, 30, h);
  c.tracar(0.1, 0.1, 0.3, 0.1, 0.015);
  c.atualizar();
  assert.ok(c.celulaSolida(20, 10));
  c.tracar(0, 0.1, 0.4, 0.1, 0.03, true);
  c.tracar(0, 0, 0.4, 0.3, 0.05, true);
  c.atualizar();
  assert.ok(!c.celulaSolida(20, 10));
  assert.ok(c.celulaSolida(0, 10) && c.celulaSolida(39, 10) && c.celulaSolida(20, 0));
});

test('o campo de distância incremental é igual ao recalculado do zero', () => {
  const aleatorio = semente(7);
  const a = new Cena(70, 50, h);
  const tracos = [];
  for (let k = 0; k < 25; k++) {
    const t = {
      x0: aleatorio() * 0.7, y0: aleatorio() * 0.5, x1: aleatorio() * 0.7, y1: aleatorio() * 0.5,
      raio: 0.013 + aleatorio() * 0.03, apagar: aleatorio() < 0.25,
    };
    tracos.push(t);
    a.tracar(t.x0, t.y0, t.x1, t.y1, t.raio, t.apagar);
    a.atualizar();
  }
  const b = new Cena(70, 50, h);
  b.refazer(a.tracos);
  b.atualizar();
  let pior = 0;
  for (let k = 0; k < a.sdf.length; k++) pior = Math.max(pior, Math.abs(a.sdf[k] - b.sdf[k]));
  assert.ok(pior < 1e-5, `diferença ${pior}`);
  assert.deepEqual(a.solido, b.solido);
});
