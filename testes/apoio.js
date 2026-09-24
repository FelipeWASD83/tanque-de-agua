// Utilidades dos testes: gerador aleatório com semente e medições da água.
import { CAMPOS } from '../js/fisica-cpu.js';

export function semente(s = 1) {
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function* particulas(sim) {
  for (let k = 0; k < sim.n; k++) {
    const o = k * CAMPOS;
    yield { x: sim.p[o], y: sim.p[o + 1], vx: sim.p[o + 2], vy: sim.p[o + 3] };
  }
}

export function velocidadeMaxima(sim) {
  let m = 0;
  for (const q of particulas(sim)) m = Math.max(m, Math.hypot(q.vx, q.vy));
  return m;
}

export function menorDistanciaAParede(sim) {
  let m = Infinity;
  for (const q of particulas(sim)) m = Math.min(m, sim.cena.distancia(q.x, q.y));
  return m;
}
