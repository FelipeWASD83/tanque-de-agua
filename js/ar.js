// Bolsões de ar: quais regiões de ar estão fechadas pela água (bolhas, ar
// preso num copo virado, a metade de baixo de uma ampulheta) e quanto ar tem
// cada uma. Sem DOM: roda na CPU para os dois motores e nos testes.
//
// A simulação só tem água; o ar é o espaço vazio. Sem isto, um bolsão fechado
// some (vira vácuo). Aqui, o ar fechado guarda o volume que tinha quando se
// fechou: a água só entra se o ar sair (bolha sobe e escapa, a ampulheta
// gorgoleja). A pressão que segura o volume é ajustada a cada subpasso pelo
// motor; este arquivo só diz quem é quem.
//
// Rótulos por célula: 0 = sem rótulo (vácuo, ou ar ainda não visto),
// 1 = atmosfera (ar que toca o topo do tanque), 2… = bolhas. O número de uma
// bolha é estável de um quadro para o outro (a pressão dela continua valendo).

import { MOLDURA, MAX_BOLHAS } from './constantes.js';

export const AR = 2;
export const ATMOSFERA = 1;
/** Bolha menor que isto (em células) não é acompanhada: ela "se dissolve". */
export const MIN_BOLHA_CELULAS = 4;

export class RastreadorDeAr {
  constructor(nx, ny) {
    this.nx = nx;
    this.ny = ny;
    const n = nx * ny;
    this.rotulo = new Uint32Array(n);
    this.pai = new Int32Array(n);
    this.componente = new Int32Array(n);
    /** Volume de ar de cada bolha, em células (índice = rótulo). */
    this.volumeAlvo = new Float32Array(MAX_BOLHAS);
    this.vivo = new Uint8Array(MAX_BOLHAS);
    this.primeira = true;
  }

  /** Esquece tudo (cenário novo): na próxima leitura, todo ar fechado vira bolha com o volume que tiver. */
  reiniciar() {
    this.rotulo.fill(0);
    this.volumeAlvo.fill(0);
    this.vivo.fill(0);
    this.primeira = true;
  }

  #raiz(a) {
    const p = this.pai;
    while (p[a] !== a) { p[a] = p[p[a]]; a = p[a]; }
    return a;
  }

  /**
   * Lê os tipos de célula atuais (0 parede, 1 água, 2 ar) e refaz os rótulos.
   * `pressaoAnterior(rotulo)` dá a pressão de uma bolha antiga (para uma bolha
   * nova que nasceu da divisão dela). Devolve os rótulos novos e a lista de
   * bolhas que nasceram (rótulo → pressão inicial).
   */
  atualizar(tipo, pressaoAnterior = () => 0) {
    const { nx, ny } = this;
    const n = nx * ny;
    const pai = this.pai, comp = this.componente;
    // componentes conexas do ar (vizinhança de 4), por união-busca
    for (let c = 0; c < n; c++) pai[c] = tipo[c] === AR ? c : -1;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = i + j * nx;
        if (pai[c] < 0) continue;
        if (i > 0 && pai[c - 1] >= 0) this.#unir(c, c - 1);
        if (j > 0 && pai[c - nx] >= 0) this.#unir(c, c - nx);
      }
    }
    // numera as componentes e junta o que cada uma tem de ar antigo
    const topo = ny - MOLDURA - 1;
    const numero = this.numero ??= new Int32Array(n).fill(-1); // raiz → índice da componente
    const raizes = [];
    const info = []; // { celulas, topo, atm, gas: Map(rotuloAntigo → células) }
    for (let c = 0; c < n; c++) {
      if (pai[c] < 0) { comp[c] = -1; continue; }
      const r = this.#raiz(c);
      let k = numero[r];
      if (k < 0) {
        k = info.length;
        numero[r] = k;
        raizes.push(r);
        info.push({ celulas: 0, topo: false, atm: 0, gas: null });
      }
      comp[c] = k;
      const d = info[k];
      d.celulas++;
      if (((c / nx) | 0) >= topo) d.topo = true;
      const antigo = this.rotulo[c];
      if (antigo === ATMOSFERA) d.atm++;
      else if (antigo >= 2) {
        d.gas ??= new Map();
        d.gas.set(antigo, (d.gas.get(antigo) ?? 0) + 1);
      }
    }
    for (const r of raizes) numero[r] = -1;
    // Quanto de cada bolha antiga caiu em cada componente nova. Só contam as
    // que ficam (bolhas de tamanho mínimo e a atmosfera): o ar de um
    // fragmento pequeno demais volta para os pedaços que sobram; o que caiu na
    // atmosfera escapou.
    const fica = (d) => d.topo || d.celulas >= MIN_BOLHA_CELULAS;
    const sobreposicaoTotal = new Map();
    for (const d of info) {
      if (!d.gas || !fica(d)) continue;
      for (const [r, q] of d.gas) sobreposicaoTotal.set(r, (sobreposicaoTotal.get(r) ?? 0) + q);
    }
    // volume de ar de cada componente fechada
    const novoAlvo = new Float32Array(info.length);
    const dono = new Int32Array(info.length).fill(-1); // bolha antiga com maior parte nela
    for (let k = 0; k < info.length; k++) {
      const d = info[k];
      if (!fica(d) || d.topo) continue;
      let v = 0, melhor = 0;
      if (this.primeira) v = d.celulas;
      else {
        v = d.atm; // ar da atmosfera que ficou preso agora
        if (d.gas) {
          for (const [r, q] of d.gas) {
            v += this.volumeAlvo[r] * (q / sobreposicaoTotal.get(r));
            if (q > melhor) { melhor = q; dono[k] = r; }
          }
        }
      }
      novoAlvo[k] = v;
    }
    // rótulos estáveis: cada bolha antiga fica com a componente onde está a maior parte dela
    const rotuloDaComp = new Uint32Array(info.length);
    const vivoNovo = new Uint8Array(MAX_BOLHAS);
    const ordem = [...info.keys()].filter((k) => !info[k].topo && novoAlvo[k] >= MIN_BOLHA_CELULAS)
      .sort((a, b) => novoAlvo[b] - novoAlvo[a]);
    for (const k of ordem) {
      const r = dono[k];
      if (r >= 2 && !vivoNovo[r]) { rotuloDaComp[k] = r; vivoNovo[r] = 1; }
    }
    const nascidas = [];
    let livre = 2;
    for (const k of ordem) {
      if (rotuloDaComp[k]) continue;
      while (livre < MAX_BOLHAS && (vivoNovo[livre] || this.vivo[livre])) livre++;
      if (livre >= MAX_BOLHAS) break; // bolhas demais: o resto vira vácuo
      rotuloDaComp[k] = livre;
      vivoNovo[livre] = 1;
      nascidas.push([livre, dono[k] >= 2 ? pressaoAnterior(dono[k]) : 0]);
    }
    for (let k = 0; k < info.length; k++) if (info[k].topo) rotuloDaComp[k] = ATMOSFERA;
    // grava
    this.volumeAlvo.fill(0);
    for (let k = 0; k < info.length; k++) if (rotuloDaComp[k] >= 2) this.volumeAlvo[rotuloDaComp[k]] = novoAlvo[k];
    for (let c = 0; c < n; c++) this.rotulo[c] = comp[c] >= 0 ? rotuloDaComp[comp[c]] : 0;
    this.vivo = vivoNovo;
    this.primeira = false;
    return { rotulo: this.rotulo, volumeAlvo: this.volumeAlvo, nascidas };
  }

  #unir(a, b) {
    const ra = this.#raiz(a), rb = this.#raiz(b);
    if (ra !== rb) this.pai[Math.max(ra, rb)] = Math.min(ra, rb);
  }
}
