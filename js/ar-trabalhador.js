// Web Worker do motor da GPU: roda o rastreador de bolsões de ar (ar.js) fora
// da tela, com os tipos de célula lidos da placa. Devolve os rótulos por
// célula, o volume de ar de cada bolha e as bolhas que nasceram.

import { RastreadorDeAr } from './ar.js';

let rastreador = null;
let geracao = 0;

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.tipo === 'reiniciar') {
    geracao = m.geracao;
    rastreador?.reiniciar();
    return;
  }
  // leitura de antes do reinício: não pode gastar o primeiro retrato da cena nova
  if (m.geracao !== geracao) {
    self.postMessage({ geracao: m.geracao, descartado: true });
    return;
  }
  if (!rastreador || rastreador.nx !== m.nx || rastreador.ny !== m.ny) rastreador = new RastreadorDeAr(m.nx, m.ny);
  const tipos = new Uint8Array(m.tipos, 0, m.nx * m.ny);
  const pressoes = new Float32Array(m.pressoes);
  const r = rastreador.atualizar(tipos, (b) => pressoes[b]);
  const rotulo = r.rotulo.slice();
  const alvo = r.volumeAlvo.slice();
  self.postMessage({ geracao: m.geracao, rotulo, alvo, nascidas: r.nascidas }, [rotulo.buffer, alvo.buffer]);
};
