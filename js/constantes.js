// Números da física e da escala. Os dois motores (GPU e CPU) leem daqui, então
// a água se comporta igual nos dois.

/** Escala do tanque: 1440 px de tela (CSS) = 1 metro. */
export const METROS_POR_PX = 1 / 1440;

/** Aceleração da gravidade, em m/s². */
export const GRAVIDADE = 9.81;

/** Densidade da água, em kg/m³. */
export const DENSIDADE_AGUA = 1000;

/**
 * Tensão superficial da água com o ar, em N/m (0,0728 a 20 °C). Entra como um
 * salto de pressão σ·κ na superfície (κ = curvatura). Num tanque de 1 m ela só
 * aparece em gota e filete de poucos milímetros — como na água de verdade.
 */
export const TENSAO_SUPERFICIAL = 0.0728;
/** Quantos bolsões de ar fechados a simulação acompanha ao mesmo tempo. */
export const MAX_BOLHAS = 4096;

/**
 * Bolhas: cada bolha fechada é uma incógnita a mais no solver de pressão (o ar
 * não entra nem sai, então a água só ocupa o lugar dele se ele escapar).
 * VOLTA_VOLUME_BOLHA é quanto do volume perdido por erro numérico a bolha
 * recupera por subpasso; PRESSAO_BOLHA_MAX (Pa) é um teto de segurança
 * (2 m de coluna d'água).
 */
export const VOLTA_VOLUME_BOLHA = 0.05;
export const PRESSAO_BOLHA_MAX = 20000;

/**
 * Fração do passo capilar clássico que fica estável (medido: 0,5 estável em
 * grade de 2,3 mm na GPU; na de 4 mm a instabilidade começa perto de 0,45).
 */
export const FATOR_CAPILAR = 0.4;
/** Teto absoluto do subpasso (s): nas grades grossas o limite capilar sozinho deixa o passo grande demais. */
export const PASSO_MAX = 0.008;

/** Profundidade do tanque (entrando na tela), em metros. Só entra na conta dos litros. */
export const PROFUNDIDADE = 0.05;

/** Partículas por célula em água parada (semeadas em 2×2). */
export const PARTICULAS_POR_CELULA = 4;

/** Pixels da máscara de paredes por célula da grade, em cada eixo. */
export const RES_MASCARA = 2;

/** Até onde o campo de distância das paredes é calculado, em células. */
export const SDF_MAX = 10;

/** Espessura da moldura do tanque, em células. */
export const MOLDURA = 2;

/** Raio de colisão da partícula com a parede, em células. */
export const RAIO_PARTICULA = 0.3;

/**
 * Número de Courant: quantas células a água pode andar por subpasso. Acima de
 * 1, a partícula anda em micro-passos (no máximo MICROPASSO células cada)
 * pelo campo de velocidade já projetado, batendo na parede a cada um — segue
 * a correnteza sem atravessar parede fina, e a pressão (a parte cara) roda
 * menos vezes.
 */
export const CFL = 2.0;
export const MICROPASSO = 0.8;
/** Teto de micro-passos por subpasso (e, com ele, da velocidade: MICROPASSO·MAX_MICROPASSOS células por subpasso). */
export const MAX_MICROPASSOS = 4;

/**
 * Correção de densidade: partícula que se amontoa (ou se espalha no miolo da
 * água) é devolvida à densidade de repouso com esta constante de tempo, em
 * segundos. É tempo, não fração por passo: assim a força da correção não muda
 * com o número de subpassos (por passo, ela bombeava energia e a água não
 * parava quieta).
 */
export const TAU_DENSIDADE = 0.1;
/** Limites do excesso de densidade que a correção aceita (fração da densidade de repouso). */
export const EXCESSO_MIN = -0.5;
export const EXCESSO_MAX = 1.0;

/**
 * Na GPU a pressão é resolvida por gradiente conjugado pré-condicionado com
 * multigrid (gpu/pcg.js): iterações por subpasso, partindo da pressão do
 * subpasso anterior.
 */
export const ITERACOES_PCG = 3;


/** Camadas de extrapolação da velocidade para o ar. */
export const CAMADAS_EXTRAPOLACAO = 2;

/**
 * Espuma: água rápida e rarefeita (jato, respingo, crista quebrando) pega ar
 * e fica branca. A partícula ganha espuma quando anda entre ESPUMA_V0 e
 * ESPUMA_V1 m/s numa célula com menos da metade da densidade da água, a
 * ESPUMA_TAXA por segundo, e a perde devagar (ESPUMA_VIDA). Água parada ou
 * lenta nunca espuma, e nada disso depende do tamanho do passo de tempo.
 */
export const ESPUMA_V0 = 1.1;
export const ESPUMA_V1 = 2.8;
export const ESPUMA_TAXA = 5;
/** Espuma: quanto tempo (s) a água aerada leva para clarear (constante de decaimento). */
export const ESPUMA_VIDA = 1.2;

/** Qualidades: quantas células a grade tenta ter, e o menor tamanho de célula (m). */
export const QUALIDADES = {
  ultra:  { nome: 'Ultra',  celulas: 450_000, hMin: 0.0016 },
  maximo: { nome: 'Máximo', celulas: 220_000, hMin: 0.0022 },
  alto:   { nome: 'Alto',   celulas: 130_000, hMin: 0.0026 },
  medio:  { nome: 'Médio',  celulas: 70_000,  hMin: 0.0032 },
  leve:   { nome: 'Leve',   celulas: 32_000,  hMin: 0.0040 },
  cpu:    { nome: 'CPU',    celulas: 9_000,   hMin: 0.0060 },
};

/**
 * Desenho: cada partícula vira uma mancha deste raio (em células). Maior
 * alisa os buraquinhos entre partículas; menor mostra respingo mais fino.
 */
export const RAIO_MANCHA = 2.1;

/**
 * Corante: a tinta se espalha devagar na água. A cor de cada partícula se
 * aproxima da média da célula com esta constante de tempo, em segundos.
 */
export const DIFUSAO_CORANTE = 3;

/**
 * Teto de partículas por célula de grade (reserva de memória na GPU). Com 4 por
 * célula de água, 4,6 deixa encher o tanque inteiro com folga para respingo.
 */
export const PARTICULAS_MAX_POR_CELULA = 4.6;

/**
 * Corantes: absorção (por metro de água atravessada) em R, G, B.
 * A água limpa absorve um pouco de vermelho, e é isso que a deixa ciano.
 */
export const ABSORCAO_AGUA = [9.0, 2.6, 1.6];
export const CORANTES = {
  limpa:    { nome: 'Limpa',    absorcao: [0, 0, 0],        amostra: '#d8eef0' },
  azul:     { nome: 'Azul',     absorcao: [60, 26, 3],      amostra: '#2f6fd0' },
  vermelho: { nome: 'Vermelho', absorcao: [2, 55, 42],      amostra: '#c8322c' },
  amarelo:  { nome: 'Amarelo',  absorcao: [1, 7, 60],       amostra: '#e6b41e' },
  verde:    { nome: 'Verde',    absorcao: [55, 5, 40],      amostra: '#2f9a52' },
  tinta:    { nome: 'Nanquim',  absorcao: [70, 70, 64],     amostra: '#1d2024' },
};

/**
 * Monta as dimensões da grade para um tanque de W × H metros.
 * Devolve nx, ny e o tamanho de célula h (o tanque fica com nx·h × ny·h).
 */
export function dimensoesDaGrade(larguraM, alturaM, qualidade) {
  const q = QUALIDADES[qualidade] ?? QUALIDADES.alto;
  let h = Math.sqrt((larguraM * alturaM) / q.celulas);
  h = Math.max(h, q.hMin);
  const ny = Math.max(24, Math.round(alturaM / h));
  h = alturaM / ny;
  const nx = Math.max(24, Math.round(larguraM / h));
  return { nx, ny, h };
}

/**
 * Maior passo de tempo seguro: nenhuma partícula anda mais que CFL células.
 * `vMax` é a maior velocidade medida no quadro anterior; sem medida, usa a
 * de uma queda da altura do tanque.
 */
export function passoMaximo(h, alturaM, vMax = null, dtQuadro = 1 / 60) {
  const queda = Math.sqrt(2 * GRAVIDADE * alturaM) * 1.15 + 0.5;
  const v = vMax == null ? queda : Math.min(queda, vMax * 1.2 + GRAVIDADE * dtQuadro + 0.25);
  // Onda capilar: a tensão superficial explícita só fica estável abaixo deste
  // passo. A conta clássica (√(ρh³/2πσ)) é otimista para curvatura medida a
  // partir de partículas; medido, o limite real é uns 40% dela.
  const capilar = FATOR_CAPILAR * Math.sqrt((DENSIDADE_AGUA * h * h * h) / (2 * Math.PI * TENSAO_SUPERFICIAL));
  return Math.min((CFL * h) / Math.max(v, 0.3), capilar, PASSO_MAX);
}

/** Raio do pincel de parede, em células: o mínimo impede parede fina demais (que vaza). */
export const PINCEL_MIN_CELULAS = 1.3;
export const PINCEL_MAX_CELULAS = 8;
