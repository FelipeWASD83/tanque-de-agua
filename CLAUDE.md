# CLAUDE.md — tanque-de-agua

"Tanque": simulação 2D de água incompressível num tanque visto de lado, com
paredes desenhadas à mão. HTML/CSS/JS sem dependência e sem build; física na
GPU por WebGPU (compute shaders WGSL) e cópia de referência na CPU, testada no
Node. Criado em 24/09/2026.

O `CLAUDE.md` da raiz vale aqui também. Este arquivo só acrescenta o que é
específico deste projeto. O `README.md` explica o uso para o Felipe.

---

## 1. O pedido e o escopo

**Pedido dele:** "simulação de água extremamente realista em 2D (física
realista); o usuário faz os obstáculos (copo, labirinto etc.) desenhando".

**Decisões tomadas (plano aprovado em 24/09/2026):**

- **Método: FLIP/APIC** (partículas carregam a água, grade MAC resolve a
  pressão). Descartados: SPH/PBF (comprime, parece geleia) e MPM (bom para
  neve/areia, pior para água pura).
- **Direção visual "canal de ensaio de laboratório"**: painel claro iluminado
  por trás com pontos a cada 1 cm, linhas a cada 10 cm e régua; paredes de
  acrílico grafite com sombra no painel; uma única cor de ação (laranja
  `#d4531c`) para o que está selecionado; números em mono pequeno nos cantos
  (referência: Linear, `prints/produto-linear-hero.png`). **O elemento único é
  a grade do painel entortando através da água** (refração onde a superfície
  curva). Papel milimetrado foi evitado de propósito (é do `cobrinha-simples`).
- **Escala física fixa:** 1440 px de tela (CSS) = 1 m. O tanque é do tamanho da
  janela; no celular ele é pequeno (e um copo desenhado é do tamanho de um copo).
- **Sem WebGPU → mesma física na CPU** (grade bem menor) com desenho em canvas
  2D e um aviso.

**Segunda rodada (plano aprovado em 24/09/2026)**, pedida com um print da água
"desmanchada" em espuma: ar, tensão superficial, resolução maior, gravidade
"mais forte" e o bug de a física quebrar depois de um tempo.

- **A gravidade continua a real (9,81 m/s²).** A sensação de leveza era câmera
  lenta sem querer: o teto de subpassos media o tempo de fila e deixava a água
  a 0,2× no Máximo. Agora o custo vem do carimbo de tempo da própria placa e o
  rodapé avisa quando a água está abaixo de 0,9× do relógio. Se ainda parecer
  leve, a oferta é um controle de gravidade, não mudar a constante.
- **Ar**: bolsões fechados são gás com volume (bolha implícita no solver);
  o ar ligado ao topo é a atmosfera (pressão zero).
- **Tensão superficial** com o valor real, 0,0728 N/m — sutil a 1 m de escala,
  e isso foi dito a ele.
- **Ultra** acima do Máximo, sabendo que pode rodar abaixo de 1,0×.
- O bug do print era a pressão mal convergida com muitas partículas: foi
  resolvido com pré-condicionador determinístico, correção de densidade
  limitada e teto de partículas (com aviso de "tanque cheio").

**Não é, e não deve virar sem ele pedir:** simulação 3D, objetos rígidos que
boiam, gravar a água entre visitas, compartilhar cenário por link.

---

## 2. Arquivos

| Arquivo | O que tem |
|---|---|
| `js/constantes.js` | **todos** os números da física, do desenho e da qualidade. Os dois motores leem daqui |
| `js/cena.js` | paredes como campo de distância com sinal (sdf) numa grade 2×2 por célula, calculado pela geometria de cada traço (cápsula): une ao desenhar, subtrai ao apagar, repõe a moldura. Guarda os traços em metros (desfazer, redimensionar). Testado |
| `js/fisica-cpu.js` | **o motor de referência**: P2G (APIC) → gravidade e faces de parede → tipos e rótulos do ar → superfície (fração de água, normal, curvatura) → tensão no ar vizinho → divergência (com correção de densidade) → lado direito das bolhas → pressão (MGPCG com bolhas) → gradiente → extrapolação → G2P em micro-passos, batendo na parede a cada um. Espuma, corante. Testado |
| `js/pressao-cpu.js` | o MGPCG da CPU, **espelho exato do da GPU** (mesma restrição, mesmo suavizador em blocos, mesma ordem): quando a GPU diverge, compara-se com este |
| `js/ar.js` | rastreador do ar: union-find nas células de ar, atmosfera (toca o topo) vs. bolha; rótulo estável pela sobreposição com o quadro anterior; o volume de ar de cada bolha passa para os pedaços quando ela se parte (pedaço com menos de 4 células devolve o ar) |
| `js/ar-trabalhador.js` | o `ar.js` rodando num Web Worker para o motor da GPU, com contador de geração (leitura de antes de um reinício é descartada) |
| `js/gpu/fisica.js` | os mesmos passos em compute shaders. Soma atômica em ponto fixo (a GPU não tem soma atômica de float). Partículas em dois buffers alternados; por quadro: ferramenta → partículas novas → compactação (tira as mortas) → subpassos. Despacho indireto (a CPU não sabe quantas partículas há) |
| `js/gpu/pcg.js` | pressão na GPU: **gradiente conjugado pré-condicionado com multigrid (MGPCG)**, beta de Polak-Ribière, 3 iterações por subpasso. Cada bolha é uma incógnita a mais (pressão única), somada por troca-e-compara agrupada por trecho |
| `js/gpu/multigrid.js` | o pré-condicionador: ciclo em V; suavizador em blocos 16×16 com ida e volta entre dois buffers (determinístico, simétrico); a grade mais grossa resolvida num despacho só |
| `js/gpu/render.js` | partículas → campo (manchas, meia resolução, rgba16float) → shader de tela: painel procedural, refração, Beer-Lambert, espuma, menisco, paredes com chanfro e sombra |
| `js/render-2d.js` | desenho de reserva: painel e paredes em vetor, água como tinta multiplicada (3 px por célula) |
| `js/motor.js` | a mesma interface para os dois motores; abre a GPU |
| `js/modelos.js` | cenários prontos em metros, ajustados a um "palco" central (não estica no ultrawide). O labirinto é gerado a cada clique |
| `js/principal.js` | liga tudo: tamanho, ferramentas de ponteiro, guia (cursor e prévia), desfazer, relógio com passo adaptativo, preferências, `window.tanque` para depurar |
| `servidor.py` | servidor estático **sem cache** (`Cache-Control: no-store`) na 4336: o `http.server` puro deixava o navegador com módulo velho depois de editar |
| `testes/*.test.js` | `node --test` na pasta |

---

## 3. Armadilhas (o que já quebrou e por quê)

- **A pressão na GPU tem que ser MGPCG.** Gauss-Seidel em blocos sozinho não
  converge: a água parada treme (1,3 m/s de velocidade máxima). Multigrid
  sozinho **explode** em paredes finas (uma célula grossa junta a água dos dois
  lados da parede) e a água some virando NaN. Só o multigrid dentro do
  gradiente conjugado ficou estável e parado (< 1 cm/s na água parada).
- **Correção de densidade é por tempo (`TAU_DENSIDADE`, em segundos), não por
  passo.** Por passo, a força dependia do número de subpassos e bombeava energia
  — a água não assentava. E ela só contrai célula rarefeita no **miolo** da
  água (perto do ar e da parede a densidade medida é naturalmente menor).
- **Espuma não pode depender de Δv/dt.** A transferência partícula↔grade muda
  a velocidade um pouco a cada passo; dividido por um dt pequeno vira "impacto"
  e a água inteira fica branca. O critério é água rápida **e** rarefeita, com
  taxa por segundo.
- **Kernel com layout `auto` só enxerga o que o shader usa.** Binding
  declarado e não usado some do layout e o bind group quebra. Todo kernel de
  partícula usa `P` pelo `vivas` (`min(contagem, P.maxPart)`).
- **Mesmo buffer em dois bindings graváveis do mesmo grupo é erro de
  validação.** Foi o caso do produto `p·Ap` (o segundo canal agora usa outro
  buffer).
- **`lerParticulas` faz a própria contagem.** A do relógio chega com um quadro
  de atraso; usando ela, redimensionar a janela copiava zero partículas e a água
  sumia.
- **Passo adaptativo com teto por custo.** O passo segue a maior velocidade
  medida no quadro anterior (`contagem[2]`), e o número de subpassos é limitado
  pelo tempo que a placa está levando (`ORCAMENTO_MS`). Sem esse teto, com a
  placa ocupada (CS2 aberto, em 24/09) o quadro lento pedia mais subpassos e
  travava tudo em 1 qps. Com ele, a água só anda mais devagar.
- **Parede do sdf por geometria, não por máscara.** A máscara de pixels dava
  borda em escada. A espessura mínima do pincel (`PINCEL_MIN_CELULAS`) existe
  para a água não atravessar parede fina — tem teste.
- **O custo do teto vem do carimbo de tempo da placa** (`custoPorSubpasso`).
  Medir pelo tempo de fila deixava a água em câmera lenta (0,2×) sem ninguém
  perceber. Sem `timestamp-query`, o `principal.js` volta para a medida pelo
  tempo de parede — pior, mas sem ela volta o travamento em 1 qps.
- **Pré-condicionador tem que ser determinístico e simétrico** para o
  gradiente conjugado: Gauss-Seidel em blocos lendo o vizinho enquanto outro
  grupo escreve dava resultado diferente a cada rodada, o PCG perdia a
  convergência e, com muita partícula, a água virava a espuma azul esparsa do
  print dele. Hoje é ida e volta entre dois buffers, com a ordem das cores
  espelhada na volta.
- **Soma atômica de float num endereço só vira fila.** Milhares de células de
  uma bolha somando por troca-e-compara no mesmo `acc[b]` deixavam a pressão
  ~20× mais lenta (45 ms contra 2,2 ms na ampulheta). Soma-se primeiro o
  trecho contíguo dentro do grupo e faz-se uma troca por trecho
  (`somarNoGrupo`, e o mesmo no `ladoBolhas`).
- **Ar começa esperando o primeiro retrato.** Na GPU os rótulos vêm do Worker
  com um ou dois quadros de atraso; sem esperar, o ar fechado vira vácuo e a
  bolha fecha antes de existir. Por isso `reiniciarAr()` roda ao limpar a água
  **e no construtor** (motor recriado ao redimensionar). Na CPU, `reiniciarAr`
  zera `tiposProntos`, senão a primeira leitura usa a água do cenário anterior
  (tem teste).
- **Bloco de água novo maior que `NOVAS_MAX` abria a bolha** (entrava em
  pedaços em quadros diferentes). Hoje o limite é 2²⁰ por quadro.
- **Tensão superficial**: a fração de água é estendida 3 camadas para dentro da
  parede (ângulo de 90°) antes do borrão; sem isso a curvatura explodia nos
  cantos. O passo capilar (`FATOR_CAPILAR` 0,4 do teórico, teto `PASSO_MAX`)
  foi medido: acima disso a gota quadrada oscila e explode.
- **Kernel com barreira de grupo não pode ter `return` antes dela** (controle
  de fluxo não uniforme): os kernels de soma agrupada guardam `NENHUMA` em vez
  de sair cedo.
- Para medir desempenho de verdade, usar **carimbo de tempo da GPU**
  (`tanque.motor.fisica.perfilar()` por etapa) e `tanque.estado.congelado =
  true` para parar o relógio. **Com jogo ou outro navegador usando a placa,
  os carimbos saem picotados** (zeros e picos de 17 ms): aí comparar A e B
  alternados, várias rodadas, pegando o menor tempo de parede de cada um.

---

## 4. Limitações conhecidas (física)

- **O ar é isotérmico e sem inércia**: a bolha tem uma pressão só e volta ao
  volume que tinha (`VOLTA_VOLUME_BOLHA`), sem compressão real (a bolha não
  encolhe no fundo). Bolha com menos de 4 células devolve o ar à vizinha ou
  some — microbolha e espuma de ar não existem.
- **Tensão superficial sutil a 1 m de escala** e com correntes parasitas
  pequenas na água parada (o teste tolera 2 cm/s com ela ligada).
- **Sem viscosidade explícita** (a numérica do APIC basta na escala do tanque).
- **Ultra pode rodar abaixo de 1,0×** mesmo na placa livre; o rodapé avisa.
- Na CPU a grade é bem mais grossa: respingo fino e filete somem.

---

## 5. Como verificar depois de mexer

- `node --test` na pasta: sdf (geometria, apagar, incremental = do zero) e
  física (água parada fica parada, com e sem tensão; copo não vaza nem perde
  água; parede do pincel mínimo segura jato forte; volume depois de rompimento
  de barragem; vasos comunicantes no mesmo nível; densidade de repouso; gota
  quadrada arredonda só com tensão; bolha sobe mantendo o volume; sem ar o
  buraco fecha; ampulheta com ar esvazia mais devagar; remontar cenário não
  perde a bolha).
- Mexeu na pressão da GPU? Comparar com a da CPU (`pressao-cpu.js`) no mesmo
  cenário: o `y` das bolhas e o `q` têm que bater (diferença da ordem de 1e-5).
- No navegador (preview `tanque-de-agua`, porta 4336), com o relógio congelado,
  os mesmos testes na GPU: água parada (vmax < 0,02 m/s depois de 3 s), copo
  (contagem constante), vasos comunicantes (níveis iguais). Olhar console: o
  único aviso esperado é o do `powerPreference` no Windows.
- Conferir em 3440 × 1440, no monitor em pé e em 390 px de largura (sem rolagem
  lateral; alvos de toque ≥ 44 px, menos as bolinhas de cor, com 40 px).
