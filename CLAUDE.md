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

**Não é, e não deve virar sem ele pedir:** simulação 3D, ar/bolhas com pressão
própria (ver seção 4), tensão superficial, objetos rígidos que boiam, gravar a
água entre visitas, compartilhar cenário por link.

---

## 2. Arquivos

| Arquivo | O que tem |
|---|---|
| `js/constantes.js` | **todos** os números da física, do desenho e da qualidade. Os dois motores leem daqui |
| `js/cena.js` | paredes como campo de distância com sinal (sdf) numa grade 2×2 por célula, calculado pela geometria de cada traço (cápsula): une ao desenhar, subtrai ao apagar, repõe a moldura. Guarda os traços em metros (desfazer, redimensionar). Testado |
| `js/fisica-cpu.js` | **o motor de referência**: P2G (APIC) → gravidade e faces de parede → células e divergência (com correção de densidade) → pressão (Gauss-Seidel vermelho-preto) → gradiente → extrapolação → G2P, anda, bate na parede. Espuma, corante. Testado |
| `js/gpu/fisica.js` | os mesmos passos em compute shaders. Soma atômica em ponto fixo (a GPU não tem soma atômica de float). Partículas em dois buffers alternados; por quadro: ferramenta → partículas novas → compactação (tira as mortas) → subpassos. Despacho indireto (a CPU não sabe quantas partículas há) |
| `js/gpu/pcg.js` | pressão na GPU: **gradiente conjugado pré-condicionado com multigrid (MGPCG)**, beta de Polak-Ribière |
| `js/gpu/multigrid.js` | o pré-condicionador: ciclo em V, suavizador em blocos 16×16 na memória do grupo |
| `js/gpu/render.js` | partículas → campo (manchas, meia resolução, rgba16float) → shader de tela: painel procedural, refração, Beer-Lambert, espuma, menisco, paredes com chanfro e sombra |
| `js/render-2d.js` | desenho de reserva: painel e paredes em vetor, água como tinta multiplicada (3 px por célula) |
| `js/motor.js` | a mesma interface para os dois motores; abre a GPU |
| `js/modelos.js` | cenários prontos em metros, ajustados a um "palco" central (não estica no ultrawide). O labirinto é gerado a cada clique |
| `js/principal.js` | liga tudo: tamanho, ferramentas de ponteiro, guia (cursor e prévia), desfazer, relógio com passo adaptativo, preferências, `window.tanque` para depurar |
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
- Para medir desempenho de verdade, usar **carimbo de tempo da GPU**
  (`timestamp-query`, pedido na abertura do device) e `tanque.estado.congelado
  = true` para parar o relógio. Tempo de parede no navegador mede a disputa com
  outros programas, não o app.

---

## 4. Limitações conhecidas (física)

- **Sem ar.** Bolsão de ar fechado some (vácuo); numa ampulheta fechada a água
  desce direto em vez de gorgolejar. Resolver exigiria rastrear regiões de ar
  fechadas e dar pressão a elas.
- **Sem tensão superficial nem viscosidade explícita** (a numérica do APIC
  basta na escala do tanque).
- Na CPU a grade é bem mais grossa: respingo fino e filete somem.

---

## 5. Como verificar depois de mexer

- `node --test` na pasta: sdf (geometria, apagar, incremental = do zero) e
  física (água parada fica parada, copo não vaza nem perde água, parede do
  pincel mínimo segura jato forte, volume depois de rompimento de barragem,
  vasos comunicantes no mesmo nível, densidade de repouso).
- No navegador (preview `tanque-de-agua`, porta 4336), com o relógio congelado,
  os mesmos testes na GPU: água parada (vmax < 0,02 m/s depois de 3 s), copo
  (contagem constante), vasos comunicantes (níveis iguais). Olhar console: o
  único aviso esperado é o do `powerPreference` no Windows.
- Conferir em 3440 × 1440, no monitor em pé e em 390 px de largura (sem rolagem
  lateral; alvos de toque ≥ 44 px, menos as bolinhas de cor, com 40 px).
