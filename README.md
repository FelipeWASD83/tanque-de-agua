# Tanque

Um tanque de água visto de lado, com física de fluido de verdade: você desenha
as paredes com o mouse (copo, funil, labirinto, o que quiser), joga água e ela
escorre, espirra, enche, transborda e se acomoda como água.

## Como abrir

Dê dois cliques em **`abrir.bat`**. Ele liga o servidorzinho do projeto
(`servidor.py`) e abre o navegador em `http://127.0.0.1:4336/`. Não abre
clicando direto no `index.html`: o navegador bloqueia os módulos do app vindos
de arquivo.

Precisa de um navegador com **WebGPU** (Brave, Chrome e Edge atuais, no
Windows). Sem WebGPU o app avisa e roda a mesma física na CPU, com bem menos
água e sem a refração.

## Ferramentas

| Tecla | Ferramenta | O que faz |
|---|---|---|
| `1` | Parede | Arraste para desenhar. Com **Shift** faz linha reta |
| `2` | Reta | Arraste de uma ponta à outra |
| `3` | Borracha | Apaga parede (a moldura do tanque não sai) |
| `4` | Torneira | Segure para despejar. Mexa o mouse para jogar o jato |
| `5` | Balde | Arraste um retângulo: ele vira um bloco de água |
| `6` | Esponja | Segure em cima da água para tirar |
| `7` | Mão | Arraste dentro da água para empurrar |
| `8` | Corante | Passe por cima da água para tingir |

- **Cor da água:** vale para torneira, balde e corante. "Limpa" é água sem tinta.
- **Tamanho** (`[` e `]`): espessura da parede e tamanho do jato, da esponja,
  da mão e do corante. A parede tem espessura mínima de propósito: parede fina
  demais deixaria a água vazar.
- **Montar:** cenários prontos (copo, funil, ampulheta, cascata, labirinto,
  vasos comunicantes). O labirinto sai diferente a cada clique.

Em cima, à direita: pausar (`Espaço`), câmera lenta de ¼ (`L`), desfazer a
última parede (`Ctrl+Z`), tirar toda a água, apagar todas as paredes,
qualidade e tela cheia (`F`).

## O que fica guardado

As **paredes** e as escolhas (ferramenta, cor, tamanho, qualidade) ficam
guardadas neste navegador e voltam quando você abre de novo. A **água não fica
guardada**: o tanque abre vazio, e o app mostra como encher.

## Qualidade

| Opção | Para quê |
|---|---|
| Ultra | Grade de ~450 mil células (1,6 mm). Pesada: mesmo na 3060 livre, a água pode andar abaixo da velocidade real |
| Máximo | ~220 mil células. Para a 3060 com o PC livre |
| Alto | O padrão (~130 mil) |
| Médio / Leve | Para quando a placa estiver ocupada (um jogo aberto, por exemplo) |

Se a placa não der conta, a água passa a andar em câmera lenta em vez de travar
a tela, e o canto de cima avisa quanto: "água a 0,6× (placa no limite)" quer
dizer que ela está andando a 60% da velocidade real. Sem esse aviso, ela anda
no tempo do relógio.

## A física, em uma linha

É o método que os simuladores de efeito de filme usam para água (FLIP/APIC):
as partículas carregam a água, e uma grade calcula a pressão que a deixa
incompressível. O resultado é que o volume se conserva, a água parada fica
parada e os vasos comunicantes se equilibram na mesma altura.

A gravidade é a da Terra (9,81 m/s²) e a escala é real: 1440 pixels de tela
são 1 metro.

**Ar.** Todo bolsão de ar fechado pela água é um gás de verdade, com o volume
dele: uma bolha no fundo sobe mantendo o tamanho, o ar preso num copo virado
fica lá, e a ampulheta fechada **gorgoleja** — a água só desce na medida em
que o ar de baixo consegue subir pelo gargalo. O ar que encosta no topo aberto
do tanque é a atmosfera e sai livre.

**Tensão superficial** com o valor real da água (0,0728 N/m). Na escala do
tanque ela é sutil, como na vida: arredonda gota pequena, filete fino e ponta
de respingo, mas não segura um bloco de água parado no ar. Numa gota de 1 cm
dá para ver; num balde, não.
