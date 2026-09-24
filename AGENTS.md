# AGENTS.md — tanque-de-agua

As regras deste projeto estão em **`CLAUDE.md`**, nesta mesma pasta. Ler antes
de mexer em qualquer coisa: lá estão o escopo, a direção visual, os arquivos e
as armadilhas da física.

Resumo do que não pode ser esquecido:

- **Todos os números da física ficam em `js/constantes.js`**; os dois motores
  (GPU em `js/gpu/`, CPU em `js/fisica-cpu.js`) leem de lá. Mudou regra da
  física? Mudou teste (`node --test` na pasta).
- A pressão na GPU é **MGPCG** (`js/gpu/pcg.js`). Gauss-Seidel sozinho não
  assenta a água; multigrid sozinho explode em parede fina.
- A correção de densidade é por **tempo**, não por passo.
- Não abre por `file://`: usar o `abrir.bat` ou o preview na porta 4336,
  registrada no `.claude/launch.json` da raiz.
- Sem ar e sem tensão superficial, de propósito (seção 4 do `CLAUDE.md`).
