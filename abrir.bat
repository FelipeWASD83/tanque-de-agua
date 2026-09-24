@echo off
rem Abre o Tanque: sobe um servidorzinho na porta 4336 e abre o navegador.
rem Fecha esta janela preta quando terminar de brincar com a agua.
rem (Aberto direto do arquivo, o navegador nao carrega os modulos do app.)
cd /d "%~dp0"
start "" "http://127.0.0.1:4336/"
python -m http.server 4336 --bind 127.0.0.1
