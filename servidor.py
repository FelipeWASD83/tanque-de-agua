"""Servidor local do Tanque: serve a pasta e manda o navegador nunca guardar
os arquivos em cache (senão, depois de uma atualização, ele continua rodando
os módulos velhos misturados com os novos).

Uso: python servidor.py [porta]      (padrão 4336)
"""

import http.server
import os
import sys


class SemCache(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, formato, *args):
        pass  # sem uma linha por arquivo no terminal


if __name__ == "__main__":
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 4336
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with http.server.ThreadingHTTPServer(("127.0.0.1", porta), SemCache) as s:
        print(f"Tanque em http://127.0.0.1:{porta}/  (feche esta janela para parar)")
        s.serve_forever()
