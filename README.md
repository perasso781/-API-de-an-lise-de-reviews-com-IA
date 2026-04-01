review_intel/
├── main.py                  # Onde a aplicação começa (inicia a API)
├── core/
│   ├── config.py            # Configurações do projeto (ex: variáveis de ambiente)
│   └── security.py          # Validação de acesso (API Key)
├── models/
│   └── schemas.py           # Define o formato dos dados (entrada e saída)
├── scraper/
│   └── crawler.py           # Coleta dados da internet (web scraping)
├── ml/
│   └── sentiment.py         # Parte de IA (analisa sentimento dos textos)
├── api/
│   └── routes.py            # Define as rotas da API (/health, /analyze, /train)
├── Dockerfile               # Configuração para rodar com Docker
├── docker-compose.yml       # Orquestra os containers
├── requirements.txt         # Lista de dependências do projeto
├── .env.example             # Exemplo de variáveis de ambiente
└── .gitignore               # Arquivos que não devem subir para o GitHub
