# Etapa 1: Define a imagem base do Python. É enxuta, por isso precisa de ajuda.
FROM python:3.11-slim

# Etapa 2: Define o diretório de trabalho padrão dentro do container.
WORKDIR /app

# Etapa 3: Copia apenas o arquivo de requisitos. Isso otimiza o cache do Docker.
COPY requirements.txt /app/

# --- A CORREÇÃO ESSENCIAL ESTÁ AQUI ---
# Antes de instalar pacotes Python, instala as ferramentas de build do Linux (Debian).
# 'build-essential' e 'gcc' são os pacotes que contêm os compiladores C/C++.
RUN apt-get update && apt-get install -y build-essential gcc

# Etapa 4: Agora sim, instala as dependências Python.
# O pip agora encontrará as ferramentas que instalamos na etapa anterior e conseguirá compilar tudo.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Etapa 5: Copia o resto dos arquivos da sua aplicação para o diretório de trabalho.
COPY . /app

# Etapa 6: Define o comando padrão que será executado quando o container iniciar.
CMD ["streamlit", "run", "app/app.py", "--server.port=8501"]