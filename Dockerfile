FROM python:3.9.6

WORKDIR /app

# Установка зависимостей для работы с числовыми библиотеками и TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    gcc \
    g++ \
    make \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка TA-Lib
# Modify the TA-Lib installation section in your Dockerfile
#RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
#    tar -xzf ta-lib-0.4.0-src.tar.gz && \
#    cd ta-lib/ && \
#    # Use a more reliable mirror for the config files
#    wget 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.guess' -O config.guess && \
#    wget 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.sub' -O config.sub && \
#    chmod +x config.guess config.sub && \
#    # Continue with the build
#    ./configure --prefix=/usr && \
#    make && \
#    make install && \
#    cd .. && \
#    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
# Копирование requirements.txt
COPY requirements.txt .

# Установка зависимостей Python
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data models logs output

# Проверка наличия всех зависимостей
RUN python -c "import numpy, pandas, tensorflow, sklearn, matplotlib, talib"

# Порты для REST API и WebSocket
EXPOSE 8000 8765

# Запуск приложения
CMD ["python", "-m", "src.main"]