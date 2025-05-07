#!/bin/bash

# Проверка аргументов командной строки
if [ $# -eq 0 ]; then
    echo "Использование: ./run.sh [collect|train|backtest|simulate|trade|optimize]"
    exit 1
fi

MODE=$1
shift

# Получение дополнительных аргументов
ARGS=""
while [[ $# -gt 0 ]]; do
    ARGS="$ARGS $1"
    shift
done

# Запуск в нужном режиме
case $MODE in
    collect)
        echo "Запуск сбора данных..."
        docker-compose run --rm app python -m src.main --mode collect $ARGS
        ;;
    train)
        echo "Запуск обучения модели..."
        docker-compose run --rm app python -m src.main --mode train $ARGS
        ;;
    backtest)
        echo "Запуск бэктестирования..."
        docker-compose run --rm app python -m src.main --mode backtest $ARGS
        ;;
    simulate)
        echo "Запуск симуляции..."
        docker-compose run --rm app python -m src.main --mode simulate $ARGS
        ;;
    trade)
        echo "Запуск торговли..."
        docker-compose run --rm app python -m src.main --mode trade $ARGS
        ;;
    optimize)
        echo "Запуск оптимизации параметров..."
        docker-compose run --rm app python -m src.main --mode optimize $ARGS
        ;;
    notebook)
        echo "Запуск Jupyter Notebook..."
        docker-compose up notebook
        ;;
    wss)
        echo "Запуск WebSocket сервера..."
        docker-compose up wss
        ;;
    *)
        echo "Неизвестный режим: $MODE"
        echo "Доступные режимы: collect, train, backtest, simulate, trade, optimize, notebook, wss"
        exit 1
        ;;
esac