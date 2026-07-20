from __future__ import annotations

import json
import logging
import socket
import time

import pika

from app.config import settings
from app.queue import connection_parameters
from app.services import init_database, process_daily_run


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ml-worker")
WORKER_ID = socket.gethostname()


def on_message(channel, method, properties, body):
    try:
        payload = json.loads(body.decode("utf-8"))
        run_id = int(payload["run_id"])
    except Exception:
        logger.exception("Некорректное сообщение: %r", body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    logger.info("Получен суточный расчёт run_id=%s", run_id)

    try:
        process_daily_run(run_id=run_id, worker_id=WORKER_ID)
        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Расчёт run_id=%s завершён", run_id)
    except Exception:
        logger.exception("Ошибка расчёта run_id=%s", run_id)
        channel.basic_ack(delivery_tag=method.delivery_tag)


def run_worker() -> None:
    init_database()

    while True:
        connection = None
        try:
            connection = pika.BlockingConnection(connection_parameters())
            channel = connection.channel()
            channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue=settings.rabbitmq_queue,
                on_message_callback=on_message,
                auto_ack=False,
            )
            logger.info(
                "Воркер %s ожидает задачи в очереди %s",
                WORKER_ID,
                settings.rabbitmq_queue,
            )
            channel.start_consuming()
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("Соединение потеряно, повтор через 5 секунд")
            time.sleep(5)
        finally:
            if connection is not None and connection.is_open:
                connection.close()


if __name__ == "__main__":
    run_worker()
