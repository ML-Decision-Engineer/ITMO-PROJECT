from __future__ import annotations

import json
import socket
from typing import Any

import pika

from app.config import settings


def connection_parameters() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        credentials=pika.PlainCredentials(
            settings.rabbitmq_user,
            settings.rabbitmq_password,
        ),
        heartbeat=30,
        blocked_connection_timeout=10,
        connection_attempts=5,
        retry_delay=2,
    )


def publish_daily_task(run_id: int) -> None:
    connection = pika.BlockingConnection(connection_parameters())
    try:
        channel = connection.channel()
        channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue,
            body=json.dumps({"run_id": run_id}).encode("utf-8"),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
        )
    finally:
        connection.close()


def rabbitmq_is_available() -> bool:
    try:
        with socket.create_connection(
            (settings.rabbitmq_host, settings.rabbitmq_port),
            timeout=2,
        ):
            return True
    except OSError:
        return False
