#!/usr/bin/env python
import pika
import json
from abc import ABC
from abc import abstractmethod


# TODO:
# - JSON response
# - Flush queue when starting (here and in SN)
# - Investigate hanging issue

class RabbitMQServer(ABC):

    def __init__(self, queue_name, durable=False, heartbeat=600, blocked_connection_timeout=600):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', heartbeat=heartbeat, blocked_connection_timeout=blocked_connection_timeout, port=5672))
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=durable)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=self._on_request)
        self.channel = channel

    def start(self):
        # print("Awaiting RPC requests")
        self.channel.start_consuming()

    def _on_request(self, channel, method, props, body):
        try:
            request = json.loads(body)
        except TypeError:
            request = json.loads(body.decode('utf-8'))
        response = self._get_reply(request)
        channel.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=str(response))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    @abstractmethod
    def _get_reply(self, request):
        pass
