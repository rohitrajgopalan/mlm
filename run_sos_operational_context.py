import numpy as np
import pika

from mlm_utils import calculate_sos_operational_context_mutliplier, load_model

model = load_model('sos_operational_context', ['Seconds Since Last Sent SOS', 'Multiplier'], {
    'Seconds Since Last Sent SOS': 'int16',
    'Multiplier': 'int8'
})


def predict_and_fit(seconds_since_last_sent_sos):
    inp = np.array([seconds_since_last_sent_sos])
    inputs = np.array([inp])
    predicted_value = model.predict(inputs)
    new_data = {'Seconds Since Last Sent SOS': seconds_since_last_sent_sos}
    model.add(new_data, calculate_sos_operational_context_mutliplier(
        calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos)))
    return predicted_value


# This is the same as what the Java classes use as the "default" exchange
DEFAULT_EXCHANGE_NAME = 'DEFAULT_EXCHANGE'

# Connect to the host and the queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Consumers will need to declare a queue - the below generates a random name
# Then the queue needs to be attached to an exchange (which the producers will publish on)
# It's ok to create the exchange everywhere - RabbitMQ will only create one
result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue
channel.exchange_declare(exchange=DEFAULT_EXCHANGE_NAME, exchange_type='fanout')
channel.queue_bind(exchange=DEFAULT_EXCHANGE_NAME, queue=queue_name)


def on_request(ch, method, props, body):
    seconds_since_last_sent_sos = int(body)
    response = predict_and_fit(seconds_since_last_sent_sos)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

channel.start_consuming()
