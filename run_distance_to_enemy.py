import numpy as np
import pika

from mlm_utils import load_model, calculate_distance_to_enemy_multiplier

model = load_model('distance_to_enemy',
                   ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier'], {
                       '#1 Nearest': 'int16',
                       '#2 Nearest': 'int16',
                       '#3 Nearest': 'int16',
                       '#4 Nearest': 'int16',
                       '#5 Nearest': 'int16',
                       'Multiplier': 'float8'
                   })


def predict_and_fit(nearest_values):
    inputs = np.array([nearest_values])
    predicted_value = model.predict(inputs)
    new_data = {}
    for i in range(5):
        new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
    model.add(new_data, calculate_distance_to_enemy_multiplier(nearest_values))
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
    body_list = body.split(',')
    body_list = [int(e) for e in body_list]
    nearest_values = np.array([body_list[0], body_list[1], body_list[2], body_list[3], body_list[4], body_list[5]])
    response = predict_and_fit(nearest_values)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

channel.start_consuming()
