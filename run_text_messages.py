import numpy as np
import pika
from mlm_utils import calculate_text_message_penalty, load_model

decay = 5 / 60
start_penalty = 49.625
model = load_model('text_messages', ['Age of Message', 'Penalty'], {
    'Age of Message': 'int16',
    'Penalty': 'int8'
})


def predict_and_fit(age_of_message):
    inp = np.array([age_of_message])
    inputs = np.array([inp])
    predicted_value = model.predict(inputs)
    new_data = {'Age of Message': age_of_message}
    model.add(new_data, calculate_text_message_penalty(age_of_message, start_penalty, decay))
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
    age_of_message = int(body)
    response = predict_and_fit(age_of_message)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

channel.start_consuming()
