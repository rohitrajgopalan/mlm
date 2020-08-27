import numpy as np
import pika

from mlm_utils import load_model, calculate_sos_score

base = 20
decay = 4 / 60
model = load_model('sos', ['Age of Message', 'Number of blue Nodes', 'Score'], {
    'Age of Message': 'int16',
    'Number of blue Nodes': 'int8',
    'Score': 'float16'
})


def predict_and_fit(age_of_message, number_of_blue_nodes):
    inp = np.array([age_of_message, number_of_blue_nodes])
    inputs = np.array([inp])
    predicted_value = model.predict(inputs)
    new_data = {'Age of Message': age_of_message, 'Number of blue Nodes': number_of_blue_nodes}
    model.add(new_data, calculate_sos_score(age_of_message, number_of_blue_nodes, base, decay))
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
    age_of_message = int(body_list[0])
    num_blue_nodes = int(body_list[1])
    response = predict_and_fit(age_of_message, num_blue_nodes)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

channel.start_consuming()
