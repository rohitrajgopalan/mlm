import numpy as np
import pika

from mlm_utils import load_model, calculate_blue_spots_score

model = load_model('blue_spots', ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                                  'Average Hierarchical distance',
                                  'Score'], {
                       'Distance since Last Update': 'int16',
                       'Number of blue Nodes': 'int8',
                       'Average Distance': 'int16',
                       'Average Hierarchical distance': 'int8',
                       'Score': 'float32'
                   })

look_ahead_time_in_seconds = 10
distance_error_base = 0.1


def predict_and_fit(distance_since_last_update, num_blue_nodes, average_distance, average_hierarchical_distance):
    inp = np.array(
        [distance_since_last_update, num_blue_nodes, average_distance, average_hierarchical_distance])
    inputs = np.array([inp])
    predicted_value = model.predict(inputs)
    new_data = {'Distance since Last Update': distance_since_last_update,
                'Number of blue Nodes': num_blue_nodes,
                'Average Distance': average_distance,
                'Average Hierarchical distance': average_hierarchical_distance}
    model.add(new_data, calculate_blue_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                                                   average_hierarchical_distance, look_ahead_time_in_seconds,
                                                   distance_error_base))
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
    distance_since_last_update = body_list[0]
    num_blue_nodes = body_list[1]
    average_distance = body_list[2]
    average_hierarchical_distance = body_list[3]
    response = predict_and_fit(distance_since_last_update, num_blue_nodes, average_distance,
                               average_hierarchical_distance)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

channel.start_consuming()
