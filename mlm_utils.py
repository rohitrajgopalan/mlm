import math

import numpy as np


def calculate_blue_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                               average_hierarchical_distance, look_ahead_time_in_seconds, distance_error_base):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    distance_modifier = math.pow(1 - 0.2, (average_distance / 100) - 1)
    h_distance_modifier = math.pow(1 - 0.2, average_hierarchical_distance)
    return score_for_all_nodes * distance_modifier * h_distance_modifier


def calculate_red_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                              average_hierarchical_distance, look_ahead_time_in_seconds, distance_error_base,
                              nearest_values):
    multipliers = np.where(nearest_values < 100, 0.25, 0)
    multiplier_sum = np.sum(multipliers) + 1
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    h_distance_modifier = 0.2 - (0.04 * average_hierarchical_distance)
    distance_modifier = 1 if average_distance <= 100 else math.pow(1 - h_distance_modifier,
                                                                   (average_distance / 100) - 1)
    return score_for_all_nodes * distance_modifier * h_distance_modifier * multiplier_sum


def calculate_distance_to_enemy_multiplier(nearest_values):
    multipliers = np.where(nearest_values >= 100, 0, 1 - (nearest_values / 100))
    return np.sum(multipliers) + 1


def calculate_text_message_penalty(age_of_message, start_penalty, decay):
    return start_penalty - (decay * age_of_message)


def calculate_tactical_graphics_score(age_of_message, start_cum_message_score, decay, mutliplier):
    return (start_cum_message_score - (age_of_message * decay)) * mutliplier


def calculate_sos_score(age_of_message, num_blue_nodes, base, decay):
    cum_message_score = 0
    for i in range(10):
        cum_message_score += (base - ((age_of_message + i) * decay))
    cum_message_score = max(0, cum_message_score)
    return num_blue_nodes * cum_message_score


def calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos):
    return 2 if seconds_since_last_sent_sos < 121 else 1
