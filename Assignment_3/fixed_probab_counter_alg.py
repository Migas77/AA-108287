from collections import Counter, defaultdict
from book_file_reader import read_text_to_word_list
import random
import numpy as np
import statistics

# NMEC: 108287 - Miguel Figueiredo
random.seed(108287)

def fixed_probability_counter(words_list, k):
    """Approximate counter with fixed probability."""
    assert k >= 1, "k must be greater than or equal to 1"
    word_count_map = {}
    probability = 1 / 2**k
    for word in words_list:
        if random.random() < probability:
            if word not in word_count_map:
                word_count_map[word] = 1
            else:
                word_count_map[word] += 1
    return word_count_map


if __name__ == '__main__':
    # ---------------------------------------------------
    # Matching slide 30 of Probabilistic Counters slides |
    # ---------------------------------------------------

    kp = 1
    prob = 1 / 2**kp                                                            # Probability of counting a word: 0.5

    trials = 10000
    k = 100
    words_list = list(range(k))

    counters = [sum(fixed_probability_counter(words_list, kp).values()) for _ in range(trials)]

    expected_value = k * prob
    expected_variance_1 = k * prob * (1 - prob)
    mean_value = 0 * prob + 1 * (1 - prob)                                      # E[Xi]
    mean_value_squared = mean_value ** 2                                        # {E[Xi]}^2
    expected_value_of_squares = 0**2 * prob + 1**2 * (1 - prob)                 # E[Xi^2]
    expected_variance_2 = k * (expected_value_of_squares - mean_value_squared)  # E[Xi^2] - {E[Xi]}^2
    expected_std_deviation = expected_variance_1 ** 0.5                         # Standard deviation is the square root of the variance
    print(f"Expected value: {expected_value}")
    print(f"Variance: {expected_variance_1} {expected_variance_2}")
    print(f"Standard deviation: {expected_std_deviation}\n")

    mean_counter_value = sum(counters) / trials

    mean_abs_error = sum(abs(counter - expected_value) for counter in counters) / trials
    mean_rel_error = sum(abs(counter - expected_value) / expected_value for counter in counters) / trials
    mean_acc_ratio = sum(counter / expected_value for counter in counters) / trials
    print(f"Mean Absolute Error: {mean_abs_error} {np.mean(np.abs(np.array(counters) - expected_value))}")
    print(f"Mean Relative Error: {100*mean_rel_error}% {100*np.mean(np.abs(np.array(counters) - expected_value) / expected_value)}%")
    print(f"Mean Accuracy Ratio: {100*mean_acc_ratio}% {100*np.mean(np.array(counters) / expected_value)}%\n")

    smallest_counter_value = min(counters)
    largest_counter_value = max(counters)
    print(f"Smallest counter value: {smallest_counter_value}")
    print(f"Largest counter value: {largest_counter_value}\n")

    mean_abs_deviation = sum(abs(counter - mean_counter_value) for counter in counters) / trials
    std_deviation = (sum((counter - mean_counter_value) ** 2 for counter in counters) / trials) ** 0.5 
    maximum_deviation = max(abs(counter - mean_counter_value) for counter in counters)
    variance = sum((counter - mean_counter_value) ** 2 for counter in counters) / trials
    print(f"Mean Counter Value: {mean_counter_value} {np.mean(counters)} {statistics.mean(counters)}")
    print(f"Mean Absolute Deviation: {mean_abs_deviation} {np.mean(np.abs(np.array(counters) - mean_counter_value))} {statistics.mean(abs(_count - mean_counter_value) for _count in counters)}")
    print(f"Standard Deviation: {std_deviation} {np.std(counters)} {statistics.pstdev(counters)}")
    print(f"Maximum Deviation: {maximum_deviation} {np.max(np.abs(np.array(counters) - mean_counter_value))}")
    print(f"Variance: {variance} {np.var(counters)} {statistics.pvariance(counters)}\n\n")

    
    # ---------------------------------------------------
    # Example of counting words in one of the books     |
    # ---------------------------------------------------
    book_paths_with_language = [
        ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french'),
        ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german')
    ]

    file_path, language = book_paths_with_language[0]
    words_list = read_text_to_word_list(file_path, language)
    word_count_map = fixed_probability_counter(words_list, kp)
    print(sum(word_count_map.values()) * 2, len(words_list))
    for word, count in word_count_map.items():
        # print(f"{word}: {count}")
        pass

    counter_distribution = Counter(counters)
    total_count = sum(counter_distribution.values())
    print(total_count)

    # Print the table
    print(f"Counting {k} events — {trials} trials")
    print("-" * 50)
    for value, count in sorted(counter_distribution.items()):
        print(f"Counter value: {value:2d} - {count:4d} times - {100*count/total_count:6.3f}%")


