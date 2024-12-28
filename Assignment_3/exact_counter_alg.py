from pympler import asizeof
from book_file_reader import read_text_to_word_list
from collections import defaultdict, Counter
import timeit


def exact_counter_basic(words_list):
    word_count_map = {}
    for word in words_list:
        if word not in word_count_map:
            word_count_map[word] = 1
        else:
            word_count_map[word] += 1

    return word_count_map

def exact_counter_basic_with_memory(words_list):
    word_count_map = {}
    memory_usage = [("", "", asizeof.asizeof(word_count_map), "")]
    
    for word in words_list:
        if word not in word_count_map:
            word_count_map[word] = 1
        else:
            word_count_map[word] += 1
        memory_usage.append((word, word_count_map[word], asizeof.asizeof(word_count_map), len(word_count_map)))

    return word_count_map, memory_usage


def exact_counter_default_dict(words_list):
    word_count_map = defaultdict(int)
    for word in words_list:
        word_count_map[word] += 1

    return word_count_map


def exact_counter_get(words_list):
    word_count_map = {}
    for word in words_list:
        word_count_map[word] = word_count_map.get(word, 0) + 1
    return word_count_map


def exact_counter_set_default(words_list):
    word_count_map = {}
    for word in words_list:
        word_count_map.setdefault(word, 0)
        word_count_map[word] += 1
    return word_count_map


if __name__ == '__main__':
    book_paths_with_language = [
        ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french'),
        ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german')
    ]

    file_path, language = book_paths_with_language[1]
    words_list = read_text_to_word_list(file_path, language)
    n_iter = 100000

    counter_result = Counter(words_list)

    print(
        counter_result == exact_counter_basic(words_list) == exact_counter_default_dict(words_list) == 
        exact_counter_get(words_list) == exact_counter_set_default(words_list)
    )

    assert counter_result == exact_counter_basic(words_list) == exact_counter_default_dict(words_list) == exact_counter_get(words_list) == exact_counter_set_default(words_list)

    # Timing Counter
    counter_time = timeit.timeit(lambda: Counter(words_list), number=n_iter)
    print(f"Counter took {counter_time} seconds")

    # Timing exact_counter_basic
    exact_counter_basic_time = timeit.timeit(lambda: exact_counter_basic(words_list), number=n_iter)
    print(f"exact_counter_basic took {exact_counter_basic_time/n_iter} seconds")

    # Timing exact_counter_default_dict
    exact_counter_default_dict_time = timeit.timeit(lambda: exact_counter_default_dict(words_list), number=n_iter)
    print(f"exact_counter_default_dict took {exact_counter_default_dict_time/n_iter} seconds")

    # Timing exact_counter_get
    exact_counter_get_time = timeit.timeit(lambda: exact_counter_get(words_list), number=n_iter)
    print(f"exact_counter_get took {exact_counter_get_time/n_iter} seconds")

    # Timing exact_counter_set_default
    exact_counter_set_default_time = timeit.timeit(lambda: exact_counter_set_default(words_list), number=n_iter)
    print(f"exact_counter_set_default took {exact_counter_set_default_time/n_iter} seconds")
