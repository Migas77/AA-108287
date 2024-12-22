from collections import defaultdict, Counter
import time

from text_file_reader import read_text_to_word_list


def count_words(words_list):
    word_count_map = {}
    for word in words_list:
        if word not in word_count_map:
            word_count_map[word] = 1
        else:
            word_count_map[word] += 1

    return word_count_map


def count_words_2(words_list):
    word_count_map = defaultdict(int)
    for word in words_list:
        word_count_map[word] += 1

    return word_count_map


if __name__ == '__main__':
    book_paths_with_language = [
        ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french'),
        ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german')
    ]

    file_path, language = book_paths_with_language[0]
    words_list = read_text_to_word_list(file_path, language)
    
    start_time = time.time()
    word_count_map = count_words(words_list)
    end_time = time.time()
    print(f"count_words took {end_time - start_time} seconds")

    start_time = time.time()
    word_count_map_2 = count_words_2(words_list)
    end_time = time.time()
    print(f"count_words_2 took {end_time - start_time} seconds")
    
    start_time = time.time()
    word_count_map_3 = Counter(words_list)
    end_time = time.time()
    print(f"Counter took {end_time - start_time} seconds")


    print(word_count_map == word_count_map_2 == word_count_map_3)
    print(dict(sorted(word_count_map.items(), key=lambda item: item[1], reverse=True)))
