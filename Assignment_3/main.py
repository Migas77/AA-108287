import numpy as np
from book_file_reader import read_text_to_word_list
from exact_counter_alg import exact_counter_basic, exact_counter_basic_with_memory
from fixed_probab_counter_alg import fixed_probability_counter, fixed_probability_counter_with_memory
from collections import Counter, defaultdict
from tabulate import tabulate
import time
import click
import matplotlib.pyplot as plt


@click.command()
@click.option('--language', type=click.Choice(['english', 'french', 'german']), default='english', help='Language of the Romeo and Juliet book; Languages supported: english, french, german', show_default=True)
@click.option('--algorithm', type=click.Choice(['exact_counter', 'fixed_prob_counter', 'lossy_count']), default='exact_counter', help='Algorithm to use for counting words; Algorithms supported: exact_counter, fixed_prob_counter, lossy_count', show_default=True)
def main_command(language, algorithm):
    book_args_by_language = {
        'english': ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        'french': ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french', '====================================================='),
        'german': ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german', 'http://gutenberg2000.de erreichbar.')
    }

    words_list = read_text_to_word_list(*book_args_by_language[language])
    n_iters = 10

    # To evaluate algorithms results and execution time
    if algorithm == 'exact_counter':
        headers, results, exec_time, addit_data = evaluate_exact_counter(words_list)
        algorithm_name = 'Exact Counter Algorithm'
    elif algorithm == 'fixed_prob_counter':
        headers, results, exec_time, addit_data = evaluate_fixed_prob_counter(words_list, n_iters)
        algorithm_name = f'Fixed Probability Counter Algorithm: 1/2 (n_iters={n_iters})'
    elif algorithm == 'lossy_count':
        headers, results, exec_time, addit_data = evaluate_lossy_count(words_list)
        algorithm_name = 'Lossy Counting Algorithm'

    for table_type in ['grid', 'latex']:
        for is_sorted in [True, False]:
            filepath = f'results/{algorithm}/{language}_{algorithm}_{table_type}_{"sorted" if is_sorted else "raw"}_results.txt'
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'Results {algorithm_name}\n')
                if is_sorted:
                    output_results = sorted(results, key=lambda x: x[1], reverse=True)
                else:
                    output_results = results
                f.write(tabulate(output_results, headers=headers, tablefmt=table_type))
                f.write(f'\nExecution time: {exec_time} seconds')
                f.write(f'\n{addit_data}')
                print(f'The {"sorted" if is_sorted else "raw"} {algorithm} results can be found in {filepath}')

    # To evaluate algorithms memmory usage
    if algorithm == 'exact_counter':
        headers, results = evaluate_exact_counter_memory_usage(words_list)
    elif algorithm == 'fixed_prob_counter':
        headers, results = evaluate_fixed_prob_counter_memory_usage(words_list)
    elif algorithm == 'lossy_count':
        headers, results = evaluate_lossy_count(words_list)

    filepath = f'results/{algorithm}/{language}_{algorithm}_memory_results.txt'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'{algorithm} memory usage\n')
        f.write(tabulate(results, headers=headers, tablefmt='grid'))
        print(f'The memory usage results can be found in {filepath}')

    memory_usage = [result[2] for result in results]
    words = list(range(len(memory_usage)))
    plt.scatter(words, memory_usage)
    plt.xlabel('Number of Processed Words')
    plt.ylabel('Memory Usage (bytes)')
    plt.title(f'Memory Usage (bytes) Over Number of Processed Words for {algorithm_name}')
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.tight_layout()
    plt.savefig(f'results/{algorithm}/{language}_{algorithm}_memory_usage_plot.png')
    plt.show()
    




def evaluate_exact_counter(words_list):
    counter_results = Counter(words_list)
    start_time = time.perf_counter()
    exact_counter_results = exact_counter_basic(words_list)
    exec_time = time.perf_counter() - start_time
    assert counter_results == exact_counter_results

    headers = ['Word', 'ExactCount']
    results = exact_counter_results.items()

    return headers, results, exec_time, ''

def evaluate_exact_counter_memory_usage(words_list):
    counter_results = Counter(words_list)
    exact_counter_results, memory_usage = exact_counter_basic_with_memory(words_list)
    assert counter_results == exact_counter_results
    assert len(memory_usage) == len(words_list) + 1

    headers = ['Current Word', 'Current Word Count', 'Memory Usage', '# Distinct Words']
    results = memory_usage

    return headers, results


def evaluate_fixed_prob_counter(words_list, n_iters):
    assert n_iters > 0
    exact_counter_results = exact_counter_basic(words_list)
    prob = 1/2
    inverse_prob = 1 / prob
    word_counts = defaultdict(list)
    total_counts = Counter()
    word_occurences = Counter()
    exec_times = []

    for i in range(n_iters):
        start_time = time.perf_counter()
        results = fixed_probability_counter(words_list, 0.5)
        exec_time = time.perf_counter() - start_time
        total_counts.update(results)
        word_occurences.update(results.keys())
        exec_times.append(exec_time)
        for word, count in results.items():
            word_counts[word].append(count)
    
    for word, counts in word_counts.items():
        assert sum(counts) == total_counts[word]
        while len(counts) < n_iters:
            counts.append(0)
        assert len(counts) == n_iters

    total_counts = {word: count / n_iters for word, count in total_counts.items()}    
    exec_time = sum(t for t in exec_times) / n_iters

    headers = [
        'Word', '# Occurrences', f'Mean Count', f'Mean Estimated Count (MEC)', 'Expected Count',
        f'MEC Abs. Err.',
        f'MEC Rel. Err.',
        f'Mean Abs. Err.',
        f'Max Abs. Err.',
        f'Min Abs. Err.',
        f'Mean Rel. Err.',
        f'Max Rel. Err.',
        f'Min Rel. Err.',
    ]

    for word, count in total_counts.items():
        print(count, exact_counter_results[word], inverse_prob)
        print(abs(count * inverse_prob - exact_counter_results[word])) # Mean Absolute Error
        print([count2*inverse_prob for count2 in word_counts[word] if count2 != 0]) # List of counts
        print(np.mean([abs(count2 * inverse_prob - exact_counter_results[word]) for count2 in word_counts[word] if count2 != 0])) # Mean of absolute errors
        break


    results = [
        (word, word_occurences[word], average_count, average_count * inverse_prob, exact_counter_results[word], 
         abs(average_count * inverse_prob - exact_counter_results[word]),
         abs(average_count * inverse_prob - exact_counter_results[word]) / exact_counter_results[word],
         np.mean([abs(count * inverse_prob - exact_counter_results[word]) for count in word_counts[word]]),
         max([abs(count * inverse_prob - exact_counter_results[word]) for count in word_counts[word]]),
         min([abs(count * inverse_prob - exact_counter_results[word]) for count in word_counts[word]]),
         np.mean([abs(count * inverse_prob - exact_counter_results[word]) / exact_counter_results[word] for count in word_counts[word]]),
         max([abs(count * inverse_prob - exact_counter_results[word]) / exact_counter_results[word] for count in word_counts[word]]),
         min([abs(count * inverse_prob - exact_counter_results[word]) / exact_counter_results[word] for count in word_counts[word]]))
    for word, average_count in total_counts.items()]

    mean_abs_error_all_words = np.mean([abs(count * inverse_prob - exact_counter_results[word]) for word, counts in word_counts.items() for count in counts])
    mean_rel_error_all_words = np.mean([abs(count * inverse_prob - exact_counter_results[word]) / exact_counter_results[word] for word, counts in word_counts.items() for count in counts])
    
    mean_of_mean_abs_error = np.mean([result[7] for result in results])
    mean_of_mean_rel_error = np.mean([result[10] for result in results])

    assert round(mean_abs_error_all_words, 4) == round(mean_of_mean_abs_error, 4)
    assert round(mean_rel_error_all_words, 4) == round(mean_of_mean_rel_error, 4)

    addit_data = f'Mean Absolute Error All Words: {mean_abs_error_all_words}\nMean Relative Error All Words: {mean_rel_error_all_words}'

    return headers, results, exec_time, addit_data

def evaluate_fixed_prob_counter_memory_usage(words_list):
    exact_counter_results, memory_usage = fixed_probability_counter_with_memory(words_list, 0.5)

    headers = ['Current Word', 'Current Word Count', 'Memory Usage', '# Distinct Words']
    results = memory_usage

    return headers, results

    

def evaluate_lossy_count(words_list):
    pass


if __name__ == "__main__":
    main_command()

