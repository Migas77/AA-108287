from book_file_reader import read_text_to_word_list
from exact_counter_alg import exact_counter_basic, exact_counter_basic_with_memory
from fixed_probab_counter_alg import fixed_probability_counter
from collections import Counter
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

    # To evaluate algorithms results and execution time
    if algorithm == 'exact_counter':
        headers, results, exec_time = evaluate_exact_counter(words_list)
    elif algorithm == 'fixed_prob_counter':
        headers, results, exec_time = evaluate_fixed_prob_counter(words_list)
    elif algorithm == 'lossy_count':
        headers, results, exec_time = evaluate_lossy_count(words_list)

    for table_type in ['grid', 'latex']:
        for is_sorted in [True, False]:
            filepath = f'results/{algorithm}/{language}_{algorithm}_{table_type}_{"sorted" if is_sorted else "raw"}_results.txt'
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'{algorithm} results\n')
                if is_sorted:
                    output_results = sorted(results, key=lambda x: x[1], reverse=True)
                else:
                    output_results = results
                f.write(tabulate(output_results, headers=headers, tablefmt=table_type))
                f.write(f'\nExecution time: {exec_time} seconds')
                print(f'The {"sorted" if is_sorted else "raw"} {algorithm} results can be found in {filepath}')

    # To evaluate algorithms memmory usage
    if algorithm == 'exact_counter':
        headers, results = evaluate_exact_counter_memory_usage(words_list)
    elif algorithm == 'fixed_prob_counter':
        headers, results = evaluate_fixed_prob_counter(words_list)
    elif algorithm == 'lossy_count':
        headers, results = evaluate_lossy_count(words_list)

    filepath = f'results/{algorithm}/{language}_{algorithm}_memory_results.txt'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'{algorithm} memory usage\n')
        f.write(tabulate(results, headers=headers, tablefmt='grid'))
        print(f'The memory usage results can be found in {filepath}')

    if algorithm == 'exact_counter':
        algorithm_name = 'Exact Counter Algorithm'
    elif algorithm == 'fixed_prob_counter':
        algorithm_name = 'Fixed Probability Counter Algorithm: 1/2'
    elif algorithm == 'lossy_count':
        algorithm_name = 'Lossy Counting Algorithm'

    memory_usage = [result[2] for result in results]
    words = list(range(len(memory_usage)))
    plt.scatter(words, memory_usage)
    plt.xlabel('Number of Processed Words')
    plt.ylabel('Memory Usage (bytes)')
    plt.title(f'Memory Usage (bytes) Over Number of Processed Words for {algorithm}')
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.tight_layout()
    plt.savefig(f'results/{language}_{algorithm}_memory_usage_plot.png')
    plt.show()
    




def evaluate_exact_counter(words_list):
    counter_results = Counter(words_list)
    start_time = time.perf_counter()
    exact_counter_results = exact_counter_basic(words_list)
    exec_time = time.perf_counter() - start_time
    assert counter_results == exact_counter_results

    headers = ['Word', 'ExactCount']
    results = exact_counter_results.items()

    return headers, results, exec_time

def evaluate_exact_counter_memory_usage(words_list):
    counter_results = Counter(words_list)
    exact_counter_results, memory_usage = exact_counter_basic_with_memory(words_list)
    assert counter_results == exact_counter_results
    assert len(memory_usage) == len(words_list) + 1

    headers = ['Current Word', 'Current Word Count', 'Memory Usage', '# Distinct Words']
    results = memory_usage

    return headers, results


def evaluate_fixed_prob_counter(words_list):
    _, exact_counter_results, _ = evaluate_exact_counter(words_list)
    n_iters = 10000
    fixed_probab_counter_results = []

    # for i in range(n_iters):
    start_time = time.perf_counter()
    results = fixed_probability_counter(words_list, 0.5)
    exec_time = time.perf_counter() - start_time
    fixed_probab_counter_results.append((results, exec_time))

    fixed_probab_counter_results
    exec_time = sum(exec_time for _, exec_time in fixed_probab_counter_results) / n_iters

    headers = ['Word', 'Count']
    results = [(word, count) for word, count in exact_counter_results]

    return headers, results, exec_time


    

def evaluate_lossy_count(words_list):
    pass


if __name__ == "__main__":
    main_command()

