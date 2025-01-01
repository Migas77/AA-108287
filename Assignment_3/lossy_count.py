from book_file_reader import read_text_to_word_list
from lossy_count_to_verify import MugegenLossyCounting
from collections import Counter
import math


class LossyCounting:
    def __init__(self, k):
        """
        Initialize the Lossy Counting algorithm.

        Args:
            k (int): inverse of epsilon.
        """
        self.k = k              # inverse of epsilon
        self.delta = 0          # Current bucket boundary
        self.buckets = {}       # Buckets to store items and their counts
        self.n = 0              # Total number of processed items
            
    def process_item(self, item):
        self.n += 1

        if item in self.buckets:
            self.buckets[item] += 1
        else:
            self.buckets[item] = 1 + self.delta

        candidate_delta = self.n // self.k       # same as math.floor(n / k)
        # assert candidate_delta == math.floor(self.n / self.k)
        # assert (candidate_delta != self.delta) == (self.n % self.k == 0)
        if candidate_delta != self.delta:
            self.delta = candidate_delta

            # Cleanup: Remove items that have a low estimated frequency.
            for item in list(self.buckets):
                if self.buckets[item] < self.delta:
                    del self.buckets[item]


    def get_frequent_items(self, threshold):
        """
        Get items with an estimated frequency above a certain threshold.

        Args:
            threshold (int): Minimum frequency threshold (absolute count).

        Returns:
            dict: A dictionary of items and their estimated frequencies.
        """
        return {item: count for item, count in self.buckets.items() if count >= threshold}
    

    def get_n_most_frequent_items(self, n):
        """
        Get the n most frequent items.

        Args:
            n (int): Number of most frequent items to return.

        Returns:
            dict: A dictionary of the n most frequent items and their estimated frequencies.
        """
        return dict(sorted(self.buckets.items(), key=lambda x: x[1], reverse=True)[:n])
    

# Example usage
if __name__ == "__main__":
    # Simulated data stream
    book_paths_with_language_and_start_line_mark = [
        ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french', '====================================================='),
        ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german', 'http://gutenberg2000.de erreichbar.')
    ]

    words_list = read_text_to_word_list(*book_paths_with_language_and_start_line_mark[0])[:100]
    
    # k - Maximum number of buckets
    # threshold - Minimum frequency to be considered frequent
    k = 12
    threshold = 15
    lc = LossyCounting(k)
    olc = MugegenLossyCounting(1/k)
    for word in words_list:
        lc.process_item(word)
        olc.addCount(word)

    print("\n\nBuckets:", lc.buckets)
    print("k delta n:", lc.k, lc.delta, lc.n)
    print("frequent items:", lc.get_frequent_items(threshold))
    print("\n\nAnother one Buckets:", olc.count)
    print("epsilon b_current N:", olc.epsilon, olc.b_current, olc.N)
    print("frequent items:", [s for s in olc.iterateOverThresholdCount(threshold)])

    assert lc.buckets == olc.count
    print(lc.buckets == olc.count)
    
    # filtered_exact_count = {item: count for item, count in Counter(words_list).items() if count >= threshold}
    # print("\nFiltered exact count:", filtered_exact_count)

    # # Check if Lossy Counting underestimates, gets the exact count, or overestimates (should never underestimate)
    # underestimates = {}
    # exact_matches = {}
    # overestimates = {}
    # for item, exact_count in filtered_exact_count.items():
    #     estimated_count = lc.buckets.get(item, 0)
    #     estimated_tuple = (exact_count, estimated_count, estimated_count / exact_count)
    #     if estimated_count < exact_count:
    #         underestimates[item] = estimated_tuple
    #     elif estimated_count == exact_count:
    #         exact_matches[item] = estimated_tuple
    #     else:
    #         overestimates[item] = estimated_tuple

    # epsilon = 1/k
    # n = len(words_list)
    # assert lc.n == n
    # print("\nMax Possible Error: ", epsilon * n)
    # print("\nUnderestimates:", underestimates)
    # print("\nExact matches:", exact_matches)
    # print("\nOverestimates:", overestimates)


