from collections import Counter
from book_file_reader import read_text_to_word_list
from lossy_count_to_verify import LossyCountOther


class LossyCounting:
    def __init__(self, k):
        """
        Initialize the Lossy Counting algorithm.

        Args:
            k (int): The maximum number of buckets (inverse of epsilon).
        """
        self.k = k              # Maximum number of buckets
        self.delta = 0          # Current bucket boundary
        self.buckets = {}       # Buckets to store items and their counts
        self.n = 0              # Total number of processed items

    def process_stream(self, items):
        """
        Process the stream of items.

        Args:
            items (list): A list of items to be processed.
        """
        for item in items:
            self.n += 1

            if item in self.buckets:
                self.buckets[item] += 1
            else:
                self.buckets[item] = 1 + self.delta

            candidate_delta = self.n // k       # same as math.floor(n / k)
            if candidate_delta != self.delta:
                self.delta = candidate_delta

                # Cleanup: Remove items that have a low estimated frequency.
                for item in list(self.buckets):
                    if self.buckets[item] <= self.delta:
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
    

# Example usage
if __name__ == "__main__":
    # Simulated data stream
    book_paths_with_language = [
        ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french'),
        ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german')
    ]

    file_path, language = book_paths_with_language[0]
    words_list = read_text_to_word_list(file_path, language)
    
    # k - Maximum number of buckets
    # threshold - Minimum frequency to be considered frequent
    k = 500
    threshold = 100
    lc = LossyCounting(k)
    lc.process_stream(words_list)
    print("\n\nBuckets:", lc.buckets)
    print("\nFrequent items:", lc.get_frequent_items(threshold))
    print("\nTotal number of items processed:", lc.n)

    # alc = LossyCountOther(1/k)
    # for word in words_list:
    #     print(alc.entries)
    # print("\n\nAnother lossy count not implemented by me")
    # print("\nBuckets:", alc.entries)
    # print("\nFrequent items:", alc.get(threshold))
    # print("\nTotal number of items processed:", alc.n)


    filtered_exact_count = {item: count for item, count in Counter(words_list).items() if count >= threshold}
    print("\nFiltered exact count:", filtered_exact_count)

    # Check if Lossy Counting underestimates, gets the exact count, or overestimates (should never underestimate)
    underestimates = {}
    exact_matches = {}
    overestimates = {}
    for item, exact_count in filtered_exact_count.items():
        estimated_count = lc.buckets.get(item, 0)
        estimated_tuple = (exact_count, estimated_count, estimated_count / exact_count)
        if estimated_count < exact_count:
            underestimates[item] = estimated_tuple
        elif estimated_count == exact_count:
            exact_matches[item] = estimated_tuple
        else:
            overestimates[item] = estimated_tuple

    epsilon = 1/k
    n = len(words_list)
    assert lc.n == n
    print("\nMax Possible Error: ", epsilon * n)
    print("\nUnderestimates:", underestimates)
    print("\nExact matches:", exact_matches)
    print("\nOverestimates:", overestimates)


