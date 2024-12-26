import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

def read_text_to_word_list(file_path, language, start_line_mark='*** START OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET ***'):
    supported_languages = ['english', 'french', 'german']
    assert language in supported_languages, f"Language must be one of {supported_languages}"

    # Load stop words
    stop_words = set(stopwords.words(language))

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the start and end of the actual text
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_line_mark in line:
            start_idx = i + 1
        elif '*** END OF THE PROJECT GUTENBERG EBOOK' in line:
            end_idx = i
            break

    # Extract the text body
    text_lines = lines[start_idx:end_idx]

    # Clean and process the text
    word_list = []
    for line in text_lines:
        # Remove punctuation and convert to lowercase
        line = line.translate(str.maketrans('', '', string.punctuation)).lower()
        # Remove stop words
        words = [word for word in line.split() if word not in stop_words]
        word_list.extend(words)

    return word_list

if __name__ == '__main__':
    # start_line_mark is needed because traducers often leave a description/message (not needed for the english original book)
    # the whitespace is on purpose
    book_paths_with_language_and_start_line_mark = [
        ('romeo_and_juliet_book/romeo_and_juliet_english.txt', 'english'),
        ('romeo_and_juliet_book/romeo_and_juliet_french.txt', 'french', '====================================================='),
        ('romeo_and_juliet_book/romeo_and_juliet_german.txt', 'german', 'http://gutenberg2000.de erreichbar.')
    ]

    words_list = read_text_to_word_list(*book_paths_with_language_and_start_line_mark[2])

    output_path = 'processed_words.txt'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(words_list))
    print(f"Processed words saved to {output_path}")
    print(f'Number of words: {len(words_list)}')
    print(f'First 30 words: {words_list[:30]}')
