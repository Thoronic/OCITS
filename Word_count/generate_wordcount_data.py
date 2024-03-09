import re

def prepare_list():
    with open('Word_count/book.txt', 'r') as file:
        content = file.read()
    lines = content.split('\n')
    print(lines[0:10])

    text = ' '.join(lines)
    words = re.findall(r'\b\w+\b', text)
    words = [remove_non_alphabetic(word) for word in words if word]
    words = list(filter(lambda x: len(x) > 0, words))
    return words

def remove_non_alphabetic(word):
    return re.sub('[^a-zA-Z]', '', word)

def main():
    # Les Misérables by Victor Hugo https://www.gutenberg.org/files/135/135-0.txt'
    words = prepare_list()

    with open("Word_count/data.txt", 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(words))

main()
