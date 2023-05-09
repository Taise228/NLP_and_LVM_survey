from transformers import pipeline


def main():
    classifier = pipeline('sentiment-analysis')

    texts = ['I am so excited.']

    results = classifier(texts)
    print(results)


if __name__ == '__main__':
    main()