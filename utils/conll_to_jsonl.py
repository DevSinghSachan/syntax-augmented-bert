import os
from collections import Counter
import jsonlines


def save_to_json(data, out_file_path):
    with jsonlines.open(out_file_path, 'w') as writer:
        for line in data:
            writer.write(line)


def postprocess(file_path, out_file_path, split_type):
    data = []
    c = Counter()
    num_ent, num_tokens, num_sent = 0, 0, 0
    with open(file_path) as fp:
        words, tags, pos, dep_head, dep_rel = [], [], [], [], []
        error = False
        for line in fp:
            line = line.strip()
            if line != '':
                items = line.split('\t')
                try:
                    words.append(items[1])
                    pos.append(items[3])
                    dep_head.append(int(items[6]))
                    dep_rel.append(items[7])
                    tags.append(items[10])
                    num_tokens += 1
                    if 'B' in items[10]:
                        num_ent += 1
                    c.update([items[10]])
                except ValueError:
                    print("ValueError")
                    error = True
            else:
                if not error:
                    num_sent += 1
                    data.append({'token': words, 'tags': tags, 'pos': pos, 'head': dep_head, 'deprel': dep_rel})
                else:
                    error = False
                words, tags, pos, dep_head, dep_rel = [], [], [], [], []
    print(f'{split_type}, # Entities: {num_ent}, # Tokens: {num_tokens}, # Sentences: {num_sent}')
    print(c)
    save_to_json(data, out_file_path)


if __name__ == "__main__":
    train_path = os.environ['NER_TRAIN_DATA_PATH']
    valid_path = os.environ['NER_VALIDATION_DATA_PATH']
    test_path = os.environ['NER_TEST_DATA_PATH']

    postprocess(train_path, "train.json", 'Train')
    postprocess(valid_path, "dev.json", 'Validation')
    postprocess(test_path, "test.json", 'Test')
