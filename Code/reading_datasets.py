import pickle
import json
import csv

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_task(location, split = 'train'):
    filename = location + split + '.tsv'

    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i > 0:
                tweet_id = row[0]
                sentence = row[1].strip()
                label = row[4]
                topic = row[2]
                data.append((sentence, label, topic, tweet_id))

    # print("Length of data")
    # print(len(data))
 
    return data


if __name__ == '__main__':
    location = '../Datasets/'
    split = 'train'
    
    data = read_task(location, split)
    print(len(data))

    data = read_task(location, 'dev')
    print(len(data))

    




