import os

DATA_DIR = os.getcwd()
TRAIN_DATA_FILENAME = DATA_DIR + "/train-images-idx3-ubyte"
TEST_DATA_FILENAME = DATA_DIR + "/t10k-images-idx3-ubyte"
TRAIN_LABEL_FILENAME = DATA_DIR + "/train-labels-idx1-ubyte"
TEST_LABEL_FILENAME = DATA_DIR + "/t10k-labels-idx1-ubyte"

def byte_to_int(byte_data): 
    return int.from_bytes(byte_data, 'big')

def read_images(filename, max_images=None):
    images = []
    with open(filename, 'rb') as f: 
        # unnecessary number
        _ = f.read(4)
        n_images = byte_to_int(f.read(4))
        if max_images: 
            n_images = max_images
        n_rows = byte_to_int(f.read(4))
        n_columns = byte_to_int(f.read(4))
        for image_index in range(n_images): 
            image = []
            for row_index in range(n_rows): 
                row = []
                for col_index in range(n_columns): 
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
            print(len(images))
        
    return images

def read_labels(filename, max_labels=None):
    labels = []
    with open(filename, 'rb') as f: 
        # unnecessary number
        _ = f.read(4)
        n_labels= byte_to_int(f.read(4))
        if max_labels: 
            n_labels = max_labels
        for label_index in range(n_labels): 
            label = f.read(1)
            labels.append(label)

    return labels

# converting images from 2D to a single dimensional array
def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X): 
    return [flatten_list(sample) for sample in X]

def dist(x,y): 
    return sum(
        [
            (byte_to_int(x_i) - byte_to_int(y_i))**2 
            for x_i, y_i in zip(x,y)
        ]
    ) ** (0.5)

def get_train_distances(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def get_most_freq_element(l): 
    return max(l, key=l.count)

def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample in X_test: 
        training_distances = get_train_distances(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances), 
                key=lambda x : x[1]
            )
        ]
        candidates = [
            byte_to_int(y_train[index])
            for index in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_freq_element(candidates)
        # print(candidates)
        # print(sorted_distance_indices)
        y_pred.append(top_candidate)
        print("y-pred: " + str(len(y_pred)))
    return y_pred

def main(): 

	testing_count = 10000
    X_train = read_images(TRAIN_DATA_FILENAME)
    y_train = read_labels(TRAIN_LABEL_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME, testing_count)
    y_test = read_labels(TEST_LABEL_FILENAME, testing_count)

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, y_train, X_test, 3)

    accuracy = sum([
        int(y_pred_i == byte_to_int(y_test_i))
        for y_pred_i, y_test_i in zip(y_pred, y_test)
    ]) / len(y_test)

    # print(y_pred)
    print(accuracy)

if __name__ == "__main__": 
    main()
