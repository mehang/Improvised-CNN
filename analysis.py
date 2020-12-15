import pickle
import matplotlib.pyplot as plt
import statistics as stat

path = "../graph/Cats vs Dogs/15*15/"
filter_size = 15
NORMAL = "without gabor"
GABOR_RANDOM = "with gabor random"
GABOR_SAME = "with gabor same"
GABOR_ALTERNATE = "with gabor alternate"

display_loss = False
display_val_loss = False
display_acc = False
display_val_acc = False
display_train_time = False
display_min_loss = True
display_max_acc = True
display_total_epoch = True


histories = {NORMAL:[], GABOR_RANDOM:[], GABOR_SAME:[], GABOR_ALTERNATE:[]}
for iteration in range(1, 11):
    with open(path+"without gabor/{}/cvd-{}-history-kernel-{}.p".format(iteration,iteration, filter_size), 'rb') as fp:
        histories[NORMAL].append(pickle.load(fp))

for iteration in range(1, 11):
    with open(path+"with gabor random/{}/cvd-{}-history-gabor-random-kernel-{}.p".format(iteration,iteration, filter_size), 'rb') as fp:
        histories[GABOR_RANDOM].append(pickle.load(fp))

for iteration in range(1, 11):
    with open(path+"with gabor same/{}/cvd-{}-history-gabor-same-kernel-{}.p".format(iteration,iteration, filter_size), 'rb') as fp:
        histories[GABOR_SAME].append(pickle.load(fp))

for iteration in range(1, 11):
    with open(path+"with gabor alternate/{}/cvd-{}-history-gabor-alternate-kernel-{}.p".format(iteration,iteration, filter_size), 'rb') as fp:
        histories[GABOR_ALTERNATE].append(pickle.load(fp))

if display_loss:
    for i in range(1, 11):
        fig = plt.figure(figsize=(10,6))
        plt.plot(histories[NORMAL][i-1]['loss'], '#01579b')
        plt.plot(histories[GABOR_RANDOM][i-1]['loss'], '#f44336')
        plt.plot(histories[GABOR_SAME][i-1]['loss'], '#4caf50')
        plt.plot(histories[GABOR_ALTERNATE][i-1]['loss'], '#000000')
        plt.title('Model Loss Progress - {}'.format(i))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
        plt.show()

if display_val_loss:
    for i in range(1, 11):
        fig = plt.figure(figsize=(10,6))
        plt.plot(histories[NORMAL][i-1]['val_loss'], '#01579b')
        plt.plot(histories[GABOR_RANDOM][i-1]['val_loss'], '#f44336')
        plt.plot(histories[GABOR_SAME][i-1]['val_loss'], '#4caf50')
        plt.plot(histories[GABOR_ALTERNATE][i-1]['val_loss'], '#000000')
        plt.title('Model Validation Loss Progress - {}'.format(i))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
        plt.show()

if display_acc:
    for i in range(1, 11):
        fig = plt.figure(figsize=(10,6))
        plt.plot(histories[NORMAL][i-1]['accuracy'], '#01579b')
        plt.plot(histories[GABOR_RANDOM][i-1]['accuracy'], '#f44336')
        plt.plot(histories[GABOR_SAME][i-1]['accuracy'], '#4caf50')
        plt.plot(histories[GABOR_ALTERNATE][i-1]['accuracy'], '#000000')
        plt.title('Model Accuracy Progress - {}'.format(i))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
        plt.show()

if display_val_acc:
    for i in range(1, 11):
        fig = plt.figure(figsize=(10,6))
        plt.plot(histories[NORMAL][i-1]['val_accuracy'], '#01579b')
        plt.plot(histories[GABOR_RANDOM][i-1]['val_accuracy'], '#f44336')
        plt.plot(histories[GABOR_SAME][i-1]['val_accuracy'], '#4caf50')
        plt.plot(histories[GABOR_ALTERNATE][i-1]['val_accuracy'], '#000000')
        plt.title('Model Validation Accuracy Progress - {}'.format(i))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
        plt.show()

if display_train_time:
    fig = plt.figure(figsize=(10, 6))
    plt.plot([hist['train_time'] for hist in histories[NORMAL]], '#01579b')
    plt.plot([hist['train_time'] for hist in histories[GABOR_RANDOM]], '#f44336')
    plt.plot([hist['train_time'] for hist in histories[GABOR_SAME]], '#4caf50')
    plt.plot([hist['train_time'] for hist in histories[GABOR_ALTERNATE]], '#000000')
    plt.title('Model Train Time')
    plt.ylabel('Train Time')
    plt.xlabel('Experiment')
    plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
    plt.show()


if display_min_loss:
    min_loss = [min(hist['loss']) for hist in histories[NORMAL]]
    print("average minimum loss without gabor: {}, standard deviation: {}".format(stat.mean(min_loss), stat.stdev(min_loss)))
    fig = plt.figure(figsize=(10, 6))
    plt.scatter([i+1 for i in range(10)],min_loss, c='#01579b', s=250, alpha=0.5, marker=".", label="normal")

    min_loss = [min(hist['loss']) for hist in histories[GABOR_RANDOM]]
    print("average minimum loss with gabor random: {}, standard deviation: {}".format(stat.mean(min_loss), stat.stdev(min_loss)))
    plt.scatter([i + 1 for i in range(10)], min_loss, c='#f44336', s=125, alpha=0.5, marker="^", label="random")

    min_loss = [min(hist['loss']) for hist in histories[GABOR_SAME]]
    print("average minimum loss with gabor same: {}, standard deviation: {}".format(stat.mean(min_loss), stat.stdev(min_loss)))
    plt.scatter([i + 1 for i in range(10)], min_loss, c='#4caf50', s=100, alpha=0.5, marker="*", label="same")

    min_loss = [min(hist['loss']) for hist in histories[GABOR_ALTERNATE]]
    print("average minimum loss with gabor alternate: {}, standard deviation: {}".format(stat.mean(min_loss), stat.stdev(min_loss)))
    plt.scatter([i + 1 for i in range(10)], min_loss, c='#000000', s=100, alpha=0.5, marker="+", label="alternate")

    plt.title('Average minimum loss on different experiment')
    plt.ylabel('Average minimum loss')
    plt.xlabel('Experiment')
    plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
    plt.show()

if display_max_acc:
    max_accuracy = [max(hist['accuracy']) for hist in histories[NORMAL]]
    print("average maximum accuracy without gabor: {}, standard deviation: {}".format(stat.mean(max_accuracy), stat.stdev(max_accuracy)))
    fig = plt.figure(figsize=(10, 6))
    plt.scatter([i+1 for i in range(10)],max_accuracy, c='#01579b', s=250, alpha=0.5, marker=".", label="normal")

    max_accuracy = [max(hist['accuracy']) for hist in histories[GABOR_RANDOM]]
    print("average maximum accuracy with gabor random: {}, standard deviation: {}".format(stat.mean(max_accuracy), stat.stdev(max_accuracy)))
    plt.scatter([i + 1 for i in range(10)], max_accuracy, c='#f44336', s=150, alpha=0.5, marker="^", label="random")

    max_accuracy = [max(hist['accuracy']) for hist in histories[GABOR_SAME]]
    print("average maximum accuracy with gabor same: {}, standard deviation: {}".format(stat.mean(max_accuracy), stat.stdev(max_accuracy)))
    plt.scatter([i + 1 for i in range(10)], max_accuracy, c='#4caf50', s=100, alpha=0.5, marker="*", label="same")

    max_accuracy = [max(hist['accuracy']) for hist in histories[GABOR_ALTERNATE]]
    print("average maximum accuracy with gabor alternate: {}, standard deviation: {}".format(stat.mean(max_accuracy), stat.stdev(max_accuracy)))
    plt.scatter([i + 1 for i in range(10)], max_accuracy, c='#000000', s=100, alpha=0.5, marker="+", label="alternate")

    plt.title('Average maximum accuracy on different experiment')
    plt.ylabel('Average maximum accuracy')
    plt.xlabel('Experiment')
    plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
    plt.show()

if display_total_epoch:
    total_epochs = [len(hist['accuracy']) for hist in histories[NORMAL]]
    print("average total epoch without gabor: {}, standard deviation: {}".format(stat.mean(total_epochs), stat.stdev(total_epochs)))
    fig = plt.figure(figsize=(10, 6))
    plt.scatter([i+1 for i in range(10)], total_epochs, c='#01579b', s=250, alpha=0.5, marker=".", label="normal")

    total_epochs = [len(hist['accuracy']) for hist in histories[GABOR_RANDOM]]
    print("average total epoch with gabor random: {}, standard deviation: {}".format(stat.mean(total_epochs), stat.stdev(total_epochs)))
    plt.scatter([i + 1 for i in range(10)], total_epochs, c='#f44336', s=150, alpha=0.5, marker="^", label="random")

    total_epochs = [len(hist['accuracy']) for hist in histories[GABOR_SAME]]
    print("average total epoch with gabor same: {}, standard deviation: {}".format(stat.mean(total_epochs), stat.stdev(total_epochs)))
    plt.scatter([i + 1 for i in range(10)], total_epochs, c='#4caf50', s=100, alpha=0.5, marker="*", label="same")

    total_epochs = [len(hist['accuracy']) for hist in histories[GABOR_ALTERNATE]]
    print("average total epoch with gabor alternate: {}, standard deviation: {}".format(stat.mean(total_epochs), stat.stdev(total_epochs)))
    plt.scatter([i + 1 for i in range(10)], total_epochs, c='#000000', s=100, alpha=0.5, marker="+", label="alternate")

    plt.title('Total epoch on different experiment')
    plt.ylabel('Total epoch')
    plt.xlabel('Experiment')
    plt.legend(['normal', 'random', 'same', 'alternate'], loc='upper right')
    plt.show()






