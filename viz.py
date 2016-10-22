import socket
import matplotlib.pyplot as plt
import matplotlib
import re
import time


address = ('', 5274)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(address)


def Loop():
    reobj = re.compile(r"Iter = ([\d]+), Loss = ([\d]+[.][\d]*)")
    cnt = 0
    matplotlib.rcParams.update({'font.size': 6})
    fig = plt.figure(figsize=(10, 4), dpi=150, facecolor='black')
    fig.suptitle('Training Loss of LSTM-CNN')
    plt.ion()
    ax = fig.add_subplot(1, 1, 1, axisbg='black')
    Iter = []
    Loss = []
    ax.set_title('training loss of LSTM-CNN')
    ax.set_xlabel('iter (* 500)')
    ax.set_ylabel('Loss')
    plt.show()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.title.set_color('yellow')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='both', direction='in')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_title('training loss of LSTM-CNN')
    ax.set_xlabel('iter (* 500)')
    ax.set_ylabel('Loss')
    ax.plot(Iter, Loss, color='yellow')
    plt.pause(1)

    while True:
        data, addr = s.recvfrom(2048)
        cnt += 1
        match = reobj.search(data.decode())
        iter = int(match.group(1))
        loss = float(match.group(2))
        iter = iter / 500
        print(iter)
        print(loss)
        Iter += [iter]
        Loss += [loss]

        ax.cla()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.title.set_color('yellow')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='both', direction='in')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        ax.set_title('training loss of LSTM-CNN')
        ax.set_xlabel('iter (* 500)')
        ax.set_ylabel('Loss')
        ax.plot(Iter, Loss, color='yellow')
        plt.pause(1)

Loop()
s.close()
