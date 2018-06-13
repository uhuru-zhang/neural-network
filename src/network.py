# coding=utf-8
import random

import numpy
"""
该模型为一个 “前馈学习神经网络” 实现了 “随机梯度下降算法”。
梯度通过反向传播计算。注意此处着重于代码简单，易于阅读和修改。
他并不是最优的，并且忽略了许多期望的特征。
"""

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes  # 代表神经网络的大小 例如 [3, 2, 2，3] 代表三个输入，第一层有两个神经元，第二层有两个神经元，有三个输出
        self.biases = [numpy.random.randn(b_num, 1) for b_num in sizes[1:]]

        # 同一个神经网络中，在某一层上，存在多少行取决于后者，存在多少列，取决于前轴
        self.weights = [numpy.random.randn(row_num, col_num)
                        for row_num, col_num in zip(sizes[1:], sizes[:-1])]

    def feed_forward(self, a):
        """
        一个神经网络的计算过程
        :param a: 输入数据
        :return:
        """
        for w, b in zip(self.weights, self.biases):
            # 每一次迭代向前推进一层
            a = sigmoid(numpy.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        梯度下降算法核心部分

        :param training_data: 训练数据（x，y）
        :param epochs: 训练次数
        :param mini_batch_size: 随机测试数据集大小
        :param eta:
        :param test_data:
        :return:
        """
        n_test = len(test_data) if test_data else 0

        n = len(training_data)
        for j in xrange(epochs):
            # 随机化训练集
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]

            # 梯度下降
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 计算梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    # TODO 此处以下需要重写
    def backprop(self, x, y):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(numpy.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
