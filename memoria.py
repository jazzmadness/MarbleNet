import numpy as np
from collections import deque

'''
Cria um deque, i.e, um array de tamanho maximo definido pelo buffer, que quando eh adicionado um elemento novo,
ele substitui o elemento mais antigo do array.

A memoria eh usada para guardar experiencias para rodar o Stochastic Gradient Descent
'''

class Mem():
	def __init__(self, buf):
		self.buffer = deque(maxlen = buf)

	def add(self, exp):
		self.buffer.append(exp)
	def amostra(self, batch):
		return [self.buffer[i] for i in np.random.choice(
														np.arrange(
																	len(self.buffer)
																	), 
														size = batch,
														replace = False
														)
				]