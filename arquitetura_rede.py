import tensorflow as tf
import numpy as np
import random #para epsilon greedy
import time

class DQRede:
	'''
	dim_estado = dimensao das imagens a ser processadas (vai ser em escala de cinza nesse caso,
														com n imagens empilhadas)
	tamanho_acao = numero de acoes possiveis
	'''
	def __init__(self, dim_estado, tamanho_acao, learning_rate, nome = 'DQRede'):
		self.dim_estado = dim_estado
		self.tamanho_acao = tamanho_acao
		self.learning_rate = learning_rate
		#inicializa variaveis
		with tf.variable_scope(nome):
			self.inputs = tf.placeholder(tf.float32, [None, *dim_estado], name = "INnputs") #[None, dim1, dim2, dim3]
			self.acoes = tf.place_holder(tf.float32, [None, *tamanho_acao], name = "Acoes")
			'''Q-Target = R(s,a) (recompensa no estado 's', fazendo a acao 'a') 
			+ gamma*max{a}(Q_hat(s',a')) (maximo Q-Value entre todas as acoes, 
			no proximo estado apos uma acao)
			'''
			self.Q_target = tf.placeholder(tf.float32, [None], name = "Q-Target")