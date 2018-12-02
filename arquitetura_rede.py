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
			self.inputs = tf.placeholder(tf.float32, [None, *dim_estado], name = "Inputs") #[None, dim1, dim2, dim3]
			self.acoes = tf.place_holder(tf.float32, [None, *tamanho_acao], name = "Acoes")
			'''Q-Target = R(s,a) (recompensa no estado 's', fazendo a acao 'a') 
			+ gamma*max{a}(Q_hat(s',a')) (maximo Q-Value entre todas as acoes, 
			no proximo estado apos uma acao)
			'''
			self.Q_target = tf.placeholder(tf.float32, [None], name = "Q-Target")
			"""
			Primeira Convolucao com Batch e Ativacao ELU
			Input: 224X320XN (N imagens empilhadas)
			"""
			self.conv1 = tf.layers.conv2d(inputs = self.inputs, #entrada
										  filters = 24, #numero de saidas de cada aplicadao do filtro
										  kernel_size = [6,6], #tamanho dos filtros a serem aplicados
										  strides = [2,2], #numero de "pulos" a cada aplicacao do filtro
										  padding = "VALID", #sem padding (ex:zero padding, adicionar zeros para nao reduzir a dimensionalidade)
										  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), #inicializador aleatorio dos pesos
										  name = "Conv1")
			self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
																 training = True, #normaliza com o batch, ao inves de usar estatitica movel
																 epsilon = 1e-5, #adicionar na variancia para nao dividir por zero
																 name = 'Batch_Norm1')
			self.conv1_saida = tf.nn.elu(self.conv1_batchnorm, name = "Conv1_Saida")
			'''
			Saida:110X158X24
			'''

			'''
			Primeiro Max Pooling
			'''

			self.max_pool_1 = tf.layers.max_pooling2d(inputs = self.conv1_saida,
    												  pool_size = [2,2],
    												  strides = [4,4],
    												  padding='VALID',
    												  name="Max_Pool_1")
			'''
			Saida:28X40X24
			'''

			'''
			Segunda Convolucao com Batch e Ativacao ELU
			'''
			self.conv2 = tf.layers.conv2d(inputs = self.max_pool_1, #entrada
										  filters = 32, #numero de saidas de cada aplicadao do filtro
										  kernel_size = [6,6], #tamanho dos filtros a serem aplicados
										  strides = [2,2], #numero de "pulos" a cada aplicacao do filtro
										  padding = "VALID", #sem padding (ex:zero padding, adicionar zeros para nao reduzir a dimensionalidade)
										  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), #inicializador aleatorio dos pesos
										  name = "Conv2")
			self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
																 training = True, #normaliza com o batch, ao inves de usar estatitica movel
																 epsilon = 1e-5, #adicionar na variancia para nao dividir por zero
																 name = 'Batch_Norm2')
			self.conv2_saida = tf.nn.elu(self.conv2_batchnorm, name = "Conv2_Saida")
			'''
			Saida:12X18X32
			'''

			'''
			Terceira Convolucao com Batch e Ativacao ELU
			'''
			self.conv3 = tf.layers.conv2d(inputs = self.conv2_saida, #entrada
										  filters = 128, #numero de saidas de cada aplicadao do filtro
										  kernel_size = [4,6], #tamanho dos filtros a serem aplicados
										  strides = [4,4], #numero de "pulos" a cada aplicacao do filtro
										  padding = "VALID", #sem padding (ex:zero padding, adicionar zeros para nao reduzir a dimensionalidade)
										  kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), #inicializador aleatorio dos pesos
										  name = "Conv3")
			self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
																 training = True, #normaliza com o batch, ao inves de usar estatitica movel
																 epsilon = 1e-5, #adicionar na variancia para nao dividir por zero
																 name = 'Batch_Norm3')
			self.conv3_saida = tf.nn.elu(self.conv3_batchnorm, name = "Conv3_Saida")
			'''
			Saida:3X4X128
			'''

			'''
			Flatten: Achata tudo num vetor unidimensional para entrar nas camadas densas
			'''

			self.flatten = tf.layers.flaten(self.conv3_saida)

			'''
			Saida:1536
			'''

			'''
			Camada totalmente conectada
			'''

			self.tc = tf.layers.dense(inputs = self.flatten,
									  units = 512, #512 neurons
									  activation = tf.nn.elu, #ativa com a elu
									  kernel_initializer = tf.contrib.layers.xavier_initializer(), #inicializador aleatorio dos pesos
									  name = "Tc1")

			'''
			Camada Final: Totalmente conectada, com saida sendo o numero de acoes possiveis
			'''

			self.saida = tf.layers.dense(inputs = self.tc,
										 kernel_initializer = tf.contrib.layers.xavier_initializer(), #inicializador aleatorio dos pesos
										 units = tamanho_acao #N saidas, onde N eh o numero de acoes possiveis
										 activation = None)

			#Q valor predito, eh a soma das 3 saidas vezes as 3 acoes

			self.Q = tf.reduce_sum(tf.multiply(self.saida, self.acoes), axis = 1)

			'''
			Perda: Media quadratica entre o Q-Value predito e o Q-Target
			(recompensa no estado 's', fazendo a acao 'a') 
			+ gamma*max{a}(Q_hat(s',a')) (maximo Q-Value entre todas as acoes, 
			no proximo estado apos uma acao)
			'''

			self.perda = tf.reduce_mean(tf.square(self.Q_target - self.Q))

			'''
			Otimiza usando o ADAM
			'''

			self.otimizador = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.perda)

			#metodo minimize faz o compute_gradients() e o apply_gradients()