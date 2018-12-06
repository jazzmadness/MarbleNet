#bibliotecas

import tensorflow as tf
import numpy as np
import retro
import time
import cv2
from baselines.common.atari_wrappers import FrameStack
import matplotlib.pyplot as plt

#custom
from wrappers import *
from memoria import Mem
from epsilon_greedy import eg
from arquitetura_rede_DDQN import DDQRede

print('Criando Ambiente...')

env = retro.make('MarbleMadness-Genesis', 'Level1_Sem_Timer_Subindo')

#modifica as imagens para sair em escala de cinza

env = TrataImg(env)

#modifica o step e adiciona o metodo para empilhar 4 frames dentro do env

frames_empilhados = 4
env = FrameStack(env, frames_empilhados)

#modifica o array de acoes possiveis para apenas necessarias

env = DiscretizadorAcoes(env)
	
#inicia ambiente
env.reset()

tamanho_acao = env.action_space.n
learning_rate = 0.0005
dim_estado = [*env.env.frames[0].shape, frames_empilhados] #4 frames empilhados de 136X136


###############################

#reseta o grafo (limpa alguma variavel, placeholder, etc)
tf.reset_default_graph()

print('Instanciando a Rede...')

#define a rede
DQRede = DDQRede(dim_estado, tamanho_acao, learning_rate)

with tf.Session() as sess:

	#metodo para salvar o modelo
	saver = tf.train.Saver()

	# carrega o modelo
	saver.restore(sess, "./models/modelo_DDQN_1.ckpt")

	done = False

	for i in range(1):
		
		while not done:
			env.render()
			estados_emp = np.stack(env.env.frames, axis = 2)
			#Pega a melhor acao
			Qs = sess.run(DQRede.saida, feed_dict = {DQRede.inputs: estados_emp.reshape((1, *estados_emp.shape))})
			#print(Qs)
			acao = np.argmax(Qs)
			#print(acao)
			env.step(acao)



