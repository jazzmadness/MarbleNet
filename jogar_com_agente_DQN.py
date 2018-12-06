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
from arquitetura_rede_DQN import DQRede

print('Criando Ambiente...')

env = retro.make('MarbleMadness-Genesis', 'Level1_Sem_Timer_Subindo')

#modifica as imagens para sair em escala de cinza

env = TrataImg(env)

#modifica o step e adiciona o metodo para empilhar 4 frames dentro do env

env = FrameStack(env, 4)

#modifica o array de acoes possiveis para apenas necessarias

env = DiscretizadorAcoes(env)

with tf.Session() as sess:

	#inicia ambiente
	env.reset()

	#metodo para salvar o modelo
	saver = tf.train.Saver()

	# carrega o modelo
	saver.restore(sess, "./models/modelo_DQN_1.ckpt")

	done = False

	for i in range(1):
		
		while not done:
			env.render()
			estado_emp = np.stack(env.env.frames, axis = 2)
			#Pega a melhor acao
			Qs = sess.run(DQRede.saida, feed_dict = {DQRede.inputs: estados_emp})
			acao = np.argmax(Qs)
			env.step(acao)



