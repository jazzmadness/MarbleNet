#referencia: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/A2C%20with%20Sonic%20the%20Hedgehog/sonic_env.py

import numpy as np
import gym
import cv2
from skimage import transform

#vamos por stack de frames para ter nocao de movimento. ja tem um metodo implementado em
#https://github.com/openai/baselines

#from baselines.common.atari_wrappers import FrameStack

class DiscretizadorAcoes(gym.ActionWrapper):
	'''
	Nao precisamos de todas as combinacoes de botoes que tem no gym.ActionWrapper
	Precisamos apenas dos direcionais, entao essa classe substitui um gym.ActionWrappper
	por um com as acoes necessarias
	'''
	def __init__(self, env):
		super(DiscretizadorAcoes, self).__init__(env) #chama o init do ActionWrapper
		botoes = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
		acoes = [
				["LEFT"],# ["LEFT", "UP"], ["LEFT", "DOWN"],
				["RIGHT"],# ["RIGHT", "UP"], ["RIGHT", "DOWN"],
				["UP"],# ["UP","LEFT"], ["UP", "RIGHT"],
				["DOWN"],# ["DOWN","LEFT"], ["DOWN", "RIGHT"],
				#A,B,C garante um turboboost...so preciso escolher um dos botoes ja que todos fazem a mesma coisa	
				["A","LEFT"],# ["A","LEFT", "UP"], ["A","LEFT", "DOWN"],
				["A","RIGHT"],# ["A","RIGHT", "UP"], ["A","RIGHT", "DOWN"],
				["A","UP"],# ["A","UP","LEFT"], ["A","UP", "RIGHT"],
				["A","DOWN"]#, ["A","DOWN","LEFT"], ["A","DOWN", "RIGHT"]
				]	
		self._actions = []
		'''
		Aqui, cria um array com arrays contendo os inputs que queremos
		Exemplo, a primeira acao, LEFT, sera adicionado no array um array
		[0,0,0,0,0,0,1,0,0,0,0,0]
		'''	
		for acao in acoes:
			arr = np.array([False] * 12)
			for botao in acao:
				arr[botoes.index(botao)] = True
			self._actions.append(arr)

		self.action_space = gym.spaces.Discrete(len(self._actions))

	'''
	Aqui retorna o array ao inves da acao discreta
	'''

	def action(self, a):
		return self._actions[a].copy()

	def acoes_discretizadas(self, a):
		arr_d = np.array([False] * 8)
		arr_d[a] = True
		return arr_d

'''
Modifica o Observation Wrapper para soltar imagens em escala de cinza
'''

class TrataImg(gym.ObservationWrapper):
	def __init__(self, env):
		super(TrataImg, self).__init__(env) #chama o init do ObservationWrapper

	def observation(self, frame):
		#converte para escala de cinza
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		#normaliza os pixels
		#norm_frame = frame/255.0

		#da um resize para ficarmos com uma imagem menor
		#224X320 -2 x downscale-> 112X160 - pega o meio entre os dois e transforma em quadrado -> 136x136
		#transformar em quadrado mais para simplficar a criacao dos kernels
		#resize_norm_frame = transform.resize(frame, [136,136])

		return frame


