#referencia: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/A2C%20with%20Sonic%20the%20Hedgehog/sonic_env.py

import numpy as np
import gym

#vamos por stack de frames para ter nocao de movimento. ja tem um metodo implementado em
#https://github.com/openai/baselines

from baselines.common.atari_wrappers import FrameStack

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
				["LEFT"], ["LEFT", "UP"], ["LEFT", "DOWN"],
				["RIGHT"], ["RIGHT", "UP"], ["RIGHT", "DOWN"],
				["UP"], ["UP","LEFT"], ["UP", "RIGHT"],
				["DOWN"], ["DOWN","LEFT"], ["DOWN", "RIGHT"],
				#A,B,C garante um turboboost...so preciso escolher um dos botoes ja que todos fazem a mesma coisa	
				["A","LEFT"], ["A","LEFT", "UP"], ["A","LEFT", "DOWN"],
				["A","RIGHT"], ["A","RIGHT", "UP"], ["A","RIGHT", "DOWN"],
				["A","UP"], ["A","UP","LEFT"], ["A","UP", "RIGHT"],
				["A","DOWN"], ["A","DOWN","LEFT"], ["A","DOWN", "RIGHT"]
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

	def action(self, a):
		return self._actions[a].copy()




