#referencia: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/A2C%20with%20Sonic%20the%20Hedgehog/sonic_env.py

import numpy as np
import gym
from gym import spaces
import cv2
from skimage import transform
from collections import deque



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

		#corta as bordas da esquerda e da direita que sao totalmente pretas
		frame = frame[:,15:-15]

		#normalizar os frames nao precisa, pois o proprio transform.resize faz isso

		#da um resize para ficarmos com uma imagem menor
		#224X320 --> 84X84 (Tentei tamanhos maiores mais estava demorando muito o treino)
		#transformar em quadrado mais para simplficar a criacao dos kernels
		frame = transform.resize(frame, [84,84])

		return frame


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))



class FrameStack_FrameSkip(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action, passo):
    	if passo % 4 == 0:
    		ob, reward, done, info = self.env.step(action)
    		self.frames.append(ob)
    		return self._get_ob(), reward, done, info
    	else:
    		self.env.step(action)
    		return

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))



