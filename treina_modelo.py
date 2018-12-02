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
from epislon_greedy import eg
from arquitetura_rede import DQRede

#inicia o ambiente

#importei a rom com o comando:

#python3 -m retro.import caminho/do/arquivo/marble_madness.md

env = retro.make('MarbleMadness-Genesis', 'Level1')

#modifica o array de acoes possiveis para apenas necessarias

env = wrappers.DiscretizadorAcoes(env)

#modifica as imagens para sair em escala de cinza

env = wrappers.TrataImg(env)

#modifica o step e adiciona o metodo para empilhar 4 frames dentro do env

env = FrameStack(env, 4)

#########hiperparametros#########:

#modelo
dim_estado = [224, 320, 4] #4 frames empilhados de 224x320
tamanho_acao = env.action_space.n
learning_rate = 0.0005

#treino
numero_episodios = 500
passo_decay_max = 100
batch = 64

#epsilon greedy
prob_inicial = 1
min_prob = 0.01
tx_decay = 0.0001

#Q learning
'''
delta(w) = alpha[(R + gamma*max{a}(Q_hat(s',a')) - Q_hat(s,a))] * nabla{w}(Q_hat(s,a))
w = pesos da rede
alpha = learning rate
gamma = taxa de desconto do modelo para o proximo passo
nabla{w} = gradiente do valor q atual em relacao ao peso
Q_hat = Valor Q predito pela rede neural
'''
gamma = 0.95

#memoria
pretrain = batch 	#numero de experiencias para guardar quando inicia o agente pela primeira vez 
					#(precisamos de dados para comecar)
tamanho_memoria = 1e6

#reseta o grafo (limpa alguma variavel, placeholder, etc)
tf.reset_default_graph()

#define a rede
DQRede = DQRede(dim_estado, tamanho_acao, learning_rate)

#cria e popula a memoria
memoria = Mem(buf = tamanho_memoria)

#inicia
env.reset()

#popula a memoria

for i in range(pretrain):
	if i == 0:



