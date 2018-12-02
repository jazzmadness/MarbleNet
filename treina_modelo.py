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

#modifica as imagens para sair em escala de cinza

env = TrataImg(env)

#modifica o step e adiciona o metodo para empilhar 4 frames dentro do env

env = FrameStack(env, 4)

#modifica o array de acoes possiveis para apenas necessarias

env = DiscretizadorAcoes(env)

'''
exemplo:
ob,rew,done,info = env.step(env.action_space.sample())
ob vai ser um lazyframes disso, mas nao precisamos usar
vamos usar o env.env.frames (nosso deque que eh sempre alimentado com nova obsevacao, e exclui a mais antiga)
e epmilhar ele com np.stack(env.env.frames, axis = 2)
'''

#########hiperparametros#########:

#modelo
dim_estado = [224, 320, 4] #4 frames empilhados de 224x320
tamanho_acao = env.action_space.n
learning_rate = 0.0005

#treino
numero_episodios = 500
tamanho_batch = 64

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

###############################

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
		#acao inicial
		acao_disc = env.action_space.sample()
		ob,rew,done,info = env.step(acao_disc)
		acao_array = env.action(acao_disc)
		estado_emp = np.stack(env.env.frames, axis = 2)

	#proxima acao
	prox_acao_disc = env.action_space.sample()
	prox_ob, prox_rew, prox_done, prox_info = env.step(prox_acao_disc)
	prox_acao_array = env.action(prox_acao_disc)
	prox_estado_emp = np.stack(env.env.frames, axis = 2)
	#adiciona experiencia
	memoria.add(estado_emp, acao_array, rew, prox_estado_emp, done)
	#atualizando as coisas atuais para rodar mais um futuro passo:
	#estado atual recebe o proximo estado
	estado_emp = prox_estado_emp
	#acao atual recebe a proxima acao
	acao_array = prox_acao_array
	#recomepnsa atual recebe prox recompensa
	rew = prox_rew
	#done atual recebe proximo done
	done = prox_done

#reenicia o ambiente (agora vamos comecar de verdade)
env.reset()

'''
configura o tensorboard
nao tem como salvar muita coisa, pois nao temos metrica para comparacao, pois nao temos dados,
e as imagens vao ser empilhadas, entao nao vai dar para ver muita coisa
podemos apenas ver a perda
'''

#tensorboard --logdir=/tensorboard/dqn/1

writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", DQRede.loss)
write_op = tf.summary.merge_all()

#TREINO

#acoes_possiveis = [np.array(env.action(i)) for i in range(0, env.action_space.n)]

#metodo para salvar o modelo
saver = tf.train.Saver()

with tf.Session() as sess:
	#inicializa as vars
	sess.run(tf.global_variables_initializer())

	#inicializa o decaimento
	passo_decay = 0

	for episodio in range(numero_episodios):
		recompensas_episodio = []

		#acao inicial
		acao_disc, prob_exp = ep(env, prob_inicial, min_prob, tx_decay, passo_decay, estado)
		ob,rew,done,info = env.step(acao_disc)
		acao_array = env.action(acao_disc)
		estado_emp = np.stack(env.env.frames, axis = 2)
		recompensas_episodio.append(rew)
		#memoria.add(estado_emp, acao_array, rew, prox_estado_emp, done)
		passo_decay += 1

		#entra em loop ate acabar
		while not done:
			#escolhe ou exploracao ou abusar do que ja sabe pelo epsilon greedy
			prox_acao_disc, prox_prob_exp = ep(env, prob_inicial, min_prob, tx_decay, passo_decay, estado)
			prox_ob,prox_rew,prox_done,prox_info = env.step(acao_disc)
			prox_acao_array = env.action(acao_disc)		
			prox_estado_emp = np.stack(env.env.frames, axis = 2)
			recompensas_episodio.append(prox_rew)
			memoria.add(estado_emp, acao_array, rew, prox_estado_emp, done)
			#atualizando as coisas atuais para rodar mais um futuro passo:
			#estado atual recebe o proximo estado
			estado_emp = prox_estado_emp
			#acao atual recebe a proxima acao
			acao_array = prox_acao_array
			#recomepnsa atual recebe prox recompensa
			rew = prox_rew
			#done atual recebe proximo done
			done = prox_done

			##############SGD##############
			#Sera realizado um aprendizado com um mini batch de experiencias que guardamos
			mini_batch = memoria.amostra(tamanho_batch)

			estados_mb = np.array([mb[0] for mb in mini_batch], ndim = 3)
			acoes_mb = np.array([mb[1] for mb in mini_batch])
			recompensas_mb = np.array([mb[2] for mb in mini_batch])
			prox_estados_mb = np.array([mb[3] for mb in mini_batch], ndim = 3)
			dones_mb = np.array([mb[4] for mb in mini_batch])
			Q_targets_mb = []

			# Pega Valores Q do prox estado

			Qs_prox_estado = sess.run(DQRede.output, feed_dict = {DQRede.inputs: prox_estados_mb})

			# Q Target = R se episodio terminou em s+1, senao Q_target = R + gamma*max{a}(Q_hat(s',a'))

			for i in range(0, len(batch)):
				terminou = dones_mb[i]

				if terminou:
					Q_targets_mb.append(recompensas_mb[i])
				else:
					target = recompensas_mb[i] + gamma * np.max(Qs_prox_estado[i])
					Q_targets_mb.append(target)

			targets_mb = np.array([mb for mb in Q_targets_mb])
			perda, _ = sess.run([DQRede.perda, DQRede.otimizador],
                                    feed_dict={DQRede.inputs: estados_mb,
                                               DQRede.Q_target: targets_mb,
                                               DQRede.acoes: acoes_mb})
			#manda para o TensorBoard
			sumario = sess.run(write_op, 
            						feed_dict={DQNetwork.inputs: estados_mb,
                                                   DQNetwork.Q_target: targets_mb,
                                                   DQNetwork.acoes: acoes_mb})
			writer.add_summary(sumario, episodio)
			writer.flush()

		#a cada 5 episodios salva o modelo
		if episodio % 5 == 0:
			save_path = saver.save(sess, "./models/modelo_DQN_1.ckpt")
			print("Modelo Salvo!")

		recomepensa_total = np.sum(recompensas_episodio)
		print('Episodio: {}'.format(episodio),
                              'Recompensa Total: {}'.format(recomepensa_total),
                              'Perda: {:.4f}'.format(perda),
                              'Prob. Exploracao: {:.4f}'.format(prob_exp))
		#reseta o ambiente e comeca outro episodio
		print('Acabou episodio, resetando ambiente...')
		env.reset()








