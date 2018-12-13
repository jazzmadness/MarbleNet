#bibliotecas

import tensorflow as tf
import numpy as np
import retro
import time
import cv2
import gc
from baselines.common.atari_wrappers import FrameStack
import matplotlib.pyplot as plt
import warnings # ignora os warnings do skimage
warnings.filterwarnings('ignore')

#custom
from wrappers import *
from memoria import Mem
#from epsilon_greedy import eg
from arquitetura_rede_DDQN_v2 import DDQRede

#inicia o ambiente

#importei a rom com o comando:

#python3 -m retro.import caminho/do/arquivo/marble_madness.md

print('Criando Ambiente...')

env = retro.make('MarbleMadness-Genesis', 'Level1_Sem_Timer_Subindo')

#modifica as imagens para sair em escala de cinza

env = TrataImg(env)

#modifica o step e adiciona o metodo para empilhar 4 frames dentro do env

frames_empilhados = 4

env = FrameStack(env, frames_empilhados)

#modifica o array de acoes possiveis para apenas necessarias

env = DiscretizadorAcoes(env)

print('OK')

print('Resetando Ambiente...')

#inicia
env.reset()

print('OK')

env.step(env.action_space.sample())


'''
exemplo:
ob,rew,done,info = env.step(env.action_space.sample())
ob vai ser um lazyframes disso, mas nao precisamos usar
vamos usar o env.env.frames (nosso deque que eh sempre alimentado com nova obsevacao, e exclui a mais antiga)
e epmilhar ele com np.stack(env.env.frames, axis = 2)
'''

#########hiperparametros#########:

print('Criando Hiperparametros...')

#modelo
dim_estado = [*env.env.frames[0].shape, frames_empilhados] #4 frames empilhados de 84X84
tamanho_acao = env.action_space.n #8 acoes
learning_rate = 0.0005

#treino
numero_episodios = 500
tamanho_batch = 64

#epsilon greedy
prob_inicial = 1.0
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
pretrain = tamanho_batch 	#numero de experiencias para guardar quando inicia o agente pela primeira vez 
					#(precisamos de dados para comecar)
tamanho_memoria = 25000 #estava um milhao, diminiui pois com 8gb de RAM nao tava aguentando

print('OK')

###############################

#reseta o grafo (limpa alguma variavel, placeholder, etc)
tf.reset_default_graph()

print('Instanciando a Rede...')

#define a rede
DQRede = DDQRede(dim_estado, tamanho_acao, learning_rate)

print('OK')

print('Criando Memoria...')

#cria e popula a memoria
memoria = Mem(buf = tamanho_memoria)

print('OK')

#popula a memoria

print('Populando a memoria...')

for i in range(pretrain):
	
	if i == 0: 
		#acao inicial
		acao_disc = env.action_space.sample()
		ob,rew,done,info = env.step(acao_disc)
		acao_array = env.acoes_discretizadas(acao_disc)
		estado_emp = np.stack(env.env.frames, axis = 2)

	#proxima acao
	prox_acao_disc = env.action_space.sample()
	prox_ob, prox_rew, prox_done, prox_info = env.step(prox_acao_disc)
	prox_acao_array = env.acoes_discretizadas(prox_acao_disc)
	prox_estado_emp = np.stack(env.env.frames, axis = 2)
	#adiciona experiencia
	memoria.add((estado_emp, acao_array, rew, prox_estado_emp, done))
	#atualizando as coisas atuais para rodar mais um futuro passo:
	#estado atual recebe o proximo estado
	estado_emp = prox_estado_emp
	#acao atual recebe a proxima acao
	acao_array = prox_acao_array
	#recomepnsa atual recebe prox recompensa
	rew = prox_rew
	#done atual recebe proximo done
	done = prox_done

print('OK')

#reenicia o ambiente (agora vamos comecar de verdade)

print('Resetando o Ambiente...')

env.reset()

print('OK')

'''
configura o tensorboard
nao tem como salvar muita coisa, pois nao temos metrica para comparacao, pois nao temos dados,
e as imagens vao ser empilhadas, entao nao vai dar para ver muita coisa
podemos apenas ver a perda
'''

print('Configurando TensorBoard...')

#tensorboard --logdir=/tensorboard/dqn/1

writer = tf.summary.FileWriter("/tensorboard/ddqn/1")
tf.summary.scalar("Perda", DQRede.perda)
write_op = tf.summary.merge_all()

print('OK')

#TREINO

print('Entrando no Treino...')

#acoes_possiveis = [np.array(env.action(i)) for i in range(0, env.action_space.n)]

#metodo para salvar o modelo
saver = tf.train.Saver()

with tf.Session() as sess:

	#deinfe estrategia epsilon greedy

	def eg(env, sess, prob_inicial, min_prob, tx_decay, passo_decay, estado_emp):
		#define um numero aleatorio como o tradeoff entre exploracao e tirar vantagem
		exp_vant_tradeoff = np.random.rand()

		#probabilidade de exploracao
		prob_exp = min_prob + (prob_inicial - min_prob) * np.exp(-tx_decay * passo_decay)

		if (prob_exp > exp_vant_tradeoff):
			#print('Explorando')
			#explora
			acao = env.action_space.sample()

		else:
			#print('Abusando')
			#procura melhor acao baseada na estimacao do Q-valor da rede neural
			Qs = sess.run(DQRede.saida, feed_dict = {
														DQRede.inputs: estado_emp.reshape((1, *estado_emp.shape))
													}	
						)

			#procura o indice da melhor acao
			#acao = acoes_possiveis[np.argmax(Qs)]
			acao = np.argmax(Qs)

		return acao, prob_exp

	#inicializa as vars
	sess.run(tf.global_variables_initializer())

	#inicializa o decaimento
	passo_decay = 0
	passo = 0

	for episodio in range(numero_episodios):
		recompensas_episodio = []

		#acao inicial
		#env.render()
		estado_emp = np.stack(env.env.frames, axis = 2)
		acao_disc, prob_exp = eg(env, sess, prob_inicial, min_prob, tx_decay, passo_decay, estado_emp)
		ob,rew,done,info = env.step(acao_disc)
		acao_array = env.acoes_discretizadas(acao_disc)
		estado_emp = np.stack(env.env.frames, axis = 2)
		recompensas_episodio.append(rew)
		#memoria.add(estado_emp, acao_array, rew, prox_estado_emp, done)
		passo += 1
		if passo == 15: #decaimento exponencial a cada 15 frames
			passo_decay += 1
			passo = 0
			
		passo_decay += 1

		#entra em loop ate acabar
		while not done:
			#env.render()
			#escolhe ou exploracao ou abusar do que ja sabe pelo epsilon greedy
			prox_acao_disc, prox_prob_exp = eg(env, sess, prob_inicial, min_prob, tx_decay, passo_decay, estado_emp)
			#print(prox_acao_disc)
			prox_ob,prox_rew,prox_done,prox_info = env.step(prox_acao_disc)

			#print("Recompensa:", prox_rew) #recompensa
			#print("Terminou?", prox_done) #terminou?
			#print("Infos Adicionais", prox_info) #valores setados em data.json

			prox_acao_array = env.acoes_discretizadas(prox_acao_disc)		
			prox_estado_emp = np.stack(env.env.frames, axis = 2)
			recompensas_episodio.append(prox_rew)
			memoria.add((estado_emp, acao_array, rew, prox_estado_emp, done))
			passo += 1
			
			if passo == 15: #decaimento exponencial a cada 15 frames
				passo_decay += 1
				passo = 0

			#atualizando as coisas atuais para rodar mais um futuro passo:
			#estado atual recebe o proximo estado
			estado_emp = prox_estado_emp
			#acao atual recebe a proxima acao
			acao_array = prox_acao_array
			#acao discreta atual recebe a proxima acao discreta
			acao_disc = prox_acao_disc
			#recomepnsa atual recebe prox recompensa
			rew = prox_rew
			#done atual recebe proximo done
			done = prox_done

			##############SGD##############
			#Sera realizado um aprendizado com um mini batch de experiencias que guardamos
			mini_batch = memoria.amostra(tamanho_batch)

			estados_mb = np.array([mb[0] for mb in mini_batch], ndmin = 3)
			acoes_mb = np.array([mb[1] for mb in mini_batch])
			recompensas_mb = np.array([mb[2] for mb in mini_batch])
			prox_estados_mb = np.array([mb[3] for mb in mini_batch], ndmin = 3)
			dones_mb = np.array([mb[4] for mb in mini_batch])
			Q_targets_mb = []

			# Pega Valores Q do prox estado

			Qs_prox_estado = sess.run(DQRede.saida, feed_dict = {DQRede.inputs: prox_estados_mb})

			# Q Target = R se episodio terminou em s+1, senao Q_target = R + gamma*max{a}(Q_hat(s',a'))

			for i in range(0, len(mini_batch)):
				terminou = dones_mb[i]

				if terminou:
					Q_targets_mb.append(recompensas_mb[i])
				else:
					target = recompensas_mb[i] + gamma * np.max(Qs_prox_estado[i])
					Q_targets_mb.append(target)

			targets_mb = np.array([mb for mb in Q_targets_mb])

			#consegue a perda, e calcula e aplica os gradientes nos pesos

			perda, _ = sess.run([DQRede.perda, DQRede.otimizador],
                                    feed_dict={DQRede.inputs: estados_mb,
                                               DQRede.Q_target: targets_mb,
                                               DQRede.acoes: acoes_mb})
			#manda para o TensorBoard
			sumario = sess.run(write_op, 
            						feed_dict={DQRede.inputs: estados_mb,
                                               DQRede.Q_target: targets_mb,
                                               DQRede.acoes: acoes_mb})
			writer.add_summary(sumario, episodio)
			writer.flush()

			#coleta o lixo
			gc.collect()

		#a cada 5 episodios salva o modelo
		if episodio % 5 == 0:
			save_path = saver.save(sess, "./models/modelo_DDQN_1.ckpt")
			print("Modelo Salvo!")

		recomepensa_total = np.sum(recompensas_episodio)
		print('Episodio: {}'.format(episodio),
                              'Recompensa Total: {}'.format(recomepensa_total),
                              'Perda: {:.4f}'.format(perda),
                              'Passo Decay: {:.4f}'.format(passo_decay),
                              'Prob. Exploracao: {:.4f}'.format(prox_prob_exp))
		#reseta o ambiente e comeca outro episodio
		print('Acabou episodio, resetando ambiente...')
		env.reset()

	print('Chegou ao limite de Episodios, treino acabou.')








