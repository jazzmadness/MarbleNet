import retro #biblioteca para integracao com os jogos
import time  #caso precise fazer um sleep no sistema para checar as coisas
import numpy as np
from utilitarios import DiscretizadorAcoes


#importei a rom com o comando:

#python3 -m retro.import caminho/do/arquivo/marble_madness.md

env = retro.make('MarbleMadness-Genesis', 'Level1_Sem_Timer_Subindo')

#modifica o array de acoes possiveis para apenas necessarias

env = DiscretizadorAcoes(env)

env.reset() 

done = False #done significa que o episodio ("vida") foi encerrado

passo = 1

while not done: #enquanto estamos "vivos":

	env.render() #renderiza o quadro

	acao = env.action_space.sample() #tira uma amostra de todas as acoes possiveis
	
	aux = np.zeros(env.action_space.n)

	aux[acao] = 1

	print(acao) #imprimi essa acao no terminal
	print(aux) #transformacao para vetor
 
	ob,rew,done,info = env.step(acao) #coleta todas as informacoes resultantes de um step no ambiente

	passo += 1

	print("Passo", passo)
	print("Shape da Imagem", ob.shape) #imagem para trabalhar no modelo
	print("Recompensa:", rew) #recompensa
	print("Terminou?", done) #terminou?
	print("Infos Adicionais", info) #valores setados em data.json
	time.sleep(0.05)