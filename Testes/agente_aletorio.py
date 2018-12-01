import retro #biblioteca para integracao com os jogos
import time  #caso precise fazer um sleep no sistema para checar as coisas
from utilitarios import DiscretizadorAcoes

#importei a rom com o comando:

#python3 -m retro.import caminho/do/arquivo/marble_madness.md

env = retro.make('MarbleMadness-Genesis', 'Level1')

#modifica o array de acoes possiveis para apenas necessarias

env = DiscretizadorAcoes(env)

env.reset() 

done = False #done significa que o episodio ("vida") foi encerrado

while not done: #enquanto estamos "vivos":

	env.render() #renderiza o quadro

	acao = env.action_space.sample() #tira uma amostra de todas as acoes possiveis

	print(acao) #imprimi essa acao no terminal

	ob,rew,done,info = env.step(acao) #coleta todas as informacoes resultantes de um step no ambiente

	print("Shape da Imagem", ob.shape) #imagem para trabalhar no modelo
	print("Recompensa:", rew) #recompensa
	print("Terminou?", done) #terminou?
	print("Infos Adicionais", info) #valores setados em data.json
	#time.sleep(0.1)