contagem_caiu = 0
flag_caiu_base_1 = 0
flag_caiu_base_2 = 0

function check_done()
	local flag_caiu_atual = data.flag_caiu
	if data.time == 0 or contagem_caiu == 3 then --se o tempo acabou ou a bola caiu 3 vezes (pois conta 2 a cada caida), terminamos o episodio
		return true
	elseif flag_caiu_atual - flag_caiu_base_1 == 6 then
		flag_caiu_base_2 = flag_caiu_atual -- vai atualizando o flag
		contagem_caiu = contagem_caiu + 1 -- soma no contador
	else
		return 0
	end
end

-- inicia variaveis de score e tempo

score_anterior = 0 
progresso_anterior = 0
tempo_anterior = 0

function recompensa_punicao()
	local score_atual = data.score
	local progresso_atual = data.progresso
	local tempo_atual = data.time
	local flag_caiu_atual = data.flag_caiu


	--print(score_atual)
	--print(tempo_atual)
	--print(score_anterior)
	--print(tempo_anterior)

	if tempo_atual > tempo_anterior then -- preciso manualmente atualizar quando o tempo anterior esta subindo no comeco do jogo (o meu save novo nao tem o tempo subindo, entao isso nao eh mais necessario)
		tempo_anterior = tempo_atual 
	end

	--condicoes conjuntas

	if progresso_atual ~= progresso_anterior or score_atual > score_anterior or 
												tempo_atual < tempo_anterior or 
												data.time == 0 or 
												flag_caiu_atual - flag_caiu_base_1 == 6 then --a condicao de tempo esta ao contrario do score em relacao a temporalidade pois a variavel tempo eh decrescente
		local delta_score = score_atual - score_anterior
		local delta_progresso = progresso_atual - progresso_anterior
		local delta_tempo = tempo_anterior - tempo_atual
		local delta_combinado = 2*(delta_progresso) + 0.1*(delta_score) - (delta_tempo)
		score_anterior = score_atual
		progresso_anterior = progresso_atual
		tempo_anterior = tempo_atual

		if data.time == 0 then
			delta_combinado = delta_combinado - 100 --punicao extra por ter perdido
		end
		if flag_caiu_atual - flag_caiu_base_1 == 6 then
			delta_combinado = delta_combinado - 100 -- punicao extra por ter caido
			flag_caiu_base_1 = flag_caiu_atual -- vai atualizando o flag
		end
		
		return delta_combinado

	else -- se nao aconteceu nada (o agente nao se mexeu, ou seja, nao ganhou score) ou o tempo nao mudou de valor (pois estamos olhando frame a frame, o tempo so muda a cada segundo), ou o agente nao caiu, nao da nenhuma recomepnsa
		flag_caiu_base_1 = flag_caiu_atual -- vai atualizando o flag todo frame
		return 0
	end
end
