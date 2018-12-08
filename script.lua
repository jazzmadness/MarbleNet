function check_done()
	if data.time == 0 or data.flag_caiu == 6 then --se o tempo acabou ou a bola caiu, terminamos o episodio
		return true
	else
		return 0
	end
end

-- inicia variaveis de score e tempo

score_anterior = 0 
tempo_anterior = 0
flag_caiu_base = 0

function recompensa_punicao()
	local score_atual = data.score
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

	if score_atual > score_anterior or tempo_atual < tempo_anterior or data.time == 0 or flag_caiu_atual - flag_caiu_base == 6 then --a condicao de tempo esta ao contrario do score em relacao a temporalidade pois a variavel tempo eh decrescente
		local delta_score = score_atual - score_anterior
		local delta_tempo = tempo_anterior - tempo_atual
		local delta_combinado = 2*(delta_score) - 10*(delta_tempo) -- score anda de 10 em 10, tempo anda de 1 em 1
		score_anterior = score_atual
		tempo_anterior = tempo_atual

		if data.time == 0 then
			delta_combinado = delta_combinado - 500 --punicao extra por ter perdido
		end
		if flag_caiu_atual - flag_caiu_base == 6 then
			delta_combinado = delta_combinado - 500 -- punicao extra por ter caido
			flag_caiu_base = flag_caiu_atual -- vai atualizando o flag
		end
		
		return delta_combinado

	else -- se nao aconteceu nada (o agente nao se mexeu, ou seja, nao ganhou score) ou o tempo nao mudou de valor (pois estamos olhando frame a frame, o tempo so muda a cada segundo), ou o agente nao caiu, nao da nenhuma recomepnsa
		flag_caiu_base = flag_caiu_atual -- vai atualizando o flag todo frame
		return 0
	end
end
