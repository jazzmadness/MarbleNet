function check_done()
	if data.time == 0 then --se o tempo acabou, terminamos o episodio
		return true
	else
		return 0
	end
end

-- inicia variaveis de score e tempo

score_anterior = 0 
tempo_anterior = 0

function recompensa_punicao()
	local score_atual = data.score
	local tempo_atual = data.time

	--print(score_atual)
	--print(tempo_atual)
	--print(score_anterior)
	--print(tempo_anterior)

	if tempo_atual > tempo_anterior then -- preciso manualmente atualizar quando o tempo anterior esta subindo no comeco do jogo
		tempo_anterior = tempo_atual 
	end

	if score_atual > score_anterior or tempo_anterior > tempo_atual then --a condicao de tempo esta ao contrario do score em relacao a temporalidade pois a variavel tempo eh decrescente
		local delta_score = score_atual - score_anterior
		local delta_tempo = tempo_anterior - tempo_atual
		local delta_combinado = 2*(delta_score) - 10*(delta_tempo) -- score anda de 10 em 10, tempo anda de 1 em 1
		score_anterior = score_atual
		tempo_anterior = tempo_atual

		if data.time == 0 then
			delta_combinado = delta_combinado - 100 --punicao extra por ter perdido
		end
		return delta_combinado
	else -- se nao aconteceu nada (o agente nao se mexeu, ou seja, nao ganhou score) ou o tempo nao mudou de valor (pois estamos olhando frame a frame, o tempo so muda a cada segundo), nao da nenhuma recomepnsa
		return 0
	end
end
	