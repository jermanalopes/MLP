close all
clear all
clc
%% ************** INFORMAÇÕES SOBRE O BANCO DE DADOS **********************
%      6 padrões, porém foram retirados 8 padrões que faltavam     %
%      informações, assim, temos 358 padrões para treinamento e teste     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                                         %
%  CódigoClasse     Classe:                    Padrões     Saída Desejada %
%       1           psoriasis			         112          100000      %
%       2           seboreic dermatitis          61           010000      %
%       3           lichen planus                72           001000      %
%       4           pityriasis rosea             49           000100      %
%       5           cronic dermatitis            52           000010      %    
%       6           pityriasis rubra pilaris     20           000001      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Inicio Código

%Carrega banco de dados 

[ x , y ] = carregaDados(8,'dados.txt',6);


%Parametros Inciais MLP
n_ent = 8;    %números de entrada     
n_saida = 6;     %números de saída  
n_NeuroSaida    = 6;
n_NeuroEsc      = 20;
epocas          = 700;         
realizacoes     = 5;
taxa_aprend     = 0.2;

%Normalizar Dados Entradas
xmax = max(max(abs(x)));
x = x / xmax;
 
tam         = size(x); %tamanho da entrada
qtdAmostras = tam(1); 

dados = bias(x,qtdAmostras);  % Acrescentando BIAS
x = dados;

%Aleatorização dos dados
aleatorio = randperm(length(x)); %Aleatorizar Dados

x = x(aleatorio,:); %Aleatorizar x
y = y(aleatorio,:); %Aleatorizar y

%Separação dos dados entre treinamento e teste
quantTreine = 48;
xtreine = x(1:48,:); %Valores para Treinamento x
ytreine = y(1:48,:);%Valores para Treinamento y

quantTeste = 16;
xteste = x(49:64,:);%Valores para Teste x
yteste = y(49:64,:);%Valores para Teste y

%TREINAMENTO
for r = 1:realizacoes
eqm_teste=0;
%Inicialização dos Pesos W(camada oculta) e M(camada de saída)
pesoW  = rand(n_ent+1,n_NeuroEsc);
pesoM = rand(n_NeuroEsc+1,n_NeuroSaida);

pesoW = pesoW + 0.004*pesoW - 0.02; 
pesoM = pesoM + 0.004*pesoM - 0.02; 

%Início Treinamento
 for  e = 1:epocas
  eqm=0;
    
  alea = randperm(length(xtreine)); %Aleatorizar Dados para treinamento
  xtreine = xtreine(alea,:); %Aleatorizar x para treinamento
  ytreine = ytreine(alea,:); %Aleatorizar y para treinamento
  
for a=1:quantTreine 
 
 %Camada Oculta
 ui = pesoW'*(xtreine(a,:))';  %Ativação do neurônio da camada oculta  
 zi = 1./(1+ exp(-ui));  %Saidas da camada oculta
 ziBias = bias(zi',1);   %Inserir Bias no zi
 
 %Camada de Saída
 uk = ziBias * pesoM;   %Ativação do neurônio da camada de saída
 yk = 1./(1+ exp(-uk)); %Saidas da camada de saída

%Cálculos do Erro
errog = ytreine(a,:) - yk; %Cálculo do erro da camada_saida
errok = 0.5 * sum(errog).^2; %Cálculo do somatório do erro
eqm = eqm + errok; %Cálculo erro quadrático médio

%Gradientes da camada de saída + atualização dos pesos saída
der_saida = yk.*(1 - yk); %Derivada da função Logistica
grad_saida= errog .* der_saida; %Gradiente da camada de saída
pesoM = pesoM + (taxa_aprend*ziBias'*grad_saida);
 
%Gradientes da camada oculta + atualização dos pesos ocultos
der_oculta = (zi.*(1 - zi)); % derivada da camada oculta
grad_oculta = der_oculta .* ( pesoM(2:end,:)* grad_saida');                                           
pesoW = pesoW + (taxa_aprend*grad_oculta*xtreine(a,:))';

end
%Erro Médio
erroplot(e) = eqm / quantTreine;

end

%TESTE
for b=1:quantTeste
%Camada Oculta   
ui2 = xteste(b,:)*pesoW;  %Ativação do neurônio da camada oculta
zi2 = 1./(1+ (exp(-ui2)));  %Saidas da camada oculta
zi2Bias = bias(zi2,1);  

%Camada Saída
uk2 = zi2Bias*pesoM;   %Ativação do neurônio da camada de saída
yk2 = 1./(1+ (exp(-uk2))); %Saidas da camada de saída
 
%Cálculo do Erro de Teste
erro_saida= yteste(b,:) - yk2; %Cálculo do erro da camada_saida
saida2(b,:)= yk2;
end
errow = 0.5 * sum(erro_saida).^2; %Cálculo do somatório do erro
eqm_teste = eqm_teste + errow; %Cálculo erro quadrático médio
  
cont = 0;

for i=1:length(yteste) 
[valueTeste posTeste]=max(yteste(i,:));
[valueSaida posSaida]=max(saida2(i,:));
if(posTeste==posSaida)
  cont=cont+1;
end
end
taxa_acerto(r) = 100*cont/length(xteste)
taxa = taxa_acerto;
erroplot_teste(r) = eqm_teste / quantTeste;

end
acuracia = sum(taxa)/realizacoes

%Plotar erro médio Treinamento
figure(1), hold on
xlabel ('Números de Épocas');
ylabel ('Erro Médio de Treinamento');
title ('Erro médio por Épocas');
handle = plot(erroplot,'-r','LineWidth',2);

%Plotar erro médio Teste
figure(2), hold on
xlabel ('Números de Realizações');
ylabel ('Erro Médio de Teste');
title ('Erro médio por Realizações');
handle = plot(erroplot_teste,'-r','LineWidth',2);

