function [y] = bias(dados,qtdAmostras)

bias = -1;
for i= 1:1:qtdAmostras
    matrizBias(i,1) = bias;
end

y = [matrizBias dados];     

end
    
          