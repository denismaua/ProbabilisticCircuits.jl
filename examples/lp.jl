using JuMP
using Cbc
using LinearAlgebra
myModel = Model(Cbc.Optimizer)
prices = [99.74, 91.22, 98.71, 103.75, 97.15]
cashFlows = [4 5 2.5 5 4; 4 5 2.5 5 4; 4 5 2.5 5 4; 4 5 2.5 5 4; 4 5 102.5 5 4;4 5 0 105 104;4 105 0 0 0; 104 0 0 0 0]
Liab_CFs = [5, 7, 7, 6, 8, 7, 20, 0]*1000
nt=size(cashFlows,1)
nb=size(cashFlows,2)
Rates = [0.01, 0.015, -0.017, 0.019, 0.02, 0.025, 0.027, 0.029]
Disc= [0.99009901, 0.970661749, 0.950686094, 0.927477246, 0.90573081, 0.862296866, 0.829863942, 0.795567442]'
nBonds = [10, 100, 20, 30, 5]*1000
@variable(myModel, 0<= x[b = 1:length(nBonds)] <= nBonds[b])
@objective(myModel,Min,prices' *x)
B=[1.01, 2.020024752, 1.02101183, 1.02502363, 1.024009823, 1.050370059, 1.039082218, 1.043109481]
A=Liab_CFs-cashFlows*x
M=zeros(length(A))
M=A[1]
for k=2:length(A)
    M(k)=M[k-1]*B[k-1]+A[k]
end
@constraint(myModel, constraint1,M.*Disc .<=0.05*53.844)
print(myModel)
optimize!(myModel)
println("Optimal Solutions:")
println("x = ", JuMP.value.(x))
