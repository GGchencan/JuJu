"""
Input
A :  transition probability
B : observation probability
P_i: Initialization probability
O :  observation sequence
Output
Delta ：The matrix formed by the maximum probability of all single paths with state I at time t is N*K order, that is, the number of rows represents the number of states,
 and the number of columns represents the number of times.
Psi矩阵： The matrix formed by the t-1st node of the path with the highest probability
in all single paths with state I at time t, is N*K order, that is, the number of rows represents the number of states and the number of columns represents the time.
And the first column is 0.
P：The optimal path probability of observation sequence O.
I：The optimal path of observation sequence O

"""

"""
Input
A :  transition probability
B : observation probability
P_i: Initialization probability
O :  observation sequence
Output
Delta ：The matrix formed by the maximum probability of all single paths with state I at time t is N*K order, that is, the number of rows represents the number of states,
 and the number of columns represents the number of times.
Psi矩阵： The matrix formed by the t-1st node of the path with the highest probability
in all single paths with state I at time t, is N*K order, that is, the number of rows represents the number of states and the number of columns represents the time.
And the first column is 0.
P：The optimal path probability of observation sequence O.
I：The optimal path of observation sequence O

Usage
example：（李航《统计学习方法》P186例10.3）

Input
A =[0.5000    0.2000    0.3000;    0.3000    0.5000    0.2000;  0.2000    0.3000    0.5000]
B =[ 0.5000    0.5000;0.4000    0.6000;   0.7000    0.3000]
Pi = [ 0.2000; 0.4000;  0.4000;]
O1 =  [ 1;2; 1]

Output
[D,Psi,P,I] = Viterbi(A,B,Pi,O1)
D =
    0.1000    0.0280    0.0076
    0.1600    0.0504    0.0101
    0.2800    0.0420    0.0147
Psi =
     0     3     2
     0     3     2
     0     3     3
P =
    0.0147
I =
     3
     3
     3

"""

function Viterbi(A,B,P_i,O)
    N , M = size(A); #Number of states
    K= size(O)[1]; #Number of observation sequence

    # Evaluate the first column of the Delta matrix
    Delta = zeros(N,K);
    for i = 1:M
        Delta[i,1] = P_i[i] * B[i,O[1,1]];
    end

    #Recursively evaluate the remaining values of the Delta matrix
    Delta_j = zeros(M);

    Psi = zeros(M,K);



    for t = 2:K
        for j = 1:N
            for i = 1:M
                Delta_j[i,1] = Delta[i,t-1] * A[i,j] * B[j,O[t,1]];
            end
            max_delta_j = maximum(Delta_j); #Find the maximum probability value
            psi = argmax(Delta_j); #Find the maximum probability position
            Psi[j,t]= psi; #Remember  the position  of leading node
            Delta[j,t] = max_delta_j;  #Remember  the value  of maximum probability
        end
    end

    P_better = maximum(Delta[:,K]);
    psi_k = argmax(Delta[:,K]);
    #println(psi_k)
    P = P_better; #Optimal path probability
    I = zeros(K);
    I[K] = psi_k;
    #println(I)
    #println(Psi)
    for t = K-1:-1:1
        #println(t)
        I[t] = Psi[convert(Int, I[t+1]),t+1]; #Path backtracking to get the optimal path
        #println(I)
    end

    return Delta,Psi,P,I
    #return 0
end



A =[0.50 0.20 0.30; 0.30 0.50 0.20 ; 0.20 0.30 0.50]
B =[ 0.50 0.50; 0.40 0.60 ;0.70 0.30]
P_i =[ 0.20; 0.40 ;0.40]
O =[1; 2; 1]
#I = viterbi(A,B,P_i,O)
#println(I)
Delta,Psi,P,I = Viterbi(A,B,P_i,O)
println(Delta,Psi,P,I)
println("****************")
