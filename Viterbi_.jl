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
A :  probability
Output
Delta ：The matrix formed by the maximum probability of all single paths with state I at time t is N*K order, that is, the number of rows represents the number of states,
 and the number of columns represents the number of times.
Psi矩阵： The matrix formed by the t-1st node of the path with the highest probability
in all single paths with state I at time t, is N*K order, that is, the number of rows represents the number of states and the number of columns represents the time.
And the first column is 0.
P：The optimal path probability of observation sequence O.
I：The optimal path of observation sequence O

Usage
input：
4 


A =[0.5 0.2 0.3;
    0.3 0.5 0.2 ;
    0.2 0.3 0.5;
    0.2 0.4 0.4;
    0.1 0.8 0.1;
    ]

julia> Delta
5×3 Array{Float64,2}:
 0.5  0.1   0.12
 0.3  0.25  0.08
 0.2  0.15  0.2
 0.2  0.2   0.16
 0.1  0.4   0.04

julia> Psi
5×3 Array{Float64,2}:
 0.0  1.0  5.0
 0.0  1.0  5.0
 0.0  1.0  5.0
 0.0  1.0  5.0
 0.0  1.0  5.0

julia> I
3-element Array{Float64,1}:
 1.0
 5.0
 3.0
 """
function viterbi(A)
    M, N = size(A); #Number of states
    K = N;  #Number of observation sequence

    # Evaluate the first column of the Delta matrix
    Delta = zeros(M,N);
    Delta[:,1] = A[:,1]

    #Recursively evaluate the remaining values of the Delta matrix
    Delta_j = zeros(M);
    Psi = zeros(M,K);
    for j = 2:N
        for i = 1:M
            for t =1:M
                Delta_j[t] = Delta[t,j-1] * A[i,j];
            end
            max_delta_j = maximum(Delta_j); #Find the maximum probability value
            psi = argmax(Delta_j); #Find the maximum probability position
            Psi[i,j]= psi; #Remember  the position  of leading node
            Delta[i,j] = max_delta_j;  #Remember  the value  of maximum probability
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



A =[0.5 0.2 0.3;
    0.3 0.5 0.2 ;
    0.2 0.3 0.5;
    0.2 0.4 0.4;
    0.1 0.8 0.1;
    ]


Delta,Psi,P,I = viterbi(A)
println(Delta,Psi,P,I)
println("****************")
