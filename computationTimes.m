clear; close all; clc;

% Code for the paper "On the effect of quadratic regularization in direct
% data-driven LQR" by Manuel Klädtke, Feiran Zhao, Florian Dörfler, and
% Moritz Schulze Darup submitted to IFAC WC 2026

% @ 2025 Manuel Klädtke

% This code uses CVX for the solution SDPs
% CVX Research, Inc. CVX: Matlab software for disciplined convex
% programming, version 2.0. https://cvxr.com/cvx, April 2011.
% M. Grant and S. Boyd. Graph implementations for nonsmooth convex programs,
% Recent Advances in Learning and Control (a tribute to M. Vidyasagar), V.
% Blondel, S. Boyd, and H. Kimura, editors, pages 95-110, Lecture Notes in
% Control and Information Sciences, Springer, 2008.
% http://stanford.edu/~boyd/graph_dcp.html.

% Compare computations for quadratic regularization tr(G*P*G'),
% projection-based quadratic regularization tr(Pi_perp*G*P*G'), and their
% respective equivalent lower-dimensional parametrizations in (15), namely
% {1, 2, 3} and {1}.
% Do this for different data sizes (ell)

% State and input dimensions
n = 2; m = 1;
% Eigenpairs
lambda1 = 0.85; % Note that lambda1 = lambda2 is not a good idea, since it yields (A,B) not controllable
lambda2 = 0.2;
v1 = [1; 1]; v2 = [-1; 1];
T = [v1, v2];
% True system parameters
A = T\diag([lambda1, lambda2])*T;
B = [1;0];
% LQR weight matrices
Q = eye(n); R = 0.1*eye(m);
% Disturbance w ~ M(mu, Sigma) with Sigma = diag(sigma, sigma)
mu = zeros(n,1); sigma = sqrt(0.01);

% Try different sizes for data matrices. Note that too large ell lead to
% out-of-memory errors for tr(G*P*G') and tr(Pi_perp*G*P*G').
% Furthermore, when using SDPT3, tr(G*P*G') and tr(Pi_perp*G*P*G') already
% seem to fail for ell > 140 even though 'out-of-memory' error is not
% encountered at that point, yet.
ells = [30; 60; 90; 120];
nRuns = 2; % Number of computation runs to average over

% Regularization weight is chosen arbitrarily here, since it does not seem
% to impact computation times. That is, if lambda is not chosen way too
% small or way too big.
lambda = 100;

% Allocate empty matrices for computation times
cmpTimesQuad = zeros(nRuns, length(ells));
cmpTimesProj = zeros(nRuns, length(ells));
cmpTimes123 = zeros(nRuns, length(ells));
cmpTimes1 = zeros(nRuns, length(ells));

T_preproc = zeros(nRuns, length(ells));

for j = 1:nRuns
    for i = 1:length(ells)
        ell = ells(i);

        % Use same way of data generation as for the figures, for comparison.

        % Generate X0 heaviliy skewed in direction of v2
        X0 = 10*v2+randn(n, ell);
        % Explore near a (barely stabilizing) controller
        K_explore = [-2.8, 6.8];
        sigma_u = sqrt(1);
        U = K_explore*X0 + sigma_u*randn(m, ell);
        X = A*X0 + B*U + sigma*randn(n,ell);

        Z = [X0; U];
        D = [Z; X];

        Pi_perp = eye(ell)-pinv(Z)*Z;
        
        % Measure preprocessing time associated with the parametrization in
        % (15)
        tic
        M_LS = X*pinv(Z);
        K_LS = U*pinv(X0);

        A_LS = M_LS(:, 1:n);
        B_LS = M_LS(:, n+1:n+m);

        dX = X-M_LS*Z;
        dU = U-K_LS*X0;

        Sigma_dx = 1/ell*dX*dX';
        Sigma_du = 1/ell*dU*dU';
        Sigma_x0 = 1/ell*X0*X0';
        Sigma_z = 1/ell*Z*Z';

        T_preproc(j, i) = toc;
        
        % Quadratic regularization
        % H(G, P) = lambda*trace(G*P*G')
        % Proposed in "Low-complexity learning of Linear Quadratic
        % Regulators from noisy data" by Claudio De Persis and Pietro Tesi
        tic
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable S(ell, n)
            variable L(m, m) semidefinite
            variable N(ell, ell) semidefinite
            minimize(trace(Q*P)+trace(R*L)+lambda*trace(N))
            subject to
                X0*S == P;
                P >= eye(n);
                [P-eye(n), X*S; (X*S)', P] >= 0;
                [L, U*S; (U*S)', P] >= 0;
                [N, S; S', P] >= 0;
        cvx_end
        cmpTimesQuad(j, i) = toc;

        
        % Projection-based variant
        % H(G, P) = lambda*trace(Pi_perp*G*P*G')
        % Adjacent to the one proposed in "On the Certainty-Equivalence
        % Approach to Direct Data-Driven LQR Design" by Florian Dörfler,
        % Pietro Tesi, and Claudio De Persis
        tic
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable S(ell, n)
            variable L(m, m) semidefinite
            variable N(ell, ell) semidefinite
            minimize(trace(Q*P)+trace(R*L)+lambda*trace(Pi_perp*N))
            subject to
                X0*S == P;
                P >= eye(n);
                [P-eye(n), X*S; (X*S)', P] >= 0;
                [L, U*S; (U*S)', P] >= 0;
                [N, S; S', P] >= 0;
        cvx_end
        cmpTimesProj(j, i) = toc;


        % This version directly solves the problem
        % min tr(Q*P) + tr(K'*R*K*P) + lambda tr((A_cl-(A_LS+B_LS*K))'*P*(A_cl-(A_LS+B_LS*K))*Sigma_dx^(-1)) + lambda * tr((K-K_LS)'*P*(K-K_LS)*Sigma_du^(-1)) + lambda * tr(Sigma_x0^(-1)*P)
        % s.t. A_cl'*P*A_cl - P + I <= 0
        % To convexify this, we introduce the variables Ktilde = K*P and Acltilde = Acl*P
        % See Eq. (14) in the paper
    
        % {1, 2, 3}: H = lambda*(H_1 + H_2 + H_3)
        tic
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable Ktilde(m, n)
            variable Acltilde(n,n)
            variable N(m, m) semidefinite
            variable M(n, n) semidefinite
            variable L(m, m) semidefinite
            minimize(trace(Q*P) + trace(R*L) + lambda/ell*trace(M*inv(Sigma_dx)) + lambda/ell*trace(N*inv(Sigma_du)) + lambda/ell*trace(P*inv(Sigma_x0))  )
            subject to
                P >= eye(n);
                [L, Ktilde;
                 Ktilde', P] >= 0;
                [N,                     Ktilde - K_LS*P;
                (Ktilde - K_LS*P)',    P]                  >= 0;
                [M,                     Acltilde - (A_LS*P+B_LS*Ktilde);
                (Acltilde - (A_LS*P+B_LS*Ktilde))',    P]                  >= 0;
                [P-eye(n),              Acltilde;
                 Acltilde',     P]                  >= 0;
        cvx_end
        cmpTimes123(j, i) = toc;

        % {1}: H = lambda*H_1
        tic
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable Ktilde(m, n)
            variable Acltilde(n,n)
            variable M(n, n) semidefinite
            variable L(m, m) semidefinite
            minimize(trace(Q*P) + trace(R*L) + lambda/ell*trace(M*inv(Sigma_dx))) 
            subject to
                P >= eye(n);
                [L, Ktilde;
                 Ktilde', P] >= 0;
                [M,                     Acltilde - (A_LS*P+B_LS*Ktilde);
                (Acltilde - (A_LS*P+B_LS*Ktilde))',    P]                  >= 0;
                [P-eye(n),              Acltilde;
                 Acltilde',     P]                  >= 0;
        cvx_end
        cmpTimes1(j, i) = toc;


    end
end
% Compute mean times over nRuns
T_quad = mean(cmpTimesQuad)';
T_123 = mean(cmpTimes123+T_preproc)';
T_proj = mean(cmpTimesProj)';
T_1 = mean(cmpTimes1+T_preproc)';

meantimesTable = table(ells, T_quad, T_123, T_proj, T_1)