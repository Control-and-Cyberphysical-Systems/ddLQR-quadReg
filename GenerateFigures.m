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

% Set this flag to true to generate random data for a new experiment.
% The statistics of randomized quantities match those described in the
% paper, but their realizations will be different.
randomize_experiment = false;

% Set these flags to true to (re-)compute the data for fig1 and fig2,
% respectively. Warning: This may take 15+ minutes, depending on your
% system.
% If randomize_experiment is true, these will automatically overwritten as 
% true, since there is no pre-computed data available for a new 
% randomized experiment.
compute_figure1 = false;
compute_figure2 = false;

if randomize_experiment
    compute_figure1 = true;
    compute_figure2 = true;
else
    rng(1);
end

% Text sizes and colors for figures
labelsize = 12;
Cmap = [215,25,28;
        253,174,97;
        96,190,82; 
        43,131,186]/255;


%% Numerical experiment setup
n = 2; m = 1; % State and input dimensions
% System matrices
A = [0.525, -0.325;
    -0.325, 0.525];
B = [1; 0];

Q = eye(n); R = 0.1*eye(m); % LQR weight matrices

% Disturbance w ~ N(mu, Sigma) with Sigma = diag(sigma, sigma)
mu_x = zeros(n,1); sigma_x = sqrt(0.01);

% Exploration parameters
ell = 30; % Number of data columns
v = [-1; 1]; % Main direction for initial state exploration
K_explore = [-2.8, 6.8]; % % Explore near a  (barely stabilizing) controller
% Statistics of randomized exploration components
mu_x0 = zeros(n, 1); sigma_x0 = 1;
mu_u = zeros(m, 1); sigma_u = 1;

%% Generate data and compute associated parameters
% Data X0 heavily skewed towards v
X0 = 10*v + mu_x0 + sigma_x0*randn(n, ell);
% Data U heavily skewed towards K_explore*x0
U = K_explore*X0 + mu_u + sigma_u*randn(m, ell);
% Resulting future state data for each column
X = A*X0 + B*U + mu_x + sigma_x*randn(n,ell);

% Condensed data matrices
Z = [X0; U]; 
D = [Z; X];

% Linear least-squares parameters
M_LS = X*pinv(Z);
K_LS = U*pinv(X0);
A_LS = M_LS(:, 1:n);
B_LS = M_LS(:, n+1:n+m);
% Linear least-squares residuals
dX = X-M_LS*Z;
dU = U-K_LS*X0;
% Empirical (error) covariances
Sigma_dx = 1/ell*dX*dX';
Sigma_du = 1/ell*dU*dU';
Sigma_x0 = 1/ell*X0*X0';
Sigma_z = 1/ell*Z*Z';

% Projection matrix for projection-based regularization
Pi_perp = eye(ell)-pinv(Z)*Z;


%% Fig1: Visualizing the effect of the "first term" involving 
% the closed loop matrix A_cl being pushed towards A_LS + B_LS * K

if ~compute_figure1
    % Load pre-computed data
    load('fig1_data')
else
    % Re-compute data
    nPoints = 100;
    lambda = logspace(-4, 6, 100);
    
    % Allocate empty matrices for (vectorized) results
    Acl_vec_H1 = zeros(n*n, nPoints);
    Acl_vec_H12 = zeros(n*n, nPoints);
    Acl_vec_H13 = zeros(n*n, nPoints);
    Acl_vec_H123 = zeros(n*n, nPoints);
    Acl_vec_proj = zeros(n*n, nPoints);
    Acl_vec_full = zeros(n*n, nPoints);
    
    K_vec_H1 = zeros(m*n, nPoints);
    K_vec_H12 = zeros(m*n, nPoints);
    K_vec_H13 = zeros(m*n, nPoints);
    K_vec_H123 = zeros(m*n, nPoints);
    K_vec_proj = zeros(m*n, nPoints);
    K_vec_full = zeros(m*n, nPoints);
    
    Acl_normDiff_H1 = zeros(1,nPoints);
    Acl_normDiff_H12 = zeros(1,nPoints);
    Acl_normDiff_H13 = zeros(1,nPoints);
    Acl_normDiff_H123 = zeros(1,nPoints);
    Acl_normDiff_proj = zeros(1,nPoints);
    Acl_normDiff_full = zeros(1,nPoints);
    
    for i = 1:nPoints
        % Equivalent reformulations using parametric effect of quadratic
        % regularization
    
        % This version directly solves the problem
        % min tr(Q*P) + tr(K'*R*K*P) + lambda tr((A_cl-(A_LS+B_LS*K))'*P*(A_cl-(A_LS+B_LS*K))*Sigma_dx^(-1)) + lambda * tr((K-K_LS)'*P*(K-K_LS)*Sigma_du^(-1)) + lambda * tr(Sigma_x0^(-1)*P)
        % s.t. A_cl'*P*A_cl - P + I <= 0
        % To convexify this, we introduce the variables Ktilde = K*P and Acltilde = Acl*P
        % See Eq. (14) in the paper
    
        % For comparisons, use different subsets of the individual
        % regularization terms H_i
    
        % {1, 2, 3}: H = lambda*(H_1 + H_2 + H_3)
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable Ktilde(m, n)
            variable Acltilde(n,n)
            variable N(m, m) semidefinite
            variable M(n, n) semidefinite
            variable L(m, m) semidefinite
            minimize(trace(Q*P) + trace(R*L) + lambda(i)/ell*trace(M*inv(Sigma_dx)) + lambda(i)/ell*trace(N*inv(Sigma_du)) + lambda(i)/ell*trace(P*inv(Sigma_x0))  )
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
    
        K = Ktilde*inv(P);
        Acl = Acltilde*inv(P);
        Acl_vec_H123(:, i) = reshape(Acl, [], 1);
        K_vec_H123(:, i) = reshape(K, [], 1);
        Acl_normDiff_H123(i) = norm(Acl-(A_LS+B_LS*K), 'fro');
    
        % {1}: H = lambda*H_1
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable Ktilde(m, n)
            variable Acltilde(n,n)
            variable M(n, n) semidefinite
            variable L(m, m) semidefinite
            minimize(trace(Q*P) + trace(R*L) + lambda(i)/ell*trace(M*inv(Sigma_dx))) 
            subject to
                P >= eye(n);
                [L, Ktilde;
                 Ktilde', P] >= 0;
                [M,                     Acltilde - (A_LS*P+B_LS*Ktilde);
                (Acltilde - (A_LS*P+B_LS*Ktilde))',    P]                  >= 0;
                [P-eye(n),              Acltilde;
                 Acltilde',     P]                  >= 0;
        cvx_end
    
        K = Ktilde*inv(P);
        Acl = Acltilde*inv(P);
        Acl_vec_H1(:, i) = reshape(Acl, [], 1);
        K_vec_H1(:, i) = reshape(K, [], 1);
        Acl_normDiff_H1(i) = norm(Acl-(A_LS+B_LS*K), 'fro');
    
        % {1, 2}: H = lambda*(H_1 + H_2)
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable Ktilde(m, n)
            variable Acltilde(n,n)
            variable N(m, m) semidefinite
            variable M(n, n) semidefinite
            variable L(m, m) semidefinite
            minimize(trace(Q*P) + trace(R*L) + lambda(i)/ell*trace(M*inv(Sigma_dx)) + lambda(i)/ell*trace(N*inv(Sigma_du))) 
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
    
        K = Ktilde*inv(P);
        Acl = Acltilde*inv(P);
        Acl_vec_H12(:, i) = reshape(Acl, [], 1);
        K_vec_H12(:, i) = reshape(K, [], 1);
        Acl_normDiff_H12(i) = norm(Acl-(A_LS+B_LS*K), 'fro');
    
        % {1, 3}: H = lambda*(H_1 + H_3)
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable Ktilde(m, n)
            variable Acltilde(n,n)
            variable N(m, m) semidefinite
            variable M(n, n) semidefinite
            variable L(m, m) semidefinite
            minimize(trace(Q*P) + trace(R*L) + lambda(i)/ell*trace(M*inv(Sigma_dx)) + lambda(i)/ell*trace(P*inv(Sigma_x0))  ) 
            subject to
                P >= eye(n);
                [L, Ktilde;
                 Ktilde', P] >= 0;
                [M,                     Acltilde - (A_LS*P+B_LS*Ktilde);
                (Acltilde - (A_LS*P+B_LS*Ktilde))',    P]                  >= 0;
                [P-eye(n),              Acltilde;
                 Acltilde',     P]                  >= 0;
        cvx_end
    
        K = Ktilde*inv(P);
        Acl = Acltilde*inv(P);
        Acl_vec_H13(:, i) = reshape(Acl, [], 1);
        K_vec_H13(:, i) = reshape(K, [], 1);
        Acl_normDiff_H13(i) = norm(Acl-(A_LS+B_LS*K), 'fro');
        

        % Regularizations proposed in the literature for comparison
        
        % Quadratic regularization
        % H(G, P) = lambda*trace(G*P*G')
        % Proposed in "Low-complexity learning of Linear Quadratic
        % Regulators from noisy data" by Claudio De Persis and Pietro Tesi
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable S(ell, n)
            variable L(m, m) semidefinite
            variable N(ell, ell) semidefinite
            minimize(trace(Q*P)+trace(R*L)+lambda(i)*trace(N))
            subject to
                X0*S == P;
                P >= eye(n);
                [P-eye(n), X*S; (X*S)', P] >= 0;
                [L, U*S; (U*S)', P] >= 0;
                [N, S; S', P] >= 0;
        cvx_end
        K = U*S*inv(P);
        Acl = X*S*inv(P);
        Acl_vec_full(:, i) = reshape(Acl, [], 1);
        K_vec_full(:, i) = reshape(K, [], 1);
        Acl_normDiff_full(i) = norm(Acl-(A_LS+B_LS*K), 'fro');
    
        % Projection-based variant
        % H(G, P) = lambda*trace(Pi_perp*G*P*G')
        % Adjacent to the one proposed in "On the Certainty-Equivalence
        % Approach to Direct Data-Driven LQR Design" by Florian Dörfler,
        % Pietro Tesi, and Claudio De Persis
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable S(ell, n)
            variable L(m, m) semidefinite
            variable N(ell, ell) semidefinite
            minimize(trace(Q*P)+trace(R*L)+lambda(i)*trace(Pi_perp*N))
            subject to
                X0*S == P;
                P >= eye(n);
                [P-eye(n), X*S; (X*S)', P] >= 0;
                [L, U*S; (U*S)', P] >= 0;
                [N, S; S', P] >= 0;
        cvx_end
        K = U*S*inv(P);
        Acl = X*S*inv(P);
        Acl_vec_proj(:, i) = reshape(Acl, [], 1);
        K_vec_proj(:, i) = reshape(K, [], 1);
        Acl_normDiff_proj(i) = norm(Acl-(A_LS+B_LS*K), 'fro');
    end
end


% Plot results
figure
% Comparison of {1, 2, 3} with tr(G*P*G')
loglog(lambda, Acl_normDiff_H123, 'Color', Cmap(1,:), 'LineWidth', 0.7)
hold on
loglog(lambda, 1.03*Acl_normDiff_full, ... % Slight shift to see both lines
    '--', 'Color', Cmap(1,:), 'LineWidth', 0.7)
% {1, 3}
loglog(lambda, Acl_normDiff_H13, 'Color', Cmap(2,:), 'LineWidth', 0.7)
% {1, 2}
loglog(lambda, Acl_normDiff_H12, 'Color', Cmap(3,:), 'LineWidth', 0.7)
% Comparison of {1} with tr(Pi_perp*G*P*G')
loglog(lambda, Acl_normDiff_H1, 'Color', Cmap(4,:), 'LineWidth', 0.7)
loglog(lambda, Acl_normDiff_proj, '--', 'Color', Cmap(4,:), 'LineWidth', 0.7)
% Warning to explain different results. In particular, case {1, 2} does not
% converge to 0
if any(abs(eig(A_LS+B_LS*K_LS))>0)
    title('Warning! $K_\mathrm{LS}$ does not stabilize $(A_\mathrm{LS}, B_\mathrm{LS})$!', 'Interpreter', 'latex')
end

% Figure labeling etc
set(gca, 'fontsize',labelsize)
xlabel('$\lambda$', 'Interpreter','latex')
ylabel('$$\|\overline{{A}}_\mathrm{cl} - ({A}_\mathrm{LS} + {B}_\mathrm{LS} \overline{{K}})\|_{\mathrm{F}}$$', 'Interpreter','latex')
legend('$\{1, 2, 3\}$', '$\mathrm{tr}({G} {P} {G}^\top)$', '$\{1, 3\}$', '$\{1, 2\}$', '$\{1\}$', '$\mathrm{tr}(\Pi_{\perp} {G} {P} {G}^\top)$' ...
    , 'Interpreter', 'latex', 'Location', 'west')
xticks([1e-6, 1e-3, 1e0, 1e3, 1e6])
xlim([1e-6, 1e6])
yticks([1e-6, 1e-3, 1e0])
grid on; box on;
set(gcf,'position',0.6*[0 0, 780 390])
set(gcf,'PaperPositionMode','Auto')

%% Second experiment: Visualizing the effect of the "second term" involving
% the policy K being pushed towards K_LS

% Compute set of all stabilizing controllers for (A_LS, B_LS) for
% visualization purposes
% These expressions were derived analytically by applying the Jury
% criterion to a controllable canonical form of the system.
% Note that they are only valid for the considered case n = 2, m = 1, and
% that the set of stabilizing controllers is neither a polytope nor convex,
% in general!
% Controllable canonical form:
S = ctrb(A_LS, B_LS);
Sinv = inv(S);
sT = Sinv(end, :);
T = zeros(n, n);
for i = 1:n
    T(i,:) = sT*A_LS^(i-1);
end
Tinv = inv(T);
Actrb = T*A_LS*Tinv;
Ftilde = [-1, 0;
    1, 0;
    1, 1;
    1, -1];
a1 = Actrb(n, 1); a2 = Actrb(n, 2);
e = [a1+1;
    -a1+1;
    1-a1-a2;
    1-a1+a2];
F = Ftilde*Tinv';
% The set of stabilizing controllers is given by F*K <= e

% Compute vertices for visualization
combs = nchoosek(1:4, 2);
V_stab = [];
for i = 1:size(combs,1)
    if rcond(F(combs(i,:), :)) > eps
        V_stab = [V_stab, linsolve(F(combs(i,:), :), e(combs(i,:)))];
    end
end
V_stab = V_stab(:, convhull(V_stab'));

% Compute certainty equivalent LQR for comparison
[~, K_CE, ~] = idare(A_LS, B_LS, Q, R, [], []); 
K_CE = -K_CE;

if ~compute_figure2
    % Load pre-computed data
    load('fig2_data')
else
    % Re-compute data
    nPoints = 100;
    lambda = [0, logspace(-10, 10, nPoints)];
    % Allocate empty matrices for (vectorized) results
    K_vec_H23 = zeros(n*m, nPoints);
    K_vec_H2 = zeros(n*m, nPoints);
    K_vec_covar = zeros(n*m, nPoints);
    for i = 1:nPoints
        % Equivalent reformulations using parametric effect of quadratic
        % regularization for the covariance parametrization
    
        % This version directly solves the problem
        % min tr(Q*P) + tr(K'*R*K*P) + lambda * tr((K-K_LS)'*P*(K-K_LS)*Sigma_du^(-1)) + lambda * tr(Sigma_x0^(-1)*P)
        % s.t. (A_LS + B_LS * K)'*P*(A_LS + B_LS * K) - P + I <= 0
        % To convexify this, we introduce the variable Ktilde = K*P 
        % See Eq. (15) in the paper.
    
        % For comparisons, use different subsets of the individual
        % regularization terms H_i

        % {2, 3}: H = lambda*(H_2 + H_3)
        cvx_begin SDP
        variable P(n, n) semidefinite
        variable Ktilde(m, n)
        variable Acltilde(n,n)
        variable N(m, m) semidefinite
        variable L(m, m) semidefinite
        minimize(trace(Q*P) + trace(R*L) + lambda(i)/ell*trace(N*inv(Sigma_du)) + lambda(i)/ell*trace(P*inv(Sigma_x0))  )
        subject to
            P >= eye(n);
            [L, Ktilde;
             Ktilde', P] >= 0;
            [N,                     Ktilde - K_LS*P;
            (Ktilde - K_LS*P)',    P]                  >= 0;
            [P-eye(n),              A_LS*P+B_LS*Ktilde;
             (A_LS*P+B_LS*Ktilde)',     P]                  >= 0;
        cvx_end
        K_vec_H23(:,i) = reshape(Ktilde*inv(P), [], 1);
    
        % {2}: H = lambda*H_2
        cvx_begin SDP
        variable P(n, n) semidefinite
        variable Ktilde(m, n)
        variable Acltilde(n,n)
        variable N(m, m) semidefinite
        variable L(m, m) semidefinite
        minimize(trace(Q*P) + trace(R*L) + lambda(i)/ell*trace(N*inv(Sigma_du)) )
        subject to
            P >= eye(n);
            [L, Ktilde;
             Ktilde', P] >= 0;
            [N,                     Ktilde - K_LS*P;
            (Ktilde - K_LS*P)',    P]                  >= 0;
            [P-eye(n),              A_LS*P+B_LS*Ktilde;
             (A_LS*P+B_LS*Ktilde)',     P]                  >= 0;
        cvx_end
        K_vec_H2(:,i) = reshape(Ktilde*inv(P), [], 1);
    
        % Regularization proposed in the literature for comparison 
        % with {2, 3}
        
        % Quadratic regularization for covariance parametrization
        % H(V, P) = lambda*trace(V*P*V'*Sigma_z)
        % Proposed in "Regularization for Covariance Parameterization of
        % Direct Data-Driven LQR Control" by Feiran Zhao, Allesandro Chiuso
        % and Florian Dörfler
        X0_bar = X0*Z'/ell;
        U_bar = U*Z'/ell;
        X_bar = X*Z'/ell;
        cvx_begin SDP
            variable P(n, n) semidefinite
            variable S(m+n, n)
            variable L(m, m) semidefinite
            variable N(m+n, m+n) semidefinite
            minimize(trace(Q*P)+trace(R*L)+lambda(i)*trace(N*Sigma_z))
            subject to
                X0_bar*S == P;
                P >= eye(n);
                [P-eye(n), X_bar*S; (X_bar*S)', P] >= 0;
                [L, U_bar*S; (U_bar*S)', P] >= 0;
                [N, S; S', P] >= 0;
        cvx_end
        K_vec_covar(:, i) = U_bar*S*inv(P);
    

    end    
end

figure
% Plot set of stabilizing controller for (A_LS, B_LS)
patch(V_stab(1,:)', V_stab(2,:)', 'w', 'FaceColor', 'none')
hold on
% {2}
plot(K_vec_H2(1,:), K_vec_H2(2, :), 'Color', Cmap(1,:), 'LineWidth', 1) 
% Comparison of {2, 3} with tr(Pi_perp*G*P*G')
% Shift these up/down a little, since they overlap
plot(K_vec_H23(1,:), K_vec_H23(2, :)+0.05, 'Color', Cmap(4,:), 'LineWidth', 0.5) % {2, 3}
plot(K_vec_covar(1,:), K_vec_covar(2, :)-0.05, '--', 'Color', Cmap(4,:), 'LineWidth', 0.5) % trace(V*P*V'*Sigma_z)
% Mark end points to verify they converge to different values
plot(K_vec_H2(1, end), K_vec_H2(2, end), 'x', 'Color', Cmap(1,:))
plot(K_vec_H23(1, end), K_vec_H23(2, end), 'x', 'Color', Cmap(4,:))
% End point of {2} coincides with K_LS. Add a label to it
plot(K_LS(1), K_LS(2), 'x', 'Color', Cmap(1,:))
text(K_LS(1)+0.1, K_LS(2)+0.65, '${K}_{\mathrm{LS}}$', 'Interpreter','latex', 'fontsize',labelsize)

% All three regularizations coince with the certainty equivalent (CE)
% solution for lambda = 0. Mark this point and add a label.
plot(K_CE(1), K_CE(2), 'xk')
text(K_CE(1)+0.15, K_CE(2)+0.5, '${K}_{\mathrm{CE}}$', 'Interpreter','latex', 'fontsize',labelsize)

% Note that, if K_LS is not stabilizing (A_LS, B_LS), then case {2} won't
% converge to K_LS. This is a natural consequence of A_cl = A_LS + B_LS*K
% being enforced as an equality constraint. Due to space restrictions, this
% effect is not discussed in the paper.
% Warning to explain different results. 
if any(abs(eig(A_LS+B_LS*K_LS))>0)
    title('Warning! $K_\mathrm{LS}$ does not stabilize $(A_\mathrm{LS}, B_\mathrm{LS})$!', 'Interpreter', 'latex')
end

% Figure labeling etc
set(gca, 'fontsize',labelsize)
legend('', '$\{2\}$', '$\{2, 3\}$', '$\mathrm{tr}(V P V^{\top}\hat{\Sigma}_{z})$', 'Interpreter', 'latex')
grid on; box on
xlabel('$\overline{{K}}_1$', 'Interpreter', 'latex'); 
ylabel('$\overline{{K}}_2$', 'Interpreter', 'latex'); 
set(gcf,'position',0.6*[0 0, 780 390])
set(gcf,'PaperPositionMode','Auto')



%% Third experiment: Visualizing the effect of the "third term" involving
% the initial state covariance Sigma_x0

% This part does not have a flag to turn the computations on/off, since
% they are quite fast

% Visualzation for lambda in {0, 10^(decStart), 10^(decStart+1), 10^(decStart+2)} 
decStart = 0;
lambda = [0, logspace(decStart, decStart+2, 3)];
% Specify number of grid points in each direction
n1 = 21;
n2 = 21;
% Create grid for evaluation
x1 = linspace(-1, 1, n1);
x2 = linspace(-1, 1, n2);
[XX1, XX2] = meshgrid(x1, x2);
dXXplus1 = zeros(n1, n2, 4);
dXXplus2 = zeros(n1, n2, 4);

% Visualization of example trajectories are only provided for the data
% used in the paper, since they were hand-placed for that specific
% figure.
% Initial states for example trajectories
X0s = [0,  0.9, 0.9, 0.9,  0,   -0.9, -0.9, -0.9;
      0.9, 0.9, 0,  -0.9, -0.9, -0.9,  0,    0.9];
nSim = 20;
% Allocate empty matrix for (vectorized) simulation results
XSim_vec = zeros(n*(nSim+1), size(X0s, 2), 4);


for i = length(lambda):-1:1
    % Covariance parametrization with case {3}
    cvx_begin SDP
        variable P(n, n) semidefinite
        variable Ktilde(m, n)
        variable Acltilde(n,n)
        variable L(m, m) semidefinite
        minimize(trace(Q*P) + trace(R*L) + lambda(i)*trace(P*inv(Sigma_x0))  )
        subject to
            P >= eye(n);
            [L, Ktilde;
             Ktilde', P] >= 0;
            [P-eye(n),              A_LS*P+B_LS*Ktilde;
             (A_LS*P+B_LS*Ktilde)',     P]                  >= 0;
    cvx_end
    % Optimal controller
    K = Ktilde*inv(P);
    % Synthesized closed-loop matrix
    A_cl = A_LS+B_LS*K;
    % System matrix of dx(k+1) = x(k+1) - x(k) = (A_cl - I)*x(k)
    % for quiver plot
    dA_cl = A_cl-eye(n);
    % Evaluate dA_cl on grid
    for j = 1:numel(XX1)
        dXplus = dA_cl*[XX1(j); XX2(j)];
        [row, col] = ind2sub([n1, n2], j);
        dXXplus1(row, col, i) = dXplus(1);
        dXXplus2(row, col, i) = dXplus(2);
    end
    
    % Example trajectories for the data used in the paper
    if ~randomize_experiment
        % Generate 8 example trajectories 
        for j = 1:size(X0s, 2)
            XSim_tmp = [X0s(:,j), zeros(n, nSim)];
            for k = 1:nSim
                XSim_tmp(:, k+1) = A_cl*XSim_tmp(:,k);
            end
            % Save (vectorized) results
            XSim_vec(:, j, i) = reshape(XSim_tmp, [], 1);
        end
    end
end

% Plot results
figure
t = tiledlayout(2, 2, 'TileSpacing', 'tight');
subplots = [];
for i = length(lambda):-1:1
    subplots = [subplots, nexttile(i)];
    if ~isMATLABReleaseOlderThan("R2024b")
        % The following code scales the quiver plots equivalently across all
        % subplots for easier comparison, which is the version in the paper
        % Unfortunately, it seems to be incompatible with Matlab R2023b 
        % (and presumably also older realses)
        if i == 4
            q = quiver(XX1, XX2, dXXplus1(:, :, i), dXXplus2(:, :, i), 'Color', Cmap(4,:));
            qScale = q.ScaleFactor;
        else
            quiver(XX1, XX2, qScale*dXXplus1(:, :, i), qScale*dXXplus2(:, :, i), 'off', 'Color', Cmap(4,:));
        end
    else
        % This version scales the quiver plots individually for all
        % subplots
        quiver(XX1, XX2, dXXplus1(:, :, i), dXXplus2(:, :, i), 'Color', Cmap(4,:));
    end
    hold on
    % Visualization of example trajectories. Only done for the data in the
    % paper, since the following code is quite specific
    if ~randomize_experiment
        for j = 1:size(X0s, 2)
            % Extract relevant data from results
            XSim = reshape(XSim_vec(:, j, i), n, []);
            plot(XSim(1,:), XSim(2,:), 'k')
            plot(XSim(1,1), XSim(2,1), '.k', 'MarkerSize', 10)
            h = annotation('arrow');
            if i == 4 && (j == 4 || j == 8)
                % Place these separately, since they are too close to the
                % starting point, otherwise
                set(h,'parent', gca, ...
                'position', [XSim(1,1),XSim(2,1),0.8*(XSim(1,2)-XSim(1,1)),0.8*(XSim(2,2)-XSim(2,1))], ...
                'HeadLength', 3, 'HeadWidth', 3, 'HeadStyle', 'cback1', 'Color', 'k');
            else
                set(h,'parent', gca, ...
                'position', [XSim(1,1),XSim(2,1),0.5*(XSim(1,2)-XSim(1,1)),0.5*(XSim(2,2)-XSim(2,1))], ...
                'HeadLength', 3, 'HeadWidth', 3, 'HeadStyle', 'cback1', 'Color', 'k');
            end
        end
    end
    % Figure labeling etc
    xlim([-1, 1]); ylim([-1, 1])
    grid on
    box on
    set(gca, 'FontSize', labelsize-3)
    if i == 1
        title('$\lambda = 0$','interpreter','Latex', 'FontSize', labelsize-1)
    else
        title(sprintf('$\\lambda = 30 \\cdot 10^{%i}$', floor(log10(lambda(i)))) ,'interpreter','Latex', 'FontSize', labelsize-1)
    end
end

% Figure labeling etc
xlabel(t, '$x_1$', 'Interpreter', 'latex','fontsize',labelsize)
t.XLabel.VerticalAlignment = 'baseline';
ylabel(t, '$x_2$', 'Interpreter', 'latex','fontsize',labelsize)
t.YLabel.VerticalAlignment = 'top';
set(gcf,'position',0.5*[0 0, 780 700])
set(gcf,'PaperPositionMode','Auto')


