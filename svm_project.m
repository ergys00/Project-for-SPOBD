clear;
clc;
close all;

% Preparazione dei dati
data = readtable('loan_data.csv');

y = data.loan_status;
A = [data.person_age, data.person_income, data.person_emp_exp, ...
     data.loan_amnt, data.loan_int_rate, data.loan_percent_income, ...
     data.cb_person_cred_hist_length, data.credit_score];

% Dummizzazione delle variabili categoriali
home      = grp2idx(categorical(data.person_home_ownership));
intent    = grp2idx(categorical(data.loan_intent));
education = grp2idx(categorical(data.person_education));
gender    = grp2idx(categorical(data.person_gender));
default   = grp2idx(categorical(data.previous_loan_defaults_on_file));

A = [A, home, intent, education, gender, default];
A = normalize(A);
y(y==0) = -1;

n = size(A,1);
cv = cvpartition(n,'HoldOut',0.3);
A_tr = A(training(cv),:);
y_tr = y(training(cv));
A_te = A(test(cv),:);
y_te = y(test(cv));

[m,d] = size(A_tr);

% Parametri del modello
C = 1;
lambda = 3e-2;
rho = 1;         
N = 5;
maxit = 30;
tol = 1e-4;

% SVM Centralizzata
fprintf('\n *** SVM CENTRALIZZATA ***\n');
tic;
cvx_begin quiet
    cvx_solver sedumi
    variables w(d) b xi(m)
    minimize( 0.5*sum_square(w) + lambda*sum_square(w) + C*sum(xi) )
    subject to
        y_tr.*(A_tr*w + b) >= 1 - xi;
        xi >= 0;
cvx_end
t_c = toc;

scores = A_te*w + b;
ypred = sign(scores); ypred(ypred==0)=1;
acc_c = mean(ypred == y_te);
fprintf('Accuratezza centralizzata: %.2f%% (tempo = %.3fs)\n', acc_c*100, t_c);

% Variabili per i grafici
w_centralized = w;
b_centralized = b;


% SVM Distribuita - SPLIT PER ESEMPI
fprintf('\n*** SVM DISTRIBUITA (Split-by-Examples) ***\n');

% Partizionamento dei campioni in N blocchi
idx = crossvalind('Kfold', m, N);
S = arrayfun(@(j)find(idx==j),1:N,'uni',0);  % S_j = insieme dei campioni del nodo j

% Inizializzazione variabili ADMM
w_j = zeros(d,N);      
b_j = zeros(1,N);      
u_j = zeros(d,N);      
z = zeros(d,1);        

prE = zeros(maxit,1);  % residuo primale
drE = zeros(maxit,1);  % residuo duale

% Storico per grafici
z_hist = zeros(d,maxit);          
b_examples_hist = zeros(maxit,1);  % media dei bias locali per iterazione

tic;
for k = 1:maxit
    % 1: Aggiornamento locale w_j, b_j in ogni nodo
    for j = 1:N
        A_j = A_tr(S{j},:);
        y_j = y_tr(S{j});
        
        cvx_begin quiet
            cvx_solver sedumi
            variable w_local(d)
            variable b_local
            minimize( sum(max(0,1 - y_j.*(A_j*w_local + b_local))) ...
                      + lambda*sum_square(w_local) ...
                      + (rho/2)*sum_square(w_local - z + u_j(:,j)) )
        cvx_end
        
        w_j(:,j) = w_local;
        b_j(j) = b_local;
    end
    
    % 2: Aggiornamento della variabile di consenso z
    z_old = z;
    z = mean(w_j + u_j, 2);

    % 3: Aggiornamento delle variabili duali u_j
    for j = 1:N
        u_j(:,j) = u_j(:,j) + (w_j(:,j) - z);
    end
    
    % Calcolo residui
    prE(k) = sqrt(sum(vecnorm(w_j - z).^2));
    drE(k) = sqrt(N)*norm(rho*(z - z_old));

    % Storico
    z_hist(:,k) = z;
    b_examples_hist(k) = mean(b_j);
    
    if prE(k)<tol && drE(k)<tol
        fprintf('Convergenza (Split-by-Examples) raggiunta alla iterazione %d\n', k);
        break;
    end
end
t_e = toc;
K_e = k;

% Soluzione
w_examples = z;
b_examples = mean(b_j);

scores = A_te*w_examples + b_examples;
ypred = sign(scores); ypred(ypred==0)=1;
acc_e = mean(ypred == y_te);

fprintf('Accuratezza Split-by-Examples: %.2f%% (tempo = %.3fs, iter = %d)\n', ...
        acc_e*100, t_e, K_e);



% SVM Distribuita - SPLIT PER FEATURE
fprintf('\n*** SVM DISTRIBUITA (Split-by-Features) ***\n');

% Partizionamento delle feature in N blocchi
feat_per_node = floor(d/N);
F = cell(N,1);  
for i = 1:N-1
    F{i} = ((i-1)*feat_per_node+1):(i*feat_per_node);
end
F{N} = ((N-1)*feat_per_node+1):d;

% Inizializzazione variabili ADMM
x_i = cell(N,1);  
for i = 1:N
    x_i{i} = zeros(length(F{i}),1);
end

z_mean = zeros(m,1);  
u = zeros(m,1);      
b = 0;               

prF = zeros(maxit,1); 
drF = zeros(maxit,1);

% Storico
wF_hist = zeros(d,maxit);
bF_hist = zeros(maxit,1);

tic;
for k = 1:maxit

    Ax_old = zeros(m,1);
    for i = 1:N
        A_i = A_tr(:, F{i});
        Ax_old = Ax_old + A_i * x_i{i};
    end
    
    % 1: Aggiornamento di x(i) (regressione ridge)
    for i = 1:N
        A_i = A_tr(:, F{i});
        Ax_minus_i = Ax_old - A_i*x_i{i};  
        s_k = z_mean - u - Ax_minus_i - b;

        d_i = length(F{i});
        x_i{i} = (A_i'*A_i + (2*lambda/rho)*eye(d_i)) \ (A_i'*s_k);
    end

    % Calcolo nuovo Ax
    Ax = zeros(m,1);
    for i = 1:N
        Ax = Ax + A_tr(:, F{i}) * x_i{i};
    end

    % 2: Aggiornamento bias b
    b = mean(z_mean - u - Ax);

    % 3: Aggiornamento z (operatore prox della hinge loss)
    z_old = z_mean;

    C_hinge = 10;  
    gamma = C_hinge / rho;
    c = Ax + b + u;

    for l = 1:m
        y_l = y_tr(l);
        yc = y_l * c(l);

        if yc >= 1
            z_mean(l) = c(l);
        elseif yc <= 1 - gamma
            z_mean(l) = c(l) + gamma*y_l;
        else
            z_mean(l) = y_l;
        end
    end

    % 4: Aggiornamento variabile duale u
    u = u + (Ax + b - z_mean);

    % Calcolo residui
    prF(k) = norm(Ax + b - z_mean);
    drF(k) = rho * norm(z_mean - z_old);

    % Storico
    w_temp = zeros(d,1);
    for i = 1:N
        w_temp(F{i}) = x_i{i};
    end
    wF_hist(:,k) = w_temp;
    bF_hist(k)   = b;

    if prF(k)<tol && drF(k)<tol
        fprintf('Convergenza (Split-by-Features) raggiunta alla iterazione %d\n', k);
        break;
    end
end
t_f = toc;
K_f = k;

% Ricostruzione vettore w
w_features = zeros(d,1);
for i = 1:N
    w_features(F{i}) = x_i{i};
end

scores = A_te*w_features + b;
ypred = sign(scores); ypred(ypred==0)=1;
acc_f = mean(ypred == y_te);

fprintf('Accuratezza Split-by-Features: %.2f%% (tempo = %.3fs, iter = %d)\n', ...
        acc_f*100, t_f, K_f);



% PCA + ADMM - Split-by-Features
fprintf('\n*** PCA MULTIPLA (Split-by-Features) ***\n');

maxPC = min(13, d);
results = zeros(maxPC,4); % [accuracy, time, iterations, n_components]

% PCA sul dataset completo
[V, A_pca, ~, ~, explained] = pca(A);

% Mostra varianza spiegata
for k_pca = 1:maxPC
    fprintf('%d\t%.2f\n', k_pca, sum(explained(1:k_pca)));
end

for k_pca = 1:maxPC
    fprintf('\n--- PCA con %d componenti ---\n', k_pca);

    % Estrazione delle prime k componenti
    A_k = A_pca(:, 1:k_pca);
    A_k_tr = A_k(training(cv), :);
    A_k_te = A_k(test(cv), :);
    [m_k, d_k] = size(A_k_tr);

    % Partizionamento delle feature
    feat_per_node_k = floor(d_k/N);
    F_k = cell(N,1);
    for i = 1:N-1
        F_k{i} = ((i-1)*feat_per_node_k+1):(i*feat_per_node_k);
    end
    F_k{N} = ((N-1)*feat_per_node_k+1):d_k;

    % Inizializzazione ADMM
    x_i_k = cell(N,1);
    for i=1:N
        x_i_k{i} = zeros(length(F_k{i}),1);
    end
    z_mean_k = zeros(m_k,1);
    u_k = zeros(m_k,1);
    b_k = 0;

    tic;
    for iter = 1:maxit

        Ax_old = zeros(m_k,1);
        for i = 1:N
            A_i = A_k_tr(:, F_k{i});
            Ax_old = Ax_old + A_i*x_i_k{i};
        end

        % Aggiornamento x^(i)
        for i=1:N
            A_i = A_k_tr(:, F_k{i});
            Ax_minus_i = Ax_old - A_i*x_i_k{i};
            s_k = z_mean_k - u_k - Ax_minus_i - b_k;

            d_i = length(F_k{i});
            x_i_k{i} = (A_i'*A_i + (2*lambda/rho)*eye(d_i)) \ (A_i'*s_k);
        end
        
        Ax = zeros(m_k,1);
        for i=1:N
            Ax = Ax + A_k_tr(:, F_k{i})*x_i_k{i};
        end

        % Aggiornamento b
        b_k = mean(z_mean_k - u_k - Ax);

        % Aggiornamento z_mean 
        z_old = z_mean_k;
        c = Ax + b_k + u_k;
        C_hinge = 10; 
        gamma = C_hinge/rho;

        for l=1:m_k
            y_l = y_tr(l);
            yc = y_l*c(l);
            if yc>=1
                z_mean_k(l) = c(l);
            elseif yc<=1-gamma
                z_mean_k(l) = c(l) + gamma*y_l;
            else
                z_mean_k(l) = y_l;
            end
        end

        % Aggiornamento u
        u_k = u_k + (Ax + b_k - z_mean_k);

        % Verifica convergenza
        if norm(Ax+b_k-z_mean_k) < tol && rho*norm(z_mean_k-z_old) < tol
            break;
        end
    end
    t_pca = toc;

    % Ricostruzione w
    w_k = zeros(d_k,1);
    for i=1:N
        w_k(F_k{i}) = x_i_k{i};
    end

    % Predizione
    scores = A_k_te*w_k + b_k;
    ypred = sign(scores); ypred(ypred==0)=1;
    acc_pca = mean(ypred==y_te);

    fprintf('  k=%d  accuratezza=%.2f%%  tempo=%.3fs  iterazioni=%d\n', ...
            k_pca, acc_pca*100, t_pca, iter);

    results(k_pca,:) = [acc_pca, t_pca, iter, k_pca];
end

% Risultati PCA
fprintf('\n=== RISULTATI MULTI-PCA ===\n');
fprintf('k\tAccuratezza(%%)\tTempo(s)\tIter\n');
for i=1:maxPC
    fprintf('%d\t%.2f\t\t%.3f\t%d\n', ...
            results(i,4), results(i,1)*100, results(i,2), results(i,3));
end

% Miglior trade-off
[~, best_idx] = min(abs(results(:,1) - acc_c) + 0.01*results(:,2));
fprintf('\n*** MIGLIOR TRADE-OFF: %d componenti (acc=%.2f%%, tempo=%.3fs)\n', ...
        results(best_idx,4), results(best_idx,1)*100, results(best_idx,2));


fontSize = 16;   
% PLOT residui ADMM (Split-by-Examples)
figure;
subplot(1,2,1);
semilogy(prE(1:K_e), 'LineWidth', 1.5);
title('Residuo Primale - Split-by-Examples', 'FontSize', fontSize);
xlabel('Iterazione', 'FontSize', fontSize);
ylabel('||w_j - z||', 'FontSize', fontSize);
set(gca,'FontSize',fontSize-2);
grid on;

subplot(1,2,2);
semilogy(drE(1:K_e), 'LineWidth', 1.5);
title('Residuo Duale - Split-by-Examples', 'FontSize', fontSize);
xlabel('Iterazione', 'FontSize', fontSize);
ylabel('||z_k - z_{k-1}||', 'FontSize', fontSize);
set(gca,'FontSize',fontSize-2);
grid on;

% PLOT residui ADMM (Split-by-Features)
figure;
subplot(1,2,1);
semilogy(prF(1:K_f), 'LineWidth', 1.5);
title('Residuo Primale - Split-by-Features', 'FontSize', fontSize);
xlabel('Iterazione', 'FontSize', fontSize);
ylabel('||Ax + b - z||', 'FontSize', fontSize);
set(gca,'FontSize',fontSize-2);
grid on;

subplot(1,2,2);
semilogy(drF(1:K_f), 'LineWidth', 1.5);
title('Residuo Duale - Split-by-Features', 'FontSize', fontSize);
xlabel('Iterazione', 'FontSize', fontSize);
ylabel('||z_k - z_{k-1}||', 'FontSize', fontSize);
set(gca,'FontSize',fontSize-2);
grid on;

% PLOT accuratezza vs numero componenti PCA
figure;
acc_pca_vec = results(:,1);
plot(1:maxPC, acc_pca_vec, '-o','LineWidth',1.5);
grid on;
title('Accuratezza vs Numero Componenti PCA', 'FontSize', fontSize);
xlabel('Componenti PCA', 'FontSize', fontSize);
ylabel('Accuratezza', 'FontSize', fontSize);
set(gca,'FontSize',fontSize-2);

% PLOT varianza spiegata 
cumVar = cumsum(explained(1:maxPC));
figure;
plot(1:maxPC, cumVar, '-o','LineWidth',1.5);
grid on;
title('Varianza Spiegata Cumulativa PCA', 'FontSize', fontSize);
xlabel('Componenti', 'FontSize', fontSize);
ylabel('Varianza (%)', 'FontSize', fontSize);
ylim([min(cumVar)-3 100]);
set(gca,'FontSize',fontSize-2);

% PLOT tempo vs numero componenti
figure;
plot(1:maxPC, results(:,2), '-o','LineWidth',1.5);
grid on;
title('Tempo ADMM (PCA) vs Numero Componenti', 'FontSize', fontSize);
xlabel('Componenti PCA', 'FontSize', fontSize);
ylabel('Tempo (s)', 'FontSize', fontSize);
set(gca,'FontSize',fontSize-2);

% Istogramma accuratezze finali
figure;
accuracies_summary = [acc_c, acc_e, acc_f, max(results(:,1))];
bar(accuracies_summary, 'LineWidth',1.2);
set(gca,'XTickLabel',{'Centralizzata','Split-Examples','Split-Features','PCA-best'}, ...
         'FontSize',fontSize-2);
ylabel('Accuratezza', 'FontSize', fontSize);
title('Confronto Accuratezze Finali', 'FontSize', fontSize);
ylim([min(accuracies_summary)-0.02, max(accuracies_summary)+0.01]);
grid on;

% DISAGREEMENT (residuo primale)
K_max = max(K_e, K_f);

dis_examples = NaN(K_max,1);
dis_features = NaN(K_max,1);
dis_examples(1:K_e) = prE(1:K_e);
dis_features(1:K_f) = prF(1:K_f);

figure;
semilogy(1:K_max, dis_examples, 'LineWidth',1.5); hold on;
semilogy(1:K_max, dis_features, 'LineWidth',1.5);
grid on;
xlabel('Iterazioni', 'FontSize', fontSize);
ylabel('Disagreement (residuo primale)', 'FontSize', fontSize);
title('Confronto Disagreement: Split-by-Examples vs Split-by-Features', ...
      'FontSize', fontSize);
legend('Split-by-Examples','Split-by-Features','Location','SouthWest', ...
       'FontSize', fontSize-1);
set(gca,'FontSize',fontSize-2);

% Accuratezza vs iterazioni
acc_c_vec = acc_c * ones(K_max,1);

acc_e_iter = NaN(K_max,1);
for t = 1:K_e
    w_tmp = z_hist(:,t);
    scores_t = A_te*w_tmp + b_examples_hist(t);
    ypred_t = sign(scores_t); ypred_t(ypred_t==0) = 1;
    acc_e_iter(t) = mean(ypred_t == y_te);
end

acc_f_iter = NaN(K_max,1);
for t = 1:K_f
    w_tmp = wF_hist(:,t);
    scores_t = A_te*w_tmp + bF_hist(t);
    ypred_t = sign(scores_t); ypred_t(ypred_t==0) = 1;
    acc_f_iter(t) = mean(ypred_t == y_te);
end

figure;
plot(1:K_max, acc_c_vec, 'k--','LineWidth',1.5); hold on;
plot(1:K_max, acc_e_iter, 'r','LineWidth',1.5);
plot(1:K_max, acc_f_iter, 'b','LineWidth',1.5);
grid on;
xlabel('Iterazioni', 'FontSize', fontSize);
ylabel('Accuratezza di Classificazione', 'FontSize', fontSize);
title('Confronto Accuratezza: Centralizzata vs Distribuita', 'FontSize', fontSize);
legend('Centralizzata','Split-by-Examples','Split-by-Features', ...
       'Location','SouthEast','FontSize',fontSize-1);
set(gca,'FontSize',fontSize-2);
