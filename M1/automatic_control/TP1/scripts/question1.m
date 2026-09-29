%% Problem 1: Discretization
fprintf('\n--- Problem 1 ---\n');
Ts = 0.1; % 샘플링 주기 Ts = 0.1s

% Q1.1: Continuous transfer function 정의 및 ZOH 이산화
s = tf('s');
G = (2*s + 1) / (s^2 + 2*s + 1);
Gd = c2d(G, Ts, 'zoh');

disp('연속 전달함수 G(s):');
disp(G);
disp('이산 전달함수 Gd(z) [Ts = 0.1s]:');
disp(Gd);

% Q1.2: 연속계 및 이산계 계단 응답(Step Response) 비교 시각화
f1 = figure(1);
step(G, 'r-', Gd, 'b--');
legend('Continuous G(s)', 'Discrete Gd(z)', 'Location', 'Best');
setup_figure(f1, 'Question 1.2: Step Response Comparison (G vs Gd)');