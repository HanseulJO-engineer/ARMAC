%% Problem 3: Stability Study
fprintf('\n--- Problem 3 ---\n');
Ts = 0.1;
z = tf('z', Ts);

% Q3.1: 개루프 시스템 및 극점 확인
H = 1 / ((z - 0.4)*(z - 0.8));
p_H = pole(H);
fprintf('Q3.1 개루프 극점: %f, %f\n', p_H(1), p_H(2));

if all(abs(p_H) < 1)
    fprintf('개루프 시스템 상태: 안정 (Stable)\n');
else
    fprintf('개루프 시스템 상태: 불안정 (Unstable)\n');
end

% Q3.2 & Q3.3: 3가지 K값에 대한 폐루프 전달함수 생성
G_stable   = feedback(0.5 * H, 1);   % K = 0.5 (Stable)
G_lim      = feedback(0.68 * H, 1);  % K = 0.68 (Limit of Stability)
G_unstable = feedback(1.0 * H, 1);   % K = 1.0 (Unstable)

disp('K = 1.0 일 때 폐루프 전달함수 G(z):');
disp(G_unstable);

% Q3.4: 근궤적(rlocus) 분석 및 단위원 표시
f3_1 = figure(3);
rlocus(H);
hold on;
th = linspace(0, 2*pi, 300);
plot(cos(th), sin(th), 'k--', 'LineWidth', 1.5);
setup_figure(f3_1, 'Question 3.4: Root Locus of H(z)');

% Q3.4: K값별 계단 응답 비교 (시간축을 2.5초로 조정하여 3가지 상태 명확화)
t_vec = 0:Ts:2.5; % 발산 초기 특성을 보기 위해 2.5초로 제한

f3_2 = figure(4);

% 상단: 한 그래프 내 비교
subplot(2,1,1);
step(G_stable, 'g', G_lim, 'b', G_unstable, 'r', t_vec);
legend('K = 0.5 (Stable)', 'K = 0.68 (Limit)', 'K = 1.0 (Unstable)', 'Location', 'NorthWest');
setup_figure(f3_2, 'Question 3.4: Step Response Comparison (t = 0 ~ 2.5s)');

% 하단: 3가지 동작 특성 강조 표시
subplot(2,1,2);
[y_s, ~] = step(G_stable, t_vec);
[y_l, ~] = step(G_lim, t_vec);
[y_u, ~] = step(G_unstable, t_vec);

plot(t_vec, y_s, 'g-o', 'LineWidth', 1.2, 'MarkerSize', 4); hold on;
plot(t_vec, y_l, 'b-s', 'LineWidth', 1.2, 'MarkerSize', 4);
plot(t_vec, y_u, 'r-^', 'LineWidth', 1.2, 'MarkerSize', 4);
grid on; box on;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Amplitude', 'FontWeight', 'bold');
title('Detailed Behavioral Comparison', 'FontWeight', 'bold');
legend('K = 0.5 (Decaying / Stable)', ...
    'K = 0.68 (Sustained Oscillation / Limit)', ...
    'K = 1.0 (Growing Oscillation / Unstable)', ...
    'Location', 'NorthWest');