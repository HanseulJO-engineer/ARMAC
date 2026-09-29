%% Problem 2: Discrete transfer function
fprintf('\n--- Problem 2 ---\n');
Ts = 0.1;

% Q2.1: tf 및 zpk 형식 작성
z = tf('z', Ts);
Hd_tf = (0.047*z + 0.046) / (z^2 - 1.81*z + 0.9);
Hd_zpk = zpk(Hd_tf);

disp('Hd(z) - TF Form:');
disp(Hd_tf);
disp('Hd(z) - ZPK Form:');
disp(Hd_zpk);

% Q2.2: pzmap 시각화
f2 = figure(2);
pzmap(Hd_tf);
setup_figure(f2, 'Question 2.2: Pole-Zero Map of Hd(z)');

% Q2.3: pole 명령을 이용한 극점 계산
poles_Hd = pole(Hd_tf);
disp('Hd(z)의 극점(Poles):');
disp(poles_Hd);