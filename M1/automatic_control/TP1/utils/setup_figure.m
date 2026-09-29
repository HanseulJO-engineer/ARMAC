function setup_figure(fig_handle, title_text)
figure(fig_handle);
grid on; box on;

% figure 내부의 실제 axes 객체 탐색
ax = findobj(fig_handle, 'Type', 'axes');
if isempty(ax)
    ax = gca;
end

% 축 라벨 및 제목 적용
xlabel(ax, 'Time (s) / Real Axis', 'FontSize', 11, 'FontWeight', 'bold');
ylabel(ax, 'Amplitude / Imaginary Axis', 'FontSize', 11, 'FontWeight', 'bold');
title(ax, title_text, 'FontSize', 12, 'FontWeight', 'bold');

% 지원하는 속성만 안전하게 적용 (StepPlot 객체 오류 방지)
for k = 1:length(ax)
    if isprop(ax(k), 'LineWidth')
        set(ax(k), 'LineWidth', 1.2);
    end
    if isprop(ax(k), 'FontSize')
        set(ax(k), 'FontSize', 10);
    end
end
end