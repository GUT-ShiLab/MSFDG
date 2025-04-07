% 读取数据并去除第一列，保留样本名（第一行自动作为列索引）
ba_raw = readmatrix('similarity_matrix_braycurtis_Cirrhosis_A.csv');
sample_names = readcell('similarity_matrix_braycurtis_Cirrhosis_A.csv', 'Range', 'A2:A500'); % 读取第一列样本名
ba = ba_raw(:, 2:end); % 去除第一列，但保留第一行

bm_raw = readmatrix('similarity_matrix_braycurtis_Cirrhosis_M.csv');
bm = bm_raw(:, 2:end);

ca_raw = readmatrix('similarity_matrix_cosine_Cirrhosis_A.csv');
ca = ca_raw(:, 2:end);

cm_raw = readmatrix('similarity_matrix_cosine_Cirrhosis_M.csv');
cm = cm_raw(:, 2:end);

ja_raw = readmatrix('similarity_matrix_jaccard_Cirrhosis_A.csv');
ja = ja_raw(:, 2:end);

jm_raw = readmatrix('similarity_matrix_jaccard_Cirrhosis_M.csv');
jm = jm_raw(:, 2:end);

% 进行SNF操作
[Similarity] = SNF({ba, bm, ca, cm, ja, jm}, 4, 5);

% 将样本名添加到最终矩阵中
Similarity_with_labels = array2table(Similarity, 'RowNames', sample_names, 'VariableNames', sample_names);

% 保存结果为 CSV 文件
writetable(Similarity_with_labels, 'fused_Cirrhosis_matrix.csv', 'WriteRowNames', true);
