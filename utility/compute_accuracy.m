function acc = compute_accuracy(ground_lables, actual_ids, k)

total_cluster_num = 0;
for idx = 1 : k
    table = tabulate(ground_lables(actual_ids == idx));   
    if ~isempty(table)
        [~, row_idx] = max(table(:,3));              
        total_cluster_num = total_cluster_num + table(row_idx, 2);
    end
end
acc = total_cluster_num / length(ground_lables);

