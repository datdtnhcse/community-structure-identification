import numpy as np

def fowlkes_mallows_index(ground_truth, clustering_result):
    tp = fp = tn = fn = 0
    n = len(ground_truth)

    for i in range(n):
        for j in range(i + 1, n):
            same_ground_truth = ground_truth[i] == ground_truth[j]
            same_cluster = clustering_result[i] == clustering_result[j]

            if same_ground_truth and same_cluster:
                tp += 1
            elif not same_ground_truth and not same_cluster:
                tn += 1
            elif not same_ground_truth and same_cluster:
                fp += 1
            else:
                fn += 1

    return tp / np.sqrt((tp + fp) * (tp + fn))
    
def mutual_information(ground_truth, clustering_result):
    n = len(ground_truth)
    ground_truth_counts = {}
    clustering_counts = {}
    joint_counts = {}
    mutual_info = 0.0

    for i in range(n):
        ground_truth_label = ground_truth[i]
        clustering_label = clustering_result[i]

        if ground_truth_label in ground_truth_counts:
            ground_truth_counts[ground_truth_label] += 1
        else:
            ground_truth_counts[ground_truth_label] = 1

        if clustering_label in clustering_counts:
            clustering_counts[clustering_label] += 1
        else:
            clustering_counts[clustering_label] = 1

        if (ground_truth_label, clustering_label) in joint_counts:
            joint_counts[(ground_truth_label, clustering_label)] += 1
        else:
            joint_counts[(ground_truth_label, clustering_label)] = 1

    for (ground_truth_label, clustering_label), count in joint_counts.items():
        p_joint = count / n
        p_ground_truth = ground_truth_counts[ground_truth_label] / n
        p_clustering = clustering_counts[clustering_label] / n

        mutual_info += p_joint * np.log2(p_joint / (p_ground_truth * p_clustering))

    return mutual_info

# Tính Entropy
def entropy(labels):
    n = len(labels)
    label_counts = {}

    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    entropy = 0.0
    for count in label_counts.values():
        p = count / n
        entropy -= p * np.log2(p)

    return entropy

# Tính Normalized Mutual Information
def normalized_mutual_information(ground_truth, clustering_result):
    mi = mutual_information(ground_truth, clustering_result)
    h_ground_truth = entropy(ground_truth)
    h_clustering = entropy(clustering_result)

    nmi = 2 * mi / (h_ground_truth + h_clustering)

    return nmi

def adjusted_rand_index(ground_truth, clustering_result):
    n = len(ground_truth)
    a = b = c = d = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_ground_truth = ground_truth[i] == ground_truth[j]
            same_cluster = clustering_result[i] == clustering_result[j]

            if same_ground_truth and same_cluster:
                a += 1
            elif same_ground_truth and not same_cluster:
                b += 1
            elif not same_ground_truth and same_cluster:
                c += 1
            else:
                d += 1

    rand_index = (a + d) / (a + b + c + d)
    expected_index = ((a + b) * (a + c) + (c + d) * (b + d)) / (a + b + c + d)**2
    adjusted_rand_index = (rand_index - expected_index) / (1 - expected_index)

    return adjusted_rand_index
