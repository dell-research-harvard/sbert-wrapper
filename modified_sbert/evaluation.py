import os
import numpy as np
import csv
from itertools import combinations
from typing import List

from sentence_transformers import evaluation, LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score

import logging
from transformers import logging as lg
import wandb

import cluster_fns


lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


class BinaryClassificationEvaluator_wandb(evaluation.BinaryClassificationEvaluator):

    def compute_metrices(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True], ['manhattan', 'Manhattan-Distance', manhattan_distances, False], ['euclidean', 'Euclidean-Distance', euclidean_distances, False], ['dot', 'Dot-Product', dot_scores, True]]:
            # Note: newer versions of sbert have updated the spelling on manhatten to manhattan

            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                'accuracy': acc,
                'accuracy_threshold': acc_threshold,
                'f1': f1,
                'f1_threshold': f1_threshold,
                'precision': precision,
                'recall': recall,
                'ap': ap
            }

            wandb.log({
                f"Classification Accuracy {name}": acc,
                f"Classification Accuracy threshold {name}": acc_threshold,
                f"Classification F1 {name}": f1,
                f"Classification F1 threshold {name}": f1_threshold,
                f"Classification Precision {name}": precision,
                f"Classification Recall {name}": recall,
                f"Classification Average precision {name}": ap
            })

        return output_scores


class EmbeddingSimilarityEvaluator_wandb(evaluation.EmbeddingSimilarityEvaluator):

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}\n".format(
            eval_pearson_dot, eval_spearman_dot))

        wandb.log({
            "Embedding Similarity Cosine-Similarity Pearson": eval_pearson_cosine,
            "Embedding Similarity Cosine-Similarity Spearman": eval_spearman_cosine,
            "Embedding Similarity Manhattan-Distance Pearson": eval_pearson_manhattan,
            "Embedding Similarity Manhattan-Distance Spearman": eval_spearman_manhattan,
            "Embedding Similarity Euclidean-Distance Pearson": eval_pearson_euclidean,
            "Embedding Similarity Euclidean-Distance Spearman": eval_spearman_euclidean,
            "Embedding Similarity Dot-Product Pearson": eval_pearson_dot,
            "Embedding Similarity Dot-Product Spearman": eval_spearman_dot,
        })

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])

        if self.main_similarity == evaluation.SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == evaluation.SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == evaluation.SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == evaluation.SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")


class TripletEvaluator_wandb(evaluation.TripletEvaluator):

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        num_triplets = 0
        num_correct_cos_triplets, num_correct_manhattan_triplets, num_correct_euclidean_triplets = 0, 0, 0

        embeddings_anchors = model.encode(
            self.anchors, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_positives = model.encode(
            self.positives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_negatives = model.encode(
            self.negatives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )

        # Cosine distance
        pos_cos_distance = paired_cosine_distances(embeddings_anchors, embeddings_positives)
        neg_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_negatives)

        # Manhattan
        pos_manhattan_distance = paired_manhattan_distances(embeddings_anchors, embeddings_positives)
        neg_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_negatives)

        # Euclidean
        pos_euclidean_distance = paired_euclidean_distances(embeddings_anchors, embeddings_positives)
        neg_euclidean_distances = paired_euclidean_distances(embeddings_anchors, embeddings_negatives)

        for idx in range(len(pos_cos_distance)):
            num_triplets += 1

            if pos_cos_distance[idx] < neg_cos_distances[idx]:
                num_correct_cos_triplets += 1

            if pos_manhattan_distance[idx] < neg_manhattan_distances[idx]:
                num_correct_manhattan_triplets += 1

            if pos_euclidean_distance[idx] < neg_euclidean_distances[idx]:
                num_correct_euclidean_triplets += 1

        accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_manhattan = num_correct_manhattan_triplets / num_triplets
        accuracy_euclidean = num_correct_euclidean_triplets / num_triplets

        logger.info("Accuracy Cosine Distance:   \t{:.2f}".format(accuracy_cos * 100))
        logger.info("Accuracy Manhattan Distance:\t{:.2f}".format(accuracy_manhattan * 100))
        logger.info("Accuracy Euclidean Distance:\t{:.2f}\n".format(accuracy_euclidean * 100))

        wandb.log({
            "Triplet Accuracy Cosine Distance": accuracy_cos,
            "Triplet Accuracy Manhattan Distance": accuracy_manhattan,
            "Triplet Accuracy Euclidean Distance": accuracy_euclidean
        })

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

        if self.main_distance_function == evaluation.SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == evaluation.SimilarityFunction.MANHATTAN:
            return accuracy_manhattan
        if self.main_distance_function == evaluation.SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhattan, accuracy_euclidean)


class ClusterEvaluator_wandb(evaluation.SentenceEvaluator):   

    """
    Evaluate a model based on allocation of texts into correct clusters.
    Embeddings are clustered with the specified clustering algorithm using cosine distance. Best clustering parameters
    (distance threshold) are found using an approximate search method to speed to evaluation time.

    All possible combination of articles are split into pairs, with positives being in the same cluster and negatives
    being in different clusters.
    Metrics are precision, recall and F1.

    Returned metrics are F1 along with the optimal clustering threshold.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    :param cluster_type: Clustering algoritm to use. Supports "agglomerative" (hierarchical), "SLINK", "HDBScan"

    Modelled on: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py
    """

    def __init__(
            self,
            sentences1: List[str],
            sentences2: List[str],
            labels: List[int],
            name: str = '',
            batch_size: int = 512,
            show_progress_bar: bool = False,
            write_csv: bool = True,
            cluster_type: str = "agglomerative"
    ):

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.cluster_type = cluster_type

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "clustering_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "accuracy_threshold", "f1", "precision", "recall", "f1_threshold"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Cluster Evaluation of the model on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model)

        #Main score is F1
        main_score = scores['f1']

        file_output_data = [epoch, steps]

        for score in self.csv_headers:
            if score in scores:
                file_output_data.append(scores[score])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score

    def compute_metrices(self, model):

        sentences = []
        labels = []
        for i in range(len(self.sentences1)):

            if self.sentences1[i] not in sentences:
                sentences.append(self.sentences1[i])
            s1_id = sentences.index(self.sentences1[i])
            if self.sentences2[i] not in sentences:
                sentences.append(self.sentences2[i])
            s2_id = sentences.index(self.sentences2[i])

            if self.labels[i] == 1:
                labels.append([s1_id, s2_id])

        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar)

        # Normalize the embeddings to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def cluster_eval(threshold, embeddings, labels, cluster_type='agglomerative'):

            clustered_ids = cluster_fns.cluster(
                cluster_type,
                cluster_params={"threshold": threshold, "clustering linkage": 'average', "metric": 'cosine', "min cluster size": 2},
                corpus_embeddings=embeddings
            )

            # Convert every pair in a cluster into an edge
            cluster_edges = cluster_fns.edges_from_clusters(clustered_ids)

            metrics = cluster_fns.evaluate(pred_edges=cluster_edges, gt_edges=labels, print_metrics=False)

            total = len(list(combinations(range(len(embeddings)), 2)))
            cluster_tn = total - metrics["tps"] - metrics["fps"] - metrics["fns"]

            metrics["accuracy"] = (metrics["tps"] + cluster_tn)/total

            return metrics

        tenths = {}
        for threshold in [0.01] + [round(x, 2) for x in (np.linspace(0.1, 0.9, 9))] + [0.99]:
            tenths[threshold] = cluster_eval(threshold, embeddings, labels, cluster_type=self.cluster_type)

        def best_threshold(dictionary, metric):

            ths = list(dictionary.keys())
            scores = []

            for th in ths:
                scores.append(dictionary[th][metric])

            sorted_scores = sorted(scores)

            best_score = sorted_scores[-1]
            second_best_score = sorted_scores[-2]
            third_best_score = sorted_scores[-3]

            if best_score == second_best_score:
                best_indices = [i for i, x in enumerate(scores) if x == best_score]
                best_thresholds = []
                for idx in best_indices:
                    best_thresholds.append(ths[idx])

                best_th = max(best_thresholds)
                second_best_th = min(best_thresholds)

            elif second_best_score == third_best_score:
                second_indices = [i for i, x in enumerate(scores) if x == second_best_score]
                second_thresholds = []
                for idx in second_indices:
                    second_thresholds.append(ths[idx])

                best_th = max(second_thresholds)
                second_best_th = min(second_thresholds)

            else:
                best_idx = scores.index(best_score)
                best_th = ths[best_idx]

                second_best_idx = scores.index(second_best_score)
                second_best_th = ths[second_best_idx]

            return best_th, second_best_th

        max_f1_th, second_f1_th = best_threshold(tenths, metric='f_score')
        max_acc_th, second_acc_th = best_threshold(tenths, metric='accuracy')

        min_th = min(max_f1_th, second_f1_th, max_acc_th, second_acc_th)
        max_th = max(max_f1_th, second_f1_th, max_acc_th, second_acc_th)

        hundreths = {}
        for threshold in np.arange(min_th, max_th, 0.01):
            hundreths[threshold] = cluster_eval(threshold, embeddings, labels)

        hd_max_f1_th, _ = best_threshold(hundreths, 'f_score')
        hd_max_acc_th, _ = best_threshold(hundreths, 'accuracy')

        acc = hundreths[hd_max_acc_th]['accuracy']
        acc_threshold = hd_max_acc_th

        f1 = hundreths[hd_max_f1_th]['f_score']
        precision = hundreths[hd_max_f1_th]['precision']
        recall = hundreths[hd_max_f1_th]['recall']
        f1_threshold = hd_max_f1_th

        logger.info("Cluster Accuracy:           {:.2f}\t(Threshold: {:.2f})".format(acc * 100, acc_threshold))
        logger.info("Cluster F1:                 {:.2f}\t(Threshold: {:.2f})".format(f1 * 100, f1_threshold))
        logger.info("Cluster Precision:          {:.2f}".format(precision * 100))
        logger.info("Cluster Recall:             {:.2f}\n".format(recall * 100))

        output_scores = {
            'accuracy': acc,
            'accuracy_threshold': acc_threshold,
            'f1': f1,
            'f1_threshold': f1_threshold,
            'precision': precision,
            'recall': recall,
        }

        wandb.log({  
            "Cluster Accuracy": acc,
            "Cluster Accuracy threshold": acc_threshold,
            "Cluster F1": f1,
            "Cluster F1 threshold": f1_threshold,
            "Cluster Precision": precision,
            "Cluster Recall": recall
        })

        return output_scores


class CEBinaryClassificationEvaluator_wandb(CEBinaryClassificationEvaluator):

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CEBinaryClassificationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

        acc, acc_threshold = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)
        f1, precision, recall, f1_threshold = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(pred_scores, self.labels, True)
        ap = average_precision_score(self.labels, pred_scores)

        logger.info("Accuracy:           {:.2f}\t(Threshold: {:.4f})".format(acc * 100, acc_threshold))
        logger.info("F1:                 {:.2f}\t(Threshold: {:.4f})".format(f1 * 100, f1_threshold))
        logger.info("Precision:          {:.2f}".format(precision * 100))
        logger.info("Recall:             {:.2f}".format(recall * 100))
        logger.info("Average Precision:  {:.2f}\n".format(ap * 100))

        wandb.log({
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Average precision": ap
        })

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall, ap])

        return f1


class CEClusterEvaluator_wandb():   

    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name

        self.csv_file = "CEClusterEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "accuracy_threshold", "f1", "f1_threshold", "precision", "recall"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CEClusterEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

        acc, acc_threshold, f1, precision, recall, f1_threshold = self.find_best_acc_and_f1(pred_scores, self.sentence_pairs, self.labels)

        logger.info("Cluster Accuracy:           {:.2f}\t(Threshold: {:.2f})".format(acc * 100, acc_threshold))
        logger.info("Cluster F1:                 {:.2f}\t(Threshold: {:.2f})".format(f1 * 100, f1_threshold))
        logger.info("Cluster Precision:          {:.2f}".format(precision * 100))
        logger.info("Cluster Recall:             {:.2f}\n".format(recall * 100))

        wandb.log({     
            "Cluster Accuracy": acc,
            "Cluster Accuracy threshold": acc_threshold,
            "Cluster F1": f1,
            "Cluster F1 threshold": f1_threshold,
            "Cluster Precision": precision,
            "Cluster Recall": recall
        })

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall])

        return f1

    @staticmethod
    def find_best_acc_and_f1(scores, sentence_pairs, labels):

        assert len(scores) == len(labels)

        sentences = []
        pair_ids = []
        for pair in sentence_pairs:
            if pair[0] not in sentences:
                sentences.append(pair[0])
            s1_id = sentences.index(pair[0])
            if pair[1] not in sentences:
                sentences.append(pair[1])
            s2_id = sentences.index(pair[1])
            pair_ids.append([s1_id, s2_id])

        gt_edges = [pair_ids[i] for i in range(len(pair_ids)) if labels[i] == 1]
        gt_edges = cluster_fns.edges_from_clusters(cluster_fns.clusters_from_edges(gt_edges))   # Impose transitivity

        total_possible_edges = len(labels)

        thds = list(set([round(score, 2) for score in scores]))
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for th in thds:

            preds = [pair_ids[i] for i in range(len(pair_ids)) if scores[i] > th]

            # Impose transitivity
            pred_edges = cluster_fns.edges_from_clusters(cluster_fns.clusters_from_edges(preds))

            metrics = cluster_fns.evaluate(pred_edges=pred_edges, gt_edges=gt_edges, print_metrics=False)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics['f_score'])

            cluster_tn = total_possible_edges - metrics["tps"] - metrics["fps"] - metrics["fns"]
            accuracies.append((metrics["tps"] + cluster_tn) / total_possible_edges)

        # Find max values
        max_acc = max(accuracies)
        acc_idx = accuracies.index(max_acc)
        acc_threshold = thds[acc_idx]

        max_f1 = max(f1s)
        f1_idx = f1s.index(max_f1)
        precision = precisions[f1_idx]
        recall = recalls[f1_idx]
        f1_threshold = thds[f1_idx]

        return max_acc, acc_threshold, max_f1, precision, recall, f1_threshold
