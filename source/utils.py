import random

from sklearn.manifold import TSNE

from .constants import QUESTION_TYPES


def position_accuracy(true_answers, answers):
    cnt_correct = 0
    cnt_all = 0
    for ans, tans in zip(answers, true_answers):
        if ans is None:
            continue
        true_ans = tans[0]
        cnt_all += len(true_ans)
        for ans_symb, true_ans_symb in zip(list(ans), list(true_ans)):
            cnt_correct += ans_symb == true_ans_symb
    if cnt_all:
        return cnt_correct / cnt_all
    return None


def non_position_accuracy(true_answers, answers):
    cnt_correct = 0
    cnt_all = 0
    for ans, tans in zip(answers, true_answers):
        if ans is None:
            continue
        true_ans = tans[0]
        cnt_all += len(true_ans)
        for ans_symb in ans:
            cnt_correct += ans_symb in true_ans
    if cnt_all:
        return cnt_correct / cnt_all
    return None


def evaluate_model_on_different_tasks(answers, tasks, metrics=(position_accuracy, non_position_accuracy),
                                      need_print=True):
    types = QUESTION_TYPES
    result = {}
    for task_type in types:
        task_answers = []
        task_true_answers = []
        type_name = types[task_type]
        for i, task in enumerate(tasks):
            if task['type'] == task_type:
                task_answers.append(answers[i])
                task_true_answers.append(task['answers'])
        for metric in metrics:
            value = metric(task_true_answers, task_answers) or 0.0
            result[(metric.__name__, type_name)] = value
            if need_print:
                print('%s on %s: %.3f' % (metric.__name__, type_name, value))
        if need_print:
            print('---------')

    if not need_print:
        return result


def cross_validation(model_class, model_args, tasks, n_iter=5):
    metrics = []
    train_size = int(len(tasks) * 0.7)
    for i in range(n_iter):
        new_args = []
        for arg in model_args:
            arg_model = arg['model'](*arg.get('args', []), **arg.get('kwargs', {}))
            new_args.append(arg_model)
        model = model_class(*new_args)
        random.shuffle(tasks)
        train, test = tasks[:train_size], tasks[train_size:]
        model.train(train)
        print('evaluating.........')
        answers = model(test)
        metrics.append(evaluate_model_on_different_tasks(answers, test, need_print=False))
    final_metrics = {}
    for item in metrics:
        for elem in item:
            if elem not in final_metrics:
                final_metrics[elem] = 0
            final_metrics[elem] += item[elem] / n_iter
    return final_metrics


class Dummy(object):
    def fit(self, *args):
        pass

    def transform(self, X):
        return X


class TSNEWithTransform(TSNE):
    def transform(self, X):
        return self.fit_transform(X)
