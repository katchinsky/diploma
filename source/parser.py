import json
import os
import re

import tqdm

from .constants import QUESTION_TYPES


class Parser(object):
    def __init__(self, foldername):
        self.foldername = foldername
        self.tasks = []
        self.types = QUESTION_TYPES
        self.without_answer_cnt = 0
        self.all_tasks_cnt = 0
        self.with_image_cnt = 0
        self.with_table_cnt = 0

    def __call__(self, tasks=None):
        if not tasks:
            self.parse_folder()
        elif not isinstance(tasks, list):
            tasks = [tasks]
        if tasks:
            for elem in tqdm(tasks):
                task = self.preprocess(elem)
                if task:
                    self.tasks.append(task)
        return self.tasks

    @staticmethod
    def parse_task(task):
        is_number_options = len(re.findall(r'\d\)', task)) > 1
        is_letter_options = len(re.findall(r'[а-жa-z]\)', task)) > 1
        number_options = {}
        letter_options = {}
        task_description = ''
        if not is_number_options and not is_letter_options:
            question = task

            if 'процент' in question or 'вероятность' in question:
                question_type = 3
            elif re.search(r'запишите.*число', question):
                question_type = 5
            else:
                question_type = 0

        elif is_number_options and not is_letter_options:
            question_type = 1
            splitted_task = re.split(r'\d\)', task)
            question, number_options = splitted_task[0], splitted_task[1:]
        elif not is_number_options and is_letter_options:
            question_type = 1
            splitted_task = re.split(r'[а-жa-z]\)', task)
            question, letter_options = splitted_task[0], splitted_task[1:]
        else:
            question_type = 2
            first_number = task.find('1)')
            first_letter = task.find('a)') if 'a)' in task else task.find('а)')
            if first_number < first_letter:
                splitted_task = re.split(r'\d\)', task)
                question, number_options, letter_options = splitted_task[0], splitted_task[1:-1], splitted_task[-1]
                letter_options = re.split(r'[а-жa-z]\)', letter_options)
                number_options.append(letter_options[0])
                letter_options = letter_options[1:]
            else:
                splitted_task = re.split(r'[а-жa-z]\)', task)
                question, letter_options, number_options = splitted_task[0], splitted_task[1:-1], splitted_task[-1]
                number_options = re.split(r'\d\)', number_options)
                letter_options.append(number_options[0])
                number_options = number_options[1:]

        for i in range(len(number_options)):
            number_options[i] = number_options[i].split('\xa0')[0]
            number_options[i] = number_options[i].split('запишите в ответ')[0]
            number_options[i] = number_options[i].split('выбери')[0]

        for i in range(len(letter_options)):
            letter_options[i] = letter_options[i].split('\xa0')[0]
            letter_options[i] = letter_options[i].split('запишите в ответ')[0]
            letter_options[i] = letter_options[i].split('выбери')[0]

        options = {'number_options': list(filter(lambda x: len(x), map(lambda x: x.strip(), number_options))),
                   'letter_options': list(filter(lambda x: len(x), map(lambda x: x.strip(), letter_options)))}
        if options['letter_options'] and not options['number_options'] and question_type == 2:
            question_type = 1

        if question_type == 1 and re.search(r'(укажи|установи|расположи).*(последовательност|поряд)', question):
            question_type = 4
        return question, options, question_type

    def preprocess(self, task):
        self.all_tasks_cnt += 1
        if 'answer' not in task:
            self.without_answer_cnt += 1
            return None
        if task.get('image'):
            self.with_image_cnt += 1
            return None
        if 'таблиц' in task['text'].lower():
            self.with_table_cnt += 1
            return None
        answer = task['answer'].lower()
        answers = list(set(map(lambda x: x.strip(), answer[answer.find(':') + 1:].split('|'))))
        title = task['id'].lower()
        text = task['text'].lower()
        text = text[text.find(title) + len(title):]
        text = text[:text.find('пояснение')]
        text = text.replace('\xa0', ' ')
        question, options, question_type = self.parse_task(text)
        return {
            'question': question,
            'options': options,
            'type': question_type,
            'answers': answers
        }

    def parse_file(self, filename):
        with open(filename, 'r') as f:
            json_data = json.load(f)
        for raw_task in json_data.get('tasks', []):
            task = self.preprocess(raw_task)
            if task:
                self.tasks.append(task)

    def parse_folder(self):
        print(self.foldername)
        print(os.listdir(self.foldername))
        for filename in tqdm(os.listdir(self.foldername)):
            self.parse_file('{dirname}/{filename}'.format(dirname=self.foldername, filename=filename))
