'''
    This is the code for evaluating self-correction of LLM.
    Please pay attention to some key parameters, such as:
        model: 'gpt-3.5-turbo', 'gpt-3.5-turbo-1106', or 'gpt-4'.
        request_timeout: when time exceeds, the code will continue in order to avoid getting stuck.
        temperature: [0,2], showing the variation of the answer.
'''

# import openai 
#from openai import OpenAI
import ollama
import json 
import time 
import re 
import numpy as np 
import os 
from tqdm import tqdm


def read_data(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data

def save_result(messages, path):
    f = open(path, 'a+')
    json.dump(messages, f)
    f.write('\n')
    f.close()

def normalize_answer(ans):
    # ans = ans.lower()
    ans = ans.replace(',', '')
    ans = ans.replace('.', '')
    ans = ans.replace('?', '')
    ans = ans.replace('!', '')
    ans = ans.replace('\'', '')
    ans = ans.replace('\"', '')
    ans = ans.replace(';', '')
    ans = ans.replace(':', '')
    ans = ans.replace('-', '')
    ans = ans.replace('_', '')
    ans = ans.replace('(', '')
    ans = ans.replace(')', '')
    ans = ans.replace('[', '')
    ans = ans.replace(']', '')
    ans = ans.replace('{', '')
    ans = ans.replace('}', '')
    ans = ans.replace('/', '')
    ans = ans.replace('\\', '')
    ans = ans.replace('|', '')
    ans = ans.replace('<', '')
    ans = ans.replace('>', '')
    ans = ans.replace('=', '')
    ans = ans.replace('+', '')
    ans = ans.replace('*', '')
    ans = ans.replace('&', '')
    ans = ans.replace('^', '')
    ans = ans.replace('%', '')
    ans = ans.replace('$', '')
    # ans = ans.replace('#', '')
    ans = ans.replace('@', '')
    ans = ans.replace('~', '')
    ans = ans.replace('`', '')
    ans = ans.replace(' ', '')
    return ans

def get_answer_from_text(sentence):
    try:
        sentence = sentence.replace(',', '')     # To remove the punctuation in number, e.g., $2,000
        pattern = re.compile(r'##(.*?)##')
        ans = re.findall(pattern, sentence)
        if len(ans):
            ans = ans[-1]
            ans = normalize_answer(ans)
            try:
                ans = float(ans)
            except:
                ans = float(10086100100)
        else:
            ans = float(10086100100)
        return ans
    except:
        return float(10086100100)


def chat(messages, model_version):
    try:
        response = ollama.chat(
            model=model_version,
            messages=messages,
            temperature=0,
            #request_timeout=50,
        )

        text = response['message']['content']
    except:
        text = -1
    return text

def majority_vote(responses):
    votes = {}
    for response in responses:
        answer = get_answer_from_text(response)  
        if answer in votes:
            votes[answer].append(response) 
        else:
            votes[answer] = [response]

    max_votes_answer = max(votes, key=lambda x: len(votes[x]))
    return votes[max_votes_answer][0]


def Self_Consistency(prompt, model, num_trials):
    responses = []
    for _ in range(num_trials):
        response = chat(prompt, model)
        print('Think path:', _ + 1, 'Answer:', get_answer_from_text(response))
        responses.append(response)
    final_response = majority_vote(responses)
    return final_response

import random

def read_random_data(file, num_samples=5):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    random.shuffle(data)
    return data[:num_samples]

path = 1

def main(i, data, model):
    QAs = dict()
    QAs['index'] = i
    question = data['question']
    answer = float(data['answer'])
    extractor = " Your final answer should be put between two ##, like ## 1 ## (if your final answer is 1), at the end of your response."

    question = question + " Explain your reasoning step-by-step." + extractor
    QAs['Q1'] = {'role': 'user', 'content': question}
    messages=[{'role': 'user', 'content': question}]
    response_1 = Self_Consistency(messages, model, path)
    if response_1==-1:
        return -1
    QAs['A1'] = {'role': 'assistant', 'content':response_1}
    messages.append({'role': 'assistant', 'content':response_1})
    
    question = "Review your previous answer and find problems with your answer."
    QAs['Q2'] = {'role': 'user', 'content': question}
    messages.append({'role': 'user', 'content': question})
    response_2 = chat(messages, model)
    if response_2==-1:
        return -1
    QAs['A2'] = {'role': 'assistant', 'content':response_2}
    messages.append({'role': 'assistant', 'content':response_2})

    question = "Based on the problems you found, improve your answer. Please reiterate your answer." + extractor
    QAs['Q3'] = {'role': 'user', 'content': question}
    messages.append({'role': 'user', 'content': question})
    response_3 = Self_Consistency(messages, model, path)
    if response_3==-1:
        return -1
    QAs['A3'] = {'role': 'assistant', 'content':response_3}
    messages.append({'role': 'assistant', 'content':response_3})

    QAs['answer'] = answer
    QAs['P1_ans'] = get_answer_from_text(response_1)
    QAs['P2_ans'] = get_answer_from_text(response_2)
    QAs['P3_ans'] = get_answer_from_text(response_3)

    return QAs 


if __name__=='__main__':
    """
    Flag: 
        1: Run the dataset to collect responses from LLM. 
        2: Evaluate the results.
        3: Run and Evaluate.
    
    Dataset:
        GSM8K, SVAMP. 

    Model:
        'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', or 'gpt-4'.  
    """
    flag = 3

    dataset = 'SVAMP'
    model = "gpt-3.5-turbo-0613"
    input_dir = "dataset/"
    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_input = f'{input_dir}/{dataset}.jsonl'
    path_output = f'{output_dir}/{dataset}_{model}_{path}_baseline.jsonl'

    skip_list, skip_list2 = [], []  # We use "skip_list2" to save these unsolved questions and solve them again in the next loop.
    if flag==1 or flag==3:
        data = read_random_data(path_input)
        # data = read_data(path_input)
        # data = data[:20]
        skip_list2 = range(len(data))  
        print(f"data size: {len(skip_list2)}, output: {path_output}")
        while len(skip_list2)!=0:
            skip_list = []
            for i in tqdm(skip_list2): 
                start = time.time()
                messages = main(i, data[i], model)
                if messages==-1:
                    print(f"I={i}, Skip this round. Next one!")
                    skip_list.append(i)
                    continue 
                save_result(messages, path_output)
                end = time.time()
            print(f"Skip list: {skip_list}.")
            skip_list2 = skip_list 

    if flag==2 or flag==3:
        data_est = read_data(path_output)
        length = len(data_est)
        count_1 = 0 # The accuracy of standard prompt.
        count_2 = 0 # The accuracy of IoE prompt.
        count_3 = 0 # The accuracy of Refinement.
        for i in range(length): 
            if data_est[i]['answer']==data_est[i]['P1_ans']:
                count_1 += 1
            if data_est[i]['answer']==data_est[i]['P3_ans']:
                count_2 += 1

        print("Model:", model,"Dataset:", dataset, "Path:", path)
        print(f"The accuracy of standard Prompt: {count_1/length*100}.")
        print(f"The accuracy of critical prompt: {count_2/length*100}.") 
