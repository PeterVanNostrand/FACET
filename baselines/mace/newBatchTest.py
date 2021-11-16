import multiprocessing as mp
from baselines.mace.generateSATExplanations import genExp


def generateExplanationsWithMaxTime(maxTime,
                                    approach_string,
                                    explanation_file_name,
                                    model_trained,
                                    dataset_obj,
                                    factual_sample,
                                    norm_type_string,
                                    observable_data_dict,
                                    standard_deviations):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=generateExplanationsWithQueueCatch, args=(q,
                                                                     approach_string,
                                                                     explanation_file_name,
                                                                     model_trained,
                                                                     dataset_obj,
                                                                     factual_sample,
                                                                     norm_type_string,
                                                                     observable_data_dict,
                                                                     standard_deviations,)
                    )
    p.start()
    p.join(maxTime)
    if p.is_alive():
        p.terminate()
        print("killing after", maxTime, "second")
        return {
            'counterfactual_sample': None,
        }
    else:
        return q.get()


def generateExplanationsWithQueueCatch(queue,
                                       approach_string,
                                       explanation_file_name,
                                       model_trained,
                                       dataset_obj,
                                       factual_sample,
                                       norm_type_string,
                                       observable_data_dict,
                                       standard_deviations):
    try:
        generateExplanationsWithQueue(queue,
                                      approach_string,
                                      explanation_file_name,
                                      model_trained,
                                      dataset_obj,
                                      factual_sample,
                                      norm_type_string,
                                      observable_data_dict,
                                      standard_deviations)
    except:
        print("solver returned error for", approach_string)
        queue.put({
            'counterfactual_sample': None,
        })


def generateExplanationsWithQueue(queue,
                                  approach_string,
                                  explanation_file_name,
                                  model_trained,
                                  dataset_obj,
                                  factual_sample,
                                  norm_type_string,
                                  observable_data_dict,
                                  standard_deviations):
    queue.put(generateExplanations(
        approach_string,
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string,
        observable_data_dict,
        standard_deviations
    ))


def getEpsilonInString(approach_string):
    tmp_index = approach_string.find('eps')
    epsilon_string = approach_string[tmp_index + 4: tmp_index + 8]
    return float(epsilon_string)


def generateExplanations(
        approach_string,
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string,
        observable_data_dict,
        standard_deviations):

    # MACE Counterfactual
    return genExp(
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string,
        'mace',
        getEpsilonInString(approach_string)
    )
